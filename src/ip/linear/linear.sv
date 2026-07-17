`timescale 1ns / 1ps
//==============================================================================
// linear.sv — Core Datapath & Control FSM
//
// Computes S = Q × Kᵀ  (Scaled Dot-Product Attention Score)
//   K : [D_HEAD × D_MODEL]  — loaded into BRAM, preloaded into PE weights
//   Q : [SEQ_LEN × D_MODEL] — streamed in via AXI-Stream
//   S : [SEQ_LEN × D_HEAD]  — serialized out via AXI-Stream
//
// Key design decisions:
//   - Weight-Stationary: K loaded once per inference call
//   - Double-buffer ping-pong: serialize row N while MAC computes row N+1
//   - No mac_stall in normal flow; stall only on M_AXIS backpressure
//   - tdest removed (Option C): state machine alone distinguishes K vs Q phase
//   - K_RAM port widths fixed at 32-bit (matches BRAM IP config, depth=4096)
//   - K_RAM port B read latency = 2 cycles (pipeline delay chain in preload)
//==============================================================================

module linear #(
    parameter int D_MODEL    = 64,
    parameter int SEQ_LEN    = 64,
    parameter int DATA_WIDTH = 16,
    parameter int N_PE       = 64,
    parameter int D_HEAD     = 64
)(
    input  logic        iclk,
    input  logic        irst_n,

    // Control from AXI-Lite slave
    input  logic        i_start_attn_score,
    output logic        o_attn_score_done,
    output logic        o_busy,

    // AXI-Stream Slave (DMA → IP): K first, then Q
    input  logic [31:0] i_s_axis_tdata,
    input  logic        i_s_axis_tvalid,
    input  logic        i_s_axis_tlast,
    output logic        o_s_axis_tready,

    // AXI-Stream Master (IP → DMA): attention scores
    output logic [31:0] o_m_axis_tdata,
    output logic        o_m_axis_tvalid,
    output logic        o_m_axis_tlast,
    input  logic        i_m_axis_tready
);

    //--------------------------------------------------------------------------
    // Local parameters
    //--------------------------------------------------------------------------
    localparam int K_DEPTH    = D_HEAD * D_MODEL;       // BRAM entries used
    localparam int SQRT_SHIFT = $clog2(D_MODEL) / 2;   // approx divide by sqrt(D_MODEL)

    // N_TILES = ceil(D_HEAD / N_PE), elaboration-time integer division
    localparam int N_TILES = (D_HEAD + N_PE - 1) / N_PE;

    // Counter widths — guard against clog2(1)=0
    localparam int J_W = (D_HEAD  > 1) ? $clog2(D_HEAD)  : 1;
    localparam int K_W = (D_MODEL > 1) ? $clog2(D_MODEL) : 1;
    localparam int P_W = (N_PE    > 1) ? $clog2(N_PE)    : 1;
    localparam int S_W = (SEQ_LEN > 1) ? $clog2(SEQ_LEN) : 1;
    localparam int T_W = (N_TILES > 1) ? $clog2(N_TILES) : 1;

    //--------------------------------------------------------------------------
    // Synthesis-time validity assertions (HANDOFF_TILING.md §7)
    //--------------------------------------------------------------------------
    // synthesis translate_off
    initial begin
        assert (N_PE <= D_HEAD)
            else $fatal(1, "[linear] N_PE=%0d must be <= D_HEAD=%0d", N_PE, D_HEAD);
        assert (D_MODEL >= D_HEAD)
            else $fatal(1, "[linear] D_MODEL=%0d >= D_HEAD=%0d required for ping-pong safety", D_MODEL, D_HEAD);
        assert (N_PE * N_TILES >= D_HEAD)
            else $fatal(1, "[linear] N_TILES calculation error: N_PE=%0d * N_TILES=%0d < D_HEAD=%0d", N_PE, N_TILES, D_HEAD);
    end
    // synthesis translate_on

    //--------------------------------------------------------------------------
    // FSM
    //--------------------------------------------------------------------------
    typedef enum logic [2:0] {
        ST_IDLE        = 3'b000,
        ST_LOAD_K      = 3'b001,
        ST_PRELOAD_MAC = 3'b010,
        ST_COMPUTE     = 3'b011,
        ST_DONE        = 3'b100
    } state_t;
    state_t state;

    //--------------------------------------------------------------------------
    // Internal signals
    //--------------------------------------------------------------------------

    // K-RAM interface (port widths fixed at 32-bit per BRAM IP config)
    logic        k_ram_wea;
    logic [31:0] k_ram_addra;
    logic [31:0] k_ram_dina;
    logic [31:0] k_ram_addrb;
    logic [31:0] k_ram_doutb;

    // K load
    logic k_load_done;

    // Preload pipeline (2-cycle BRAM port-B latency compensation)
    // preload_row_j: LOCAL PE index within a tile, range 0..N_PE-1
    // preload_tile_idx: which tile bank being filled, range 0..N_TILES-1
    logic [P_W-1:0] preload_row_j;
    logic [K_W-1:0] preload_col_k;
    logic [T_W-1:0] preload_tile_idx;
    logic            preload_active;
    logic            preload_done;

    logic [P_W-1:0] preload_j_d1, preload_j_d2;
    logic [K_W-1:0] preload_k_d1, preload_k_d2;
    logic [T_W-1:0] preload_tile_d1, preload_tile_d2;
    logic            preload_en_d1, preload_en_d2;
    logic preload_last_addr;
    logic preload_frozen;


    // matmul_ip control
    logic            matmul_preload_en;
    logic [P_W-1:0]  matmul_preload_j;
    logic [K_W-1:0]  matmul_preload_k;
    logic [T_W-1:0]  matmul_preload_tile_sel;
    logic            matmul_data_valid;
    logic            matmul_acc_clear;
    logic [K_W-1:0]  matmul_k_index;
    logic [T_W-1:0]  matmul_tile_sel;

    // matmul_ip output
    logic                                   matmul_result_valid;
    logic signed [N_PE-1:0][DATA_WIDTH-1:0] matmul_result;
    logic [J_W-1:0]                         matmul_result_col_base; // = tile_idx*N_PE, delay-aligned to matmul_result_valid

    // Q row buffer — holds one full Q row (D_MODEL elements) for the
    // duration of the internal tile loop (step 2b), so Q does not need
    // to be re-streamed from AXI-Stream per tile.
    logic signed [DATA_WIDTH-1:0] q_row_buf [0:D_MODEL-1];
    logic [K_W-1:0] q_col_k;       // current k index within one Q row (load, step 2a)
    logic            q_transfer;    // Q beat accepted this cycle (step 2a only)

    // Tile-loop control (step 2b)
    logic [T_W-1:0] tile_idx;       // current tile being computed, 0..N_TILES-1
    logic [K_W-1:0] tile_k_cnt;     // MAC cycle index within current tile, 0..D_MODEL-1
    logic            compute_phase; // 0 = step 2a (load Q row), 1 = step 2b (tile loop)
    logic            tile_mac_en;   // drives matmul_data_valid during step 2b
    logic [T_W-1:0]  tile_result_cnt; // count of matmul_result_valid pulses this row

    // M_AXIS backpressure stall
    // Stall Q input only when downstream cannot accept (i_m_axis_tready=0)
    // and serializer is active
    logic serialize_active;
    logic axis_out_stall;           // serialize_active & ~i_m_axis_tready

    // Double ping-pong buffer
    // buffer[0] and buffer[1] alternate: one being written, one being read
    // Sized to D_HEAD (not N_PE) to hold a full tiled row across all tiles.
    logic signed [DATA_WIDTH-1:0] result_buffer [0:1][0:D_HEAD-1];
    logic        buf_write_sel;     // which buffer matmul writes into
    logic        buf_read_sel;      // which buffer serializer reads from

    // Serializer counters
    logic [J_W-1:0] ser_col_j;     // current column being serialized
    logic [S_W-1:0] ser_row_i;     // current row index

    // Done detection
    logic last_row_last_col;        // final handshake of final element
    logic compute_done;

    //--------------------------------------------------------------------------
    // BRAM instantiation
    //--------------------------------------------------------------------------
    k_ram u_k_ram (
        .clka  (iclk),
        .ena   (1'b1),
        .wea   (k_ram_wea),
        .addra (k_ram_addra),
        .dina  (k_ram_dina),
        .douta (),
        .clkb  (iclk),
        .enb   (preload_active),
        .web   (),
        .addrb (k_ram_addrb),
        .doutb (k_ram_doutb)
    );

    //--------------------------------------------------------------------------
    // matmul_ip instantiation
    //--------------------------------------------------------------------------
    matmul_ip #(
        .N_COLS    (D_HEAD),
        .D_MODEL   (D_MODEL),
        .N_PE      (N_PE),
        .DATA_WIDTH(DATA_WIDTH),
        .N_TILES   (N_TILES)
    ) u_matmul_ip (
        .i_clk               (iclk),
        .i_reset_n           (irst_n),

        .i_preload_en        (matmul_preload_en),
        .i_preload_j         (matmul_preload_j),
        .i_preload_k         (matmul_preload_k),
        .i_preload_data      (k_ram_doutb[DATA_WIDTH-1:0]),
        .i_preload_tile_sel  (matmul_preload_tile_sel),

        .i_data_valid        (matmul_data_valid),
        .i_acc_clear         (matmul_acc_clear),
        .i_k_index           (matmul_k_index),
        .i_a_data            (q_row_buf[tile_k_cnt]),
        .i_tile_sel          (matmul_tile_sel),
        .i_col_base          (J_W'(32'(tile_idx) * 32'(N_PE))),

        .o_result_valid      (matmul_result_valid),
        .o_result            (matmul_result),
        .o_result_col_base   (matmul_result_col_base)
    );

    //==========================================================================
    // FSM — state transitions only
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            state <= ST_IDLE;
        end else begin
            case (state)
                ST_IDLE:
                    if (i_start_attn_score)
                        state <= ST_LOAD_K;

                ST_LOAD_K:
                    if (k_load_done)
                        state <= ST_PRELOAD_MAC;

                ST_PRELOAD_MAC:
                    if (preload_done)
                        state <= ST_COMPUTE;

                ST_COMPUTE:
                    if (compute_done)
                        state <= ST_DONE;

                ST_DONE:
                    state <= ST_IDLE;

                default:
                    state <= ST_IDLE;
            endcase
        end
    end

    //==========================================================================
    // K-RAM write address counter (ST_LOAD_K)
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            k_ram_addra <= '0;
        end else begin
            if (k_ram_wea) begin
                if (i_s_axis_tlast || (k_ram_addra == 32'(K_DEPTH - 1)))
                    k_ram_addra <= '0;
                else
                    k_ram_addra <= k_ram_addra + 1;
            end
        end
    end

    //==========================================================================
    // Preload scan counters (ST_PRELOAD_MAC)
    // Scans K[tile_idx][j][k] for tile_idx=0..N_TILES-1, j=0..N_PE-1 (local
    // PE index), k=0..D_MODEL-1. Flat BRAM address = (tile_idx*N_PE + j)*D_MODEL + k.
    // Last tile may be partial (D_HEAD % N_PE != 0); scan still runs the full
    // N_PE range per handoff §5 — PE[j] for j >= (D_HEAD - tile_idx*N_PE) in
    // the last tile reads BRAM addresses beyond D_HEAD*D_MODEL-1 (garbage/
    // wrapped), but those PEs are masked out at result_buffer write time
    // (see combinational write-mask below) and never contribute to output.
    // 2-cycle delay chain aligns control with BRAM port-B output
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            preload_row_j    <= '0;
            preload_col_k    <= '0;
            preload_tile_idx <= '0;
            preload_frozen   <= 1'b0;
        end else begin
            // Reset khi re-enter ST_PRELOAD_MAC
            if (state == ST_LOAD_K && k_load_done) begin
                preload_row_j    <= '0;
                preload_col_k    <= '0;
                preload_tile_idx <= '0;
                preload_frozen   <= 1'b0;
            end else if (preload_active) begin
                if (preload_last_addr) begin
                    // Freeze counter — không increment nữa
                    preload_frozen <= 1'b1;
                end else if (!preload_frozen) begin
                    // Normal scan
                    if (preload_col_k == K_W'(D_MODEL - 1)) begin
                        preload_col_k <= '0;
                        if (preload_row_j == P_W'(N_PE - 1)) begin
                            preload_row_j    <= '0;
                            preload_tile_idx <= (preload_tile_idx == T_W'(N_TILES - 1))
                                            ? '0
                                            : preload_tile_idx + 1;
                        end else begin
                            preload_row_j <= preload_row_j + 1;
                        end
                    end else begin
                        preload_col_k <= preload_col_k + 1;
                    end
                end
            end else begin
                // Not in preload state — clear frozen flag
                preload_frozen <= 1'b0;
            end
        end
    end

    // 2-stage delay chain for BRAM port-B latency
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            preload_j_d1    <= '0; preload_j_d2    <= '0;
            preload_k_d1    <= '0; preload_k_d2    <= '0;
            preload_tile_d1 <= '0; preload_tile_d2 <= '0;
            preload_en_d1   <= '0; preload_en_d2   <= '0;
        end else begin
            preload_j_d1    <= preload_row_j;
            preload_k_d1    <= preload_col_k;
            preload_tile_d1 <= preload_tile_idx;
            preload_en_d1   <= preload_active;

            preload_j_d2    <= preload_j_d1;
            preload_k_d2    <= preload_k_d1;
            preload_tile_d2 <= preload_tile_d1;
            preload_en_d2   <= preload_en_d1;
        end
    end

    //==========================================================================
    // Q input column counter + q_row_buf capture (ST_COMPUTE, step 2a)
    // Tracks which k index within the current Q row; loads q_row_buf so
    // the tile loop (step 2b) can broadcast without re-streaming AXI-Stream.
    // Active only during compute_phase == 0 (see q_transfer combinational def).
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            q_col_k <= '0;
        end else begin
            if (q_transfer) begin
                q_row_buf[q_col_k] <= $signed(i_s_axis_tdata[DATA_WIDTH-1:0]);
                if (i_s_axis_tlast || (q_col_k == K_W'(D_MODEL - 1)))
                    q_col_k <= '0;
                else
                    q_col_k <= q_col_k + 1;
            end
        end
    end

    //==========================================================================
    // Tile-loop counters (ST_COMPUTE, step 2b)
    // tile_k_cnt: MAC cycle index within current tile, 0..D_MODEL-1
    // tile_idx:   which tile, 0..N_TILES-1. Advances with zero gap: resets
    //             tile_k_cnt to 0 the same cycle tile_idx increments.
    // compute_phase: 0 = step 2a (load Q row via AXI-Stream), 1 = step 2b
    //                (internal tile loop, tready deasserted)
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            compute_phase <= 1'b0;
            tile_idx      <= '0;
            tile_k_cnt    <= '0;
        end else begin
            if (state == ST_COMPUTE) begin
                if (!compute_phase) begin
                    // step 2a: Q row load complete -> enter tile loop
                    if (q_transfer &&
                        (i_s_axis_tlast || (q_col_k == K_W'(D_MODEL - 1)))) begin
                        compute_phase <= 1'b1;
                        tile_idx      <= '0;
                        tile_k_cnt    <= '0;
                    end
                end else begin
                    // step 2b: tile loop, zero-gap between tiles
                    if (tile_mac_en) begin
                        if (tile_k_cnt == K_W'(D_MODEL - 1)) begin
                            tile_k_cnt <= '0;
                            if (tile_idx == T_W'(N_TILES - 1)) begin
                                // last tile of this row done -> back to step 2a
                                // for next row (serializer picks up in parallel)
                                compute_phase <= 1'b0;
                                tile_idx      <= '0;
                            end else begin
                                tile_idx <= tile_idx + 1;
                            end
                        end else begin
                            tile_k_cnt <= tile_k_cnt + 1;
                        end
                    end
                end
            end else begin
                compute_phase <= 1'b0;
                tile_idx      <= '0;
                tile_k_cnt    <= '0;
            end
        end
    end

    //==========================================================================
    // tile_result_cnt — counts matmul_result_valid pulses within current row.
    // Serialize trigger fires only after N_TILES pulses (see combinational
    // serializer-start logic below), not after every single pulse.
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            tile_result_cnt <= '0;
        end else begin
            if (matmul_result_valid) begin
                tile_result_cnt <= (tile_result_cnt == T_W'(N_TILES - 1))
                                  ? '0
                                  : tile_result_cnt + 1;
            end
        end
    end

    //==========================================================================
    // Double ping-pong buffer — multi-tile write
    //
    // buf_write_sel: matmul writes result into buffer[buf_write_sel]
    // buf_read_sel:  serializer reads from buffer[buf_read_sel]
    //
    // On EVERY matmul_result_valid pulse (one per tile):
    //   1. Write up to N_PE results into buffer[buf_write_sel] at offset
    //      matmul_result_col_base (= tile_idx * N_PE). Last tile may be
    //      partial (D_HEAD % N_PE != 0) — PEs whose absolute column index
    //      (col_base + p) >= D_HEAD are masked (not written); they hold
    //      garbage/duplicate MAC results that never contributed to a valid
    //      K row (see preload note above) and must not reach result_buffer.
    //
    // Only on the pulse that completes the row (tile_result_cnt reaches
    // N_TILES-1, i.e. this is the N_TILES-th pulse of the row):
    //   2. Capture buf_read_sel = buf_write_sel (this buffer now ready to read)
    //   3. Flip buf_write_sel for next row
    //
    // Since serialize takes D_HEAD cycles and the full per-row tile loop
    // takes N_TILES * D_MODEL cycles, and N_TILES * D_MODEL >= D_HEAD
    // (guaranteed by D_MODEL >= D_HEAD >= N_PE and N_TILES >= 1), no
    // collision occurs between write and read buffers.
    //==========================================================================
    logic row_complete_pulse; // this matmul_result_valid pulse is the last tile of the row
    assign row_complete_pulse = matmul_result_valid
                              & (tile_result_cnt == T_W'(N_TILES - 1));

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            buf_write_sel <= 1'b0;
            buf_read_sel  <= 1'b0;
            for (int i = 0; i < D_HEAD; i++) begin
                result_buffer[0][i] <= '0;
                result_buffer[1][i] <= '0;
            end
        end else begin
            if (matmul_result_valid) begin
                // Write this tile's results into current write buffer,
                // masking PEs beyond D_HEAD on a partial last tile.
                for (int p = 0; p < N_PE; p++) begin
                    if ((32'(matmul_result_col_base) + p) < D_HEAD)
                        result_buffer[buf_write_sel][32'(matmul_result_col_base) + p] <= matmul_result[p];
                end
                if (row_complete_pulse) begin
                    // Serializer reads from the buffer just completed
                    buf_read_sel  <= buf_write_sel;
                    // Next row goes into the other buffer
                    buf_write_sel <= ~buf_write_sel;
                end
            end
        end
    end

    //==========================================================================
    // Serializer — output row counter and column counter
    //
    // serialize_active: high from matmul_result_valid until last element sent
    // ser_col_j: cycles 0..D_HEAD-1 per row
    // ser_row_i: increments after each row completes
    //
    // Backpressure: if axis_out_stall, counters freeze (valid stays high,
    // data stable) — compliant with AXI-Stream spec.
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            serialize_active <= 1'b0;
            ser_col_j        <= '0;
            ser_row_i        <= '0;
        end else begin
            // Start serializing only when the row is fully complete
            // (all N_TILES results written into result_buffer), not on
            // every single-tile matmul_result_valid pulse.
            if (row_complete_pulse) begin
                serialize_active <= 1'b1;
                ser_col_j        <= '0;
            end

            // Advance serializer only when downstream accepts
            if (serialize_active && i_m_axis_tready) begin
                if (ser_col_j == J_W'(D_HEAD - 1)) begin
                    // Last column of this row sent
                    ser_col_j <= '0;
                    if (ser_row_i == S_W'(SEQ_LEN - 1)) begin
                        // Last row of entire output — done
                        ser_row_i        <= '0;
                        serialize_active <= 1'b0;
                    end else begin
                        ser_row_i        <= ser_row_i + 1;
                        // Next row result will re-assert serialize_active
                        // via matmul_result_valid on next cycle
                        serialize_active <= 1'b0;
                    end
                end else begin
                    ser_col_j <= ser_col_j + 1;
                end
            end

            // Clear row counter on return to IDLE
            if (state == ST_DONE)
                ser_row_i <= '0;
        end
    end

    //==========================================================================
    // Combinational logic
    //==========================================================================

    // Status outputs
    assign o_busy            = (state != ST_IDLE);
    assign o_attn_score_done = (state == ST_DONE);

    // S_AXIS handshake
    // In ST_LOAD_K : accept unconditionally (K data)
    // In ST_COMPUTE: accept Q data only during step 2a (compute_phase==0),
    //   unless downstream stalled. tready is forced LOW throughout step 2b
    //   (compute_phase==1, internal tile loop) per handoff §5/§6 — no
    //   AXI-Stream transfer occurs during the tile loop.
    assign axis_out_stall  = serialize_active & ~i_m_axis_tready;
    assign o_s_axis_tready = (state == ST_LOAD_K)  ? 1'b1 :
                             (state == ST_COMPUTE)  ? (~compute_phase & ~axis_out_stall) :
                                                       1'b0;

    // K-RAM write path
    assign k_ram_wea   = i_s_axis_tvalid & o_s_axis_tready & (state == ST_LOAD_K);
    assign k_ram_dina  = i_s_axis_tdata;
    assign k_load_done = k_ram_wea & i_s_axis_tlast;

    // K-RAM read address (preload scan)
    // Flat address = (tile_idx * N_PE + local_j) * D_MODEL + k
    assign preload_active  = (state == ST_PRELOAD_MAC);
    assign k_ram_addrb     = preload_active
                           ? ((32'(preload_tile_idx) * 32'(N_PE) + 32'(preload_row_j)) * 32'(D_MODEL)
                              + 32'(preload_col_k))
                           : '0;
    assign preload_last_addr = preload_active
                         & (preload_row_j    == P_W'(N_PE    - 1))
                         & (preload_col_k    == K_W'(D_MODEL - 1))
                         & (preload_tile_idx == T_W'(N_TILES - 1));
    // Preload done: delayed last address (last tile, last local PE, last k)
    // has arrived at PE input
    assign preload_done = preload_en_d2
                        & (preload_tile_d2 == T_W'(N_TILES - 1))
                        & (preload_j_d2    == P_W'(N_PE    - 1))
                        & (preload_k_d2    == K_W'(D_MODEL - 1));

    // matmul_ip preload interface (delayed by 2 cycles for BRAM latency)
    assign matmul_preload_en       = preload_en_d2;
    assign matmul_preload_j        = preload_j_d2;
    assign matmul_preload_k        = preload_k_d2;
    assign matmul_preload_tile_sel = preload_tile_d2;

    // Q transfer (step 2a only — load one Q row into q_row_buf)
    assign q_transfer = (state == ST_COMPUTE)
                      & (~compute_phase)
                      & i_s_axis_tvalid
                      & o_s_axis_tready;

    // Tile-loop MAC enable (step 2b only — broadcast q_row_buf, no AXI-Stream)
    assign tile_mac_en = (state == ST_COMPUTE) & compute_phase;

    // matmul_ip compute interface: driven by tile loop during step 2b only
    assign matmul_data_valid = tile_mac_en;
    assign matmul_k_index    = tile_k_cnt;
    assign matmul_acc_clear  = tile_mac_en & (tile_k_cnt == '0);
    assign matmul_tile_sel   = tile_idx;

    // M_AXIS output
    assign o_m_axis_tvalid = serialize_active;
    assign o_m_axis_tlast = serialize_active
                              & (ser_col_j == J_W'(D_HEAD - 1))
                              & (ser_row_i == S_W'(SEQ_LEN - 1));
    assign o_m_axis_tdata  = 32'(($signed(result_buffer[buf_read_sel][ser_col_j])) >>> SQRT_SHIFT);

    // ST_COMPUTE exit condition:
    // Done after the last element of the last row has been accepted by downstream
    assign last_row_last_col = o_m_axis_tvalid
                             & o_m_axis_tlast
                             & i_m_axis_tready
                             & (ser_row_i == S_W'(SEQ_LEN - 1));

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n)
            compute_done <= 1'b0;
        else
            compute_done <= last_row_last_col;
    end

endmodule


