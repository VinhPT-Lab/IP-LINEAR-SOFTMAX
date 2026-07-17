`timescale 1ns / 1ps
//==============================================================================
// softmax.sv - Core Datapath & Control FSM
//
// Computes row-wise Softmax(S) for S : [SEQ_LEN x D_HEAD]
//   Input  : SEQ_LEN rows streamed in via AXI-Stream, D_HEAD beats/row
//   Output : SEQ_LEN rows streamed out via AXI-Stream, D_HEAD beats/row,
//            Q1.15 unsigned fraction (attention weights, each in [0,1))
//
// Reused directly from attention_top.sv (old monolithic design):
//   - exp_rom: Vivado Block Memory Generator IP, 2048 entries, LUT of
//     exp(-z), synchronous read (1-cycle latency).
//     addr = (-(x - max)) & 0x7FF  (clamped to 0 when x - max > 0, since
//     z = x - max is always <= 0 by construction)
//
// Key design decisions:
//   - Hướng B: row buffer bằng register array (giống q_row_buf trong
//     linear.sv). D_HEAD (mặc định 64) phần tử x 16-bit = 128 byte ->
//     distributed RAM, không cần BRAM IP cho row buffer.
//   - tready = 1 chỉ trong ST_LOAD_ROW. Mọi state xử lý nội bộ khác
//     (FIND_MAX / EXP_SUM / DIVIDE) đều tready = 0.
//   - Pass 3 (Divide) dùng reciprocal-LUT + multiplier (reciprocal_divider.sv),
//     thay vì toán tử "/" behavioral hoặc div_gen blackbox.
//   - SUM_WIDTH = clog2(D_HEAD) + EXP_WIDTH, tổng quát hoá theo D_HEAD
//     (bản cũ hardcode 24-bit, chỉ đúng cho SEQ_LEN=3 nhỏ).
//==============================================================================

module softmax #(
    parameter int D_HEAD          = 64,
    parameter int SEQ_LEN         = 64,
    parameter int DATA_WIDTH      = 16,    // input S element width (signed)
    parameter int EXP_WIDTH       = 16,    // exp_rom output width (unsigned)
    parameter int RECIP_ADDR_W    = 12,    // reciprocal ROM address width (mantissa bits). 4096-entry ROM,
                                            // đo được max 1 LSB sai số trên thang Q1.15 (xem recip_lut_check.py).
                                            // KHÔNG phụ thuộc D_HEAD/SEQ_LEN -- không cần đổi khi customize IP.
    parameter int RECIP_OUT_W     = 19     // reciprocal ROM output width (Q0.RECIP_OUT_W unsigned).

)(
    input  logic        iclk,
    input  logic        irst_n,

    // Control from AXI-Lite slave
    input  logic        i_start_softmax,
    output logic        o_softmax_done,
    output logic        o_busy,

    // AXI-Stream Slave (DMA -> IP): S rows in, D_HEAD beats/row, SEQ_LEN rows
    input  logic [31:0] i_s_axis_tdata,
    input  logic        i_s_axis_tvalid,
    input  logic        i_s_axis_tlast,
    output logic        o_s_axis_tready,

    // AXI-Stream Master (IP -> DMA): softmax output rows, Q1.15 unsigned
    output logic [31:0] o_m_axis_tdata,
    output logic        o_m_axis_tvalid,
    output logic        o_m_axis_tlast,
    input  logic        i_m_axis_tready
);

    //--------------------------------------------------------------------------
    // Local parameters
    //--------------------------------------------------------------------------
    // SUM_WIDTH: tổng D_HEAD giá trị exp_rom (EXP_WIDTH-bit unsigned) không
    // tràn: worst case tất cả D_HEAD phần tử đều = max exp value (2^EXP_WIDTH-1).
    localparam int SUM_WIDTH = $clog2(D_HEAD > 1 ? D_HEAD : 2) + EXP_WIDTH;

    // Dividend for reciprocal_divider: exp_row_buf[i] shifted left 15 -> Q1.15 numerator
    // scaled into the integer domain so quotient (numer/sum) lands directly
    // as a Q1.15 fraction in the low EXP_WIDTH bits of the result.
    localparam int DIVIDEND_WIDTH = EXP_WIDTH + 15;

    // Counter widths - guard against clog2(1)=0
    localparam int J_W = (D_HEAD  > 1) ? $clog2(D_HEAD)  : 1;
    localparam int S_W = (SEQ_LEN > 1) ? $clog2(SEQ_LEN) : 1;

    // reciprocal_divider tự quản lý toàn bộ bit-width nội bộ (multiply +
    // shift tường minh trong RTL), không cần các hằng số "measured" như
    // div_gen cũ (DIV_FRAC_BASE, DIV_RAW_WIDTH...). Không có localparam
    // nào ở đây cần theo dõi/re-đo khi D_HEAD hay EXP_WIDTH thay đổi.


    //--------------------------------------------------------------------------
    // Synthesis-time validity assertions
    //--------------------------------------------------------------------------
    // synthesis translate_off
    initial begin
        assert (D_HEAD >= 1)
            else $fatal(1, "[softmax] D_HEAD=%0d must be >= 1", D_HEAD);
        assert (SEQ_LEN >= 1)
            else $fatal(1, "[softmax] SEQ_LEN=%0d must be >= 1", SEQ_LEN);
        assert (RECIP_OUT_W > EXP_WIDTH)
            else $fatal(1, "[softmax] RECIP_OUT_W=%0d must be > EXP_WIDTH=%0d for adequate reciprocal precision",
                        RECIP_OUT_W, EXP_WIDTH);
    end
    // synthesis translate_on

    //--------------------------------------------------------------------------
    // FSM
    //--------------------------------------------------------------------------
    typedef enum logic [3:0] {
        ST_IDLE      = 4'd0,
        ST_LOAD_ROW  = 4'd1,
        ST_FIND_MAX  = 4'd2,
        ST_EXP_SUM   = 4'd3,
        ST_DIV_ISSUE = 4'd4,   // issue D_HEAD dividend/divisor beats into reciprocal_divider
        ST_DIV_DRAIN = 4'd5,   // wait for remaining pipelined results to return
        ST_SERIALIZE = 4'd6,
        ST_DONE      = 4'd7
    } state_t;
    state_t state;

    //--------------------------------------------------------------------------
    // Internal signals
    //--------------------------------------------------------------------------

    // Row buffers (Hướng B - register array, distributed RAM)
    logic signed [DATA_WIDTH-1:0] s_row_buf   [0:D_HEAD-1]; // input row
    logic        [EXP_WIDTH-1:0]  exp_row_buf [0:D_HEAD-1]; // exp(x-max) per element
    logic        [EXP_WIDTH-1:0]  out_row_buf [0:D_HEAD-1]; // Q1.15 divide result

    // ST_LOAD_ROW
    logic [J_W-1:0] load_col;
    logic           row_transfer;
    logic           row_load_done;

    // ST_FIND_MAX - one compare per cycle, index 0..D_HEAD-1
    logic signed [DATA_WIDTH-1:0] max_val;
    logic [J_W-1:0]               findmax_idx;
    logic                         findmax_done;

    // ST_EXP_SUM - 2-stage pipeline: address-scan stage -> ROM 1-cycle
    // latency -> accumulate stage. exp_rom_addr driven combinationally from
    // scan_idx; exp_rom_data (1 cycle later) corresponds to rom_idx_q.
    logic [J_W-1:0]        scan_idx;      // index being addressed this cycle
    logic                  scan_active;   // scan_idx valid this cycle
    logic [J_W-1:0]        rom_idx_q;     // index whose ROM data returns THIS cycle
    logic                  rom_valid_q;   // rom_idx_q / exp_rom_data valid this cycle
    logic signed [DATA_WIDTH-1:0] z_val;
    logic [10:0]           exp_rom_addr;
    logic [EXP_WIDTH-1:0]  exp_rom_data;
    logic [SUM_WIDTH-1:0]  sum_acc;
    logic [SUM_WIDTH-1:0]  sum_latched;
    logic                  expsum_done;

    // ST_DIV_ISSUE / ST_DIV_DRAIN - reciprocal_divider, fixed 3-cycle
    // latency pipeline, in-order return (xem reciprocal_divider.sv).
    logic [DIVIDEND_WIDTH-1:0]            div_dividend;
    logic [SUM_WIDTH-1:0]                 div_divisor;
    logic                                 div_tvalid_out;
    logic [EXP_WIDTH-1:0]                 div_result;   // Q1.15 unsigned result trực tiếp, không cần bit-slice thủ công
    logic [J_W-1:0]                       div_in_idx;
    logic                                 div_issue_active;
    logic [J_W-1:0]                       div_out_idx;
    logic                                 div_all_returned;

    // Serializer (ST_SERIALIZE)
    logic [J_W-1:0] ser_col_j;
    logic [S_W-1:0] ser_row_i;
    logic           serialize_active;
    logic           axis_out_stall;
    logic           row_serialize_done;

    // Done detection (whole frame)
    logic last_row_last_col;
    logic compute_done;

    //--------------------------------------------------------------------------
    // exp_rom instantiation (reused directly from attention_top.sv)
    // Vivado Block Memory Generator IP, single-port ROM, 2048 entries,
    // synchronous read (1-cycle latency)
    //--------------------------------------------------------------------------
    always_comb begin
        if (z_val <= 0)
            exp_rom_addr = 11'(32'(-32'(z_val)) & 32'h7FF);
        else
            exp_rom_addr = 11'h000;
    end

    assign z_val = scan_active ? (s_row_buf[scan_idx] - max_val) : '0;

    exp_rom u_exp_rom (
        .clka  (iclk),
        .ena   (1'b1),
        .addra (exp_rom_addr),
        .douta (exp_rom_data)
    );

    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // reciprocal_divider instantiation - thay thế Xilinx div_gen blackbox.
    // Toàn bộ bit-width/bit-position tự suy ra từ tham số RTL tường minh
    // (xem reciprocal_divider.sv), không có "measured constant" nào cần
    // re-đo khi D_HEAD/SEQ_LEN/EXP_WIDTH thay đổi qua Customize IP.
    // Latency cố định 3 cycle (so với DIV_LATENCY=55 của div_gen cũ).
    //--------------------------------------------------------------------------
    reciprocal_divider #(
        .DIVIDEND_WIDTH (DIVIDEND_WIDTH),
        .DIVISOR_WIDTH  (SUM_WIDTH),
        .EXP_WIDTH      (EXP_WIDTH),
        .ADDR_W         (RECIP_ADDR_W),
        .OUT_W          (RECIP_OUT_W)
    ) u_softmax_recip_div (
        .iclk      (iclk),
        .irst_n    (irst_n),

        .i_tvalid  (div_issue_active),
        .i_dividend(div_dividend),
        .i_divisor (div_divisor),

        .o_tvalid  (div_tvalid_out),
        .o_result  (div_result)
    );

    //==========================================================================
    // FSM - state transitions only
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            state <= ST_IDLE;
        end else begin
            case (state)
                ST_IDLE:
                    if (i_start_softmax)
                        state <= ST_LOAD_ROW;

                ST_LOAD_ROW:
                    if (row_load_done)
                        state <= ST_FIND_MAX;

                ST_FIND_MAX:
                    if (findmax_done)
                        state <= ST_EXP_SUM;

                ST_EXP_SUM:
                    if (expsum_done)
                        state <= ST_DIV_ISSUE;

                ST_DIV_ISSUE:
                    // last dividend/divisor beat issued this cycle
                    if (div_in_idx == J_W'(D_HEAD - 1))
                        state <= ST_DIV_DRAIN;

                ST_DIV_DRAIN:
                    if (div_all_returned)
                        state <= ST_SERIALIZE;

                ST_SERIALIZE:
                    if (row_serialize_done)
                        state <= (ser_row_i == S_W'(SEQ_LEN - 1)) ? ST_DONE : ST_LOAD_ROW;

                ST_DONE:
                    state <= ST_IDLE;

                default:
                    state <= ST_IDLE;
            endcase
        end
    end

    //==========================================================================
    // ST_LOAD_ROW - tready=1, capture D_HEAD beats into s_row_buf
    //==========================================================================
    assign row_transfer = (state == ST_LOAD_ROW) & i_s_axis_tvalid & o_s_axis_tready;

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            load_col      <= '0;
            row_load_done <= 1'b0;
        end else begin
            row_load_done <= 1'b0;
            if (state != ST_LOAD_ROW) begin
                load_col <= '0;
            end else if (row_transfer) begin
                s_row_buf[load_col] <= $signed(i_s_axis_tdata[DATA_WIDTH-1:0]);
                if (i_s_axis_tlast || (load_col == J_W'(D_HEAD - 1))) begin
                    load_col      <= '0;
                    row_load_done <= 1'b1;
                end else begin
                    load_col <= load_col + 1;
                end
            end
        end
    end

    //==========================================================================
    // ST_FIND_MAX - scan s_row_buf[0..D_HEAD-1], running max, 1 elem/cycle
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            max_val      <= {1'b1, {(DATA_WIDTH-1){1'b0}}}; // most negative value
            findmax_idx  <= '0;
            findmax_done <= 1'b0;
        end else begin
            findmax_done <= 1'b0;
            if (state == ST_LOAD_ROW && row_load_done) begin
                // Prime the scan for the row just loaded
                max_val     <= {1'b1, {(DATA_WIDTH-1){1'b0}}};
                findmax_idx <= '0;
            end else if (state == ST_FIND_MAX) begin
                if (s_row_buf[findmax_idx] > max_val)
                    max_val <= s_row_buf[findmax_idx];

                if (findmax_idx == J_W'(D_HEAD - 1)) begin
                    findmax_idx  <= '0;
                    findmax_done <= 1'b1;
                end else begin
                    findmax_idx <= findmax_idx + 1;
                end
            end
        end
    end

    //==========================================================================
    // ST_EXP_SUM - scan s_row_buf again, lookup exp_rom(-(x-max)), accumulate
    // sum, write exp_row_buf.
    //
    // Timing: scan_idx advances every cycle while scan_active (issues
    // D_HEAD addresses, cycles 0..D_HEAD-1). exp_rom has 1-cycle synchronous
    // read latency, so exp_rom_data for scan_idx=N appears the cycle AFTER
    // scan_idx=N was presented -- captured via rom_idx_q/rom_valid_q, a
    // 1-cycle delayed copy of scan_idx/scan_active. This mirrors the
    // WAIT_ROM state of the original softmax_exp_sum module but pipelined
    // (1 elem/cycle instead of the old 3-cycle-per-elem FSM).
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            scan_idx    <= '0;
            scan_active <= 1'b0;
            rom_idx_q   <= '0;
            rom_valid_q <= 1'b0;
            sum_acc     <= '0;
            sum_latched <= '0;
            expsum_done <= 1'b0;
        end else begin
            expsum_done <= 1'b0;

            if (state == ST_FIND_MAX && findmax_done) begin
                // Entering ST_EXP_SUM next cycle: prime scan + accumulator
                scan_idx    <= '0;
                scan_active <= 1'b1;
                rom_valid_q <= 1'b0;
                sum_acc     <= '0;
            end else if (state == ST_EXP_SUM) begin
                // Advance address scan (stops issuing after last index)
                if (scan_active) begin
                    if (scan_idx == J_W'(D_HEAD - 1)) begin
                        scan_active <= 1'b0;
                    end else begin
                        scan_idx <= scan_idx + 1;
                    end
                end

                // Delay chain: rom_idx_q/rom_valid_q trail scan_idx/scan_active
                // by exactly 1 cycle, matching exp_rom's read latency
                rom_idx_q   <= scan_idx;
                rom_valid_q <= scan_active;

                // Accumulate stage: exp_rom_data valid this cycle corresponds
                // to rom_idx_q (address issued last cycle)
                if (rom_valid_q) begin
                    exp_row_buf[rom_idx_q] <= exp_rom_data;
                    sum_acc <= sum_acc + SUM_WIDTH'(exp_rom_data);

                    if (rom_idx_q == J_W'(D_HEAD - 1)) begin
                        sum_latched <= sum_acc + SUM_WIDTH'(exp_rom_data);
                        expsum_done <= 1'b1;
                    end
                end
            end
        end
    end

    //==========================================================================
    // ST_DIV_ISSUE / ST_DIV_DRAIN - issue D_HEAD reciprocal-divides through
    // reciprocal_divider (3-cycle fixed latency), collect results into
    // out_row_buf.
    //
    // Dividend = exp_row_buf[i] << 15  (Q1.15 numerator, DIVIDEND_WIDTH bits)
    // Divisor  = sum_latched           (SUM_WIDTH bits, same denominator for
    //                                    the whole row)
    // reciprocal_divider trả thẳng kết quả Q1.15 unsigned qua o_result,
    // không cần bit-slice thủ công như div_gen cũ (không có FRACTIONAL
    // field / quotient field nào phải tự suy luận vị trí).
    //
    // Pipeline fixed-latency, nhận 1 beat/cycle, trả kết quả cùng thứ tự
    // (in-order) sau đúng số cycle cố định -- div_in_idx (issue) và
    // div_out_idx (collect) không cần giao tiếp trực tiếp; div_out_idx chỉ
    // đếm số lần o_tvalid pulse theo đúng thứ tự.
    //==========================================================================
    assign div_dividend  = {exp_row_buf[div_in_idx], 15'd0};
    assign div_divisor   = sum_latched;

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            div_in_idx       <= '0;
            div_issue_active <= 1'b0;
        end else begin
            if (state == ST_EXP_SUM && expsum_done) begin
                // Entering ST_DIV_ISSUE next cycle
                div_in_idx       <= '0;
                div_issue_active <= 1'b1;
            end else if (state == ST_DIV_ISSUE && div_issue_active) begin
                if (div_in_idx == J_W'(D_HEAD - 1)) begin
                    div_issue_active <= 1'b0;
                end else begin
                    div_in_idx <= div_in_idx + 1;
                end
            end
        end
    end

    // Collect divider outputs - fixed-latency pipeline returns in order
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            div_out_idx      <= '0;
            div_all_returned <= 1'b0;
        end else begin
            div_all_returned <= 1'b0;
            if (state == ST_EXP_SUM && expsum_done) begin
                div_out_idx <= '0;
            end else if (div_tvalid_out) begin
                // reciprocal_divider trả thẳng kết quả Q1.15 unsigned qua
                // o_result (div_result) -- không cần bit-slice thủ công như
                // div_gen cũ (không có QUOTIENT/FRACTIONAL field ẩn nào cần
                // định vị bằng hằng số đo đạc).
                out_row_buf[div_out_idx] <= div_result;
                
                if (div_out_idx == J_W'(D_HEAD - 1)) begin
                    div_out_idx      <= '0;
                    div_all_returned <= 1'b1;
                end else begin
                    div_out_idx <= div_out_idx + 1;
                end
            end
        end
    end

    //==========================================================================
    // Serializer - stream out_row_buf via M_AXIS, then loop to next row
    // (or finish frame). Backpressure: counters freeze while
    // i_m_axis_tready=0, valid/data stay stable - AXI-Stream compliant.
    //==========================================================================
    assign axis_out_stall = serialize_active & ~i_m_axis_tready;

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            serialize_active <= 1'b0;
            ser_col_j        <= '0;
            ser_row_i        <= '0;
        end else begin
            if (state == ST_DIV_DRAIN && div_all_returned) begin
                serialize_active <= 1'b1;
                ser_col_j        <= '0;
            end

            if (serialize_active && i_m_axis_tready) begin
                if (ser_col_j == J_W'(D_HEAD - 1)) begin
                    ser_col_j        <= '0;
                    serialize_active <= 1'b0;
                    if (ser_row_i == S_W'(SEQ_LEN - 1))
                        ser_row_i <= '0;
                    else
                        ser_row_i <= ser_row_i + 1;
                end else begin
                    ser_col_j <= ser_col_j + 1;
                end
            end

            if (state == ST_DONE)
                ser_row_i <= '0;
        end
    end

    assign row_serialize_done = serialize_active
                              & (ser_col_j == J_W'(D_HEAD - 1))
                              & i_m_axis_tready;

    //==========================================================================
    // Combinational logic
    //==========================================================================

    // Status outputs
    assign o_busy         = (state != ST_IDLE);
    assign o_softmax_done = (state == ST_DONE);

    // S_AXIS handshake: only accept during ST_LOAD_ROW
    assign o_s_axis_tready = (state == ST_LOAD_ROW);

    // M_AXIS output
    assign o_m_axis_tvalid = serialize_active;
    assign o_m_axis_tlast  = serialize_active
                            & (ser_col_j == J_W'(D_HEAD - 1))
                            & (ser_row_i == S_W'(SEQ_LEN - 1));
    assign o_m_axis_tdata  = 32'(out_row_buf[ser_col_j]);

    // Whole-frame done condition (last element of last row accepted downstream)
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

`timescale 1ns / 1ps
//==============================================================================
// reciprocal_divider.sv
//
// Thay thế Xilinx Divider Generator (div_gen) trong softmax.sv bằng phương
// pháp reciprocal-LUT + multiplier. Toàn bộ bit-width/bit-position do RTL
// này định nghĩa tường minh -- không có "measured constant" nào phụ thuộc
// vào cấu hình ẩn của 1 IP blackbox, nên thay đổi D_HEAD/SEQ_LEN/EXP_WIDTH
// KHÔNG đòi hỏi phải re-generate hay re-đo lại bất kỳ thứ gì bên ngoài.
//
// Thuật toán (range-reduction reciprocal, chuẩn fixed-point):
//   1. Tìm vị trí bit MSB=1 cao nhất của divisor (sum), gọi là msb_pos.
//      => divisor = mantissa * 2^msb_pos, với mantissa trong [1.0, 2.0)
//         (bit MSB=1 luôn ngầm định, không cần lưu trong ROM).
//   2. Lấy ADDR_W bit tiếp theo (ngay dưới MSB) làm địa chỉ ROM.
//   3. ROM (Vivado Block Memory Generator IP, customize từ recip_rom.coe
//      sinh bởi golden_model.py -- CÙNG FLOW với exp_rom hiện có) chứa
//      sẵn reciprocal(mantissa) dạng Q0.OUT_W không dấu.
//   4. weight = (dividend * recip_lut[addr]) >> (OUT_W + msb_pos - 15)
//      (dịch phải để bù lại phần "* 2^-msb_pos" và scale Q1.15 mong muốn).
//
// Độ chính xác: với ADDR_W=12 (ROM 4096 entry, ~ kích thước exp_rom hiện
// có), sai số tối đa đo được so với golden integer-division là 1 LSB trên
// thang Q1.15, verify bằng recip_lut_check.py trên 200k+ mẫu ngẫu nhiên
// D_HEAD in {16,64,128}.
//
// Latency: 3 cycle cố định (find-MSB -> ROM read -> multiply+shift), so
// với DIV_LATENCY=55 cycle của div_gen cũ -- nhanh hơn đáng kể, và không
// cần drain-pipeline phức tạp vì latency ngắn, có thể xử lý tuần tự đơn
// giản (không cần FIFO in-order return riêng).
//==============================================================================
module reciprocal_divider #(
    parameter int DIVIDEND_WIDTH = 31,   // = EXP_WIDTH + 15 (Q1.15 numerator)
    parameter int DIVISOR_WIDTH  = 22,   // = clog2(D_HEAD) + EXP_WIDTH (sum width)
    parameter int EXP_WIDTH      = 16,   // output width (Q1.15 unsigned result)
    parameter int ADDR_W         = 12,   // recip ROM address width (mantissa bits)
    parameter int OUT_W          = 19    // recip ROM output width (Q0.OUT_W)
)(
    input  logic                          iclk,
    input  logic                          irst_n,

    input  logic                          i_tvalid,
    input  logic [DIVIDEND_WIDTH-1:0]     i_dividend,   // exp_row_buf[i] << 15
    input  logic [DIVISOR_WIDTH-1:0]      i_divisor,    // sum_latched (same for whole row)

    output logic                          o_tvalid,
    output logic [EXP_WIDTH-1:0]          o_result      // Q1.15 unsigned quotient
);

    // MSB_W: đủ bit để biểu diễn vị trí bit cao nhất (0..DIVISOR_WIDTH-1)
    localparam int MSB_W = (DIVISOR_WIDTH > 1) ? $clog2(DIVISOR_WIDTH) : 1;

    //--------------------------------------------------------------------
    // Stage 0 (combinational): tìm vị trí bit MSB=1 cao nhất của divisor
    // bằng priority-encoder tổ hợp (divisor > 0 luôn đúng theo construction
    // của softmax, vì sum_latched là tổng >=1 phần tử exp_rom_data > 0).
    //--------------------------------------------------------------------
    logic [MSB_W-1:0] msb_pos_c;
    logic             found_c;
    always_comb begin
        msb_pos_c = '0;
        found_c   = 1'b0;
        for (int b = DIVISOR_WIDTH-1; b >= 0; b--) begin
            if (!found_c && i_divisor[b]) begin
                msb_pos_c = MSB_W'(b);
                found_c   = 1'b1;
            end
        end
    end

    // Địa chỉ ROM = ADDR_W bit ngay dưới MSB (mantissa fractional bits).
    // Nếu msb_pos < ADDR_W (divisor nhỏ, ít bit ý nghĩa hơn ADDR_W), dịch
    // trái để bù (zero-pad phần LSB còn thiếu) thay vì tràn âm.
    logic [ADDR_W-1:0] recip_addr_c;
    always_comb begin
        if (int'(msb_pos_c) >= ADDR_W)
            recip_addr_c = i_divisor[int'(msb_pos_c)-1 -: ADDR_W];
        else
            recip_addr_c = ADDR_W'(i_divisor) << (ADDR_W - int'(msb_pos_c));
    end

    //--------------------------------------------------------------------
    // Stage 1 (registered): latch dividend/msb_pos, đọc ROM (1-cycle sync
    // read, giống exp_rom).
    //
    // recip_rom = Vivado Block Memory Generator IP (KHÔNG phải module hành
    // vi tay) -- customize giống hệt cách tạo exp_rom:
    //   Width  = OUT_W (19), Depth = 2^ADDR_W (4096)
    //   Load Init File = recip_rom.coe (sinh bởi golden_model.py, hàm
    //   generate_recip_lut()/write_coe_generic())
    //   Port A: Primitives Output Register ON, Core Output Register OFF
    //   (latency=1, đúng bằng exp_rom hiện có -- xác nhận trong Summary
    //   tab của IP customize dialog trước khi generate).
    // Nếu ADDR_W/OUT_W tham số đổi, phải re-customize IP này lại cho khớp
    // (Width/Depth) và re-run golden_model.py để sinh lại recip_rom.coe.
    //--------------------------------------------------------------------
    logic                          s1_valid;
    logic [DIVIDEND_WIDTH-1:0]     s1_dividend;
    logic [MSB_W-1:0]              s1_msb_pos;
    logic [OUT_W-1:0]              recip_rom_data;

    recip_rom u_recip_rom (
        .clka  (iclk),
        .ena   (1'b1),
        .addra (recip_addr_c),
        .douta (recip_rom_data)
    );

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            s1_valid    <= 1'b0;
            s1_dividend <= '0;
            s1_msb_pos  <= '0;
        end else begin
            s1_valid    <= i_tvalid;
            s1_dividend <= i_dividend;
            s1_msb_pos  <= msb_pos_c;
        end
    end

    //--------------------------------------------------------------------
    // Stage 2 (registered): multiply dividend * reciprocal, tính shift
    // tổng quát = OUT_W + msb_pos - 15 (dẫn xuất trực tiếp từ công thức
    // toán học, không phải hằng số đo đạc).
    //--------------------------------------------------------------------
    localparam int PROD_WIDTH = DIVIDEND_WIDTH + OUT_W;
    logic [PROD_WIDTH-1:0] prod_c;
    assign prod_c = s1_dividend * recip_rom_data;

    // shift_total = OUT_W + msb_pos (KHÔNG trừ 15 nữa, vì i_dividend đầu vào
    // đã là "exp_val << 15" (Q1.15 numerator) sẵn, không phải exp_val trần -
    // phần "-15" đã tự triệt tiêu với "<<15" có sẵn trong dividend. Suy ra:
    //   weight = exp_val * 2^15 / sum
    //          = exp_val * 2^15 * recip_q / (2^OUT_W * 2^msb_pos)
    //          = (exp_val << 15) * recip_q >> (OUT_W + msb_pos)
    //          = i_dividend * recip_q >> (OUT_W + msb_pos)
    // (bug trước: trừ nhầm 15 lần nữa dù dividend đã pre-shift, khiến kết
    // quả lớn hơn đúng 2^15 lần - phát hiện qua tb_recip_isolated.sv).
    localparam int SHIFT_W = $clog2(PROD_WIDTH+1) + 1;
    logic signed [SHIFT_W-1:0] shift_total_c;
    assign shift_total_c = SHIFT_W'(OUT_W) + SHIFT_W'(s1_msb_pos);

    logic [PROD_WIDTH-1:0] shifted_c;
    always_comb begin
        if (shift_total_c >= 0)
            shifted_c = prod_c >> shift_total_c;
        else
            shifted_c = prod_c << (-shift_total_c);
    end

    // Clamp về EXP_WIDTH bit (Q1.15 unsigned, giá trị luôn < 1.0 theo
    // construction của softmax nên về lý thuyết không bao giờ saturate,
    // nhưng giữ clamp để an toàn trước sai số làm tròn ở biên).
    localparam logic [EXP_WIDTH-1:0] MAX_Q15 = {EXP_WIDTH{1'b1}};

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            o_tvalid <= 1'b0;
            o_result <= '0;
        end else begin
            o_tvalid <= s1_valid;
            if (shifted_c > PROD_WIDTH'(MAX_Q15))
                o_result <= MAX_Q15;
            else
                o_result <= EXP_WIDTH'(shifted_c);
        end
    end

endmodule