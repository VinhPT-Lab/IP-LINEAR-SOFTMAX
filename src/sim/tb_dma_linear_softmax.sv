`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////

import axi_vip_pkg::*;
import dma_mix_ip_linear_axi_vip_0_0_pkg::*;

module tb_dma_linear_softmax();

    logic aclk;
    logic aresetn;
    logic mm2s_introut_0;
    logic s2mm_introut_0;
    
    //User path
    `define MEM_Q "E:/DOWNLOAD/HCMUT/TTKS/src/mem files/golden model/q_ram.mem"
    `define MEM_K "E:/DOWNLOAD/HCMUT/TTKS/src/mem files/golden model/k_ram.mem"
    `define GOLDEN_SCORE "E:/DOWNLOAD/HCMUT/TTKS/src/mem files/golden model/golden_score.mem"
    `define GOLDEN_SOFTMAX "E:/DOWNLOAD/HCMUT/TTKS/src/mem files/golden model/golden_softmax.mem"  
    `define RTL_SCORE "E:/DOWNLOAD/HCMUT/TTKS/src/mem files/rtl/softmax_score.mem"
    
    //User localparam
    localparam time clk_period = 10ns;
    localparam logic [31:0] SRC_BASE = 32'h0000_0000;
    localparam logic [31:0] DST_BASE = 32'h0001_0000;
    localparam logic [31:0] DMA_BASE = 32'h4000_0000;
    localparam logic [31:0] IP_LINEAR_BASE = 32'h4001_0000;
    localparam logic [31:0] IP_SOFTMAX_BASE = 32'h4002_0000;
    
    localparam logic [31:0] MM2S_DMACR = DMA_BASE + 32'h00;
    localparam logic [31:0] MM2S_DMASR = DMA_BASE +32'h04;
    localparam logic [31:0] MM2S_SA = DMA_BASE + 32'h18;
    localparam logic [31:0] MM2S_LENGTH = DMA_BASE + 32'h28;
    localparam logic [31:0] S2MM_DMACR = DMA_BASE + 32'h30;
    localparam logic [31:0] S2MM_DMASR = DMA_BASE + 32'h34;
    localparam logic [31:0] S2MM_DA = DMA_BASE + 32'h48;
    localparam logic [31:0] S2MM_LENGTH = DMA_BASE +32'h58;
    //IL = IP LINEAR IS = IP SOFTMAX
    localparam logic [31:0] IL_S00_SLV0 = IP_LINEAR_BASE + 32'h0000_0000;
    localparam logic [31:0] IL_S00_SLV1 = IP_LINEAR_BASE + 32'h0000_0004;
    localparam logic [31:0] IS_S00_SLV0 = IP_SOFTMAX_BASE + 32'h0000_0000;
    localparam logic [31:0] IS_S00_STATUS = IP_SOFTMAX_BASE + 32'h0000_0004;
    
    localparam int DMA_CR_RS = 0;
    localparam int DMA_CR_RESET = 2;
    localparam int DMA_SR_IDLE = 1;
    localparam int DMA_SR_IOC_IRQ = 12;
    localparam int DMA_SR_ERR_IRQ = 14;
    
    //User declare logic
    dma_mix_ip_linear_axi_vip_0_0_mst_t vip_mst;
    real t_start, t_end;
    
    //User submodule
    dma_mix_ip_linear_wrapper dut(
        .aclk(aclk),
        .aresetn(aresetn),
        .mm2s_introut_0(mm2s_introut_0),
        .s2mm_introut_0(s2mm_introut_0)
    );
    
    //User task
    //axi write data into addr
    task automatic axi_write(input xil_axi_ulong addr, input logic [31:0] data);
        xil_axi_resp_t resp;
        begin
            vip_mst.AXI4LITE_WRITE_BURST(addr, 0, data, resp);
            if(resp != XIL_AXI_RESP_OKAY) begin
                $fatal(1, "AXI write failed: addr=0x%08h data=0x%08h resp=%0d", addr, data, resp);
            end
        end
    endtask
    
    //axi read data from addr
    task automatic axi_read(input xil_axi_ulong addr, output logic [31:0] data);
        xil_axi_resp_t resp;
        begin
            vip_mst.AXI4LITE_READ_BURST(addr, 0, data, resp);
            if(resp != XIL_AXI_RESP_OKAY) begin
                $fatal(1, "AXI read failed: addr=0x%08h resp=%0d", addr, resp);
            end
        end
    endtask

    //init source data
    localparam int SEQ_LEN = 16;
    localparam int D_HEAD = 16;
    localparam int D_MODEL = 64;
    localparam int NUM_WORDS_K = D_HEAD*D_MODEL;
    localparam int NUM_BYTES_K = NUM_WORDS_K*4;
    localparam int NUM_WORDS_Q = SEQ_LEN*D_MODEL;
    localparam int NUM_BYTES_Q = NUM_WORDS_Q*4;
    localparam int NUM_WORDS_OUT = SEQ_LEN*D_HEAD;
    localparam int NUM_BYTES_OUT = NUM_WORDS_OUT*4;
    
    localparam logic [31:0] K_BASE = SRC_BASE;
    localparam logic [31:0] Q_BASE = SRC_BASE + NUM_BYTES_K;
    
    logic [31:0] src_data [0:NUM_WORDS_K + NUM_WORDS_Q-1];
    logic [31:0] golden_score [0:NUM_WORDS_OUT-1];
    task automatic init_src_data();
        $readmemh(`MEM_K, src_data, 0, NUM_WORDS_K-1);
        $readmemh(`MEM_Q, src_data, NUM_WORDS_K, NUM_WORDS_K + NUM_WORDS_Q-1);
        $readmemh(`GOLDEN_SOFTMAX, golden_score);
        $display("init data src done");
    endtask
    
    //preload data k,q into bram src
    task automatic preload_bram_src();
        for(int i=0; i<NUM_WORDS_K + NUM_WORDS_Q; i++) begin
            axi_write(SRC_BASE + i*4, src_data[i]);
        end
        $display("preload data into bram src done");
    endtask 
    
    //check bram src: check data loaded in bram
    task automatic check_bram_src();
        logic [31:0] rd;
        for(int i=0; i<NUM_WORDS_K + NUM_WORDS_Q; i++) begin
            axi_read(SRC_BASE + i*4, rd); 
            if( rd!== src_data[i]) begin
                $fatal(1, "SRC mismatch at word %0d; exp=0x%08h got=0x%08h", i, src_data[i], rd);
            end
        end
        $display("check bram src done");
    endtask 
    
    initial begin
        aclk=0;
        forever #(clk_period/2) aclk=~aclk;
    end
    
    //dma reset
    task automatic dma_reset();
        logic [31:0] mm2s_cr;
        logic [31:0] s2mm_cr;
        
        axi_write(MM2S_DMACR, 32'h0000_0004);
        axi_write(S2MM_DMACR, 32'h0000_0004);
        
        do begin
            axi_read(MM2S_DMACR, mm2s_cr);
        end while(mm2s_cr[DMA_CR_RESET]);
        
        do begin
            axi_read(S2MM_DMACR, s2mm_cr);
        end while(s2mm_cr[DMA_CR_RESET]);
    endtask 
    
    //init dma
    task automatic dma_start_transfer();
        // Start IP, then feed K.
        axi_write(MM2S_DMACR, 32'h0000_0001);
        $display("1");
        axi_write(MM2S_SA, K_BASE);
        $display("2");
        axi_write(IL_S00_SLV0, 32'h0000_0001);
        $display("3");
        axi_write(MM2S_LENGTH, NUM_BYTES_K);
        $display("4");
    
        wait_mm2s_done();
        axi_write(MM2S_DMASR, 32'h0000_1000);
        $display("5");
        
        axi_write(IS_S00_SLV0, 32'h0000_0001);
        $display("6");
    
        // Arm output right before feeding Q.
        axi_write(S2MM_DMACR, 32'h0000_0001);
        $display("7");
        axi_write(S2MM_DA, DST_BASE);
        $display("8");
        axi_write(S2MM_LENGTH, NUM_BYTES_OUT);
        $display("9");
    
        // Feed Q. If linear is still PRELOAD_MAC, MM2S will wait on TREADY.
        axi_write(MM2S_SA, Q_BASE);
        $display("10");
        axi_write(MM2S_LENGTH, NUM_BYTES_Q);
        $display("11");
   
        wait_mm2s_done();
        wait_s2mm_done();
    
        $display("dma transfer done");
    endtask 
    
    //wait mm2s done
    task automatic wait_mm2s_done();
        logic [31:0] mm2s_sr;
        int timeout;
        timeout=0;
        do begin
            axi_read(MM2S_DMASR, mm2s_sr);
            if(mm2s_sr[DMA_SR_ERR_IRQ]) begin
                $fatal(1, "MM2S DMA error, DMASR=0x%08h", mm2s_sr);
            end
            timeout++;
            if(timeout>200000) begin
                $fatal(1, "Timeout waiting MM2S done, DMASR=0x%08h", mm2s_sr);
            end
        end while(!mm2s_sr[DMA_SR_IOC_IRQ]);
        $display("mm2s done");
    endtask
    
    //wait s2mm done
    task automatic wait_s2mm_done();
        logic [31:0] s2mm_sr;
        int timeout;
        timeout=0;
        do begin
            axi_read(S2MM_DMASR, s2mm_sr);
            if(s2mm_sr[DMA_SR_ERR_IRQ]) begin
                $fatal(1, "S2MM DMA error, DMASR=0x%08h", s2mm_sr);
            end
            timeout++;
            if(timeout>200000) begin
                $fatal(1, "Timeout waiting S2MM done, DMASR=0x%08h", s2mm_sr);
            end
        end while(!s2mm_sr[DMA_SR_IOC_IRQ]);
        $display("s2mm done");
    endtask
    
    //compare with golden score
    task automatic compare_bram_dst_with_golden_model();
        logic [31:0] rd;
        int fail_show_max;
        int shown;
        int pass_cnt;
        int fail_cnt;
        
        pass_cnt=0;
        fail_cnt=0;
        fail_show_max=20;
        shown=0;
        
        for(int i=0; i<NUM_WORDS_OUT; i++) begin
            axi_read(DST_BASE + i*4, rd);
            if(rd === golden_score[i]) begin
                pass_cnt++;
            end
            else begin
                fail_cnt++;
                if(shown < fail_show_max) begin
                    $display("[FAIL] idx=%d exp=0x%08h got=0x%08h delta=%0d", i, golden_score[i], rd, $signed(rd)-$signed(golden_score[i]));
                    shown++;
                end
            end
        end
        
        if(fail_cnt == 0) begin
            $display("PASS: all output words match golden model");
        end
        else begin
            $display("FAIL: %0d output word mismatch", fail_cnt);
        end
    endtask 
    
    //write output mem file
    task automatic write_bram_dst_to_mem(input string file_path);
        int fd;
        logic [31:0] rd;
        fd=$fopen(file_path, "w");
        if(fd==0) begin
            $fatal(1, "Cannot open output mem file: %s", file_path);
        end
        
        $fdisplay(fd, "// DMA + ip_axi_linear RTL output");
        $fdisplay(fd, "// Words: %0d", NUM_WORDS_OUT);
        $fdisplay(fd, "// Format: one 32-bit hex word per line");
    
        for (int i = 0; i < NUM_WORDS_OUT; i++) begin
          axi_read(DST_BASE + i * 4, rd);
          $fdisplay(fd, "%08h", rd);
        end
    
        $fclose(fd);
        $display("Wrote RTL output mem file: %s", file_path);
    endtask
    
    //main
    initial begin
        //init
        aresetn = 1'b0;
        repeat (20) @(posedge aclk);
        aresetn = 1'b1;
        
        repeat(5)@(posedge aclk);
        vip_mst = new("vip_mst", dut.dma_mix_ip_linear_i.vip_ctrl.inst.IF);
        vip_mst.start_master();
        
        //programme
        t_start = $realtime;
        init_src_data();
        t_end = $realtime;
        $display("init src data time: %.2f ns", t_end-t_start);
        
        t_start = $realtime;
        preload_bram_src();
        t_end = $realtime;
        $display("preload_bram_src time: %.2f ns", t_end-t_start);
        //check_bram_src();
        t_start = $realtime;
        dma_reset();
        t_end = $realtime;
        $display("dma_reset time: %.2f ns", t_end-t_start);
        begin
            logic [31:0] dbg;
            axi_read(MM2S_DMASR, dbg);
            $display("DEBUG: MM2S_DMASR right after reset = 0x%08h", dbg);
        end
        t_start = $realtime;
        dma_start_transfer();
        t_end = $realtime;
        $display("dma_start_transfer time: %.2f ns", t_end-t_start);
        
        t_start = $realtime;
        compare_bram_dst_with_golden_model();
        write_bram_dst_to_mem(`RTL_SCORE);
        t_end = $realtime;
        $display("compare and write file time: %.2f ns", t_end-t_start);
        repeat(5)@(posedge aclk);
        $finish;
    end

endmodule
