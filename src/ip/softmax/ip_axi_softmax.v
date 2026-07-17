`timescale 1 ns / 1 ps

	module ip_axi_softmax #
	(
		// Users to add parameters here
		parameter integer D_HEAD      = 64,
		parameter integer SEQ_LEN     = 64,
		parameter integer DATA_WIDTH  = 16,
		parameter integer EXP_WIDTH   = 16,
		parameter int RECIP_ADDR_W    = 12,
		parameter int RECIP_OUT_W     = 19,
		// User parameters ends
		// Do not modify the parameters beyond this line


		// Parameters of Axi Slave Bus Interface S00_AXI
		parameter integer C_S00_AXI_DATA_WIDTH	= 32,
		parameter integer C_S00_AXI_ADDR_WIDTH	= 4
	)
	(
		// Users to add ports here
		input  logic [31:0] s00_axis_tdata,
		input  logic        s00_axis_tvalid,
		input  logic        s00_axis_tlast,
		output logic        s00_axis_tready,

		output logic [31:0] m00_axis_tdata,
		output logic        m00_axis_tvalid,
		output logic        m00_axis_tlast,
		input  logic        m00_axis_tready,
		// User ports ends
		// Do not modify the ports beyond this line


		// Ports of Axi Slave Bus Interface S00_AXI
		input logic  s00_axi_aclk,
		input logic  s00_axi_aresetn,
		input logic [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_awaddr,
		input logic [2 : 0] s00_axi_awprot,
		input logic  s00_axi_awvalid,
		output logic  s00_axi_awready,
		input logic [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_wdata,
		input logic [(C_S00_AXI_DATA_WIDTH/8)-1 : 0] s00_axi_wstrb,
		input logic  s00_axi_wvalid,
		output logic  s00_axi_wready,
		output logic [1 : 0] s00_axi_bresp,
		output logic  s00_axi_bvalid,
		input logic  s00_axi_bready,
		input logic [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_araddr,
		input logic [2 : 0] s00_axi_arprot,
		input logic  s00_axi_arvalid,
		output logic  s00_axi_arready,
		output logic [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_rdata,
		output logic [1 : 0] s00_axi_rresp,
		output logic  s00_axi_rvalid,
		input logic  s00_axi_rready
	);
	//User signal
	logic start_softmax;
	logic softmax_done;
	logic busy;
// Instantiation of Axi Bus Interface S00_AXI
	ip_axi_softmax_slave_lite_v1_0_S00_AXI # ( 
		.C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
		.C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
	) ip_axi_softmax_slave_lite_v1_0_S00_AXI_inst (
		.S_AXI_ACLK(s00_axi_aclk),
		.S_AXI_ARESETN(s00_axi_aresetn),
		.S_AXI_AWADDR(s00_axi_awaddr),
		.S_AXI_AWPROT(s00_axi_awprot),
		.S_AXI_AWVALID(s00_axi_awvalid),
		.S_AXI_AWREADY(s00_axi_awready),
		.S_AXI_WDATA(s00_axi_wdata),
		.S_AXI_WSTRB(s00_axi_wstrb),
		.S_AXI_WVALID(s00_axi_wvalid),
		.S_AXI_WREADY(s00_axi_wready),
		.S_AXI_BRESP(s00_axi_bresp),
		.S_AXI_BVALID(s00_axi_bvalid),
		.S_AXI_BREADY(s00_axi_bready),
		.S_AXI_ARADDR(s00_axi_araddr),
		.S_AXI_ARPROT(s00_axi_arprot),
		.S_AXI_ARVALID(s00_axi_arvalid),
		.S_AXI_ARREADY(s00_axi_arready),
		.S_AXI_RDATA(s00_axi_rdata),
		.S_AXI_RRESP(s00_axi_rresp),
		.S_AXI_RVALID(s00_axi_rvalid),
		.S_AXI_RREADY(s00_axi_rready),

		.o_start_softmax(start_softmax),
		.i_softmax_done (softmax_done),
		.i_busy         (busy)
	);

	// Add user logic here
	softmax #(
		.D_HEAD          (D_HEAD),
		.SEQ_LEN         (SEQ_LEN),
		.DATA_WIDTH      (DATA_WIDTH),
		.EXP_WIDTH       (EXP_WIDTH),
		.RECIP_ADDR_W(RECIP_ADDR_W),
		.RECIP_OUT_W(RECIP_OUT_W)
	) u_softmax (
		.iclk(s00_axi_aclk),
		.irst_n(s00_axi_aresetn),

		.i_start_softmax(start_softmax),
		.o_softmax_done (softmax_done),
		.o_busy         (busy),

		.i_s_axis_tdata (s00_axis_tdata),
		.i_s_axis_tvalid(s00_axis_tvalid),
		.i_s_axis_tlast (s00_axis_tlast),
		.o_s_axis_tready(s00_axis_tready),

		.o_m_axis_tdata (m00_axis_tdata),
		.o_m_axis_tvalid(m00_axis_tvalid),
		.o_m_axis_tlast (m00_axis_tlast),
		.i_m_axis_tready(m00_axis_tready)
	);
	// User logic ends

	endmodule