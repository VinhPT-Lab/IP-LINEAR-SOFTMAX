#include "xil_io.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xstatus.h"
#include "attn_test_data.h"

#define SRC_BASE        0xA0000000U
#define DST_BASE        0xA0010000U
#define DMA_BASE        0xA0020000U
#define IP_LINEAR_BASE  0xA0030000U
#define IP_SOFTMAX_BASE  0xA0040000U

#define MM2S_DMACR      (DMA_BASE + 0x00U)
#define MM2S_DMASR      (DMA_BASE + 0x04U)
#define MM2S_SA         (DMA_BASE + 0x18U)
#define MM2S_LENGTH     (DMA_BASE + 0x28U)
#define S2MM_DMACR      (DMA_BASE + 0x30U)
#define S2MM_DMASR      (DMA_BASE + 0x34U)
#define S2MM_DA         (DMA_BASE + 0x48U)
#define S2MM_LENGTH     (DMA_BASE + 0x58U)

#define IL_START        (IP_LINEAR_BASE + 0x00U)
#define IL_STATUS       (IP_LINEAR_BASE + 0x04U)

#define SM_START        (IP_SOFTMAX_BASE + 0x00U)
#define SM_STATUS       (IP_SOFTMAX_BASE + 0x04U)

#define DMA_CR_RS       (1U << 0)
#define DMA_CR_RESET    (1U << 2)
#define DMA_SR_IDLE     (1U << 1)
#define DMA_SR_IOC_IRQ  (1U << 12)
#define DMA_SR_ERR_IRQ  (1U << 14)

#define SEQ_LEN         16U
#define D_HEAD          16U
#define D_MODEL         64U

#define NUM_WORDS_K     (D_HEAD * D_MODEL)
#define NUM_BYTES_K     (NUM_WORDS_K * 4U)
#define NUM_WORDS_Q     (SEQ_LEN * D_MODEL)
#define NUM_BYTES_Q     (NUM_WORDS_Q * 4U)
#define NUM_WORDS_OUT   (SEQ_LEN * D_HEAD)
#define NUM_BYTES_OUT   (NUM_WORDS_OUT * 4U)

#define K_BASE          SRC_BASE
#define Q_BASE          (SRC_BASE + NUM_BYTES_K)

/* k_data, q_data, golden_score, golden_softmax provided by attn_test_data.h (static const u32) */

static int wait_dma_done(u32 sr_addr, const char *name)
{
    for (u32 timeout = 0; timeout < 10000000U; timeout++) {
        u32 sr = Xil_In32(sr_addr);

        if ((sr & DMA_SR_ERR_IRQ) != 0U) {
            xil_printf("%s DMA error: DMASR=0x%08lx\r\n", name, sr);
            return XST_FAILURE;
        }

        if ((sr & DMA_SR_IOC_IRQ) != 0U) {
            return XST_SUCCESS;
        }
    }

    xil_printf("%s DMA timeout: DMASR=0x%08lx\r\n", name, Xil_In32(sr_addr));
    return XST_FAILURE;
}

static int dma_reset(void)
{
    Xil_Out32(MM2S_DMACR, DMA_CR_RESET);
    Xil_Out32(S2MM_DMACR, DMA_CR_RESET);

    for (u32 timeout = 0; timeout < 1000000U; timeout++) {
        if (((Xil_In32(MM2S_DMACR) & DMA_CR_RESET) == 0U) &&
            ((Xil_In32(S2MM_DMACR) & DMA_CR_RESET) == 0U)) {
            return XST_SUCCESS;
        }
    }

    xil_printf("DMA reset timeout\r\n");
    return XST_FAILURE;
}

static void preload_bram(void)
{
    for (u32 i = 0; i < NUM_WORDS_K; i++) {
        Xil_Out32(K_BASE + i * 4U, k_data[i]);
    }

    for (u32 i = 0; i < NUM_WORDS_Q; i++) {
        Xil_Out32(Q_BASE + i * 4U, q_data[i]);
    }
}

static int run_linear_dma(void)
{
    xil_printf("1\r\n");
    Xil_Out32(MM2S_DMACR, DMA_CR_RS);
    xil_printf("2\r\n");
    Xil_Out32(MM2S_SA, K_BASE);
    xil_printf("3\r\n");
    Xil_Out32(IL_START, 0x00000001U);
    xil_printf("4\r\n");
    Xil_Out32(MM2S_LENGTH, NUM_BYTES_K);
    xil_printf("5\r\n");
    if (wait_dma_done(MM2S_DMASR, "MM2S-K") != XST_SUCCESS) {
        return XST_FAILURE;
    }
    xil_printf("6\r\n");
    Xil_Out32(MM2S_DMASR, DMA_SR_IOC_IRQ);
    xil_printf("7\r\n");
    Xil_Out32(SM_START, 0x00000001U);
    xil_printf("8\r\n");
    Xil_Out32(S2MM_DMACR, DMA_CR_RS);
    Xil_Out32(S2MM_DA, DST_BASE);
    Xil_Out32(S2MM_LENGTH, NUM_BYTES_OUT);

    Xil_Out32(MM2S_SA, Q_BASE);
    Xil_Out32(MM2S_LENGTH, NUM_BYTES_Q);

    if (wait_dma_done(MM2S_DMASR, "MM2S-Q") != XST_SUCCESS) {
        return XST_FAILURE;
    }

    if (wait_dma_done(S2MM_DMASR, "S2MM") != XST_SUCCESS) {
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

static void dump_output(void)
{
    xil_printf("---DUMP_BEGIN---\r\n");
    for (u32 i = 0; i < NUM_WORDS_OUT; i++) {
        u32 got = Xil_In32(DST_BASE + i * 4U);
        xil_printf("%lu,0x%08lx\r\n", i, got);
    }
    xil_printf("---DUMP_END---\r\n");
}

static int compare_output(void)
{
    u32 fail_cnt = 0U;

    /* DST_BASE holds the SOFTMAX weight output (SM_START runs before S2MM
     * captures it), so it must be checked against golden_softmax, NOT
     * golden_score (which is the raw linear/attention score - a different
     * value domain entirely: signed fixed-point score vs. Q1.15 unsigned
     * weight). */
    for (u32 i = 0; i < NUM_WORDS_OUT; i++) {
        u32 got = Xil_In32(DST_BASE + i * 4U);
        u32 exp = golden_softmax[i];

        if (got != exp) {
            if (fail_cnt < 20U) {
                xil_printf("FAIL idx=%lu exp=0x%08lx got=0x%08lx\r\n", i, exp, got);
            }
            fail_cnt++;
        }
    }

    if (fail_cnt == 0U) {
        xil_printf("PASS: all output words match golden softmax\r\n");
        return XST_SUCCESS;
    }

    xil_printf("FAIL: %lu mismatches\r\n", fail_cnt);
    return XST_FAILURE;
}

int main(void)
{
    Xil_DCacheDisable();

    xil_printf("IP Linear DMA bare-metal bring-up\r\n");

    preload_bram();

    if (dma_reset() != XST_SUCCESS) {
        return XST_FAILURE;
    }

    if (run_linear_dma() != XST_SUCCESS) {
        return XST_FAILURE;
    }

    xil_printf("IP STATUS=0x%08lx\r\n", Xil_In32(IL_STATUS));

    dump_output();

    return compare_output();
}