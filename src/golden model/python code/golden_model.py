import numpy as np
import os
import math

# ==============================================================================
# golden_model.py — Bit-true reference model for ip_axi_linear
#
# Parameter mapping (must match ip_axi_linear / linear.sv):
#   D_MODEL    : inner dimension of Q×Kᵀ dot product  → linear.sv D_MODEL
#   SEQ_LEN    : number of Q rows (output rows of S)  → linear.sv SEQ_LEN
#   D_HEAD     : number of K rows (output cols of S)  → linear.sv D_HEAD
#   N_PE       : number of parallel PEs               → linear.sv N_PE (= D_HEAD, 1-tile)
#   DATA_WIDTH : element bit width, signed fixed-point → linear.sv DATA_WIDTH
#   FRAC_BITS  : fractional bits in Q<INT>.<FRAC> rep  → pe_unit FRAC_BITS = DATA_WIDTH/2
#   SQRT_SHIFT : output scaling right-shift            → linear.sv $clog2(D_MODEL)/2
#
# Scaling pipeline (bit-true to RTL):
#   mac_sum  = Σ Q_int[i][k] * K_int[j][k]           (38-bit accumulator, pe_unit)
#   pe_out   = round_half_to_even(mac_sum / 2^FRAC_BITS)  → DATA_WIDTH bits
#   score    = arithmetic_right_shift(pe_out, SQRT_SHIFT)  → DATA_WIDTH bits (linear.sv)
# ==============================================================================

# ==============================================================================
# DEFAULT PARAMETERS — match ip_axi_linear defaults
# ==============================================================================
DEFAULT_D_MODEL    = 64
DEFAULT_SEQ_LEN    = 64
DEFAULT_D_HEAD     = 64
DEFAULT_N_PE       = 64     # informational only; must equal D_HEAD for 1-tile mode
DEFAULT_DATA_WIDTH = 16
DEFAULT_RUN_MODE   = 2
DEFAULT_UNIFORM    = 50

ROM_DEPTH = 2048            # exp LUT depth (fixed)

# ==============================================================================
# OUTPUT PATHS
# ==============================================================================
COE_OUT_PATH = r"E:\DOWNLOAD\HCMUT\TTKS\src\coe files\golden model"
MEM_OUT_PATH = r"E:\DOWNLOAD\HCMUT\TTKS\src\mem files\golden model"

# ==============================================================================
# INPUT UTILITIES
# ==============================================================================
def get_int_input(prompt, default, min_val=None, max_val=None):
    while True:
        raw = input(f"{prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("  -> Invalid integer.")
            continue
        if min_val is not None and val < min_val:
            print(f"  -> Must be >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"  -> Must be <= {max_val}.")
            continue
        return val


def get_mode_input(default):
    while True:
        raw = input(f"RUN_MODE (1=Uniform, 2=Random) [default={default}]: ").strip()
        if raw == "":
            return default
        if raw in ("1", "2"):
            return int(raw)
        print("  -> Enter 1 or 2.")


# ==============================================================================
# DERIVED PARAMETERS (bit-true to RTL)
# ==============================================================================
def derive_params(D_MODEL, DATA_WIDTH):
    """
    Compute derived constants exactly as RTL does.
    pe_unit   : FRAC_BITS  = DATA_WIDTH / 2          (integer division)
    linear.sv : SQRT_SHIFT = $clog2(D_MODEL) / 2     (integer division)
    """
    frac_bits  = DATA_WIDTH // 2
    clog2_dm   = int(math.ceil(math.log2(D_MODEL))) if D_MODEL > 1 else 1
    sqrt_shift = clog2_dm // 2
    return frac_bits, sqrt_shift


# ==============================================================================
# FIXED-POINT CONVERSION
# ==============================================================================
def float_to_fixed(val_array, frac_bits):
    """Convert float array to signed fixed-point integer (DATA_WIDTH bits)."""
    scaled = np.round(val_array * float(1 << frac_bits))
    return np.clip(scaled, -32768, 32767).astype(np.int64)


# ==============================================================================
# ROUNDING — bit-true to pe_unit round-to-nearest-even
#
# pe_unit RTL:
#   round_up = acc[FRAC_BITS-1] & (|acc[FRAC_BITS-2:0] | acc[FRAC_BITS])
#   o_result = acc[FRAC_BITS+DATA_WIDTH-1 : FRAC_BITS] + round_up
#
# This is round-half-to-even (banker's rounding) on the integer accumulator,
# truncating FRAC_BITS LSBs.
# ==============================================================================
def round_half_to_even_shift(acc_array, shift):
    """
    Arithmetic right-shift by `shift` bits with round-half-to-even,
    applied element-wise on a numpy int64 array.
    Matches pe_unit RTL accumulator truncation exactly.
    """
    half      = np.int64(1) << np.int64(shift - 1)   # 2^(shift-1)
    low_mask  = (np.int64(1) << np.int64(shift)) - np.int64(1)  # mask for shift LSBs
    remainder = acc_array & low_mask                  # bits being dropped
    quotient  = acc_array >> np.int64(shift)          # truncated result (arithmetic)

    # Half-boundary: remainder == half (exact 0.5)
    at_half   = (remainder == half)
    above_half = (remainder > half)

    # LSB of quotient (for even check)
    lsb       = quotient & np.int64(1)

    # Round up when: above half, OR at exactly half AND result is odd (round-to-even)
    round_up  = above_half | (at_half & (lsb != 0))

    return quotient + round_up.astype(np.int64)


# ==============================================================================
# ARITHMETIC RIGHT SHIFT (linear.sv SQRT_SHIFT stage)
# Truncation only — matches Verilog >>> on signed.
# ==============================================================================
def arith_right_shift(arr, shift):
    """Arithmetic right shift with truncation (no rounding). Matches >>> in Verilog."""
    return arr >> np.int64(shift)


# ==============================================================================
# EXP LUT
# ==============================================================================
def generate_exp_lut(frac_bits):
    lut = []
    for i in range(ROM_DEPTH):
        x     = -i / float(1 << frac_bits)
        val   = np.exp(x)
        q_val = int(np.round(val * float(1 << frac_bits)))
        if q_val == 0 and val > 0:
            q_val = 1
        lut.append(q_val)
    return lut


# ==============================================================================
# RECIPROCAL LUT — for reciprocal_divider.sv (replaces div_gen in softmax.sv)
#
# Table of reciprocal(mantissa) for mantissa in [1.0, 2.0), addressed by the
# RECIP_ADDR_W bits immediately below the divisor's leading 1-bit (implicit,
# not stored) — standard fixed-point range-reduction reciprocal.
#
# Fixed size, independent of D_HEAD/SEQ_LEN/EXP_WIDTH — generate ONCE, never
# needs regenerating when attention dimensions change (unlike div_gen, which
# needed re-customizing + re-measuring bit offsets every time SUM_WIDTH
# changed). See reciprocal_divider.sv header comment for the full derivation.
#
# RECIP_ADDR_W=12 (4096-entry ROM) / RECIP_OUT_W=19 (Q0.19 unsigned) verified
# to give max 1 LSB error vs golden integer division on Q1.15 scale, across
# 200k+ random trials spanning D_HEAD in {16,64,128}.
# ==============================================================================
RECIP_ADDR_W = 12
RECIP_OUT_W  = 19

def generate_recip_lut(addr_w=RECIP_ADDR_W, out_w=RECIP_OUT_W):
    n = 1 << addr_w
    lut = []
    for i in range(n):
        mantissa = 1.0 + i / float(n)          # mantissa in [1.0, 2.0)
        recip    = 1.0 / mantissa                # in (0.5, 1.0]
        q_val    = int(round(recip * float(1 << out_w)))
        if q_val >= (1 << out_w):
            q_val = (1 << out_w) - 1
        lut.append(q_val)
    return lut


# ==============================================================================
# FILE OUTPUT HELPERS
# ==============================================================================
def _ensure(path):
    os.makedirs(path, exist_ok=True)

def write_coe_16(filename, data):
    try:
        _ensure(COE_OUT_PATH)
        fp = os.path.join(COE_OUT_PATH, filename)
        with open(fp, 'w') as f:
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")
            for i, v in enumerate(data):
                sep = ";" if i == len(data) - 1 else ","
                f.write(f"{int(v) & 0xFFFF:04X}{sep}\n")
        print(f"[OK] COE 16-bit : {fp}")
    except Exception as e:
        print(f"[ERR] {filename}: {e}")

def write_coe_32(filename, data):
    try:
        _ensure(COE_OUT_PATH)
        fp = os.path.join(COE_OUT_PATH, filename)
        with open(fp, 'w') as f:
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")
            for i, v in enumerate(data):
                sep = ";" if i == len(data) - 1 else ","
                f.write(f"{int(v) & 0xFFFFFFFF:08X}{sep}\n")
        print(f"[OK] COE 32-bit : {fp}")
    except Exception as e:
        print(f"[ERR] {filename}: {e}")

def write_coe_generic(filename, data, width_bits):
    """
    COE writer cho width bất kỳ (không giới hạn 16/32-bit) — dùng cho ROM
    có output width lẻ như recip_rom (RECIP_OUT_W=19). Vivado Block Memory
    Generator chấp nhận hex string với đúng số hex digit ceil(width/4);
    giá trị được mask đúng width_bits trước khi format để tránh lệch digit
    khi width không phải bội số của 4 (bug đã gặp: 19-bit ghi dư 1 hex
    digit khiến $readmemh/BMG đọc lệch giá trị).
    """
    try:
        _ensure(COE_OUT_PATH)
        fp = os.path.join(COE_OUT_PATH, filename)
        hex_digits = (width_bits + 3) // 4
        mask = (1 << width_bits) - 1
        with open(fp, 'w') as f:
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")
            for i, v in enumerate(data):
                sep = ";" if i == len(data) - 1 else ","
                f.write(f"{int(v) & mask:0{hex_digits}X}{sep}\n")
        print(f"[OK] COE {width_bits}-bit : {fp}")
    except Exception as e:
        print(f"[ERR] {filename}: {e}")

def write_mem_16(filename, data):
    try:
        _ensure(MEM_OUT_PATH)
        fp = os.path.join(MEM_OUT_PATH, filename)
        with open(fp, 'w') as f:
            for v in data:
                f.write(f"{int(v) & 0xFFFF:04X}\n")
        print(f"[OK] MEM 16-bit : {fp}")
    except Exception as e:
        print(f"[ERR] {filename}: {e}")

def write_mem_32(filename, data):
    try:
        _ensure(MEM_OUT_PATH)
        fp = os.path.join(MEM_OUT_PATH, filename)
        with open(fp, 'w') as f:
            for v in data:
                f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")
        print(f"[OK] MEM 32-bit : {fp}")
    except Exception as e:
        print(f"[ERR] {filename}: {e}")

def write_golden_score(score_int):
    try:
        _ensure(MEM_OUT_PATH)
        fp = os.path.join(MEM_OUT_PATH, "golden_score.mem")
        with open(fp, 'w') as f:
            for v in score_int.flatten():
                f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")
        print(f"[OK] golden_score.mem : {fp}")
    except Exception as e:
        print(f"[ERR] golden_score.mem: {e}")

def write_golden_softmax(weights_q15):
    try:
        _ensure(MEM_OUT_PATH)
        fp = os.path.join(MEM_OUT_PATH, "golden_softmax.mem")
        with open(fp, 'w') as f:
            for v in weights_q15.flatten():
                f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")
        print(f"[OK] golden_softmax.mem : {fp}")
    except Exception as e:
        print(f"[ERR] golden_softmax.mem: {e}")


# ==============================================================================
# C HEADER EXPORT — ports mem_to_c_arrays.py logic in-process.
#
# Consumes the same int64 arrays already held in memory (Q_int, K_int,
# score_int) instead of re-parsing the .mem files written above. This is now
# the single source of truth for the mask/width applied to each word — must
# stay bit-identical to write_mem_32()/write_golden_score() (`& 0xFFFFFFFF`,
# 32-bit word, one element per word, no packing). If those two masking rules
# ever diverge, .mem (used by RTL sim) and .h (used by bare-metal SW) produce
# different DMA payloads for the same logical data — silent divergence,
# because both now come from one script run instead of two independent
# commands with independent chances to fail loudly.
# ==============================================================================
def _emit_c_array(name, flat_int_array):
    """
    flat_int_array : 1-D array-like of Python/numpy ints, any sign.
    Emits `static const u32 name[N] = { ... };` with each element masked to
    32-bit two's complement, matching write_mem_32()/write_golden_score().
    """
    words = [int(v) & 0xFFFFFFFF for v in flat_int_array]
    lines = [f"static const u32 {name}[{len(words)}] = {{"]
    for i in range(0, len(words), 4):
        chunk = words[i:i + 4]
        vals = ", ".join(f"0x{w:08x}U" for w in chunk)
        comma = "," if i + 4 < len(words) else ""
        lines.append(f"    {vals}{comma}")
    lines.append("};")
    return "\n".join(lines)


def write_c_header(filename, k_flat, q_flat, golden_flat, softmax_flat):
    """
    k_flat, q_flat, golden_flat, softmax_flat : 1-D int64 arrays (already
    flattened, row-major, same element order as bram_src/bram_dst layout in
    main.c).
      golden_flat   : attention SCORE (linear.sv output, signed fixed-point)
      softmax_flat  : softmax WEIGHTS (Q1.15 unsigned) — this is what DST_BASE
                      actually holds in main.c's combined linear+softmax
                      pipeline (SM_START is triggered before S2MM captures
                      output), so compare_output() must check against this,
                      NOT against golden_score.
    """
    try:
        _ensure(MEM_OUT_PATH)
        fp = os.path.join(MEM_OUT_PATH, filename)
        text = "\n\n".join([
            "#pragma once",
            '#include "xil_types.h"',
            "",
            _emit_c_array("k_data", k_flat),
            _emit_c_array("q_data", q_flat),
            _emit_c_array("golden_score", golden_flat),
            _emit_c_array("golden_softmax", softmax_flat),
            "",
        ])
        with open(fp, 'w') as f:
            f.write(text)
        print(f"[OK] C header : {fp}")
        print(f"     k_data: {len(k_flat)} words, q_data: {len(q_flat)} words, "
              f"golden_score: {len(golden_flat)} words, "
              f"golden_softmax: {len(softmax_flat)} words")
    except Exception as e:
        print(f"[ERR] {filename}: {e}")


# ==============================================================================
# GOLDEN COMPUTE — Phase 1: Q × Kᵀ scaled dot-product
#
# Matches RTL pipeline exactly:
#   1. pe_unit MAC: acc = Σ Q_int[i][k] * K_int[j][k]   (int64, no overflow for DATA_WIDTH=16)
#   2. pe_unit out: round_half_to_even(acc, FRAC_BITS)   → DATA_WIDTH bits
#   3. linear.sv  : arith_right_shift(pe_out, SQRT_SHIFT) → DATA_WIDTH bits (output reg)
#   4. linear.sv  : zero-extend to 32 bits for M_AXIS TDATA
# ==============================================================================
def compute_attention_score(Q_int, K_int, frac_bits, sqrt_shift, data_width):
    """
    Q_int : [SEQ_LEN × D_MODEL]  int64
    K_int : [D_HEAD  × D_MODEL]  int64
    returns Score_int : [SEQ_LEN × D_HEAD]  int64
    """
    # Step 1: accumulate (exact integer, no overflow for 16-bit inputs × 64 elements)
    mac_sum = np.dot(Q_int, K_int.T)                          # [SEQ_LEN × D_HEAD], int64

    # Step 2: pe_unit truncation with round-half-to-even at FRAC_BITS
    pe_out = round_half_to_even_shift(mac_sum, frac_bits)     # [SEQ_LEN × D_HEAD], int64

    # Clip to DATA_WIDTH signed range (pe_unit output register)
    max_val = (1 << (data_width - 1)) - 1
    min_val = -(1 << (data_width - 1))
    pe_out  = np.clip(pe_out, min_val, max_val)

    # Step 3: SQRT_SHIFT in linear.sv (arithmetic right shift, truncation)
    score = arith_right_shift(pe_out, sqrt_shift)             # [SEQ_LEN × D_HEAD], int64

    # Clip to DATA_WIDTH signed range (result_buffer register)
    score = np.clip(score, min_val, max_val)

    return score


# ==============================================================================
# GOLDEN COMPUTE — Phase 2: Softmax (row-wise)
# ==============================================================================
def compute_softmax(score_int, exp_lut, frac_bits):
    """
    score_int  : [SEQ_LEN × D_HEAD]  int64
    exp_lut    : list of int (ROM_DEPTH entries)
    returns weights_int : [SEQ_LEN × D_HEAD]  int64
    """
    exp_lut_arr = np.array(exp_lut, dtype=np.int64)
    max_score   = np.max(score_int, axis=1, keepdims=True)
    Z_int       = score_int - max_score

    exp_Z = np.zeros_like(Z_int)
    for i in range(Z_int.shape[0]):
        for j in range(Z_int.shape[1]):
            val = int(Z_int[i, j])
            if val <= 0:
                addr = (-val) & 0x7FF
                exp_Z[i, j] = exp_lut_arr[addr] if addr < ROM_DEPTH else 0

    sum_exp      = np.sum(exp_Z, axis=1, keepdims=True)
    weights_int  = (exp_Z * (1 << frac_bits)) // sum_exp
    return exp_Z, sum_exp, weights_int


# ==============================================================================
# GOLDEN COMPUTE — Phase 3: Softmax bit-true to softmax.sv (ip_axi_softmax)
#
# NOTE: this is a *separate* model from compute_softmax() above.
# compute_softmax() above reconstructs the OLD attention_top.sv scaling
# (Q<frac_bits>.<frac_bits>, i.e. weights_int = exp_Z * 2^frac_bits // sum_exp)
# and must not be reused for ip_axi_softmax's golden_softmax.mem, because
# softmax.sv's ip_axi_softmax core does NOT use frac_bits at all for the
# final division — per HANDOFF.md section 5, it fixes the output format to
# Q1.15 unsigned regardless of DATA_WIDTH/FRAC_BITS, by construction:
#
#   SUM_WIDTH      = clog2(D_HEAD) + EXP_WIDTH      (softmax.sv localparam)
#   DIVIDEND_WIDTH = EXP_WIDTH + 15                  (softmax.sv localparam)
#   div_dividend   = exp_row_buf[i] << 15            (Q1.15 numerator, DIVIDEND_WIDTH bits)
#   div_divisor    = sum_latched                     (SUM_WIDTH bits)
#   div_dout       = div_dividend / div_divisor       (DIVIDEND_WIDTH+SUM_WIDTH bits, quotient-only)
#   out_row_buf[i] = div_dout[DIVIDEND_WIDTH-1 -: EXP_WIDTH]   (top EXP_WIDTH bits = Q1.15 result)
#
# exp_Z here must be the SAME exp_lut lookup used by exp_rom (i.e. produced by
# generate_exp_lut()/compute_softmax() above) — only the final division step
# differs from compute_softmax(). Because numerator and denominator share the
# same LUT scale, the LUT's own fractional width cancels out of the quotient;
# what matters is bit-true replication of the RTL's shift-then-truncate
# integer division, not the LUT's true numeric precision.
# ==============================================================================
def recip_divide_bit_true(dividend, divisor, sum_width, recip_lut, addr_w, out_w, exp_width):
    """
    Bit-true reimplementation of reciprocal_divider.sv (Stage 0/1/2), for a
    SINGLE (dividend, divisor) pair. Mirrors the RTL exactly:
      1. msb_pos = position of divisor's highest set bit within [sum_width-1:0]
         (priority encoder, MSB-first — divisor assumed > 0 by construction).
      2. recip_addr = addr_w bits immediately below msb_pos (mantissa bits),
         left-shifted (zero-padded) if msb_pos < addr_w.
      3. recip_rom_data = recip_lut[recip_addr]  (Q0.out_w unsigned).
      4. prod = dividend * recip_rom_data.
      5. shift_total = out_w + msb_pos  (always >= 0 here, no left-shift branch
         needed since msb_pos >= 0 and out_w > 0 for all valid configs).
      6. result = prod >> shift_total, clamped to exp_width-bit MAX_Q15.
    """
    if divisor == 0:
        return 0  # dead branch in practice (sum_latched always >= 1 exp value)

    # --- Stage 0: priority-encoder MSB search, MSB-first, within sum_width bits ---
    msb_pos = -1
    for b in range(sum_width - 1, -1, -1):
        if (divisor >> b) & 1:
            msb_pos = b
            break
    if msb_pos < 0:
        return 0  # divisor == 0, unreachable given the check above

    # --- Address = addr_w bits below MSB (zero-pad on the left if not enough bits) ---
    if msb_pos >= addr_w:
        recip_addr = (divisor >> (msb_pos - addr_w)) & ((1 << addr_w) - 1)
    else:
        recip_addr = (divisor << (addr_w - msb_pos)) & ((1 << addr_w) - 1)

    recip_rom_data = recip_lut[recip_addr]

    # --- Stage 2: multiply + shift ---
    prod = dividend * recip_rom_data
    shift_total = out_w + msb_pos
    shifted = prod >> shift_total  # shift_total always >= 0 for valid configs

    max_q15 = (1 << exp_width) - 1
    if shifted > max_q15:
        shifted = max_q15
    return shifted


def compute_softmax_golden(score_int, exp_lut, exp_width, d_head,
                            recip_lut, recip_addr_w=RECIP_ADDR_W, recip_out_w=RECIP_OUT_W):
    """
    score_int : [SEQ_LEN x D_HEAD] int64  (same fixed-point scale as softmax.sv input S)
    exp_lut   : list of int, ROM_DEPTH entries (exp_rom contents, EXP_WIDTH-bit unsigned)
    exp_width : softmax.sv EXP_WIDTH parameter (ROM/output data width)
    d_head    : softmax.sv D_HEAD parameter (row length)
    recip_lut : list of int, reciprocal ROM contents (from generate_recip_lut())
    recip_addr_w / recip_out_w : reciprocal_divider.sv ADDR_W / OUT_W parameters

    returns weights_q15 : [SEQ_LEN x D_HEAD] int64, Q1.15 unsigned (0 .. 32767),
                           bit-true to softmax.sv's ST_EXP_SUM + ST_DIV_ISSUE/DRAIN
                           pipeline, INCLUDING the reciprocal-LUT quantization
                           (max 1 LSB error is a property of the RTL divider itself,
                           not something this golden model should paper over with
                           an exact integer division).
    """
    exp_lut_arr = np.array(exp_lut, dtype=np.int64)
    rom_depth   = len(exp_lut)

    # --- ST_FIND_MAX + ST_EXP_SUM: row-wise max, then exp_rom lookup ---
    # addr = (-(x - max)) & 0x7FF  (softmax.sv line ~162); x - max is always <= 0
    # so addr = (max - x) & 0x7FF in practice, matching the RTL's two's-complement
    # negate-and-mask exactly for the in-range case (D_HEAD, DATA_WIDTH assumed
    # small enough that max - x never exceeds 0x7FF — true for the intended
    # 16x16 / 64x64 test configs).
    max_score = np.max(score_int, axis=1, keepdims=True)
    z_val     = score_int - max_score                      # always <= 0

    exp_Z = np.zeros_like(z_val)
    for i in range(z_val.shape[0]):
        for j in range(z_val.shape[1]):
            zv = int(z_val[i, j])
            if zv <= 0:
                addr = (-zv) & 0x7FF
            else:
                addr = 0                                    # clamp branch, dead in practice
            exp_Z[i, j] = exp_lut_arr[addr] if addr < rom_depth else 0

    # --- ST_EXP_SUM sum_acc / sum_latched: SUM_WIDTH-bit unsigned accumulator ---
    sum_width   = int(math.ceil(math.log2(max(d_head, 2)))) + exp_width
    sum_mask    = (1 << sum_width) - 1
    sum_latched = np.sum(exp_Z, axis=1, keepdims=True) & sum_mask

    # --- ST_DIV_ISSUE/DRAIN: Q1.15 fixed-point divide ---
    # div_dividend = exp_row_buf[i] << 15   (Q1.15 numerator, DIVIDEND_WIDTH bits)
    # div_divisor  = sum_latched            (SUM_WIDTH bits)
    #
    # quotient = dividend // divisor is ALREADY the Q1.15 result sitting in the
    # low EXP_WIDTH bits (verified numerically: e.g. exp=[100,300,600,1000],
    # sum=2000 -> dividend//divisor = [1638,4915,9830,16384] = frac*32768,
    # exactly the expected Q1.15 encoding). No further shift is applied here.
    #
    # exp_Z[i,j] <= sum_latched[i] by construction of softmax (each element's
    # exp value cannot exceed the row's own sum), so quotient < 2^15 always,
    # i.e. it fits the Q1.15 unsigned range [0, 32768) without saturation.
    dividend_width = exp_width + 15
    dividend_mask  = (1 << dividend_width) - 1
    div_dividend   = (exp_Z << 15) & dividend_mask
    div_divisor    = sum_latched

    # --------------------------------------------------------------------
    # Bit-true divide: reciprocal_divider.sv does NOT compute an exact
    # integer quotient — it approximates it via reciprocal-LUT + multiply +
    # shift (see reciprocal_divider.sv header). That approximation has a
    # documented max error of 1 LSB on the Q1.15 scale (recip_lut_check.py,
    # 200k+ trials). To be bit-true to the RTL (rather than mathematically
    # "more correct" than it), replicate the same LUT-based computation here,
    # element-by-element, instead of doing an exact `//` division.
    # --------------------------------------------------------------------
    seq_len_, d_head_ = exp_Z.shape
    weights_q15 = np.zeros((seq_len_, d_head_), dtype=np.int64)
    for i in range(seq_len_):
        divisor_i = int(div_divisor[i, 0])
        for j in range(d_head_):
            dividend_ij = int(div_dividend[i, j])
            weights_q15[i, j] = recip_divide_bit_true(
                dividend_ij, divisor_i, sum_width,
                recip_lut, recip_addr_w, recip_out_w, exp_width)

    return weights_q15.astype(np.int64)


# ==============================================================================
# PRINT / REPORT
# ==============================================================================
def print_report(Q_int, K_int, score_int, exp_Z, sum_exp, weights_int,
                 frac_bits, sqrt_shift, label=""):
    tag = f"[{label}] " if label else ""
    N_PRINT = 4   # elements to preview per row

    print("\n" + "=" * 70)
    print(f"  {tag}GOLDEN MODEL REPORT")
    print("=" * 70)

    print(f"\n[PARAMS] FRAC_BITS={frac_bits}  SQRT_SHIFT={sqrt_shift}"
          f"  divisor={1 << frac_bits} × {1 << sqrt_shift}"
          f" = {(1 << frac_bits) * (1 << sqrt_shift)}")

    SEQ_LEN, D_MODEL = Q_int.shape
    D_HEAD = K_int.shape[0]

    print(f"\n--- INPUT Q [{SEQ_LEN}×{D_MODEL}] and K [{D_HEAD}×{D_MODEL}]"
          f" (first {N_PRINT} cols) ---")
    for t in range(SEQ_LEN):
        q_str = " ".join(f"{int(Q_int[t,i]) & 0xFFFF:04x}({int(Q_int[t,i]):6})"
                         for i in range(min(N_PRINT, D_MODEL)))
        print(f"  Q[{t:2}]: {q_str}")
    for t in range(D_HEAD):
        k_str = " ".join(f"{int(K_int[t,i]) & 0xFFFF:04x}({int(K_int[t,i]):6})"
                         for i in range(min(N_PRINT, D_MODEL)))
        print(f"  K[{t:2}]: {k_str}")

    print(f"\n--- PHASE 1: ATTENTION SCORE [{SEQ_LEN}×{D_HEAD}] ---")
    flat = score_int.flatten()
    for i, v in enumerate(flat):
        vi = int(v)
        print(f"  [{i:3}]  {vi & 0xFFFFFFFF:08x}  ({vi:8})  {vi / float(1 << frac_bits):.4f}")

    print(f"\n--- PHASE 2: EXP VALUES [{SEQ_LEN}×{D_HEAD}] ---")
    flat_exp = exp_Z.flatten()
    for i, v in enumerate(flat_exp):
        vi = int(v)
        print(f"  [{i:3}]  {vi & 0xFFFF:04x}  ({vi:5})  {vi / float(1 << frac_bits):.4f}")
    for t in range(SEQ_LEN):
        sv = int(sum_exp[t, 0])
        print(f"  SUM_EXP row {t:2}: {sv}  ({sv / float(1 << frac_bits):.4f})")

    print(f"\n--- PHASE 3: SOFTMAX WEIGHTS [{SEQ_LEN}×{D_HEAD}] ---")
    flat_w = weights_int.flatten()
    for i, v in enumerate(flat_w):
        vi = int(v)
        print(f"  [{i:3}]  {vi & 0xFFFF:04x}  ({vi:5})  {vi / float(1 << frac_bits):.4f}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":

    print("=" * 70)
    print("  ip_axi_linear GOLDEN MODEL")
    print("  Parameters must match ip_axi_linear / linear.sv exactly.")
    print("=" * 70)

    # ── 1. PARAMETERS (match ip_axi_linear port parameters) ─────────────────
    print("\n[STEP 1] RTL Parameters")
    D_MODEL    = get_int_input("  D_MODEL    (inner dot-product dim, RTL param D_MODEL)",
                               DEFAULT_D_MODEL, min_val=2)
    SEQ_LEN    = get_int_input("  SEQ_LEN    (Q rows = S rows,        RTL param SEQ_LEN)",
                               DEFAULT_SEQ_LEN, min_val=1)
    D_HEAD     = get_int_input("  D_HEAD     (K rows = S cols,         RTL param D_HEAD)",
                               DEFAULT_D_HEAD, min_val=1)
    N_PE       = get_int_input("  N_PE       (parallel PEs, must = D_HEAD for 1-tile)",
                               DEFAULT_N_PE, min_val=1)
    DATA_WIDTH = get_int_input("  DATA_WIDTH (element bit width, signed fixed-point)",
                               DEFAULT_DATA_WIDTH, min_val=2, max_val=32)

    # Validate N_PE
    if N_PE != D_HEAD:
        print(f"  [WARN] N_PE={N_PE} != D_HEAD={D_HEAD}. "
              f"Tiling mode: golden outputs N_PE cols per tile. "
              f"Script will compute full D_HEAD output (no tiling loop here).")

    # Validate D_MODEL is power-of-2 (SQRT_SHIFT is exact only for pow2)
    if (D_MODEL & (D_MODEL - 1)) != 0:
        print(f"  [WARN] D_MODEL={D_MODEL} is not power-of-2. "
              f"$clog2(D_MODEL)/2 in RTL may not equal log2(D_MODEL)/2. "
              f"Verify SQRT_SHIFT matches RTL manually.")

    # Derived — exactly as RTL
    FRAC_BITS, SQRT_SHIFT = derive_params(D_MODEL, DATA_WIDTH)
    print(f"\n  [DERIVED] FRAC_BITS={FRAC_BITS}  (pe_unit: DATA_WIDTH/2)")
    print(f"  [DERIVED] SQRT_SHIFT={SQRT_SHIFT} (linear.sv: $clog2({D_MODEL})/{2} = "
          f"{int(math.ceil(math.log2(D_MODEL))) if D_MODEL > 1 else 1}/{2})")
    print(f"  [DERIVED] Total scaling divisor = {1 << (FRAC_BITS + SQRT_SHIFT)}")

    # ── 2. RUN MODE ──────────────────────────────────────────────────────────
    print("\n[STEP 2] Data Generation Mode")
    RUN_MODE = get_mode_input(DEFAULT_RUN_MODE)

    UNIFORM_VAL = DEFAULT_UNIFORM
    if RUN_MODE == 1:
        UNIFORM_VAL = get_int_input(
            "  UNIFORM_VAL (integer fixed-point value, e.g. 50 = 50 LSB)",
            DEFAULT_UNIFORM)

    # ── 3. GENERATE Q, K, V ─────────────────────────────────────────────────
    print("\n[STEP 3] Generating Q, K, V")

    if RUN_MODE == 1:
        print(f"  Mode 1: Uniform, all elements = {UNIFORM_VAL}")
        Q_int = np.full((SEQ_LEN, D_MODEL), UNIFORM_VAL, dtype=np.int64)
        K_int = np.full((D_HEAD,  D_MODEL), UNIFORM_VAL, dtype=np.int64)
        V_int = np.full((SEQ_LEN, D_MODEL), UNIFORM_VAL, dtype=np.int64)

    else:  # RUN_MODE == 2
        # Normal(0, 1.0) gives enough dynamic range to produce spread-out scores
        # after the double-scaling pipeline (FRAC_BITS + SQRT_SHIFT).
        # With D_MODEL=64, FRAC_BITS=8, SQRT_SHIFT=3: total divisor = 2048.
        # Normal sigma=1.0 → input_lsb σ ≈ 256 → E[score] σ ≈ sqrt(64)*256²/2048 ≈ 256
        # → score range roughly ±500..±700 with good spread across all output elements.
        # Clipped to DATA_WIDTH signed range to prevent int64 MAC overflow
        # (max_mac = (3σ×256)² × 64 ≈ 1.1e9, well within int64 = 9.2e18).
        print("  Mode 2: Random, seed=42, Normal(0, 1.0) → fixed-point Q<INT>.<FRAC> encoding")
        np.random.seed(42)
        Q_f   = np.random.normal(0.0, 1.0, (SEQ_LEN, D_MODEL)).astype(np.float32)
        K_f   = np.random.normal(0.0, 1.0, (D_HEAD,  D_MODEL)).astype(np.float32)
        V_f   = np.random.normal(0.0, 1.0, (SEQ_LEN, D_MODEL)).astype(np.float32)
        Q_int = float_to_fixed(Q_f, FRAC_BITS)
        K_int = float_to_fixed(K_f, FRAC_BITS)
        V_int = float_to_fixed(V_f, FRAC_BITS)

    print(f"  Q shape: {Q_int.shape}  K shape: {K_int.shape}  V shape: {V_int.shape}")
    print(f"  Q range: [{Q_int.min()}, {Q_int.max()}]")
    print(f"  K range: [{K_int.min()}, {K_int.max()}]")

    # ── 4. EXP LUT ───────────────────────────────────────────────────────────
    print("\n[STEP 4] Generating exp LUT")
    exp_lut = generate_exp_lut(FRAC_BITS)
    write_coe_16("exp_rom.coe", exp_lut)
    write_mem_16("exp_rom.mem", exp_lut)

    # ── 4b. RECIPROCAL LUT (for ip_axi_softmax's reciprocal_divider) ────────
    # Fixed-size table, independent of D_MODEL/SEQ_LEN/D_HEAD above — only
    # needs generating once. Re-run this step only if RECIP_ADDR_W/
    # RECIP_OUT_W themselves change (not tied to attention dimensions).
    print("\n[STEP 4b] Generating reciprocal LUT (for softmax.sv reciprocal_divider)")
    recip_lut = generate_recip_lut(RECIP_ADDR_W, RECIP_OUT_W)
    write_coe_generic("recip_rom.coe", recip_lut, RECIP_OUT_W)
    print(f"  RECIP_ADDR_W={RECIP_ADDR_W} (ROM depth={1<<RECIP_ADDR_W})  RECIP_OUT_W={RECIP_OUT_W}")
    print(f"  -> In Vivado: Block Memory Generator, Width={RECIP_OUT_W}, "
          f"Depth={1<<RECIP_ADDR_W}, Load Init File = recip_rom.coe, "
          f"instance name 'recip_rom', Port A latency=1 (Core Output "
          f"Register OFF, Primitives Output Register ON — default single "
          f"sync-read stage, matches exp_rom's existing config).")

    # ── 5. COMPUTE ───────────────────────────────────────────────────────────
    print("\n[STEP 5] Computing attention score (Phase 1)")
    score_int = compute_attention_score(Q_int, K_int, FRAC_BITS, SQRT_SHIFT, DATA_WIDTH)
    print(f"  Score range: [{score_int.min()}, {score_int.max()}]")

    print("\n[STEP 6] Computing softmax (Phase 2)")
    exp_Z, sum_exp, weights_int = compute_softmax(score_int, exp_lut, FRAC_BITS)

    print("\n[STEP 6b] Computing bit-true softmax golden (ip_axi_softmax, Q1.15)")
    weights_q15 = compute_softmax_golden(score_int, exp_lut,
                                         exp_width=DATA_WIDTH, d_head=D_HEAD,
                                         recip_lut=recip_lut,
                                         recip_addr_w=RECIP_ADDR_W,
                                         recip_out_w=RECIP_OUT_W)

    # ── 6. WRITE FILES ───────────────────────────────────────────────────────
    print("\n[STEP 7] Writing output files")

    # Q, K, V — flatten row-major, 32-bit words (16-bit data zero-padded to 32)
    write_coe_32("q_ram.coe", Q_int.flatten())
    write_coe_32("k_ram.coe", K_int.flatten())
    write_coe_32("v_ram.coe", V_int.flatten())

    write_mem_32("q_ram.mem", Q_int.flatten())
    write_mem_32("k_ram.mem", K_int.flatten())
    write_mem_32("v_ram.mem", V_int.flatten())

    # golden_score.mem: SEQ_LEN × D_HEAD elements, 32-bit, row-major
    write_golden_score(score_int)

    # golden_softmax.mem: SEQ_LEN × D_HEAD elements, Q1.15 unsigned, 32-bit words
    write_golden_softmax(weights_q15)

    # attn_test_data.h: k_data/q_data/golden_score/golden_softmax for main.c
    # bare-metal bring-up. Same flatten order as write_mem_32(Q_int.flatten())/
    # write_mem_32(K_int.flatten())/write_golden_score(score_int)/
    # write_golden_softmax(weights_q15) above — must not diverge from that
    # order, since K_BASE/Q_BASE offsets in main.c assume row-major flatten
    # identical to what the .mem files already encode.
    # NOTE: main.c's combined linear+softmax pipeline captures DST_BASE AFTER
    # softmax runs, so it must compare against golden_softmax, not golden_score.
    write_c_header("attn_test_data.h",
                    K_int.flatten(), Q_int.flatten(), score_int.flatten(),
                    weights_q15.flatten())

    # ── 7. REPORT ────────────────────────────────────────────────────────────
    label = "UNIFORM" if RUN_MODE == 1 else "RANDOM"
    print_report(Q_int, K_int, score_int, exp_Z, sum_exp, weights_int,
                 FRAC_BITS, SQRT_SHIFT, label=label)

    # ── 8. SUMMARY ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  D_MODEL={D_MODEL}  SEQ_LEN={SEQ_LEN}  D_HEAD={D_HEAD}"
          f"  N_PE={N_PE}  DATA_WIDTH={DATA_WIDTH}")
    print(f"  FRAC_BITS={FRAC_BITS}  SQRT_SHIFT={SQRT_SHIFT}")
    print(f"  Q  : [{SEQ_LEN} × {D_MODEL}]  →  q_ram.mem  ({SEQ_LEN*D_MODEL} words)")
    print(f"  K  : [{D_HEAD}  × {D_MODEL}]  →  k_ram.mem  ({D_HEAD*D_MODEL} words)")
    print(f"  S  : [{SEQ_LEN} × {D_HEAD}]   →  golden_score.mem   ({SEQ_LEN*D_HEAD} words)")
    print(f"  W  : [{SEQ_LEN} × {D_HEAD}]   →  golden_softmax.mem ({SEQ_LEN*D_HEAD} words, Q1.15)")
    print(f"  H  : k_data/q_data/golden_score  →  attn_test_data.h")
    print(f"  Mode: {label}")
    print(f"  Files written to: {MEM_OUT_PATH}")
    print("=" * 70)