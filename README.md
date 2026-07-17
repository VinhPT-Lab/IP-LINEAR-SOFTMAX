# Dự án tích hợp: DMA + IP Linear (Q×Kᵀ) + IP Softmax trên Zynq KV260

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Memory map](#3-memory-map)
4. [IP Linear (Q×Kᵀ) — tóm tắt](#4-ip-linear-qkᵀ--tóm-tắt)
5. [IP Softmax — tóm tắt](#5-ip-softmax--tóm-tắt)
6. [Trình tự vận hành phần mềm (bare-metal)](#6-trình-tự-vận-hành-phần-mềm-bare-metal)
7. [Kiểm chứng](#7-kiểm-chứng)
8. [Kết quả](#8-kết-quả).

---

## 1. Tổng quan

Hệ thống tính **Softmax(Q × Kᵀ)** — 2 giai đoạn đầu của Scaled Dot-Product Attention — trên phần cứng Zynq UltraScale+ (KV260), điều khiển bằng bare-metal C qua Vitis.

- **IP Linear**: tính `S = Q × Kᵀ`.
- **IP Softmax**: nhận `S`, tính softmax theo từng hàng, trả về attention weight (Q1.15 unsigned).
- Kết nối 2 IP qua DMA (không stream trực tiếp IP-to-IP): `linear` ghi kết quả `S` ra BRAM qua S2MM, sau đó `softmax` đọc lại từ BRAM qua MM2S riêng — 2 IP không share cùng 1 kênh AXI-Stream.
- Điều khiển: mỗi IP có control-plane AXI4-Lite riêng (start/status), data-plane qua `axi_dma_0` (1 kênh MM2S + 1 kênh S2MM dùng chung cho cả 2 IP, tuần tự).

**Đã tích hợp và test PASS trên KV260** với cấu hình tham số:
```
SEQ_LEN = D_HEAD = N_PE = DATA_WIDTH = 16
```
(riêng `D_MODEL` của linear giữ mặc định — xem mục 4).

---

## 2. Kiến trúc hệ thống

<p align="center">
  <img src="https://github.com/user-attachments/assets/90197d86-ffbc-48a5-b931-99e4b7906829" alt="Sơ đồ kiến trúc hệ thống" width="700">
  <br>
  <em>Sơ đồ kiến trúc hệ thống</em>
</p>

- `bram_ctrl_src`: chứa `K` rồi `Q` (preload bởi PS trước khi chạy).
- `bram_ctrl_dst`: dùng chung 2 lần — lần 1 nhận output `S` từ linear, lần 2 bị **ghi đè** bởi output softmax (đọc `S` từ chính `bram_dst`, ghi weight trở lại `bram_dst`). Xem chi tiết thứ tự ở mục 6.
- Interconnect: AXI SmartConnect (1 clock domain, không CDC nội bộ).

---

## 3. Memory map

| Base address | Vùng | Ghi chú |
|---|---|---|
| `0xA000_0000` | `SRC_BASE` (bram_ctrl_src) | K tại offset 0, Q tại offset `D_HEAD×D_MODEL×4` |
| `0xA001_0000` | `DST_BASE` (bram_ctrl_dst) | Output S (từ linear) → bị ghi đè bởi output softmax |
| `0xA002_0000` | `DMA_BASE` (axi_dma_0 S_AXI_LITE) | Thanh ghi điều khiển DMA |
| `0xA003_0000` | `IP_LINEAR_BASE` (ip_axi_linear_0 S00_AXI) | Control-plane linear |
| `0xA004_0000` | `IP_SOFTMAX_BASE` (ip_axi_softmax_0 S00_AXI) | Control-plane softmax |

**DMA register offset (dùng chung cho cả 2 lượt MM2S/S2MM):**

| Offset từ `DMA_BASE` | Tên | Ý nghĩa |
|---|---|---|
| `0x00` | `MM2S_DMACR` | bit[0]=RS, bit[2]=Reset |
| `0x04` | `MM2S_DMASR` | bit[1]=Idle, bit[12]=IOC_IRQ, bit[14]=ERR_IRQ |
| `0x18` | `MM2S_SA` | Source address |
| `0x28` | `MM2S_LENGTH` | Length (byte) — ghi = trigger transfer |
| `0x30` | `S2MM_DMACR` | bit[0]=RS, bit[2]=Reset |
| `0x34` | `S2MM_DMASR` | bit[1]=Idle, bit[12]=IOC_IRQ, bit[14]=ERR_IRQ |
| `0x48` | `S2MM_DA` | Destination address |
| `0x58` | `S2MM_LENGTH` | Length (byte) — ghi = trigger transfer |

---

## 4. IP Linear (Q×Kᵀ) — tóm tắt

<p align="center">
  <img src="https://github.com/user-attachments/assets/413d0b3e-ae3a-4b55-87dd-84382f12c2bc" alt="Sơ đồ FSM IP LINEAR" width="700">
  <br>
  <em>Sơ đồ FSM IP LINEAR</em>
</p>

**Tham số:**

| Parameter | Giá trị đã test | Mô tả |
|---|:---:|---|
| `D_MODEL` | 64 | Số chiều embedding |
| `SEQ_LEN` | 16 | Số hàng Q / hàng output |
| `D_HEAD` | 16 | Số hàng K / cột output |
| `N_PE` | 16 | Số PE song song (`N_PE ≤ D_HEAD`, hỗ trợ tiling khi nhỏ hơn) |
| `DATA_WIDTH` | 16 | Bit width signed fixed-point |

Ràng buộc bắt buộc: `N_PE ≤ D_HEAD`, `D_MODEL ≥ D_HEAD`, `N_PE × N_TILES ≥ D_HEAD`.

**FSM — 5 state:** `IDLE → LOAD_K → PRELOAD_MAC → COMPUTE → DONE → (IDLE)`
- `LOAD_K`: nhận K qua AXI-Stream, ghi vào BRAM.
- `PRELOAD_MAC`: nạp weight K vào từng PE (multi-bank nếu tiling).
- `COMPUTE`: row-outer/tile-inner — mỗi hàng Q chạy qua các tile PE, MAC tích lũy, ping-pong buffer serialize output song song với hàng kế tiếp.
- `o_m_axis_tlast` chỉ assert ở word cuối **toàn frame** (bắt buộc cho AXI DMA S2MM direct mode).

**Register map (offset 0x0/0x4/0x8/0xC, giống cấu trúc chuẩn):** bit[0] tại `0x0` = Start; `0x4` đọc trả `{busy, done}` tổ hợp trực tiếp (không qua FF trung gian).

Chi tiết đầy đủ (kiến trúc PE array, double-buffer, tiling, testbench, timing): xem `README` riêng của IP linear.

---

## 5. IP Softmax — tóm tắt

<p align="center">
  <img src="https://github.com/user-attachments/assets/06eb6b05-990c-48bc-9484-df38490e8be0" width="700">
  <br>
  <em>Sơ đồ FSM IP SOFTMAX</em>
</p>

**Tham số:**

| Parameter | Giá trị đã test | Mô tả |
|---|:---:|---|
| `D_HEAD` | 16 | Số phần tử mỗi hàng |
| `SEQ_LEN` | 16 | Số hàng |
| `DATA_WIDTH` | 16 | Bit width input `S` (signed) |
| `EXP_WIDTH` | 16 | Bit width `exp_rom` / kết quả (unsigned) |
| `RECIP_ADDR_W` | 12 | Địa chỉ ROM reciprocal — không phụ thuộc `D_HEAD`/`SEQ_LEN` |
| `RECIP_OUT_W` | 19 | Output ROM reciprocal (Q0.19), ràng buộc `RECIP_OUT_W > EXP_WIDTH` |

Output: **Q1.15 unsigned**, mỗi phần tử trong `[0, 1)`.

**FSM — 8 state:** `IDLE → LOAD_ROW → FIND_MAX → EXP_SUM → DIV_ISSUE → DIV_DRAIN → SERIALIZE → (loop hoặc DONE)`
- `FIND_MAX`: ổn định số học, trừ max trước khi tính `exp`.
- `EXP_SUM`: pipeline 2-tầng qua `exp_rom` (BRAM IP, 1-cycle latency), cộng dồn `sum_acc`.
- `DIV_ISSUE/DRAIN`: chia bằng `reciprocal_divider` (reciprocal-LUT `recip_rom` + nhân + dịch, latency cố định 3-cycle, thay cho Xilinx `div_gen` 55-cycle).
- Xử lý tuần tự từng hàng, không pipeline chồng giữa các hàng.

**Register map:** `0x00` bit[0] = Start (tự clear khi busy rising); `0x04` bit[0] = Done (latched), bit[1] = Busy (tổ hợp).

**ROM phụ trợ:** `exp_rom.coe` (2048 entry), `recip_rom.coe` (4096 entry) — cả hai là Block Memory Generator IP, sinh bởi `golden_model.py`.

Chi tiết đầy đủ (flow tính toán per-cycle, thuật toán reciprocal-divider, bug lịch sử): xem `README` riêng của IP softmax.

---

## 6. Trình tự vận hành phần mềm (bare-metal)

Đúng theo `main.c` đã chạy PASS trên KV260:

```
// Phase 1: Preload
1. Ghi K vào SRC_BASE                              (D_HEAD × D_MODEL × 4 byte)
2. Ghi Q vào SRC_BASE + NUM_BYTES_K                 (SEQ_LEN × D_MODEL × 4 byte)

// Phase 2: Reset DMA
3. MM2S_DMACR[RESET]=1, S2MM_DMACR[RESET]=1, poll tới khi tự clear

// Phase 3: Linear — Load K, kick FSM, stream K
4. MM2S_DMACR[RS] = 1
5. MM2S_SA = K_BASE
6. IL_START = 0x1                                    (kick linear FSM → ST_LOAD_K)
7. MM2S_LENGTH = NUM_BYTES_K                          (trigger MM2S: K → linear S_AXIS)
8. Poll MM2S_DMASR[IOC_IRQ], clear IOC

// Phase 4: kick Softmax NGAY (trước khi linear compute xong) + arm S2MM(1) + stream Q
9. SM_START = 0x1                                    (kick softmax FSM — softmax sẽ tự đợi
                                                        dữ liệu tới qua AXI-Stream riêng của nó,
                                                        không tranh chấp với linear vì 2 IP có
                                                        S00_AXIS/M00_AXIS độc lập)
10. S2MM_DMACR[RS] = 1
11. S2MM_DA = DST_BASE
12. S2MM_LENGTH = NUM_BYTES_OUT                       (arm output S2MM — nhận output SAU CÙNG
                                                        của toàn chuỗi, thực chất là output
                                                        SOFTMAX chứ không phải linear — xem lưu ý)
13. MM2S_SA = Q_BASE
14. MM2S_LENGTH = NUM_BYTES_Q                         (trigger MM2S: Q → linear S_AXIS)

// Phase 5: đợi hoàn tất
15. Poll MM2S_DMASR[IOC_IRQ]                          (Q transfer xong)
16. Poll S2MM_DMASR[IOC_IRQ]                          (DST_BASE đã có dữ liệu cuối cùng)

// Phase 6: đọc kết quả
17. Đọc DST_BASE — đây là OUTPUT SOFTMAX (Q1.15), KHÔNG phải score thô của linear
```

**Lưu ý bắt buộc — thứ tự phase 3/4 không được đảo:**
`IL_START` phải ghi **trước** `MM2S_LENGTH` của K, nếu không FSM linear vẫn ở `IDLE` (`tready=0`) khi DMA đã trigger, gây stall MM2S ngay beat đầu.

**Lưu ý bắt buộc — ý nghĩa dữ liệu tại `DST_BASE`:**
`S2MM_LENGTH` (bước 12) chỉ arm **một lần**, nhận đúng `NUM_BYTES_OUT` byte cuối cùng ghi vào `DST_BASE` trong toàn bộ chuỗi thao tác. Vì `SM_START` được kick trước khi arm S2MM và chuỗi lệnh chỉ dùng 1 cặp MM2S/S2MM cho cả 2 IP, `DST_BASE` sau khi hoàn tất **chứa output của softmax**, không phải score thô `S` của linear — dù cùng 1 vùng địa chỉ vật lý được dùng làm đích trung gian rồi đích cuối. So sánh kết quả (golden) phải dùng `golden_softmax`, không dùng `golden_score`.

---

## 7. Kiểm chứng

- Testbench mức IP riêng (`tb_ip_axi_linear`, `tb_ip_axi_softmax`, `tb_dma_*`): dùng AXI VIP, so khớp golden model theo từng IP — **PASS** ở nhiều cấu hình tham số khác nhau (xem README riêng từng IP).
- Testbench tích hợp cả 2 IP + DMA (`tb_dma_linear_softmax`): AXI VIP làm master thay PS — **PASS**.
- Bring-up trên KV260 thật (Vitis bare-metal, `main.c`), cấu hình `SEQ_LEN=D_HEAD=N_PE=DATA_WIDTH=16`: **PASS** — output tại `DST_BASE` khớp `golden_softmax` 100%.
- Golden data: `golden_model.py` xuất `golden_score.mem` (input softmax) và `golden_softmax.mem` (kết quả cuối, Q1.15), cùng 2 file `.coe` cho `exp_rom`/`recip_rom`.

---
## 8. Kết quả

<p align="center">
  <img src="https://github.com/user-attachments/assets/a9ddefc3-a80e-48d2-ad77-fcec934fc4f8" alt="KẾT QUẢ BARE METAL DMA + IP LINEAR" width="700">
  <br>
  <em>KẾT QUẢ BARE METAL DMA + IP LINEAR</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ef67266-fff5-45d2-bacc-2d2511215467" alt="KẾT QUẢ BARE METAL DMA + IP LINEAR + IP SOFTMAX" width="700">
  <br>
  <em>KẾT QUẢ BARE METAL DMA + IP LINEAR + IP SOFTMAX</em>
</p>

---
