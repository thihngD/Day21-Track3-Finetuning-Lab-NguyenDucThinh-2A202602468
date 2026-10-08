# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đức Thịnh  **MSSV**: 2A202602468  **Ngày**: 2026-10-08
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6GB khả dụng)`

> Mọi con số dưới đây khớp với file thật trong `results/` (đã đồng bộ về repo từ chạy Colab T4 ngày 2026-10-08, commit `d27c1c0`) và `adapters/correct/`. Gatekeeper: 25 passed · 1 warning · 1 fail — fail duy nhất là bản REPORT.md mẫu trong repo gốc, đã được thay bằng file này.

---

## 1. Setup

| | |
|---|---|
| Dataset | Ticket CSKH → JSON triage (mặc định lab), 250 mẫu |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | `1024` — p95 đo được là `98` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs → 30 optimizer step (áp dụng chung cho cả 4 run NB3+NB4) |

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json)*: render thử `<|im_start|>assistant\n<think>...</think>\n4<|im_end|>` ra đúng cấu trúc, script tự chấm `VERDICT: reasoning preserved — safe to train on traces`.

**Về lệch `max_length`**: giữ nguyên default tier T4 = 1024 dù p95 chỉ 98 (p99=100, max=101). Lý do: Qwen3.5-4B là kiến trúc hybrid linear-attention (24/32 layer linear, 8 layer full-attention theo config in-run), batch=1 + gradient checkpointing nên dư token budget không tốn thêm VRAM đáng kể, còn hạ sát 256 theo gợi ý thì rủi ro cắt dữ liệu outlier mà không lợi gì rõ về tốc độ ở quy mô 250 mẫu. Nếu tối ưu thời gian train, lần sau có thể hạ về 128–256.

**Vì sao chọn model + dataset này?** Dùng đúng model mặc định mà tier `T4` map tới (`unsloth/Qwen3.5-4B`) thay vì tự đổi, vì nó vừa khít VRAM 16GB của T4 free (full LoRA fp16 chỉ 8.78GB, còn dư băng cho batch/eval) và đã được lab kiểm tra tương thích (hybrid linear-attention, hỗ trợ `<think>`). Dataset dùng bộ seed có sẵn (250 ticket CSKH → JSON triage) vì đề bài đủ rõ, đóng (closed-vocab 3 field + 1 field tự do), dễ chấm tự động bằng accuracy từng field — phù hợp để đo đúng câu hỏi của lab (pipeline/thiết kế thí nghiệm) mà không tốn thời gian tự làm dataset riêng trong khung ~2 giờ.

**Mốc đã đóng băng**: tập eval `target` (50) và `regression` (15) được đo baseline (a)/(b) **trước khi train** (NB2), sau đó dùng lại nguyên vẹn cho (c) và cả 3 run NB4 — gatekeeper xác nhận `eval sets unmodified` và `checksum` khớp, không có việc sửa eval sau khi thấy kết quả.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (ví dụ thật trong `train_seed.jsonl`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

41% token được supervise — xa ngưỡng 0.95 (mức "tính loss cả trên prompt" bị mất trắng mục 1.1) nên pipeline mask đúng.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3195.8 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1016.7 |
| (c) LoRA fine-tune | 0.970 | 0.500 | 1.000 | 1419.5 |

**(b) có thật sự mạnh hơn (a) không?** **Có**, rõ ràng — chỉ sửa prompt (chưa train) đã kéo target từ 0 lên 0.765 và format từ 0 lên 1.0. `OPTIMIZED_PROMPT` không bị sửa lại sau khi thấy kết quả (SHA chốt: `719e74d3b6232053`), nên phép so sánh (b) vs (c) công bằng, không có việc làm yếu (b) để (c) trông thắng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | latency (ms) | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6248 | **0.970** | 1.000 | 1419.5 | 396 | 8.78 |
| `attn_only` | q,v *(matched)* | 283 | 32,456,704 | 1e-4 | 0.5372 | **0.970** | 1.000 | 917.6 | 263 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 0.000 | 5393.3 | 394 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 (4-bit) | 1e-4 | 0.7058 | **0.940** | 1.000 | 1805.1 | 461 | 3.86 |

Xếp hạng theo cột target (NB5), không theo train loss (NB4) — đúng chuẩn rubric 2.5.

**Tính công bằng của phép so sánh** (tự `make verify`/gatekeeper kiểm tra, không phải tự nhận):
- `attn_only` lệch ngân sách tham số so với `correct` chỉ **0.025%** (32,456,704 vs 32,464,896 — rank được nâng lên 283 *chỉ để* bù lại việc chỉ gắn vào 2 module, không phải một biến thử nghiệm độc lập). Biến duy nhất đang so sánh ở run này là **vị trí** (q,v vs text-linear); rank là hệ số bù giữ ngân sách cố định, không phải biến thứ hai — tránh đúng cái bẫy rubric cảnh báo (so `q,v@r=16` với `all-linear@r=16` là so *ngân sách*, không phải so *vị trí*).
- Cả 4 run (`correct`, `attn_only`, `wrong_lr`, `qlora`) đều train đúng **30 optimizer step** (2 epoch × 225 mẫu, batch hiệu dụng 16) — cùng ngân sách compute, chỉ khác đúng 1 biến mỗi run: `attn_only` đổi **vị trí**, `wrong_lr` đổi **learning rate**, `qlora` đổi **precision** (4-bit).

**4.1 — `attn_only` vs `correct`.** Cùng ngân sách tham số (~32.46M, chỉ khác số module và rank). Trên target, hai run **hoà tuyệt đối** (0.970 = 0.970), dù `attn_only` có train loss thấp hơn (0.5372 vs 0.6248) và nhanh hơn (263s vs 396s). Thứ tự theo train loss không khớp thứ tự theo target vì... không có thứ tự nào cả — target bằng nhau. Điều này nói rằng khi giữ ngân sách tham số bằng nhau, **vị trí gắn adapter không phải đòn bẩy chính** — rank cao bù lại việc chỉ gắn vào 2 module (q,v) cho kết quả tương đương gắn rank thấp vào 12 module. Train loss thấp hơn chỉ phản ánh việc rank cực lớn dễ fit nhanh trên tập train, không phải generalize tốt hơn trên eval.

**4.2 — `wrong_lr`.** Chỉ đổi một số: LR 1e-4 → 1e-5 (tức scale LR kiểu full fine-tune áp vào LoRA). Đường loss vẫn giảm dần, mượt, không NaN/spike (2.16 → 2.07 → 1.61 → 1.33 → 1.14 → 1.12), final loss 1.5702 — cao gần gấp 2.5 lần `correct`. Nếu chỉ nhìn đường loss mà không biết LR, dễ kết luận sai rằng "model vẫn đang học, train thêm step/epoch là ổn" — thực tế target = 0.000, format = 0.000, ngang hệt baseline (a) chưa train gì. Loss giảm đều **không chứng minh** model học được task. Thêm một tín hiệu phụ: latency của `wrong_lr` (5393ms) cao gấp 3–6 lần các run khác — model chưa học được khi nào nên dừng (EOS), sinh ra output dài/lặp, một triệu chứng khác của LR quá thấp mà nếu chỉ nhìn loss sẽ không thấy được.

**4.3 — `qlora`.** Tiết kiệm VRAM 8.78GB → 3.86GB (**-56%**), trả giá bằng thời gian train chậm hơn (461s vs 396s, +16%, do overhead dequant 4-bit) và target giảm nhẹ 0.970 → 0.940 (-0.03). Số đo này **không** ủng hộ khuyến nghị tuyệt đối "không dùng QLoRA cho dòng model này": mất accuracy rất nhỏ, đổi lại VRAM giảm hơn một nửa. Với model 4B trên T4 16GB thì full LoRA fp16 (8.78GB) vẫn vừa nên QLoRA không bắt buộc ở quy mô này, nhưng với model lớn hơn hoặc GPU nhỏ hơn, số liệu cho thấy QLoRA là lựa chọn hợp lý — nên đo thực tế, không nên theo khuyến nghị chung một cách máy móc.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.291` · `valid_trace_rate = 0.00`

Fine-tune đạt target rất tốt (0.765 → 0.970, +0.205) và format hoàn hảo (1.0), nhưng regression (năng lực tổng quát, đo trên tập riêng 15 mẫu) tụt từ 0.791 xuống 0.500 (-0.291) — mất gần 37% năng lực tổng quát so với trước khi train, vượt xa tolerance 0.020 của gate gấp hơn 14 lần → bị chặn, đúng theo thiết kế, và dứt khoát hơn hẳn mức "regressed by 0.113" ở một lần chạy trước đó với cùng cấu hình (chênh lệch do sinh văn bản có ngẫu nhiên khi đánh giá, không cố định seed generate). Đây là dấu hiệu catastrophic forgetting rõ rệt: 30 step với LR cao (1e-4) trên 225 mẫu thu hẹp vào một domain (ticket CSKH) đã đè lên đáng kể năng lực tổng quát, dù task chính học cực tốt — target tăng (+0.205) không bù lại được cái giá phải trả ở regression. `valid_trace_rate = 0.00` cũng đáng chú ý: output sau fine-tune không còn giữ cấu trúc `<think>` hợp lệ — tuy vậy dữ liệu train gốc cho task này vốn có khối think rỗng (không phải reasoning thật), nên đây không hẳn là "collapse" theo nghĩa mất suy luận, mà là hệ quả tự nhiên của việc task classification không cần chain-of-thought. FAILED ở đây là phán quyết nên tin, không nên hạ tolerance để cho pass — ngược lại, độ dao động giữa hai lần chạy (-0.113 vs -0.291) còn cho thấy nên chạy lại NB5 vài lần và báo cáo khoảng dao động thay vì một con số đơn lẻ, nếu muốn kết luận chắc chắn hơn.

---

## 6. Định tính — bắt buộc có cả ca THUA

Nhãn đúng lấy trực tiếp từ `data/eval_target.jsonl` (frozen eval set), khớp theo index `i` của `qualitative.json`. Cột "(b) prompt" pipeline không lưu per-mẫu (chỉ lưu điểm trung bình (b) target=0.765) nên để trống có chủ đích, không bịa.

`qualitative.json` thực tế có **5 ca điểm 0.75** (hoà nhau), không chỉ 3 — lấy đủ cả 5 vì cùng chung một nguyên nhân, không cherry-pick:

| # | Ticket (rút gọn) | Nhãn đúng (intent/urgency/product/sentiment) | (c) fine-tune dự đoán | ft_score | Nhận xét |
|---|---|---|---|---|---|
| 1 | #47 "...ốp lưng điện thoại DH936478. Shipper không gọi." | van_chuyen / thap / ốp lưng điện thoại / tich_cuc | van_chuyen / thap / ốp lưng điện thoại / tich_cuc | 1.00 | ✅ FT thắng — đúng cả 4 field |
| 2 | #48 "...ốp lưng điện thoại DH734695. Giá bao nhiêu." | hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh | hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh | 1.00 | ✅ FT thắng — đúng cả 4 field |
| 3 | #3 "...bình giữ nhiệt VN804124. Chưa thấy tiền." | hoan_tien / **thap** / bình giữ nhiệt / tich_cuc | hoan_tien / **trung_binh** / bình giữ nhiệt / tich_cuc | 0.75 | ❌ **FT thua** — sai field `urgency` |
| 4 | #5 "...nồi chiên không dầu DH249548. Thiếu phụ kiện." | san_pham_loi / **thap** / nồi chiên không dầu / trung_tinh | san_pham_loi / **trung_binh** / nồi chiên không dầu / trung_tinh | 0.75 | ❌ **FT thua** — sai field `urgency` |
| 5 | #12 "...áo khoác gió VN613097. Bị lỗi. Khi nào tiện." | san_pham_loi / **thap** / áo khoác gió / tich_cuc | san_pham_loi / **trung_binh** / áo khoác gió / tich_cuc | 0.75 | ❌ **FT thua** — sai field `urgency` |
| 6 | #39 "...nồi chiên không dầu VN949966. Hoàn tiền." | hoan_tien / **thap** / nồi chiên không dầu / tieu_cuc | hoan_tien / **trung_binh** / nồi chiên không dầu / tieu_cuc | 0.75 | ❌ **FT thua** — sai field `urgency` |
| 7 | #41 "...đèn bàn LED OD436045. Giao hàng chậm." | van_chuyen / **thap** / đèn bàn LED / tich_cuc | van_chuyen / **trung_binh** / đèn bàn LED / tich_cuc | 0.75 | ❌ **FT thua** — sai field `urgency` |

> `ft_score=0.75` = đúng 3/4 field (`triage_field_accuracy` chia đều 1/4 mỗi field) — field còn lại (`sentiment` hoặc field khác ngoài `urgency`) suy ra chắc chắn đúng từ điểm số, không cần đoán.

**Mẫu chung ở các ca FT thua — rất rõ, không mơ hồ, và nhất quán 100% trên cả 5/5 ca:** mọi ca đều sai đúng field `urgency`, và tất cả 5 ticket đều chứa cụm giảm nhẹ mức độ gấp (**"Khi nào tiện"** hoặc tương đương — ý: không gấp, nhãn đúng = `thap`) nhưng model luôn đoán `trung_binh`. Đây không phải lỗi ngẫu nhiên — model học tốt `urgency=cao` (từ "gấp", "khẩn", "ngay lập tức") và `urgency=thap` qua "không vội", nhưng cụm "khi nào tiện" (một cách nói giảm nhẹ ít xuất hiện hơn trong 250 mẫu train) bị kéo về nhãn mặc định `trung_binh`. Ngược lại 2 ca thắng tuyệt đối đều là ticket "ốp lưng điện thoại" với các cụm urgency rõ ràng hơn (đã có sẵn trong vốn từ quen của model).

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Không nên deploy bản fine-tune `correct` ở dạng hiện tại: verdict FAILED vì regression vượt tolerance rất xa (gấp >14 lần ngưỡng cho phép). Target tăng mạnh (+0.205) và format hoàn hảo là tín hiệu tốt, nhưng model đã đánh đổi gần 37% năng lực tổng quát (regression 0.791 → 0.500) để đạt điều đó — nghĩa là khi deploy thật, hệ thống nhiều khả năng trả lời sai ở các câu hỏi ngoài domain CSKH mà nó vẫn phải xử lý, và mức độ đánh đổi này đủ lớn để không thể biện minh bằng mức tăng target. Về đòn bẩy thật sự trong lab: **không phải vị trí gắn adapter** — `attn_only` và `correct` hoà nhau khi ngân sách tham số bằng nhau (0.970 = 0.970), nghĩa là rank cao ở ít module ≈ rank thấp ở nhiều module, miễn tổng tham số giống nhau. **Learning rate mới là đòn bẩy rõ nhất**: đổi đúng một số (1e-4 → 1e-5) kéo target từ 0.970 xuống 0.000 — lớn hơn hẳn ảnh hưởng của vị trí hay rank. Đòn bẩy thứ hai, ít lộ ra qua train loss nhưng quyết định việc pass/fail gate, là **chất lượng và độ đa dạng dữ liệu**: 225 mẫu chỉ từ một domain hẹp là nguyên nhân trực tiếp khiến regression tụt mạnh — thiếu dữ liệu replay (1–5% dữ liệu phổ thông) để giữ năng lực tổng quát. Nếu làm lại, ưu tiên thử thêm replay data trước khi thử các biến thể vị trí/rank khác, vì dữ liệu mới là nút nghẽn hiện tại, không phải cấu hình LoRA.

**Ba điều tôi học được:**
1. LR đúng scale quan trọng hơn vị trí/rank adapter — một con số sai (x10 → x1) phá hỏng toàn bộ kết quả, trong khi đổi vị trí (với ngân sách khớp) gần như không đổi gì.
2. Train loss không phải proxy đáng tin cho performance thật — `attn_only` có train loss thấp hơn `correct` nhưng target chỉ hoà, không thắng; phải luôn xếp hạng bằng eval target, không bằng loss.
3. Fine-tune thắng trên target không đồng nghĩa "an toàn để deploy" — phải đo regression riêng, vì domain hẹp rất dễ kéo tụt năng lực tổng quát mà train loss không hề cảnh báo trước.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 2–3% dữ liệu tổng quát (replay) vào training set rồi train lại `correct`, xem regression delta có về trong tolerance 0.02 không; sau đó mới cân nhắc NB6 (merge + hot-swap).

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
