# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

`attn_only` (gắn adapter chỉ vào q,v, rank nâng lên 283 để khớp ngân sách tham số) cho target bằng đúng `correct` (gắn vào toàn bộ text-linear, r=16): 0.965 = 0.965. Trước khi chạy tôi nghĩ gắn vào nhiều module (text-linear) chắc phải tốt hơn vì "phủ" nhiều phần model hơn. Thực tế không khác gì — chỉ cần tổng tham số bằng nhau.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

NB4 (3 run đối chứng) — 22.4 phút, hơn cả NB2+NB3+NB5 cộng lại. Đúng như dự đoán vì đây là 3 lần train full, nhưng không ngờ `qlora` lại là run train *chậm nhất* (483s) trong khi tốn ít VRAM nhất (3.86GB) — tưởng 4-bit sẽ nhanh hơn vì ít dữ liệu di chuyển, hoá ra overhead dequant làm chậm lại.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tin rằng "loss giảm đều là tín hiệu tốt". Run `wrong_lr` loss giảm mượt từ 2.16 xuống 1.12 qua 30 step, không hề có dấu hiệu bất thường (không NaN, không spike) — nhưng target = 0.000, coi như không học được gì. Nhìn riêng đường loss sẽ đánh giá sai hoàn toàn.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng để đọc lại log chạy (`Lab21_RUN_ALL.ipynb`) và tổng hợp số liệu rải rác qua nhiều cell thành report. Giới hạn thấy rõ: log console cắt bảng qualitative (chuỗi dự đoán bị truncate ở ~80 ký tự, không có cột nhãn đúng) nên AI không tự bịa ra được — phải tự mở `results/qualitative.json` thật để điền đủ mục 6 của report.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đo baseline (b) — prompt tối ưu trên model chưa train — trước khi động vào LoRA. Trong lab này riêng việc sửa prompt đã kéo target từ 0 lên 0.765; nhiều bài toán thực tế có thể dừng ở đây mà không cần train, tiết kiệm cả thời gian và rủi ro catastrophic forgetting.
