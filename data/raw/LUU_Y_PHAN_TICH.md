# Lưu ý khi làm sạch và phân tích `lazada_tai_nghe.csv`

Bộ dữ liệu: 444 dòng x 72 cột, tai nghe bluetooth trên Lazada, thu thập 13-14/11/2026 (xem `scraper/README.md`).

## Ngoại lai và giá trị bất thường
- **`battery_hours_num`**: trung vị 6 giờ nhưng có giá trị tới **300 giờ**, có thể shop tính cả hộp sạc hoặc nhập sai. Cần xử lý ngoại lai (IQR/cắt ngưỡng) và nêu rõ trong báo cáo. Giá trị > 500 đã được đặt thành rỗng.
- `price` trải rộng từ 15.000 đ đến 14,99 triệu đ (lệch phải mạnh), nên cân nhắc dùng log(price) khi phân tích.

## Cột thiếu nhiều hoặc không dùng được
- `chat_response_rate`: **trống 100%** (Lazada không còn hiển thị). Thay bằng `response_time_raw` (51%, cần tách số phút từ "Trung bình X phút").
- `is_sponsored`: tất cả bằng 0, **bỏ**.
- `star_1`..`star_5`: chỉ có ở **48 dòng** (`detail_status = ok`, lấy từ API), **không dùng cho toàn bộ dữ liệu**.
- Thông số do shop tự khai, thiếu nhiều: giờ pin 39%, bảo hành (tháng) 53%, chống nước 56% (lẫn "Có" và mã IPX, cần chuẩn hóa), trở kháng 68%.

## Hai nguồn dữ liệu chi tiết
- `detail_status = ok` (48 dòng): lấy từ API, có phân bố sao.
- `detail_status = ok_dom` (393 dòng): đọc từ giao diện, không có phân bố sao nhưng có `shop_age_years`, `reviews_with_media`, `review_tag_repurchase`.
- `rating_score`/`review_count` (trang tìm kiếm) và `rating_avg_detail`/`rating_count_detail` (trang chi tiết) lấy ở 2 trang và 2 thời điểm khác nhau, chỉ nên chọn **một cặp** và dùng nhất quán.
- 3 dòng không có chi tiết (2 `app_only`, 1 `timeout`) nhưng các cột listing vẫn dùng được.

## Mất cân bằng nhóm (ảnh hưởng ANOVA)
- `product_type`: "Tai nghe không dây" 343/444. Nên gộp các nhóm nhỏ (chụp tai, hở tai, một bên...) trước khi chạy one-way ANOVA.
- `brand`: "No Brand" + "OEM" khoảng 220 dòng, nên tạo cột có thương hiệu / không thương hiệu.

## Yêu cầu của thầy cần nhớ (NOTE DATASET.txt)
- Heatmap tương quan dùng cả **Pearson, Spearman và Kendall**.
- **One-way ANOVA** tự tách nhóm từ DataFrame rồi gọi `stats.f_oneway`, so p-value với 0.05.
- Mỗi phần thăm dò phải có **phát hiện**.
