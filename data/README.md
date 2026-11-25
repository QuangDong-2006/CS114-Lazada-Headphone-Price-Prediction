# Dữ liệu: tai nghe bluetooth trên Lazada Việt Nam

File: `lazada_tai_nghe_1500.csv` (UTF-8 có BOM, đọc bằng `pd.read_csv(..., encoding="utf-8-sig")`)

| Mục | Giá trị |
|---|---|
| Kích thước | **1550 dòng × 73 cột**, mỗi dòng 1 sản phẩm, `item_id` không trùng |
| Nguồn | https://www.lazada.vn, tự thu thập (chỉ phục vụ học tập) |
| Từ khóa tìm kiếm (`search_query`) | tai nghe bluetooth (561), tai nghe nhét tai bluetooth (285), tai nghe không dây (284), tai nghe chụp tai bluetooth (210), tai nghe true wireless (210) |
| Thời gian thu thập | 13/11/2026 – 25/11/2026 |

## Mức độ đầy đủ của trang chi tiết (`detail_status`)

| Giá trị | Số dòng | Ý nghĩa |
|---|---|---|
| `ok_dom` | 1348 | Chi tiết đọc từ giao diện trang sản phẩm (shop, đánh giá, thông số...) |
| `ok` | 48 | Chi tiết lấy từ API trang tự gọi, có thêm phân bố sao `star_1..star_5` |
| `timeout` | 150 | Trang nạp quá chậm, chỉ có các cột từ trang tìm kiếm |
| `app_only` | 4 | Sản phẩm chỉ xem được trên ứng dụng di động |

→ **1396/1550 dòng (90%) có đủ chi tiết.** Các cột từ trang tìm kiếm (giá, giảm giá, lượt bán, số đánh giá, nơi gửi...) có cho mọi dòng (một số ô trống nếu sản phẩm không hiển thị).

## Nhóm cột chính
- **Trang tìm kiếm:** `name`, `brand`, `price` (biến mục tiêu), `original_price`, `discount_pct`, `rating_score`, `review_count`, `sold_num`, `location`, `seller_name`, `is_lazmall`, `is_freeship`, `in_stock`, `url`, `search_query`...
- **Trang chi tiết:** `product_type`, `positive_seller_rating`, `ship_on_time_rate`, `store_sold_num`, `shop_age_years`, `rating_avg_detail`, `rating_count_detail`, `reviews_with_media`, `n_variants`, `bluetooth_version_num`, `battery_hours_num`, `impedance_ohm`, `warranty_months`, `waterproof`, `is_authentic_guarantee`, `shipping_fee`, `specs_raw` (toàn bộ thông số dạng JSON)...

## Lưu ý khi dùng
- `chat_response_rate` trống toàn bộ; `is_sponsored` toàn 0 → bỏ.
- `star_1..star_5` chỉ có ở 48 dòng `ok` → không dùng cho toàn bộ dữ liệu.
- Thông số do shop tự khai, thiếu nhiều (pin, chống nước, bảo hành); `battery_hours_num` có ngoại lai tới vài trăm giờ.
- `price` lệch phải mạnh → cân nhắc log(price) khi huấn luyện mô hình.
- Không chứa thông tin cá nhân người mua; tên shop, giá, đánh giá là thông tin công khai trên trang.
