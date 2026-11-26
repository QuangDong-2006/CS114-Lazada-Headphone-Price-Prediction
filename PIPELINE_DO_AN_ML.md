# Pipeline đồ án Machine Learning – Dự đoán giá tai nghe Bluetooth trên Lazada

Cập nhật: 26/11/2026. **Hạn nộp: 06/12/2026 (10 ngày, xem lịch ở mục 3).** Nhóm 3 người, mỗi người phụ trách một mô hình. Dùng chung bộ dữ liệu tự thu thập của đồ án DS111 (Lazada, tai nghe Bluetooth).

Ký hiệu: ✅ có sẵn từ dữ liệu đã cào · ⬜ cần làm · ⭐ phần nâng cao, làm nếu còn thời gian

---

## 0. Phân biệt với đồ án DS111 (quan trọng)

Thầy DS111 không cho một đồ án dùng cho nhiều môn. Hai đồ án dùng **chung dữ liệu thô** nhưng **khác mục tiêu và khác phần việc**:

| | Đồ án DS111 (Phân tích dữ liệu) | Đồ án Machine Learning (file này) |
|---|---|---|
| Câu hỏi chính | Yếu tố nào liên quan tới giá và lượt bán? | Dự đoán giá chính xác đến đâu, mô hình nào tốt nhất, vì sao? |
| Trọng tâm | EDA, tương quan, ANOVA, dashboard | Đặc trưng, mô hình, tinh chỉnh, đánh giá, giải thích, triển khai |
| Mô hình | Chỉ hồi quy tuyến tính/đa thức (Bài 05) | 3 họ thuật toán, tinh chỉnh sâu, ensemble |
| Dùng lại | Dữ liệu thô + bước làm sạch | Chỉ dùng lại dữ liệu thô và bước làm sạch, mọi phần còn lại làm mới |

Nên viết một câu như trên vào báo cáo cả hai môn để chủ động giải thích.

---

## 1. Sơ đồ tổng thể

```
[B0] Định nghĩa bài toán + thang đo                    Người 3
  ↓
[B1] Làm sạch bộ 1.550 dòng                             Người 1
  ↓
[B2] EDA hướng ML: biến mục tiêu, rò rỉ, khuyết         Người 2
  ↓
[B3] Feature engineering + tiền xử lý (Pipeline chung)  Người 1
  ↓
[B4] Giao thức đánh giá: chia test, CV theo nhóm shop   Người 2
  ↓
[B5] Baseline + hàm chấm điểm chung                     Người 3
  ↓
[B6] 3 mô hình song song ──┬─ Người 1: Linear + regularization
                           ├─ Người 2: Tree ensemble (RF, Boosting)
                           └─ Người 3: Kernel / láng giềng (SVR, k-NN)
  ↓
[B7] So sánh, kiểm định, ensemble, mở tập test          cả nhóm
  ↓
[B8] Giải thích mô hình + phân tích lỗi                 cả nhóm
  ↓
[B9] Triển khai demo dự đoán giá                        ⭐
  ↓
[B10] Báo cáo + bảo vệ
```

## 2. Chi tiết từng bước

### B0. Định nghĩa bài toán ⬜

- **Bài toán:** hồi quy có giám sát. Đầu vào là thông tin có sẵn **lúc đăng bán** một tai nghe (tên, thương hiệu, thông số, shop). Đầu ra là giá bán.
- **Biến mục tiêu:** `log10(price)` (giá lệch phải rất mạnh, log giúp sai số tương đối đều giữa hàng rẻ và hàng đắt). Báo cáo kết quả đổi lại về VNĐ.
- **Ứng dụng:** người bán mới nhập thông tin sản phẩm và nhận giá gợi ý; người mua phát hiện sản phẩm bị định giá quá cao.
- **Thang đo:** chính là **RMSE trên log giá** (dùng để chọn mô hình). Phụ: R², MAE (VNĐ), MAPE (%), và tỉ lệ sản phẩm dự đoán sai dưới 20%.
- **Mục tiêu đặt trước:** thắng baseline rõ ràng và đạt **R² ≥ 0,55 theo GroupKFold (chia theo shop)**.
  - Lần thử trước đây (R² ≈ 0,58) chia ngẫu nhiên nên bị "đẹp giả": cùng shop nằm ở cả train và test.
  - Thử nhanh khi chia theo shop: hồi quy tuyến tính R² ≈ 0,36, Ridge + TF-IDF tên R² ≈ 0,49. Vì vậy mục tiêu 0,70 là không thực tế; 0,55 là mức "tốt", 0,60 trở lên là "rất tốt".
  - Luôn báo cáo thêm điểm KFold ngẫu nhiên bên cạnh để cho thấy chênh lệch.
- ⭐ Bài toán phụ: phân loại phân khúc giá (rẻ / trung / cao) để có thêm thang đo phân loại (F1, confusion matrix). Chỉ làm nếu thầy yêu cầu có cả phân loại.

### B1. Làm sạch dữ liệu ⬜ (Người 1)

- Nguồn: tự cào Lazada.vn (Python `requests` cho trang tìm kiếm, Agent AI đọc trang chi tiết). Code và log trong `DoAn_DS111/scraper/`.
- **Bộ dùng chính thức: `data/raw/lazada_tai_nghe_1500.csv`** (1.550 dòng × 73 cột, 323–330 shop, 5 từ khóa tìm kiếm). Bộ 444 dòng cũ chỉ dùng để vẽ learning curve ở B7. Không dựa vào `data/clean/` của DS111 (không có trong thư mục này).
- Luật làm sạch (ghi số dòng còn lại sau mỗi bước vào báo cáo):
  1. Bỏ trùng `item_id`.
  2. Bỏ dòng có `product_type` không phải tai nghe: "Phụ kiện thời trang khác", "Phụ kiện trò chơi điện tử khác", "Loa & hệ thống giao tiếp", "Bảo vệ tai" (~16 dòng).
  3. Giữ giá trong **30.000 – 20.000.000 đ** (bỏ ~11 dòng, trong đó 1 sản phẩm 77 triệu).
  4. **150 dòng `detail_status = timeout`**: thiếu tên shop và mọi thông số chi tiết, chỉ có cột listing. Giữ lại (vẫn có tên, giá, nơi bán) nhưng phải xử lý `seller_name` thiếu ở B4.
- Dự kiến còn ~1.520 dòng.

### B2. EDA hướng Machine Learning ⬜

Không lặp lại EDA của DS111. Chỉ làm những gì phục vụ mô hình:
- Phân phối biến mục tiêu trước và sau log; ngoại lệ giá (hàng < 30k, > 3 triệu) giữ hay bỏ.
- **Kiểm tra rò rỉ dữ liệu:** `original_price` (giá gốc × giảm giá = giá), lượt bán, số đánh giá, điểm sao, tỉ lệ mua lại đều **phải loại**, vì sản phẩm mới chưa có hoặc chúng chứa luôn đáp án.
- Tỉ lệ khuyết từng cột → quyết định điền giá trị hay thêm cờ khuyết.
- Số mức của biến phân loại (thương hiệu ~80 mức) → gộp mức hiếm.
- **Trùng lặp gần giống:** cùng shop đăng nhiều sản phẩm gần như giống hệt (tên khác vài chữ). 5 shop lớn nhất chiếm ~32% số dòng (1 shop > 200 dòng). Đây là lý do dùng CV theo nhóm ở B4.
- **Cột khuyết theo đợt cào, không theo sản phẩm:** `is_lazmall`, `is_freeship`, `in_stock`, `shop_id`… trống 71% vì đợt cào 19/11 không lấy. **Loại khỏi đặc trưng** (nếu thêm cờ khuyết, mô hình chỉ học được "dòng thuộc đợt nào").
- **Ngưỡng ngoại lệ cho cột số** (đặt thành rỗng rồi để Imputer điền): `battery_hours_num` > 100 giờ, `impedance_ohm` > 600, `warranty_months` > 60, `bluetooth_version_num` ngoài 4–6,5; `n_variants` cắt trần 20; `shipping_fee`, `store_sold_num` lấy log.
- **Đầu ra của B2 (hạn 27/11):** bảng "cột giữ / bỏ / lý do" gửi Người 1 để làm B3.

### B3. Feature engineering và tiền xử lý (Pipeline dùng chung) ⬜

**Nhóm đặc trưng:**

| Nhóm | Đặc trưng |
|---|---|
| Thương hiệu | `brand` (gộp mức < 5 sản phẩm thành "Khác"), `brand_group`, `has_brand` |
| Sản phẩm | `product_group`, `bluetooth_version_num`, `battery_hours_clean`, `impedance_ohm`, `waterproof_group`, `waterproof_ip`, `warranty_months`, `n_variants_capped` |
| Shop / kênh bán | `is_authentic_guarantee`, `warranty_type`, `top_seller` (nhãn "Top" trong `seller_tags_raw`), `location_group` (HCM / Hà Nội / tỉnh khác / nước ngoài: China, Hong Kong, "Nước ngoài"), `log_store_sold`, `positive_seller_rating`, `ship_on_time_rate`, `shop_age_years`, `log_shipping_fee`. **Không dùng `is_lazmall`** (khuyết 71%, xem B2) |
| Khuyến mãi | `discount_pct`, `has_discount` (chỉ dùng % giảm, không dùng giá gốc) |
| Cờ khuyết | các cột `has_*` đi kèm cột số thiếu nhiều |
| **Từ tên sản phẩm (regex)** | có ANC / chống ồn, ENC / mic, gaming / độ trễ thấp, Pro / Max / Ultra, thể thao, sạc không dây, số giờ pin và phiên bản Bluetooth ghi trong tên |
| **Từ tên sản phẩm (NLP)** | TF-IDF trên tên (unigram + bigram, `min_df=3`, `sublinear_tf=True`), ⭐ giảm chiều bằng TruncatedSVD |

**Chuẩn hóa tên trước TF-IDF** (thống nhất cả nhóm): viết thường; bỏ ký tự trang trí như 【】[]|; **giữ dấu tiếng Việt** (bỏ dấu làm "chụp" và "chúp" trùng nhau nhưng cũng gộp sai từ khác); không xóa cụm "tai nghe bluetooth" bằng tay vì TF-IDF đã tự giảm trọng số từ xuất hiện ở mọi dòng.

**Tiền xử lý:** một `ColumnTransformer` đặt **bên trong** `Pipeline` của từng mô hình để không rò rỉ thống kê của tập kiểm tra:
- Số: `SimpleImputer(median)` → `StandardScaler`
- Phân loại: `OneHotEncoder(handle_unknown="ignore", min_frequency=5)`
- Tên: `TfidfVectorizer`

**Quy tắc chống rò rỉ:** biến đổi chỉ dùng thông tin của chính dòng đó (regex trên tên, đổi đơn vị, cắt ngưỡng) thì làm trước khi chia. Biến đổi học thống kê từ nhiều dòng (median, mean/std, danh sách thương hiệu, từ vựng TF-IDF) thì **bắt buộc nằm trong Pipeline**.

Đầu ra (hạn 28/11): notebook `01_chuan_bi_du_lieu.ipynb`, file `data/abt_gia.csv` (có cột `split` và `fold`) và hàm `tao_preprocessor()` trong `chung.py`. **Ba người chỉ đọc file này.**

### B4. Giao thức đánh giá ⬜

- **Nhóm (groups) = `seller_name`.** Dòng thiếu tên shop (150 dòng timeout) được gán nhóm riêng `"unknown_" + item_id`, tức mỗi dòng là một nhóm.
- **Tập test ~20% số dòng, chia theo shop**, để mô hình không "học thuộc" sản phẩm cùng shop rồi gặp lại ở test. **Không dùng `GroupShuffleSplit(test_size=0.2)`**: nó lấy 20% số *shop*, mà shop lớn có > 200 dòng nên test chỉ còn ~12% dòng. Thay bằng: xáo dữ liệu (`random_state=42`) → `GroupKFold(n_splits=5)` → lấy 1 phần làm test (~300 dòng). Kiểm tra lại: 0 shop chung giữa train và test. Khóa lại, chỉ mở một lần ở B7.
- **Kiểm chứng chéo trên 80% còn lại:** `GroupKFold(n_splits=5)` theo shop. So thêm với `KFold` thường để cho thấy kết quả bị "đẹp giả" bao nhiêu khi bỏ qua nhóm. Đây là điểm ML đáng giá để trình bày.
- Tinh chỉnh siêu tham số chỉ dùng CV trên tập train.
- Fold được **lưu thành cột `fold`** trong `abt_gia.csv`; cả 3 người dựng CV từ cột này (không tự tạo GroupKFold riêng) để mọi mô hình chạy trên đúng cùng 5 fold, điều kiện bắt buộc cho t-test ghép cặp ở B7.

### B5. Baseline ⬜

- `DummyRegressor(median)`: dự đoán mọi sản phẩm bằng giá trung vị.
- Trung vị giá theo `brand_group` (luật đơn giản người bán hay dùng).
- Hồi quy tuyến tính thường không chuẩn hóa hệ số.
Mọi mô hình ở B6 phải thắng các mốc này.

**Hàm chấm điểm chung** `danh_gia_cv(model, X, y, cv)` trong `chung.py`: trả về RMSE log (CV và train), độ lệch chuẩn qua fold, R², MAE (VNĐ), MAPE. Cả 3 người gọi đúng hàm này để bảng kết quả ghép thẳng được. Người 3 cũng tạo `requirements.txt` (cố định phiên bản pandas, scikit-learn, xgboost/lightgbm…). Hạn 28/11.

### B6. Ba mô hình, mỗi người một mô hình ⬜

Ba mô hình thuộc ba cách học khác nhau, nên khi so sánh sẽ có điều để nói:

| | Người 1 | Người 2 | Người 3 |
|---|---|---|---|
| Họ thuật toán | Tuyến tính có điều chuẩn | Cây quyết định và ensemble | Kernel và láng giềng gần nhất |
| Mô hình chính | **Ridge, Lasso, ElasticNet** | **Random Forest, Gradient Boosting** (XGBoost hoặc LightGBM) | **SVR (kernel RBF)**, k-NN làm mốc |
| Điểm mạnh | Dễ giải thích hệ số, hợp với TF-IDF nhiều chiều | Bắt tương tác phi tuyến, ít cần tiền xử lý, thường chính xác nhất | Định giá theo "sản phẩm tương tự", mô hình hóa quan hệ trơn |
| Notebook | `02a_linear.ipynb` | `02b_tree_ensemble.ipynb` | `02c_svr_knn.ipynb` |

**Luật công bằng khi so sánh:** cùng `abt_gia.csv`, cùng cột `fold`, cùng hàm `danh_gia_cv`; tinh chỉnh bằng `RandomizedSearchCV` với **cùng ngân sách `n_iter=50`** (Optuna thì 50 trial); `scoring="neg_root_mean_squared_error"`; ghi lại thời gian huấn luyện.

**Khung chung 8 bước cho cả 3 notebook** (để ghép báo cáo đồng đều):

1. **Lý thuyết:** ý tưởng, hàm mất mát, siêu tham số chính, ưu nhược điểm.
2. **Mô hình mặc định** trong Pipeline chung, đánh giá bằng GroupKFold.
3. **Tinh chỉnh siêu tham số:** `RandomizedSearchCV` hoặc ⭐ Optuna, cv = GroupKFold.
4. **Over/underfitting:** validation curve theo siêu tham số chính, learning curve theo số mẫu, so điểm train và CV.
5. **Đánh giá CV:** bảng RMSE, R², MAE (VNĐ), MAPE; biểu đồ thực tế vs dự đoán; phân phối residual.
6. **Giải thích:** cách riêng của mô hình (bên dưới) + `permutation_importance` để 3 người so được với nhau.
7. **Phân tích lỗi:** 10 sản phẩm đoán sai nhất, sai số theo nhóm thương hiệu và khoảng giá.
8. **Kết luận và bộ tham số tốt nhất** (ghi rõ để B7 dựng lại).

**Người 1: Linear + regularization**
- Đi từ LinearRegression → Ridge (L2) → Lasso (L1) → ElasticNet; tinh chỉnh `alpha` trên thang log, `l1_ratio`.
- Hai phiên bản đặc trưng: chỉ đặc trưng có cấu trúc, và có thêm TF-IDF tên. Cho thấy TF-IDF thêm được bao nhiêu điểm.
- Đường hệ số theo alpha (regularization path); liệt kê đặc trưng Lasso đưa về 0 (chọn đặc trưng tự động).
- Giải thích hệ số trên log giá: "có ANC thì giá cao hơn khoảng X%"; top từ trong TF-IDF làm tăng/giảm giá.
- ⭐ Kiểm tra giả định: residual vs fitted, QQ-plot, VIF.

**Người 2: Tree ensemble**
- Decision Tree đơn (`max_depth`, `min_samples_leaf`) → vẽ cây độ sâu 3 để đọc luật.
- Random Forest (`n_estimators`, `max_features`, `min_samples_leaf`), so OOB score với CV.
- Gradient Boosting: XGBoost hoặc LightGBM (`learning_rate`, `n_estimators`, `max_depth`/`num_leaves`, `subsample`, `colsample_bytree`, `reg_lambda`), early stopping.
- So độ quan trọng theo impurity và theo permutation; giải thích vì sao impurity thiên vị biến nhiều mức.
- ⭐ SHAP (summary plot, dependence plot cho Bluetooth, giờ pin, LazMall).

**Người 3: Kernel và láng giềng**
- k-NN: `n_neighbors`, `weights`, `p`; vẽ điểm theo k để minh họa over/underfitting; chứng minh chuẩn hóa là bắt buộc (so có và không có `StandardScaler`).
- SVR kernel RBF: `C`, `gamma`, `epsilon` (lưới log); so kernel linear và RBF.
- Giảm chiều trước mô hình khoảng cách: PCA hoặc TruncatedSVD trên TF-IDF; so có và không giảm chiều.
- Giải thích: với một sản phẩm, in 5 láng giềng gần nhất và giá của chúng.
- ⭐ MLPRegressor nhỏ để có thêm một mô hình phi tuyến khác.

### B7. So sánh, chọn mô hình, ensemble ⬜

Notebook `03_so_sanh_va_ensemble.ipynb`:
1. Dựng lại baseline + 3 mô hình tốt nhất bằng tham số ở bước 8, chạy trên cùng các fold → bảng so sánh RMSE, R², MAE, MAPE, thời gian huấn luyện, thời gian dự đoán.
2. Boxplot điểm theo fold; **t-test ghép cặp** (hoặc Wilcoxon) giữa mô hình nhất và nhì để biết hơn kém có ý nghĩa không.
3. **Ensemble:** trung bình (`VotingRegressor`) và **Stacking** (`StackingRegressor`, meta-model Ridge). So với mô hình đơn tốt nhất.
4. **Mở tập test một lần duy nhất:** đánh giá tất cả; nếu điểm test thấp hơn CV nhiều thì phân tích nguyên nhân.
5. Learning curve của mô hình cuối theo số mẫu (444 → ~1500): trả lời "thêm dữ liệu có giúp không", đồng thời giải thích vì sao cào nhiều hơn 500 dòng.
6. Huấn luyện lại mô hình chọn trên toàn bộ dữ liệu, lưu `models/model_gia.joblib`.

### B8. Giải thích mô hình và phân tích lỗi ⬜

- So độ quan trọng đặc trưng của 3 mô hình: các đặc trưng cả 3 cùng đồng ý (dự kiến thương hiệu, ANC, LazMall, tên có Pro/Max).
- Nhóm lỗi điển hình: hàng nhái mượn tên thương hiệu lớn, sản phẩm combo/nhiều phân loại giá, hàng xách tay.
- **Khoảng dự đoán:** dùng phân vị residual (hoặc ⭐ hồi quy phân vị với Gradient Boosting) để đưa ra "giá gợi ý 250k–320k" thay vì một con số.
- Hạn chế: dữ liệu một thời điểm, một sàn, giá niêm yết có thể bị thổi lên để "giảm giá ảo".

### B9. Triển khai demo ⭐

- App **Streamlit** một trang: nhập tên sản phẩm + vài thông tin (thương hiệu, LazMall, Bluetooth, giờ pin…) → giá gợi ý, khoảng giá, 5 sản phẩm tương tự trong dữ liệu.
- Thử với 5–10 sản phẩm mới lấy trên Lazada ngoài bộ dữ liệu; ghi lại sai số thật.
- Mô tả cách cập nhật mô hình: cào lại định kỳ → chạy lại notebook 01 → huấn luyện lại → so điểm với mô hình cũ trước khi thay.

### B10. Báo cáo và bảo vệ ⬜

**Khung báo cáo:** 1. Giới thiệu bài toán · 2. Dữ liệu và thu thập · 3. EDA hướng ML · 4. Đặc trưng và tiền xử lý · 5. Giao thức đánh giá · 6. Ba mô hình (mỗi người một mục theo khung 8 bước) · 7. So sánh và ensemble · 8. Giải thích và phân tích lỗi · 9. Demo · 10. Kết luận, hạn chế, hướng phát triển. · Phụ lục: nhật ký thử nghiệm (mục 4).

**Câu hỏi nên chuẩn bị:**
- Vì sao dự đoán log giá? Vì sao chọn RMSE log làm thang đo chính?
- Rò rỉ dữ liệu là gì, nhóm đã loại những cột nào, vì sao?
- Vì sao CV theo nhóm shop, kết quả khác KFold thường bao nhiêu?
- Vì sao Pipeline giúp tránh rò rỉ khi chuẩn hóa và mã hóa?
- Mỗi người: mô hình học thế nào, siêu tham số nào quan trọng nhất, dấu hiệu overfitting ở đâu, vì sao mô hình thắng hoặc thua?
- Stacking có tốt hơn không, có đáng thêm độ phức tạp không?

---

## 3. Phân công và lịch 10 ngày (chốt 26/11/2026)

### 3.1 Ai làm gì

| Giai đoạn | Người 1 | Người 2 | Người 3 |
|---|---|---|---|
| **GĐ1 – Nền chung** (27–28/11) | B1 làm sạch + B3 đặc trưng, `tao_preprocessor()`, chốt `abt_gia.csv` | B2 EDA hướng ML (bảng cột giữ/bỏ **gửi Người 1 trong 27/11**) + B4 cột `split`, `fold` | B0 bài toán, thang đo + B5 baseline + `danh_gia_cv()` + `requirements.txt` |
| **GĐ2 – Mô hình** (29/11–02/12) | `02a_linear.ipynb`: Linear, Ridge, Lasso, ElasticNet | `02b_tree_ensemble.ipynb`: Decision Tree, Random Forest, XGBoost/LightGBM | `02c_svr_knn.ipynb`: k-NN, SVR (+ TruncatedSVD) |
| **GĐ3 – Ghép** (03–04/12) | B7 bước 1–3: bảng so sánh, t-test ghép cặp, Voting + Stacking | B7 bước 4–6: mở test một lần, learning curve, mô hình cuối `.joblib` + B8 phân tích lỗi | B8 giải thích (permutation importance chung, khoảng dự đoán) + ⭐ B9 demo Streamlit |
| **Báo cáo** | Mục 4, 6.1, 7 | Mục 3, 5, 6.2, 8 (phân tích lỗi) | Mục 1, 2, 6.3, 8 (giải thích), 9 |
| **Vai trò thêm** | Giữ `abt_gia.csv`, `chung.py` (mọi thay đổi qua Người 1) | Ráp báo cáo, soát định dạng | Ráp slide, quản lý lịch và nhắc hạn |

### 3.2 Lịch theo ngày

| Ngày | Việc | Mốc phải xong |
|---|---|---|
| T6 27/11 | GĐ1: làm sạch, EDA, chọn thang đo | Tối 27/11: Người 2 gửi bảng cột giữ/bỏ |
| T7 28/11 | GĐ1: đặc trưng, chia fold, baseline | **Tối 28/11: chốt `abt_gia.csv` + `chung.py` + baseline** (họp 30 phút) |
| CN 29/11 | GĐ2 bước 1–2: lý thuyết, mô hình mặc định | Mỗi người có 1 điểm CV đầu tiên |
| T2 30/11 | GĐ2 bước 3: tinh chỉnh | |
| T3 01/12 | GĐ2 bước 4–5: over/underfitting, đánh giá CV | Họp giữa kỳ: so điểm, phát hiện lỗi sớm |
| T4 02/12 | GĐ2 bước 6–8: giải thích, phân tích lỗi, kết luận | **Tối 02/12: bộ tham số tốt nhất của 3 mô hình** |
| T5 03/12 | GĐ3: so sánh, kiểm định, ensemble | |
| T6 04/12 | GĐ3: mở test, mô hình cuối, giải thích, demo | **Tối 04/12: đủ mọi bảng và hình** |
| T7 05/12 | Viết và ráp báo cáo + slide | Bản nháp đầy đủ |
| CN 06/12 | Soát lại, tập bảo vệ (mỗi người trả lời câu hỏi phần mình), **nộp** | |

Viết báo cáo **song song** từ GĐ2: ai xong bước nào thì viết luôn đoạn đó, không dồn về 05/12.
Nếu trễ: cắt các mục ⭐ trước (Optuna, SHAP, demo Streamlit, MLP), không cắt B4 và B7.

### 3.3 "Xong" nghĩa là gì

- Notebook chạy lại từ đầu (Restart & Run All) không lỗi, chỉ đọc `abt_gia.csv` và import `chung.py`.
- Mỗi notebook mô hình có đủ 8 bước, và bảng kết quả ra từ `danh_gia_cv`.
- Bộ tham số tốt nhất ghi ở ô cuối dạng dict để B7 dựng lại.
- Mọi kịch bản đã chạy đều có dòng trong nhật ký thử nghiệm (mục 4).
- Một người khác trong nhóm đọc chéo trước khi đánh dấu xong (Người 1 ↔ 2, 2 ↔ 3, 3 ↔ 1).

**Quy ước để ghép không vỡ:** cùng `random_state=42` và cùng cột `fold`; không ai sửa `abt_gia.csv` sau 28/11 (cần thêm đặc trưng thì báo Người 1 cập nhật cho cả nhóm); mỗi người chỉ sửa notebook của mình; cùng tên cột thang đo và cùng bảng màu. Chia sẻ code qua một repo GitHub chung (hoặc thư mục Drive chung nếu nhóm chưa quen Git).

## 4. Kịch bản thử nghiệm

Mọi thử nghiệm đều ghi vào **nhật ký** (bảng mẫu ở 4.5), kể cả kịch bản cho kết quả xấu. Báo cáo dùng nhật ký này cho mục 7 và phụ lục, và đây cũng là bằng chứng nhóm ra quyết định dựa trên số liệu.

**Nguyên tắc:** mỗi kịch bản **chỉ đổi một yếu tố** so với mốc đang dùng · luôn chấm bằng `danh_gia_cv` trên cùng cột `fold` · **không ghi điểm tập test** cho tới B7 · mã kịch bản `KB-<nhóm>-<số>` (C = chung, L = Người 1, T = Người 2, K = Người 3, E = ensemble).

### 4.1 Đã thử (thử nhanh 26/11/2026, bộ 1.523 dòng sau làm sạch)

| Mã | Câu hỏi | Kịch bản | Kết quả | Quyết định |
|---|---|---|---|---|
| KB-C-01 | Chia ngẫu nhiên có làm điểm "đẹp giả" không? | Linear (không TF-IDF): KFold ngẫu nhiên vs GroupKFold theo shop | R² 0,57 vs **0,36**; RMSE log 0,289 vs 0,343 | Dùng GroupKFold; hạ mục tiêu R² xuống 0,55 |
| KB-C-02 | `GroupShuffleSplit(test_size=0.2)` có cho đủ 20% test? | Đếm số dòng test | Chỉ 181/1.523 dòng (**12%**), vì 1 shop > 200 dòng | Đổi sang GroupKFold lấy 1/5 → 305 dòng (20%), 0 shop chung |
| KB-C-03 | Mốc điểm phải thắng là bao nhiêu? | 4 baseline, GroupKFold | RMSE log: Dummy 0,439 · Trung vị theo thương hiệu 0,380 · Linear 0,343 · Ridge (α=3) + TF-IDF tên **0,307** (R² 0,49) | Mốc cho B6: RMSE log < 0,307 |
| KB-C-04 | Luật "trung vị theo thương hiệu" có đáng tin? | KFold ngẫu nhiên vs GroupKFold | R² 0,51 vs **0,23** | Thương hiệu bị lẫn với shop; minh chứng cho chia theo shop |
| KB-C-05 | Có dùng được `is_lazmall`? | Kiểm tra tỉ lệ khuyết theo đợt cào | Trống 71%, toàn bộ thuộc đợt 19/11 | Loại khỏi đặc trưng |
| KB-C-06 | Tên sản phẩm có thông tin giá không? | Thêm TF-IDF tên vào mô hình tuyến tính | RMSE log 0,343 → 0,307 | Giữ TF-IDF; khoảng cách train–CV tăng (0,19 vs 0,31) → cần điều chuẩn |

### 4.2 Kịch bản chung (GĐ1, 27–28/11, chạy bằng Ridge làm mô hình đo)

| Mã | Yếu tố thử | Các phương án | Người |
|---|---|---|---|
| KB-C-07 | Biến mục tiêu | `log10(price)` vs `price` (so trên MAE VNĐ và MAPE) | Người 3 |
| KB-C-08 | Ngoại lệ giá | lọc 30k–20 triệu vs không lọc vs 50k–10 triệu | Người 1 |
| KB-C-09 | Nhóm đặc trưng (ablation) | chỉ cấu trúc → + regex tên → + TF-IDF → + TF-IDF + SVD | Người 1 |
| KB-C-10 | Cờ khuyết | `add_indicator=True` vs `False` | Người 1 |
| KB-C-11 | Gộp thương hiệu hiếm | `min_frequency` = 3 / 5 / 10 | Người 1 |
| KB-C-12 | TF-IDF | unigram vs uni+bigram; `min_df` 2 / 3 / 5; có/không bỏ dấu | Người 2 |
| KB-C-13 | Dòng timeout (thiếu chi tiết) | giữ vs bỏ 150 dòng | Người 2 |

### 4.3 Kịch bản theo mô hình (GĐ2, 29/11–02/12)

| Người | Mã | Kịch bản |
|---|---|---|
| 1 – Linear | KB-L-01 | LinearRegression → Ridge → Lasso → ElasticNet (cùng đặc trưng) |
| | KB-L-02 | `alpha` từ 1e-3 đến 1e3 (thang log): vẽ validation curve, regularization path |
| | KB-L-03 | Có / không TF-IDF tên với mô hình tốt nhất |
| | KB-L-04 | ⭐ Thêm đặc trưng đa thức bậc 2 cho các cột số |
| 2 – Cây | KB-T-01 | Decision Tree: `max_depth` 2–20, `min_samples_leaf` 1–50 |
| | KB-T-02 | Random Forest: `max_features` (sqrt / 0,3 / 0,5), `n_estimators`; so OOB với CV |
| | KB-T-03 | Boosting: `learning_rate` 0,01–0,3 × `n_estimators` có early stopping |
| | KB-T-04 | Có / không TF-IDF (TF-IDF nhiều chiều thường bất lợi cho cây) |
| | KB-T-05 | Độ quan trọng impurity vs permutation |
| 3 – Kernel/láng giềng | KB-K-01 | k-NN có vs không `StandardScaler` (chứng minh chuẩn hóa bắt buộc) |
| | KB-K-02 | k-NN: `n_neighbors` 1–50, `weights` uniform/distance, `p` 1/2 |
| | KB-K-03 | SVR kernel linear vs RBF; lưới log cho `C`, `gamma`, `epsilon` |
| | KB-K-04 | TF-IDF giữ nguyên vs TruncatedSVD 50 / 100 / 300 chiều |
| | KB-K-05 | ⭐ MLPRegressor nhỏ |

Mỗi người kết thúc GĐ2 bằng **một dòng "mô hình tốt nhất"** trong nhật ký, ghi đủ tham số.

### 4.4 Kịch bản ghép (GĐ3, 03–04/12)

| Mã | Kịch bản |
|---|---|
| KB-E-01 | So 3 mô hình tốt nhất + baseline trên cùng 5 fold; t-test ghép cặp nhất vs nhì |
| KB-E-02 | VotingRegressor (trung bình) vs StackingRegressor (meta Ridge) vs mô hình đơn tốt nhất |
| KB-E-03 | Learning curve: 444 dòng (bộ cũ) → 1.523 dòng |
| KB-E-04 | Mở tập test **một lần**: so điểm test với điểm CV của từng mô hình |
| KB-E-05 | ⭐ Khoảng dự đoán: phân vị residual vs hồi quy phân vị (Gradient Boosting) |

### 4.5 Mẫu nhật ký thử nghiệm

| Mã | Ngày | Người | Giả thuyết | Thay đổi so với mốc | RMSE log CV (± std) | RMSE log train | R² CV | Thời gian | Kết luận |
|---|---|---|---|---|---|---|---|---|---|
| KB-L-02 | 30/11 | Người 1 | Tăng α giảm overfit do TF-IDF | Ridge α: 3 → 10 | … | … | … | … | Giữ / Bỏ, vì … |

Mẹo đọc kết quả (theo C2 W3): RMSE train và CV đều cao → **bias** (thêm đặc trưng, mô hình mạnh hơn); train thấp mà CV cao → **variance** (tăng điều chuẩn, thêm dữ liệu, bớt đặc trưng). Chênh lệch nhỏ hơn độ lệch chuẩn qua các fold thì coi là **chưa khác biệt**.

## 5. Cấu trúc thư mục đề xuất (ít file)

```
DoAn_ML/
├─ PIPELINE_DO_AN_ML.md          ← file này
├─ requirements.txt               phiên bản thư viện (Người 3)
├─ chung.py                       tao_preprocessor(), danh_gia_cv(), đọc fold (Người 1 giữ)
├─ data/raw/                      dữ liệu thô (không sửa)
├─ data/abt_gia.csv               bảng dùng chung (có cột split, fold)
├─ notebooks/
│  ├─ 01_chuan_bi_du_lieu.ipynb   B0–B5 (cả nhóm)
│  ├─ 02a_linear.ipynb            Người 1
│  ├─ 02b_tree_ensemble.ipynb     Người 2
│  ├─ 02c_svr_knn.ipynb           Người 3
│  └─ 03_so_sanh_va_ensemble.ipynb B7–B8
├─ models/model_gia.joblib
├─ app.py                         ⭐ demo Streamlit
└─ bao_cao/
```

Thư viện: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost hoặc lightgbm, ⭐ shap, optuna, streamlit, joblib.
