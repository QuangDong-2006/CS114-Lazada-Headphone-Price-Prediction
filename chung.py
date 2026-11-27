"""Hàm dùng chung cho đồ án ML dự đoán giá tai nghe Bluetooth (Lazada).

Người 1 giữ file này; mọi thay đổi báo Người 1.
Hiện có: phần B0, các thang đo. B3 (tao_preprocessor) và B5 (danh_gia_cv) sẽ thêm sau.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------- B0: bài toán
RANDOM_STATE = 42
TARGET = "log_price"          # log10(price), đơn vị gốc: VNĐ
NGUONG_SAI_TUONG_DOI = 0.20   # "đoán trúng" = sai dưới 20% giá thật

# Các cột KHÔNG được làm đặc trưng: sản phẩm mới đăng chưa có, hoặc chứa đáp án.
COT_RO_RI = [
    "price", "original_price", "discount_raw",          # giá gốc x giảm giá = giá
    "rating_score", "review_count", "sold_raw", "sold_num",
    "rating_avg_detail", "rating_count_detail",
    "star_1", "star_2", "star_3", "star_4", "star_5",
    "repurchase_rate", "returning_customers",
    "reviews_with_media", "review_tag_repurchase", "review_tags_raw",
    "search_page", "rank_in_page",                       # thứ hạng phụ thuộc lượt bán
]


# ---------------------------------------------------------------- B1: làm sạch
DUONG_DAN_THO = "data/raw/lazada_tai_nghe_1500.csv"
DUONG_DAN_SACH = "data/lazada_sach.csv"
LOAI_TAI_NGHE = [
    "Tai nghe không dây", "Tai nghe chụp tai cỡ lớn", "Tai nghe nhét tai",
    "Tai nghe hở tai", "Tai nghe chụp tai", "Tai nghe một bên kèm mic",
]
# Ngưỡng ngoại lệ giá = hàng rào Tukey 3×IQR trên log10(price) của bộ thô (dẫn chứng ở notebook B1.2):
# 10**(Q1 - 3·IQR) ≈ 2.921 đ, 10**(Q3 + 3·IQR) ≈ 20.141.427 đ → làm tròn. Cố định thành hằng số
# để không phụ thuộc cách chia dữ liệu. Chỉ loại 1 dòng (Kinera 77,2 triệu).
GIA_MIN, GIA_MAX = 3_000, 20_000_000


# Giá trị giữ chỗ (shop không khai) → coi là trống. So khớp NGUYÊN Ô, sau khi strip.
GIA_TRI_RONG = {"info unavailable", "default", "khác", "không áp dụng", "no", "not rated",
                "none", "n/a", "na", "-", "không", "null", ""}
COT_THONG_SO_CHU = ["bluetooth_version", "battery_hours", "waterproof", "model", "connection",
                    "driver_type", "impedance", "warranty", "warranty_type"]
# Tên thương hiệu thực chất là "không thương hiệu" (đã đọc toàn bộ 204 giá trị brand)
BRAND_KHONG_TH = {"no brand", "nobard", "oem", "tws"}
# Khoảng hợp lệ của thông số (ngoài khoảng → trống, để Imputer điền ở B3). Dẫn chứng ở notebook B1.4
KHOANG_HOP_LE = {
    "bluetooth_version_num": (4.0, 6.1),   # chuẩn Bluetooth SIG; 1 / 3 không còn trên tai nghe 2026
    "battery_hours_num": (1, 100),         # tai nghe chụp tai flagship tối đa ~80–100 h; 0 h vô nghĩa
    "impedance_ohm": (4, 200),             # tai nghe Bluetooth 16–64 Ω; 300 Ω là thông số tai nghe có dây
    "warranty_months": (0, 60),
}


def _chuan_hoa_chuoi(s):
    """Bỏ BOM, tab, khoảng trắng thừa ở đầu/cuối/giữa."""
    return (s.str.replace("﻿", "", regex=False)
             .str.replace(r"\s+", " ", regex=True).str.strip())


def lam_sach(raw):
    """Luật làm sạch B1. Mọi luật chỉ dùng thông tin của chính dòng đó (ngưỡng là hằng số cố định),
    nên làm trước khi chia train/test mà không rò rỉ.

    Trả về (df_sach, nhat_ky_dong, nhat_ky_sua):
      - nhat_ky_dong: list (bước, số dòng bỏ, số dòng còn)
      - nhat_ky_sua : list (bước, cột, số ô bị sửa) cho các luật sửa giá trị, không bỏ dòng
    """
    df = raw.copy()
    nhat_ky_dong = [("0. Dữ liệu thô", 0, len(df))]
    nhat_ky_sua = []

    def ghi_dong(buoc, truoc):
        nhat_ky_dong.append((buoc, truoc - len(df), len(df)))

    def ghi_sua(buoc, cot, mask):
        nhat_ky_sua.append((buoc, cot, int(mask.sum())))

    # ---- A. Chuẩn hóa chuỗi (làm trước để so trùng chính xác)
    for c in ["name", "seller_name", "brand", "location"] + COT_THONG_SO_CHU:
        truoc = df[c].copy()
        df[c] = _chuan_hoa_chuoi(df[c])
        ghi_sua("A1. Bỏ BOM/tab/khoảng trắng thừa", c, (truoc != df[c]) & truoc.notna())
    for c in COT_THONG_SO_CHU:
        m = df[c].str.lower().isin(GIA_TRI_RONG)
        df.loc[m, c] = np.nan
        ghi_sua("A2. Giá trị giữ chỗ → trống", c, m)
    # Tên shop: gộp các cách viết chỉ khác hoa/thường (quan trọng cho GroupKFold theo shop)
    khoa = df["seller_name"].str.lower()
    chuan = df.groupby(khoa)["seller_name"].agg(lambda s: s.value_counts().index[0])
    moi = khoa.map(chuan)
    ghi_sua("A3. Gộp cách viết tên shop", "seller_name", (moi != df["seller_name"]) & moi.notna())
    df["seller_name"] = moi
    m = df["location"].eq("Hồ Chí Minh")
    df.loc[m, "location"] = "Thành phố Hồ Chí Minh"
    ghi_sua("A4. Gộp tên nơi bán", "location", m)

    # ---- B. Bỏ dòng
    n = len(df); df = df.drop_duplicates("item_id"); ghi_dong("1. Bỏ trùng item_id", n)

    # Cùng shop đăng lại y hệt (cùng tên, cùng giá) dưới item_id khác → bản sao, giữ bản đầu.
    # seller trống (timeout) vẫn so được: cặp trống trùng tên + giá được cào cách nhau vài giây.
    n = len(df)
    df = df.drop_duplicates(["name", "seller_name", "price"], keep="first")
    ghi_dong("2. Bỏ tin đăng trùng (cùng tên + shop + giá)", n)

    n = len(df); df = df[df["price"].between(GIA_MIN, GIA_MAX)]
    ghi_dong(f"3. Bỏ ngoại lệ giá (ngoài {GIA_MIN:,} – {GIA_MAX:,} đ)", n)

    # Không bỏ dòng nào vì "không phải tai nghe": đã đọc tên mọi dòng, tất cả đều là tai nghe
    # Bluetooth; không có tai nghe chỉ có dây (notebook B1.2).

    # ---- C. Sửa giá trị sai
    sai = df["product_type"].notna() & ~df["product_type"].isin(LOAI_TAI_NGHE)
    df["sai_danh_muc"] = sai.astype(int)
    df.loc[sai, "product_type"] = np.nan
    ghi_sua("C1. product_type sai danh mục → trống", "product_type", sai)

    # Thông số ngoài khoảng hợp lệ
    m = df["impedance"].str.contains(r"mm|dB", case=False, na=False) & df["impedance_ohm"].notna()
    df.loc[m, "impedance_ohm"] = np.nan          # "14mm" là cỡ driver, "1dB/mW" là độ nhạy
    ghi_sua("C2. Trở kháng ghi nhầm đơn vị (mm, dB) → trống", "impedance_ohm", m)
    m = df["battery_hours"].str.contains(r"chờ|standby", case=False, na=False) & df["battery_hours_num"].notna()
    df.loc[m, "battery_hours_num"] = np.nan      # "Chờ trong 300 giờ" là thời gian chờ, không phải nghe nhạc
    ghi_sua("C2. Giờ pin là thời gian chờ → trống", "battery_hours_num", m)
    for c, (lo, hi) in KHOANG_HOP_LE.items():
        m = df[c].notna() & ~df[c].between(lo, hi)
        df.loc[m, c] = np.nan
        ghi_sua(f"C3. Ngoài khoảng hợp lệ {lo}–{hi} → trống", c, m)

    # Thương hiệu: gộp "không thương hiệu", gộp hoa/thường, lấy từ tiền tố tên "BRAND | ..."
    df["brand"] = df["brand"].str.rstrip(".").str.strip()
    m = df["brand"].str.lower().isin(BRAND_KHONG_TH)
    df.loc[m, "brand"] = "No Brand"
    ghi_sua("C4. Gộp các tên 'không thương hiệu' → No Brand", "brand", m & ~raw.loc[df.index, "brand"].eq("No Brand"))
    khoa = df["brand"].str.lower()
    chuan = df.groupby(khoa)["brand"].agg(lambda s: s.value_counts().index[0])
    moi = khoa.map(chuan)
    ghi_sua("C4. Gộp cách viết thương hiệu", "brand", (moi != df["brand"]) & moi.notna())
    df["brand"] = moi
    tien_to = df["name"].str.extract(r"^([^|【\[\(]{2,30}?)\s*\|")[0].str.strip()
    m = df["brand"].isna() & tien_to.notna() & ~tien_to.str.contains("tai nghe", case=False, na=False)
    df.loc[m, "brand"] = tien_to[m]
    ghi_sua("C5. Điền thương hiệu trống từ tiền tố tên 'X | ...'", "brand", m)
    df["brand"] = df["brand"].str.lower().map(
        df.groupby(df["brand"].str.lower())["brand"].agg(lambda s: s.value_counts().index[0]))

    # Chống nước: viết hoa mã IP
    m = df["waterproof"].str.match(r"(?i)^ip", na=False) & (df["waterproof"] != df["waterproof"].str.upper())
    df.loc[m, "waterproof"] = df.loc[m, "waterproof"].str.upper()
    ghi_sua("C6. Viết hoa mã IP", "waterproof", m)

    # ---- D. Cờ cho các bước sau
    df["thieu_chi_tiet"] = df["detail_status"].isin(["timeout", "app_only"]).astype(int)
    df["dot_cao"] = df["crawl_time"].str[:10]
    df[TARGET] = np.log10(df["price"])
    return df.reset_index(drop=True), nhat_ky_dong, nhat_ky_sua


def ve_vnd(y_log):
    """Đổi log10(giá) về VNĐ."""
    return np.power(10.0, np.asarray(y_log, dtype=float))


# ---------------------------------------------------------------- B0: thang đo
def rmse_log(y_true_log, y_pred_log):
    """Thang đo CHÍNH: RMSE trên log10(giá). Nhỏ hơn là tốt hơn."""
    return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))


def r2_log(y_true_log, y_pred_log):
    return float(r2_score(y_true_log, y_pred_log))


def mae_vnd(y_true_log, y_pred_log):
    """Sai số tuyệt đối trung bình, đơn vị VNĐ."""
    return float(np.mean(np.abs(ve_vnd(y_true_log) - ve_vnd(y_pred_log))))


def mape(y_true_log, y_pred_log):
    """Sai số tương đối trung bình, đơn vị %."""
    t, p = ve_vnd(y_true_log), ve_vnd(y_pred_log)
    return float(np.mean(np.abs(p - t) / t) * 100)


def ty_le_trung_20(y_true_log, y_pred_log):
    """% sản phẩm có giá dự đoán lệch dưới 20% so với giá thật."""
    t, p = ve_vnd(y_true_log), ve_vnd(y_pred_log)
    return float(np.mean(np.abs(p - t) / t < NGUONG_SAI_TUONG_DOI) * 100)


def he_so_sai(rmse_log10):
    """Diễn giải RMSE log10: giá dự đoán thường lệch khoảng x lần (nhân hoặc chia)."""
    return float(10 ** rmse_log10)


def tinh_thang_do(y_true_log, y_pred_log):
    """Trả về dict đủ 5 thang đo; tên khóa thống nhất cho cả nhóm."""
    return {
        "rmse_log": rmse_log(y_true_log, y_pred_log),
        "r2": r2_log(y_true_log, y_pred_log),
        "mae_vnd": mae_vnd(y_true_log, y_pred_log),
        "mape_pct": mape(y_true_log, y_pred_log),
        "trung_20_pct": ty_le_trung_20(y_true_log, y_pred_log),
    }
