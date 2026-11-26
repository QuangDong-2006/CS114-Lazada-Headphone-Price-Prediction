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
GIA_MIN, GIA_MAX = 30_000, 20_000_000


def lam_sach(raw):
    """Luật làm sạch B1. Chỉ dùng thông tin của từng dòng, nên làm trước khi chia train/test.

    Trả về (df_sach, nhat_ky) với nhat_ky là list (bước, số dòng bỏ, số dòng còn).
    """
    df = raw.copy()
    nhat_ky = [("0. Dữ liệu thô", 0, len(df))]

    def ghi(buoc, truoc):
        nhat_ky.append((buoc, truoc - len(df), len(df)))

    n = len(df); df = df.drop_duplicates("item_id"); ghi("1. Bỏ trùng item_id", n)

    # Mọi dòng đều là tai nghe (đã đọc tên từng dòng ở B1); 35 dòng shop chọn sai danh mục
    # ("Mật Ong", "Thắt lưng nữ"...) được GIỮ, chỉ xóa product_type sai và gắn cờ.
    n = len(df)
    sai = df["product_type"].notna() & ~df["product_type"].isin(LOAI_TAI_NGHE)
    df["sai_danh_muc"] = sai.astype(int)
    df.loc[sai, "product_type"] = np.nan
    ghi(f"2. Xóa product_type sai danh mục ({int(sai.sum())} dòng, giữ dòng)", n)

    n = len(df); df = df[df["price"].between(GIA_MIN, GIA_MAX)]
    ghi(f"3. Giữ giá {GIA_MIN:,} – {GIA_MAX:,} đ", n)

    # Chuẩn hóa nhẹ trên từng dòng
    df["name"] = (df["name"].str.replace("﻿", "", regex=False)
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    df["thieu_chi_tiet"] = df["detail_status"].isin(["timeout", "app_only"]).astype(int)
    df[TARGET] = np.log10(df["price"])
    return df.reset_index(drop=True), nhat_ky


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
