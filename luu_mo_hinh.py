"""Lưu và nạp lại mô hình đã huấn luyện, để không phải huấn luyện lại mỗi lần chạy notebook.

Cách dùng trong notebook (notebooks/ nằm dưới thư mục gốc repo):

    import sys; sys.path.append("..")
    from luu_mo_hinh import huan_luyen_hoac_tai, tinh_hoac_tai, doc_thong_tin

    # 1) Một Pipeline / mô hình bất kỳ
    ridge = huan_luyen_hoac_tai("L_ridge_tfidf", pipe_ridge, X_train, y_train)

    # 2) Tinh chỉnh siêu tham số (RandomizedSearchCV / GridSearchCV): lưu cả search,
    #    nên best_params_, best_score_, cv_results_, best_estimator_ đều còn nguyên
    search = huan_luyen_hoac_tai("T_rf_search", RandomizedSearchCV(...), X_train, y_train,
                                 groups=groups_train)
    search.best_params_

    # 3) Kết quả tính lâu khác (danh_gia_cv, learning_curve, permutation_importance...)
    kq = tinh_hoac_tai("K_svr_cv", danh_gia_cv, best_svr, X_train, y_train, cv)

Mỗi mô hình được lưu thành 2 file trong models/:
    <ten>.joblib  mô hình đã fit (nạp lại bằng joblib.load)
    <ten>.json    thông tin đọc được bằng mắt: tham số, best_params_, điểm CV, thời gian
                  huấn luyện, phiên bản thư viện. Dùng file này để B7 dựng lại mô hình.

Lần chạy sau, nếu dữ liệu, tham số mô hình và tham số fit giống hệt lần trước (so bằng mã
băm), mô hình được nạp từ file thay vì huấn luyện lại. Đổi bất kỳ thứ gì trong đó thì tự
huấn luyện lại và ghi đè. Muốn ép huấn luyện lại thì truyền lam_lai=True.

Quy ước tên: tiền tố theo người như mã kịch bản (L_ Người 1, T_ Người 2, K_ Người 3,
E_ ensemble), ví dụ "L_ridge_tfidf", "T_xgb_search", "K_svr_rbf".
"""

from __future__ import annotations

import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone

THU_MUC_MODELS = Path(__file__).resolve().parent / "models"


def _duong_dan(ten: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", ten):
        raise ValueError(f"Tên '{ten}' chỉ được chứa chữ không dấu, số, '_' hoặc '-'.")
    THU_MUC_MODELS.mkdir(parents=True, exist_ok=True)
    return THU_MUC_MODELS / f"{ten}.joblib", THU_MUC_MODELS / f"{ten}.json"


def _bam(*doi_tuong) -> str:
    """Mã băm của dữ liệu / tham số. Rơi về repr() nếu có thứ không pickle được (lambda...)."""
    try:
        return joblib.hash(doi_tuong)
    except Exception:
        return joblib.hash(repr(doi_tuong))


def _sang_json(x):
    """Đổi giá trị bất kỳ thành thứ ghi được ra JSON (estimator, numpy... thành chuỗi/số)."""
    if isinstance(x, dict):
        return {str(k): _sang_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sang_json(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 50 else f"<mảng {x.shape}>"
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)


def _phien_ban() -> dict:
    pb = {"scikit-learn": sklearn.__version__, "numpy": np.__version__}
    for goi in ("xgboost", "lightgbm", "pandas"):
        try:
            pb[goi] = __import__(goi).__version__
        except ImportError:
            pass
    return pb


def _kiem_tra_phien_ban(thong_tin: dict, ten: str) -> None:
    cu = thong_tin.get("phien_ban", {})
    moi = _phien_ban()
    lech = [f"{g} {cu[g]} -> {moi[g]}" for g in cu if g in moi and cu[g] != moi[g]]
    if lech:
        warnings.warn(
            f"'{ten}' được lưu với phiên bản thư viện khác ({', '.join(lech)}). "
            "Nếu nạp lỗi hoặc kết quả lạ, chạy lại với lam_lai=True.",
            stacklevel=3,
        )


def doc_thong_tin(ten: str) -> dict | None:
    """Đọc file <ten>.json (tham số, điểm, thời gian). Trả None nếu chưa có."""
    _, json_path = _duong_dan(ten)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _ghi(ten, doi_tuong, thong_tin):
    joblib_path, json_path = _duong_dan(ten)
    joblib.dump(doi_tuong, joblib_path, compress=3)
    thong_tin["kich_thuoc_MB"] = round(joblib_path.stat().st_size / 2**20, 2)
    json_path.write_text(json.dumps(thong_tin, ensure_ascii=False, indent=2), encoding="utf-8")
    if thong_tin["kich_thuoc_MB"] > 50:
        warnings.warn(
            f"{joblib_path.name} nặng {thong_tin['kich_thuoc_MB']} MB: GitHub từ chối file > 100 MB "
            "và cảnh báo từ 50 MB. Cân nhắc giảm n_estimators hoặc chỉ commit file .json.",
            stacklevel=3,
        )


def _nap_neu_khop(ten, ma_bam, lam_lai):
    joblib_path, _ = _duong_dan(ten)
    thong_tin = doc_thong_tin(ten)
    if lam_lai or thong_tin is None or not joblib_path.exists():
        return None, thong_tin
    if thong_tin.get("ma_bam") != ma_bam:
        print(f"[{ten}] dữ liệu hoặc tham số đã đổi so với bản lưu -> huấn luyện lại.")
        return None, thong_tin
    _kiem_tra_phien_ban(thong_tin, ten)
    doi_tuong = joblib.load(joblib_path)
    print(f"[{ten}] nạp từ models/{joblib_path.name} (lưu lúc {thong_tin['luu_luc']}, "
          f"lần đầu huấn luyện mất {thong_tin['thoi_gian_giay']} giây).")
    return doi_tuong, thong_tin


def huan_luyen_hoac_tai(ten: str, mo_hinh, X, y=None, lam_lai: bool = False, **fit_params):
    """Nạp mô hình đã lưu nếu khớp, ngược lại fit `mo_hinh` trên (X, y) rồi lưu lại.

    `mo_hinh` có thể là estimator, Pipeline, RandomizedSearchCV/GridSearchCV,
    VotingRegressor, StackingRegressor... `fit_params` được chuyển thẳng cho .fit()
    (ví dụ groups=... cho search dùng GroupKFold).
    Trả về đối tượng đã fit.
    """
    mo_hinh_goc = clone(mo_hinh)
    ma_bam = _bam(mo_hinh_goc.get_params(deep=True), type(mo_hinh_goc).__name__,
                  X, y, fit_params)

    doi_tuong, _ = _nap_neu_khop(ten, ma_bam, lam_lai)
    if doi_tuong is not None:
        return doi_tuong

    print(f"[{ten}] đang huấn luyện...")
    bat_dau = time.perf_counter()
    mo_hinh_goc.fit(X, y, **fit_params)
    giay = round(time.perf_counter() - bat_dau, 2)

    thong_tin = {
        "ten": ten,
        "loai": type(mo_hinh_goc).__name__,
        "luu_luc": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "thoi_gian_giay": giay,
        "so_dong_train": int(len(X)) if hasattr(X, "__len__") else None,
        "ma_bam": ma_bam,
        "phien_ban": _phien_ban(),
    }
    if hasattr(mo_hinh_goc, "best_params_"):  # RandomizedSearchCV / GridSearchCV
        thong_tin["best_params"] = _sang_json(mo_hinh_goc.best_params_)
        thong_tin["best_score"] = _sang_json(getattr(mo_hinh_goc, "best_score_", None))
        thong_tin["scoring"] = _sang_json(getattr(mo_hinh_goc, "scoring", None))
        thong_tin["params_mo_hinh_tot_nhat"] = _sang_json(
            mo_hinh_goc.best_estimator_.get_params(deep=False)
        ) if hasattr(mo_hinh_goc, "best_estimator_") else None
    else:
        thong_tin["params"] = _sang_json(mo_hinh_goc.get_params(deep=False))

    _ghi(ten, mo_hinh_goc, thong_tin)
    print(f"[{ten}] xong sau {giay} giây, đã lưu models/{ten}.joblib + .json")
    return mo_hinh_goc


def tinh_hoac_tai(ten: str, ham, *args, lam_lai: bool = False, **kwargs):
    """Gọi ham(*args, **kwargs) và lưu kết quả; lần sau đối số giống hệt thì nạp lại.

    Dùng cho các bước tính lâu không phải fit: danh_gia_cv, cross_validate, learning_curve,
    validation_curve, permutation_importance... Estimator trong args được so theo tham số
    (chưa fit) nên đổi tham số là tự tính lại.
    """
    def _chuan_hoa(a):
        if hasattr(a, "get_params") and hasattr(a, "fit"):
            return (type(a).__name__, clone(a).get_params(deep=True))
        return a

    ma_bam = _bam(getattr(ham, "__qualname__", repr(ham)),
                  [_chuan_hoa(a) for a in args],
                  {k: _chuan_hoa(v) for k, v in kwargs.items()})

    ket_qua, _ = _nap_neu_khop(ten, ma_bam, lam_lai)
    if ket_qua is not None:
        return ket_qua

    print(f"[{ten}] đang tính {getattr(ham, '__name__', 'ham')}...")
    bat_dau = time.perf_counter()
    ket_qua = ham(*args, **kwargs)
    giay = round(time.perf_counter() - bat_dau, 2)

    thong_tin = {
        "ten": ten,
        "loai": f"ket_qua:{getattr(ham, '__name__', 'ham')}",
        "luu_luc": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "thoi_gian_giay": giay,
        "ma_bam": ma_bam,
        "phien_ban": _phien_ban(),
    }
    if isinstance(ket_qua, dict):
        thong_tin["tom_tat"] = _sang_json(
            {k: (float(np.mean(v)) if isinstance(v, (list, np.ndarray)) and np.size(v) and
                 np.issubdtype(np.asarray(v).dtype, np.number) else v)
             for k, v in ket_qua.items()}
        )
    _ghi(ten, ket_qua, thong_tin)
    print(f"[{ten}] xong sau {giay} giây, đã lưu models/{ten}.joblib + .json")
    return ket_qua


def danh_sach_mo_hinh():
    """Bảng các mô hình / kết quả đã lưu trong models/ (cần pandas)."""
    import pandas as pd

    dong = []
    for p in sorted(THU_MUC_MODELS.glob("*.json")):
        tt = json.loads(p.read_text(encoding="utf-8"))
        dong.append({k: tt.get(k) for k in
                     ("ten", "loai", "luu_luc", "thoi_gian_giay", "best_score", "kich_thuoc_MB")})
    return pd.DataFrame(dong)
