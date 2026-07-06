# -*- coding: utf-8 -*-


import os
import re
import json
import math
import time
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import qmc
    HAS_QMC = True
except Exception:
    qmc = None
    HAS_QMC = False

try:
    from sklearn.cluster import KMeans
    HAS_KMEANS = True
except Exception:
    KMeans = None
    HAS_KMEANS = False


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class Config:
    CITY_NAME: str = "Shanghai"
    DATA_PATH: str = "/public/home/sszzkli/.local/Step2/shanghai.csv"
    OUT_ROOT: str = "/public/home/sszzkli/.local/Step2/shanghai_sclpso_individual_model_outputs"

    RANDOM_STATE: int = 20260521

    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    TARGET_PRIORITY: Tuple[str, ...] = (
        "PM2.5", "PM₂.₅", "PM25", "PM2_5", "pm25", "pm2.5", "pm₂.₅"
    )




    STRICT_FORECAST_ONLY_LAG_FEATURES: bool = False


    DROP_ALL_IAQI_COLUMNS: bool = False


    TARGET_LAGS: Tuple[int, ...] = (1, 2, 3, 5, 7, 14, 21, 30)
    TARGET_ROLL_WINDOWS: Tuple[int, ...] = (3, 7, 14, 21, 30)
    COVARIATE_LAGS: Tuple[int, ...] = (1, 3, 7)


    MODEL_NAMES: Tuple[str, ...] = ("SVR", "RandomForest", "ExtraTrees", "MLP")






    OPTIMIZERS: Tuple[str, ...] = (
        "SCLPSO",
    )
    REFERENCE_OPTIMIZER: str = "None"
    REPORT_OPTIMIZER: str = "SCLPSO"


    OPTIMIZE_TOP_K_MODELS: int = 0


    OPT_PARTICLES: int = 18
    OPT_ITERATIONS: int = 32
    OPT_MAX_FES: int = 0


    PSO_C1: float = 1.5
    PSO_C2: float = 2.0
    PSO_W_START: float = 0.90
    PSO_W_END: float = 0.35
    PSO_VMAX_RATIO: float = 0.25


    SCLPSO_NO_STATE: bool = True
    SCLPSO_ENABLE_ESCAPE_RECOVERY: bool = False
    SCLPSO_NMIN: int = 8
    SCLPSO_SHADE_RATE: float = 0.75
    SCLPSO_SHADE_H: int = 6
    SCLPSO_ARCHIVE_RATE: float = 2.0


    N_JOBS: int = 16


    FONT_PATH: str = "/public/home/sszzkli/.local/Times New Roman.ttf"
    DPI: int = 300


CFG = Config()


# ============================================================
# 2. Basic utilities
# ============================================================

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def stable_hash(text: str) -> int:
    value = 0
    for ch in str(text):
        value = (value * 131 + ord(ch)) % 1_000_003
    return value


def setup_matplotlib():
    if CFG.FONT_PATH and os.path.exists(CFG.FONT_PATH):
        font_manager.fontManager.addfont(CFG.FONT_PATH)
        prop = font_manager.FontProperties(fname=CFG.FONT_PATH)
        font_name = prop.get_name()
    else:
        font_name = "DejaVu Serif"
        prop = font_manager.FontProperties(family=font_name)

    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = CFG.DPI
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 15
    return prop


def read_csv_safely(path: str) -> Tuple[pd.DataFrame, str]:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "cp936", "latin1", "macroman"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to read CSV: {path}; last error: {last_error}")


def normalize_name(s: Any) -> str:
    s = "" if s is None else str(s)
    replacements = {
        "₂": "2", "₃": "3", "₅": "5", "₁": "1", "₀": "0",
        "．": ".", "。": ".", " ": "", "_": "", "-": "", ".": "",
        "（": "", "）": "", "(": "", ")": "",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.lower()


def parse_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.strip()
    x = x.replace({
        "—": np.nan, "-": np.nan, "--": np.nan, "nan": np.nan, "NaN": np.nan,
        "None": np.nan, "null": np.nan, "": np.nan, " ": np.nan,
    })
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace(r"[^\d\.\-\+eE]", "", regex=True)
    x = x.replace({"": np.nan, "-": np.nan, "+": np.nan})
    return pd.to_numeric(x, errors="coerce")


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    norm_to_col = {normalize_name(c): c for c in df.columns}
    y_col = norm_to_col.get("year") or norm_to_col.get("年份") or norm_to_col.get("年")
    m_col = norm_to_col.get("month") or norm_to_col.get("月份") or norm_to_col.get("月")
    d_col = norm_to_col.get("day") or norm_to_col.get("日期日") or norm_to_col.get("日")
    if y_col is not None and m_col is not None and d_col is not None:
        return "__ymd__"

    candidates = []
    for c in df.columns:
        raw = str(c)
        n = normalize_name(c)
        if n in ("date", "time", "datetime", "timestamp") or "date" in n or "time" in n or "日期" in raw or "时间" in raw:
            candidates.append(c)
    for c in candidates:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.50:
            return c
    return None


def parse_dates(df: pd.DataFrame, date_col: str) -> pd.Series:
    if date_col == "__ymd__":
        norm_to_col = {normalize_name(c): c for c in df.columns}
        y_col = norm_to_col.get("year") or norm_to_col.get("年份") or norm_to_col.get("年")
        m_col = norm_to_col.get("month") or norm_to_col.get("月份") or norm_to_col.get("月")
        d_col = norm_to_col.get("day") or norm_to_col.get("日期日") or norm_to_col.get("日")
        return pd.to_datetime({
            "year": pd.to_numeric(df[y_col], errors="coerce"),
            "month": pd.to_numeric(df[m_col], errors="coerce"),
            "day": pd.to_numeric(df[d_col], errors="coerce"),
        }, errors="coerce")
    return pd.to_datetime(df[date_col], errors="coerce")


def date_order_label(dates: pd.Series) -> str:
    x = pd.to_datetime(dates, errors="coerce").dropna()
    if len(x) < 2:
        return "unknown"
    diffs = x.diff().dropna()
    pos = (diffs > pd.Timedelta(0)).mean()
    neg = (diffs < pd.Timedelta(0)).mean()
    if pos > 0.90:
        return "ascending"
    if neg > 0.90:
        return "descending"
    return "mixed"


def detect_target_column(df: pd.DataFrame) -> str:
    norm_to_col = {normalize_name(c): c for c in df.columns}
    for p in CFG.TARGET_PRIORITY:
        npat = normalize_name(p)
        if npat in norm_to_col:
            return norm_to_col[npat]

    candidates = []
    for c in df.columns:
        n = normalize_name(c)
        if "pm25" in n or "pm2" in n:
            candidates.append(c)
    if candidates:
        for c in candidates:
            raw = str(c)
            n = normalize_name(c)
            if "iaqi" not in n and "指数" not in raw and "分指数" not in raw:
                return c
        return candidates[0]
    raise ValueError("Could not detect PM2.5 target column.")


def identify_leakage_columns(df: pd.DataFrame, target_col: str, date_col: str) -> List[str]:
    leakage = []
    target_norm = normalize_name(target_col)
    for c in df.columns:
        if c == target_col:
            continue
        if date_col != "__ymd__" and c == date_col:
            continue
        raw = str(c)
        n = normalize_name(c)

        if n in ("id", "index", "no", "number", "序号"):
            leakage.append(c)
            continue
        if n == "aqi" or "空气质量指数" in raw or "质量指数" in raw:
            leakage.append(c)
            continue
        if ("pm25iaqi" in n or "pm25分指数" in n or "pm25指数" in n or "pm25subindex" in n) and n != target_norm:
            leakage.append(c)
            continue
        label_keywords = [
            "空气质量等级", "质量等级", "质量评价", "首要污染物", "主要污染物",
            "primarypollutant", "qualitylevel", "status", "等级", "类别", "状况",
        ]
        if any(k in raw or normalize_name(k) in n for k in label_keywords):
            leakage.append(c)
            continue
        if CFG.DROP_ALL_IAQI_COLUMNS and "iaqi" in n:
            leakage.append(c)
            continue
    return sorted(set(leakage))


def safe_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        prefix + "R2": float(r2_score(y_true, y_pred)),
        prefix + "RMSE": rmse(y_true, y_pred),
        prefix + "MAE": float(mean_absolute_error(y_true, y_pred)),
        prefix + "MAPE_percent": safe_mape(y_true, y_pred),
        prefix + "ExplainedVariance": float(explained_variance_score(y_true, y_pred)),
        prefix + "Bias": float(np.mean(y_pred - y_true)),
    }


def improvement_percent(before: float, after: float, lower_is_better: bool = True) -> float:
    before = float(before)
    after = float(after)
    if abs(before) < 1e-12:
        return np.nan
    if lower_is_better:
        return (before - after) / abs(before) * 100.0
    return (after - before) / abs(before) * 100.0


# ============================================================
# 3. Data preparation
# ============================================================

def load_prepare_data(out_root: Path) -> Dict[str, Any]:
    df_raw, encoding = read_csv_safely(CFG.DATA_PATH)

    date_col = find_date_column(df_raw)
    if date_col is None:
        raise ValueError("No valid date column detected. Chronological split requires a date column.")
    dates = parse_dates(df_raw, date_col)
    raw_order = date_order_label(dates)

    target_col = detect_target_column(df_raw)
    leakage_cols = identify_leakage_columns(df_raw, target_col, date_col)

    df = df_raw.copy()
    df["__date__"] = dates

    numeric_cols = []
    for c in df.columns:
        if c == "__date__" or c in leakage_cols:
            continue
        converted = parse_numeric_series(df[c])
        if c == target_col or converted.notna().mean() >= 0.50:
            df[c] = converted
            numeric_cols.append(c)

    if target_col not in numeric_cols:
        df[target_col] = parse_numeric_series(df_raw[target_col])
        numeric_cols.append(target_col)

    df = df[["__date__"] + numeric_cols].copy()
    df = df[df["__date__"].notna()].copy()
    df = df[df[target_col].notna()].copy()
    df = df.sort_values("__date__").reset_index(drop=True)


    df = df.groupby("__date__", as_index=False).mean(numeric_only=True)
    df = df.sort_values("__date__").reset_index(drop=True)

    metadata = {
        "city_name": CFG.CITY_NAME,
        "data_path": CFG.DATA_PATH,
        "encoding": encoding,
        "raw_shape": list(df_raw.shape),
        "date_col_detected": date_col,
        "raw_date_order_before_sorting": raw_order,
        "target_col": target_col,
        "leakage_dropped_columns": leakage_cols,
        "drop_all_iaqi_columns": CFG.DROP_ALL_IAQI_COLUMNS,
        "strict_forecast_only_lag_features": CFG.STRICT_FORECAST_ONLY_LAG_FEATURES,
        "numeric_columns_after_leakage_removal": numeric_cols,
        "n_rows_after_cleaning": int(len(df)),
        "cleaning_start_date": str(df["__date__"].min()),
        "cleaning_end_date": str(df["__date__"].max()),
    }
    with open(out_root / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    pd.DataFrame([metadata]).to_csv(out_root / "preprocessing_metadata_flat.csv", index=False, encoding="utf-8-sig")
    df.to_csv(out_root / f"{CFG.CITY_NAME.lower()}_cleaned_numeric_timeseries.csv", index=False, encoding="utf-8-sig")
    return {"df": df, "target_col": target_col, "metadata": metadata}


def add_time_features(feat: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    out = feat.copy()
    d = pd.to_datetime(dates)
    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["day"] = d.dt.day
    out["dayofweek"] = d.dt.dayofweek
    out["dayofyear"] = d.dt.dayofyear
    out["quarter"] = d.dt.quarter
    out["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)
    return out


def make_features(df: pd.DataFrame, target_col: str, out_root: Path) -> Dict[str, Any]:
    df = df.sort_values("__date__").reset_index(drop=True)
    dates = pd.to_datetime(df["__date__"])
    y = df[target_col].astype(float)

    base_numeric_cols = [
        c for c in df.columns
        if c not in ("__date__", target_col) and pd.api.types.is_numeric_dtype(df[c])
    ]

    feat = pd.DataFrame(index=df.index)

    if not CFG.STRICT_FORECAST_ONLY_LAG_FEATURES:
        for c in base_numeric_cols:
            feat[c] = df[c]


    for lag in CFG.TARGET_LAGS:
        feat[f"{target_col}_lag{lag}"] = y.shift(lag)

    shifted_target = y.shift(1)
    for w in CFG.TARGET_ROLL_WINDOWS:
        min_p = max(2, w // 2)
        feat[f"{target_col}_rollmean{w}"] = shifted_target.rolling(w, min_periods=min_p).mean()
        feat[f"{target_col}_rollstd{w}"] = shifted_target.rolling(w, min_periods=min_p).std()
        feat[f"{target_col}_rollmin{w}"] = shifted_target.rolling(w, min_periods=min_p).min()
        feat[f"{target_col}_rollmax{w}"] = shifted_target.rolling(w, min_periods=min_p).max()

    for c in base_numeric_cols:
        for lag in CFG.COVARIATE_LAGS:
            feat[f"{c}_lag{lag}"] = df[c].shift(lag)

    feat = add_time_features(feat, dates)

    valid_mask = y.notna() & dates.notna()
    max_lag = max(max(CFG.TARGET_LAGS), max(CFG.COVARIATE_LAGS) if CFG.COVARIATE_LAGS else 1)
    valid_mask.iloc[:max_lag] = False

    feat = feat.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    dates = dates.loc[valid_mask].reset_index(drop=True)

    feat = feat.dropna(axis=1, how="all")

    n = len(feat)
    train_end = int(n * CFG.TRAIN_RATIO)
    val_end = int(n * (CFG.TRAIN_RATIO + CFG.VAL_RATIO))

    X_train_raw = feat.iloc[:train_end].copy()
    X_val_raw = feat.iloc[train_end:val_end].copy()
    X_test_raw = feat.iloc[val_end:].copy()

    y_train = y.iloc[:train_end].values.astype(float)
    y_val = y.iloc[train_end:val_end].values.astype(float)
    y_test = y.iloc[val_end:].values.astype(float)

    date_train = dates.iloc[:train_end].reset_index(drop=True)
    date_val = dates.iloc[train_end:val_end].reset_index(drop=True)
    date_test = dates.iloc[val_end:].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_test_imp = imputer.transform(X_test_raw)

    X_train = scaler.fit_transform(X_train_imp)
    X_val = scaler.transform(X_val_imp)
    X_test = scaler.transform(X_test_imp)

    chronological_ok = True
    try:
        chronological_ok = bool(date_train.max() < date_val.min() < date_val.max() < date_test.min())
    except Exception:
        chronological_ok = False

    split_ranges = {
        "n_total_after_feature_engineering": int(n),
        "n_features": int(feat.shape[1]),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_start_date": str(date_train.min()),
        "train_end_date": str(date_train.max()),
        "validation_start_date": str(date_val.min()),
        "validation_end_date": str(date_val.max()),
        "test_start_date": str(date_test.min()),
        "test_end_date": str(date_test.max()),
        "chronological_order_verified": chronological_ok,
    }
    pd.DataFrame([split_ranges]).to_csv(out_root / "split_date_ranges.csv", index=False, encoding="utf-8-sig")

    feature_info = {
        "feature_columns": list(feat.columns),
        "base_same_day_covariates": base_numeric_cols if not CFG.STRICT_FORECAST_ONLY_LAG_FEATURES else [],
        "target_lags": list(CFG.TARGET_LAGS),
        "target_rolling_windows": list(CFG.TARGET_ROLL_WINDOWS),
        "covariate_lags": list(CFG.COVARIATE_LAGS),
    }
    with open(out_root / "feature_engineering_metadata.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    pd.DataFrame({"feature": feat.columns}).to_csv(out_root / "feature_columns.csv", index=False, encoding="utf-8-sig")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "date_train": date_train,
        "date_val": date_val,
        "date_test": date_test,
        "feature_names": list(feat.columns),
        "imputer": imputer,
        "scaler": scaler,
        "split_ranges": split_ranges,
    }


# ============================================================
# 4. Model factories and search space
# ============================================================

def available_model_names() -> List[str]:
    return list(CFG.MODEL_NAMES)


def default_model(name: str, seed: int):
    if name == "SVR":
        return SVR(kernel="rbf", C=20.0, epsilon=0.10, gamma="scale", degree=3, coef0=0.0,
                   cache_size=1000, max_iter=30000)

    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=350, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            max_features="sqrt", bootstrap=True, random_state=seed, n_jobs=CFG.N_JOBS,
        )

    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            max_features="sqrt", bootstrap=False, random_state=seed, n_jobs=CFG.N_JOBS,
        )

    if name == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=(80, 40), activation="relu", solver="adam", alpha=1e-4,
            learning_rate="adaptive", learning_rate_init=1e-3, batch_size="auto",
            max_iter=800, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=40, random_state=seed,
        )

    raise ValueError(f"Unavailable model: {name}")


def model_from_params(name: str, params: Dict[str, Any], seed: int):
    p = params
    if name == "SVR":
        return SVR(
            kernel=p["kernel"], C=float(p["C"]), epsilon=float(p["epsilon"]),
            gamma=p["gamma"], degree=int(p["degree"]), coef0=float(p["coef0"]),
            cache_size=1000, max_iter=30000,
        )

    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=None if int(p["max_depth"]) <= 0 else int(p["max_depth"]),
            min_samples_split=int(p["min_samples_split"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=p["max_features"],
            bootstrap=bool(p["bootstrap"]),
            random_state=seed,
            n_jobs=CFG.N_JOBS,
        )

    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=None if int(p["max_depth"]) <= 0 else int(p["max_depth"]),
            min_samples_split=int(p["min_samples_split"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=p["max_features"],
            bootstrap=bool(p["bootstrap"]),
            random_state=seed,
            n_jobs=CFG.N_JOBS,
        )

    if name == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=p["hidden_layer_sizes"], activation=p["activation"],
            solver=p["solver"], alpha=float(p["alpha"]), learning_rate=p["learning_rate"],
            learning_rate_init=float(p["learning_rate_init"]), batch_size=p["batch_size"],
            max_iter=int(p["max_iter"]), early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=int(p["n_iter_no_change"]), random_state=seed,
        )

    raise ValueError(f"Unavailable model: {name}")


def choose(options: List[Any], x: float) -> Any:
    idx = int(np.clip(np.floor(np.clip(x, 0, 1) * len(options)), 0, len(options) - 1))
    return options[idx]


def int_range(x: float, lo: int, hi: int) -> int:
    return int(round(lo + np.clip(x, 0, 1) * (hi - lo)))


def log_range(x: float, lo: float, hi: float) -> float:
    return float(np.exp(np.log(lo) + np.clip(x, 0, 1) * (np.log(hi) - np.log(lo))))


def lin_range(x: float, lo: float, hi: float) -> float:
    return float(lo + np.clip(x, 0, 1) * (hi - lo))


def param_dim(name: str) -> int:
    dims = {"SVR": 7, "RandomForest": 6, "ExtraTrees": 6, "MLP": 8}
    return dims[name]


def vector_to_params(name: str, z: np.ndarray) -> Dict[str, Any]:
    z = np.asarray(z, dtype=float)

    if name == "SVR":
        gamma_mode = choose(["scale", "auto", "numeric"], z[3])
        gamma = gamma_mode if gamma_mode != "numeric" else log_range(z[4], 1e-4, 10.0)
        return {
            "C": log_range(z[0], 1e-2, 2e3),
            "epsilon": log_range(z[1], 1e-3, 5.0),
            "kernel": choose(["rbf", "poly", "sigmoid", "linear"], z[2]),
            "gamma": gamma,
            "degree": int_range(z[5], 2, 5),
            "coef0": lin_range(z[6], 0.0, 3.0),
            "gamma_mode": gamma_mode,
        }

    if name == "RandomForest":
        return {
            "n_estimators": int_range(z[0], 120, 1000),
            "max_depth": choose([0, 4, 6, 8, 10, 12, 16, 20, 28, 36, 48], z[1]),
            "min_samples_split": int_range(z[2], 2, 24),
            "min_samples_leaf": int_range(z[3], 1, 16),
            "max_features": choose(["sqrt", "log2", 0.40, 0.55, 0.70, 0.85, 1.0], z[4]),
            "bootstrap": bool(choose([False, True], z[5])),
        }

    if name == "ExtraTrees":
        return {
            "n_estimators": int_range(z[0], 120, 1000),
            "max_depth": choose([0, 4, 6, 8, 10, 12, 16, 20, 28, 36, 48], z[1]),
            "min_samples_split": int_range(z[2], 2, 24),
            "min_samples_leaf": int_range(z[3], 1, 16),
            "max_features": choose(["sqrt", "log2", 0.40, 0.55, 0.70, 0.85, 1.0], z[4]),
            "bootstrap": bool(choose([False, True], z[5])),
        }

    if name == "MLP":
        return {
            "hidden_layer_sizes": choose([
                (32,), (64,), (128,), (64, 32), (80, 40), (128, 64),
                (128, 64, 32), (160, 80), (200, 100),
            ], z[0]),
            "activation": choose(["relu", "tanh", "logistic"], z[1]),
            "solver": choose(["adam", "lbfgs"], z[2]),
            "alpha": log_range(z[3], 1e-6, 1e-1),
            "learning_rate": choose(["constant", "adaptive", "invscaling"], z[4]),
            "learning_rate_init": log_range(z[5], 1e-4, 5e-2),
            "batch_size": choose(["auto", 16, 32, 64, 128], z[6]),
            "max_iter": int_range(z[7], 400, 1500),
            "n_iter_no_change": 40,
        }

    raise ValueError(name)


def params_cache_key(name: str, params: Dict[str, Any]) -> str:
    return name + "::" + json.dumps(params, sort_keys=True, default=str)


def fit_predict(model, X_train, y_train, X_val, X_test):
    model.fit(X_train, y_train)
    p_train = model.predict(X_train)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    return model, p_train, p_val, p_test


# ============================================================
# 5. Continuous optimizers for model selection
# ============================================================

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def halton_chaos_unit_init(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    if HAS_QMC:
        sampler = qmc.Halton(d=dim, scramble=True, seed=int(rng.integers(1, 2**31 - 1)))
        h = sampler.random(n)
    else:
        h = rng.random((n, dim))
    chaos = rng.random((n, dim))
    for _ in range(5):
        chaos = 4.0 * chaos * (1.0 - chaos)
    return clip01(0.70 * h + 0.30 * chaos)


def uniform_unit_init(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n, dim))


def _evaluate_population(pop: np.ndarray,
                         objective: Callable[[np.ndarray], float],
                         maximize: bool,
                         state: Dict[str, Any]) -> np.ndarray:
    scores = []
    for x in np.asarray(pop, dtype=float):
        if state["fes"] >= state["max_fes"]:
            scores.append(-np.inf if maximize else np.inf)
            continue
        try:
            score = float(objective(clip01(x)))
            if not np.isfinite(score):
                score = -np.inf if maximize else np.inf
        except Exception:
            score = -np.inf if maximize else np.inf
        state["fes"] += 1
        scores.append(score)
    return np.asarray(scores, dtype=float)


def _is_better(a: float, b: float, maximize: bool) -> bool:
    return a > b if maximize else a < b


def _best_index(values: np.ndarray, maximize: bool) -> int:
    return int(np.nanargmax(values) if maximize else np.nanargmin(values))


def _worst_index(values: np.ndarray, maximize: bool) -> int:
    return int(np.nanargmin(values) if maximize else np.nanargmax(values))


def _make_result(algorithm: str, best_x: np.ndarray, best_score: float, history: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "algorithm": algorithm,
        "best_x": clip01(best_x),
        "best_score": float(best_score),
        "history": pd.DataFrame(history),
        "fes": int(state["fes"]),
        "max_fes": int(state["max_fes"]),
    }


def optimize_pso(algorithm: str,
                 objective: Callable[[np.ndarray], float],
                 dim: int,
                 max_fes: int,
                 pop_size: int,
                 seed: int,
                 maximize: bool,
                 use_halton: bool = False) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = {"fes": 0, "max_fes": int(max_fes)}
    n = int(max(4, pop_size))
    x = halton_chaos_unit_init(n, dim, rng) if use_halton else uniform_unit_init(n, dim, rng)
    v = rng.uniform(-CFG.PSO_VMAX_RATIO, CFG.PSO_VMAX_RATIO, size=(n, dim))
    fit = _evaluate_population(x, objective, maximize, state)

    pbest = x.copy()
    pbest_fit = fit.copy()
    bi = _best_index(fit, maximize)
    gbest = x[bi].copy()
    gbest_fit = float(fit[bi])
    history = []

    iteration = 0
    while state["fes"] < max_fes:
        progress = state["fes"] / max(1, max_fes)
        w = CFG.PSO_W_START - (CFG.PSO_W_START - CFG.PSO_W_END) * progress
        r1 = rng.random((n, dim))
        r2 = rng.random((n, dim))
        v = w * v + CFG.PSO_C1 * r1 * (pbest - x) + CFG.PSO_C2 * r2 * (gbest[None, :] - x)
        v = np.clip(v, -CFG.PSO_VMAX_RATIO, CFG.PSO_VMAX_RATIO)
        x = clip01(x + v)
        fit = _evaluate_population(x, objective, maximize, state)
        for i in range(n):
            if _is_better(fit[i], pbest_fit[i], maximize):
                pbest_fit[i] = fit[i]
                pbest[i] = x[i].copy()
            if _is_better(fit[i], gbest_fit, maximize):
                gbest_fit = float(fit[i])
                gbest = x[i].copy()
        iteration += 1
        history.append({"iteration": iteration, "fes": state["fes"], "best_score": gbest_fit})
    return _make_result(algorithm, gbest, gbest_fit, history, state)


def _cl_learning_probability(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.5], dtype=float)
    i = np.arange(n, dtype=float)
    return 0.05 + 0.45 * (np.exp(10.0 * i / (n - 1.0)) - 1.0) / (np.exp(10.0) - 1.0)


def _sample_shade_F(mean_f: float, rng: np.random.Generator) -> float:
    for _ in range(50):
        f = float(mean_f + 0.1 * rng.standard_cauchy())
        if f > 0:
            return min(f, 1.0)
    return min(max(float(mean_f), 1e-8), 1.0)


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights * values))
    if denom <= 1e-300:
        return float(np.mean(values)) if len(values) else 0.5
    return float(np.sum(weights * values * values) / denom)


def optimize_cl_sh_sclpso(algorithm: str,
                          objective: Callable[[np.ndarray], float],
                          dim: int,
                          max_fes: int,
                          pop_size: int,
                          seed: int,
                          maximize: bool) -> Dict[str, Any]:
    """Progressive variants:
    HPSO    = PSO + Halton-chaotic initialization
    CLHPSO  = HPSO + comprehensive learning
    SHCLPSO = CLHPSO + success-history current-to-pbest mutation
    SCLPSO  = SHCLPSO + linear population size reduction
    """
    if algorithm == "HPSO":
        return optimize_pso(algorithm, objective, dim, max_fes, pop_size, seed, maximize, use_halton=True)

    rng = np.random.default_rng(seed)
    state = {"fes": 0, "max_fes": int(max_fes)}

    use_lpsr = algorithm == "SCLPSO"
    use_shade = algorithm in ("SHCLPSO", "SCLPSO")

    n_init = int(max(6, pop_size))
    n_min = int(max(4, min(CFG.SCLPSO_NMIN, n_init)))

    x = halton_chaos_unit_init(n_init, dim, rng)
    v = np.zeros_like(x)
    fit = _evaluate_population(x, objective, maximize, state)
    pbest = x.copy()
    pbest_fit = fit.copy()
    bi = _best_index(fit, maximize)
    gbest = x[bi].copy()
    gbest_fit = float(fit[bi])

    pc = _cl_learning_probability(n_init)
    refresh_gap = 7
    no_improve = np.zeros(n_init, dtype=int)
    exemplars = np.tile(np.arange(n_init)[:, None], (1, dim))

    def refresh_particle(i: int):
        n = x.shape[0]
        candidates = [j for j in range(n) if j != i]
        all_own = True
        for d in range(dim):
            if rng.random() < pc[i] and len(candidates) >= 2:
                a, b = rng.choice(candidates, size=2, replace=False)
                winner = int(a if _is_better(pbest_fit[a], pbest_fit[b], maximize) else b)
                exemplars[i, d] = winner
                all_own = False
            else:
                exemplars[i, d] = i
        if all_own and len(candidates) >= 2:
            d = int(rng.integers(0, dim))
            a, b = rng.choice(candidates, size=2, replace=False)
            exemplars[i, d] = int(a if _is_better(pbest_fit[a], pbest_fit[b], maximize) else b)

    for i in range(n_init):
        refresh_particle(i)

    h_size = max(2, int(CFG.SCLPSO_SHADE_H))
    m_cr = np.full(h_size, 0.5, dtype=float)
    m_f = np.full(h_size, 0.5, dtype=float)
    memory_pos = 0
    archive = np.empty((0, dim), dtype=float)

    history = []
    iteration = 0

    while state["fes"] < max_fes and x.shape[0] >= 4:
        n = x.shape[0]
        progress = state["fes"] / max(1, max_fes)

        for i in range(n):
            if no_improve[i] >= refresh_gap:
                refresh_particle(i)
                no_improve[i] = 0

        dims = np.arange(dim)
        exemplar_pos = np.empty_like(x)
        for i in range(n):
            exemplar_pos[i] = pbest[exemplars[i], dims]

        w = CFG.PSO_W_START - (CFG.PSO_W_START - CFG.PSO_W_END) * progress
        c_cl = 1.49445
        c_g = 0.05 + 0.80 * progress
        v = w * v + c_cl * rng.random((n, dim)) * (exemplar_pos - x) + c_g * rng.random((n, dim)) * (gbest[None, :] - x)
        v = np.clip(v, -CFG.PSO_VMAX_RATIO, CFG.PSO_VMAX_RATIO)
        x = clip01(x + v)
        fit = _evaluate_population(x, objective, maximize, state)

        for i in range(n):
            if _is_better(fit[i], pbest_fit[i], maximize):
                pbest_fit[i] = fit[i]
                pbest[i] = x[i].copy()
                no_improve[i] = 0
            else:
                no_improve[i] += 1
            if _is_better(fit[i], gbest_fit, maximize):
                gbest_fit = float(fit[i])
                gbest = x[i].copy()


        if use_shade and state["fes"] < max_fes and n >= 4:
            remaining = max_fes - state["fes"]
            n_trials = int(max(1, min(n, remaining, round(CFG.SCLPSO_SHADE_RATE * n))))
            trial_indices = rng.choice(n, size=n_trials, replace=False)
            order = np.argsort(-pbest_fit if maximize else pbest_fit)
            p_num = max(2, int(round(0.20 * n)))
            top_indices = order[:min(p_num, n)]
            union = x if archive.shape[0] == 0 else np.vstack([x, archive])
            union_size = union.shape[0]
            trials = []
            cr_values = []
            f_values = []

            for i in trial_indices:
                mem_idx = int(rng.integers(0, h_size))
                cr = float(np.clip(rng.normal(m_cr[mem_idx], 0.1), 0.0, 1.0))
                f_mut = _sample_shade_F(m_f[mem_idx], rng)
                pbest_idx = int(rng.choice(top_indices))
                r1_candidates = np.array([j for j in range(n) if j != i])
                r1_idx = int(rng.choice(r1_candidates))
                while True:
                    r2_idx = int(rng.integers(0, union_size))
                    if r2_idx != i and r2_idx != r1_idx:
                        break
                mutant = x[i] + f_mut * (pbest[pbest_idx] - x[i]) + f_mut * (x[r1_idx] - union[r2_idx])
                mutant = clip01(mutant)
                cross = rng.random(dim) <= cr
                cross[int(rng.integers(0, dim))] = True
                trial = np.where(cross, mutant, x[i])
                trials.append(trial)
                cr_values.append(cr)
                f_values.append(f_mut)

            trial_arr = np.asarray(trials, dtype=float)
            trial_scores = _evaluate_population(trial_arr, objective, maximize, state)
            s_cr, s_f, delta = [], [], []
            new_archive = []
            for k, i in enumerate(trial_indices):
                if _is_better(trial_scores[k], fit[i], maximize) or abs(trial_scores[k] - fit[i]) <= 1e-15:
                    if _is_better(trial_scores[k], fit[i], maximize):
                        new_archive.append(x[i].copy())
                        s_cr.append(cr_values[k])
                        s_f.append(f_values[k])
                        delta.append(abs(trial_scores[k] - fit[i]))
                    x[i] = trial_arr[k]
                    fit[i] = trial_scores[k]
                if _is_better(trial_scores[k], pbest_fit[i], maximize):
                    pbest[i] = trial_arr[k]
                    pbest_fit[i] = trial_scores[k]
                    no_improve[i] = 0
                if _is_better(trial_scores[k], gbest_fit, maximize):
                    gbest = trial_arr[k].copy()
                    gbest_fit = float(trial_scores[k])

            if new_archive:
                archive = np.vstack([archive, np.asarray(new_archive, dtype=float)])
            archive_max = int(max(0, round(CFG.SCLPSO_ARCHIVE_RATE * x.shape[0])))
            if archive.shape[0] > archive_max and archive_max > 0:
                keep = rng.choice(archive.shape[0], size=archive_max, replace=False)
                archive = archive[keep]

            if len(s_f) > 0:
                df = np.asarray(delta, dtype=float)
                weights = df / max(np.sum(df), 1e-300)
                m_cr[memory_pos] = _weighted_lehmer(np.asarray(s_cr), weights)
                m_f[memory_pos] = _weighted_lehmer(np.asarray(s_f), weights)
                memory_pos = (memory_pos + 1) % h_size


        if use_lpsr and x.shape[0] > n_min:
            target_n = int(round(n_init - (state["fes"] / max(1, max_fes)) * (n_init - n_min)))
            target_n = int(np.clip(target_n, n_min, n_init))
            if target_n < x.shape[0]:
                keep = np.argsort(-pbest_fit if maximize else pbest_fit)[:target_n]
                x = x[keep]
                v = v[keep]
                fit = fit[keep]
                pbest = pbest[keep]
                pbest_fit = pbest_fit[keep]
                no_improve = no_improve[keep]
                pc = _cl_learning_probability(target_n)
                exemplars = np.tile(np.arange(target_n)[:, None], (1, dim))
                for i in range(target_n):
                    refresh_particle(i)

        iteration += 1
        history.append({"iteration": iteration, "fes": state["fes"], "best_score": gbest_fit, "population_size": int(x.shape[0])})

    return _make_result(algorithm, gbest, gbest_fit, history, state)


def optimize_continuous(algorithm: str,
                        objective: Callable[[np.ndarray], float],
                        dim: int,
                        max_fes: int,
                        pop_size: int,
                        seed: int,
                        maximize: bool = True) -> Dict[str, Any]:
    algorithm = algorithm.strip()
    if max_fes <= 0:
        max_fes = max(1, pop_size * CFG.OPT_ITERATIONS)



    if algorithm == "PSO":
        return optimize_pso(algorithm, objective, dim, max_fes, pop_size, seed, maximize, use_halton=False)
    if algorithm in ("HPSO", "CLHPSO", "SHCLPSO", "SCLPSO"):
        return optimize_cl_sh_sclpso(algorithm, objective, dim, max_fes, pop_size, seed, maximize)

    raise ValueError(f"Unknown optimizer in lineage-only application setting: {algorithm}")


# ============================================================
# 6. Model optimization and ensemble optimization
# ============================================================

def sclpso_like_optimize_model(optimizer: str,
                               name: str,
                               data: Dict[str, Any],
                               baseline_val_r2: float,
                               out_root: Path) -> Dict[str, Any]:
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test = data["X_test"]

    dim = param_dim(name)
    max_fes = CFG.OPT_MAX_FES if CFG.OPT_MAX_FES > 0 else CFG.OPT_PARTICLES * CFG.OPT_ITERATIONS
    seed_base = CFG.RANDOM_STATE + stable_hash(optimizer) * 17 + stable_hash(name) * 31
    cache: Dict[str, Dict[str, Any]] = {}
    eval_counter = {"n": 0}

    def objective(z: np.ndarray) -> float:
        z = clip01(z)
        params = vector_to_params(name, z)
        key = params_cache_key(name, params)
        if key in cache:
            return float(cache[key]["objective_score"])

        seed = seed_base + eval_counter["n"]
        eval_counter["n"] += 1
        model = model_from_params(name, params, seed)
        try:
            fitted, p_train, p_val, p_test = fit_predict(model, X_train, y_train, X_val, X_test)
            train_r2 = float(r2_score(y_train, p_train))
            val_r2 = float(r2_score(y_val, p_val))

            gap = max(0.0, train_r2 - val_r2 - 0.12)
            score = val_r2 - 0.03 * gap
            rec = {
                "ok": True,
                "objective_score": float(score),
                "val_R2": val_r2,
                "train_R2": train_r2,
                "params": params,
                "p_train": p_train,
                "p_val": p_val,
                "p_test": p_test,
                "model": fitted,
                "error": "",
            }
        except Exception as e:
            rec = {
                "ok": False,
                "objective_score": -1e9,
                "val_R2": -1e9,
                "train_R2": -1e9,
                "params": params,
                "error": repr(e),
            }
        cache[key] = rec
        return float(rec["objective_score"])

    print(f"    {optimizer}-{name}: dim={dim}, maxFEs={max_fes}")
    opt_result = optimize_continuous(
        algorithm=optimizer,
        objective=objective,
        dim=dim,
        max_fes=max_fes,
        pop_size=CFG.OPT_PARTICLES,
        seed=seed_base,
        maximize=True,
    )
    best_params = vector_to_params(name, opt_result["best_x"])
    best_key = params_cache_key(name, best_params)


    if best_key not in cache:
        _ = objective(opt_result["best_x"])
    best_rec = cache[best_key]

    if not best_rec.get("ok", False):

        successful = [v for v in cache.values() if v.get("ok", False)]
        if not successful:
            raise RuntimeError(f"{optimizer}-{name} failed: no successful hyperparameter set.")
        best_rec = max(successful, key=lambda r: r["objective_score"])

    hist_df = opt_result["history"].copy()
    if not hist_df.empty:
        hist_df["best_validation_R2"] = hist_df["best_score"]
        hist_df["optimizer"] = optimizer
        hist_df["model"] = name
    hist_df.to_csv(out_root / f"{optimizer}_{name}_history.csv", index=False, encoding="utf-8-sig")

    result = {
        "optimizer": optimizer,
        "model_name": name,
        "best_params": best_rec["params"],
        "best_validation_R2": float(best_rec["val_R2"]),
        "best_train_R2": float(best_rec["train_R2"]),
        "objective_score": float(best_rec["objective_score"]),
        "accepted_by_validation": bool(best_rec["val_R2"] >= baseline_val_r2),
        "train_pred": best_rec["p_train"],
        "val_pred": best_rec["p_val"],
        "test_pred": best_rec["p_test"],
        "model": best_rec["model"],
        "history": hist_df,
        "fes": opt_result["fes"],
        "evaluated_unique_params": len(cache),
    }

    with open(out_root / f"{optimizer}_{name}_best_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "optimizer": optimizer,
            "model_name": name,
            "best_params": result["best_params"],
            "best_validation_R2": result["best_validation_R2"],
            "baseline_validation_R2": baseline_val_r2,
            "accepted_by_validation": result["accepted_by_validation"],
            "fes": result["fes"],
            "evaluated_unique_params": result["evaluated_unique_params"],
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"      best validation R2={result['best_validation_R2']:.6f}, unique={len(cache)}")
    return result


# ============================================================
# 7. Plotting
# ============================================================

def add_bar_labels(ax, fmt="{:.3f}", rotation=0):
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height):
            ax.annotate(
                fmt.format(height),
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=12,
                rotation=rotation,
                xytext=(0, 3 if height >= 0 else -3),
                textcoords="offset points",
            )


def plot_before_after_metric(summary: Dict[str, float], out_root: Path, metric: str, lower_is_better: bool, after_label: str):
    before = summary[f"before_test_{metric}"]
    after = summary[f"after_test_{metric}"]
    improvement = improvement_percent(before, after, lower_is_better=lower_is_better)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Before", f"After {after_label}"]
    values = [before, after]
    ax.bar(labels, values)
    add_bar_labels(ax, "{:.4f}" if metric == "R2" else "{:.3f}")
    ax.set_ylabel(f"Test {metric}")
    ax.set_title(f"{CFG.CITY_NAME} Test {metric}: Before vs After {after_label}")
    if metric == "R2":
        ax.axhline(0, linestyle="--", linewidth=1)
        text = f"R2 improvement: {after - before:+.4f}"
    else:
        text = f"{metric} reduction: {improvement:+.2f}%"
    ax.text(0.5, 0.95, text, transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round", alpha=0.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / f"Before_After_{after_label}_Test_{metric}.jpg")
    plt.close(fig)


def plot_prediction_scatter(y_true, pred_before, pred_after, out_root: Path, after_label: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, pred_before, s=20, alpha=0.55, label="Before")
    ax.scatter(y_true, pred_after, s=20, alpha=0.55, label=f"After {after_label}")
    lo = float(np.nanmin([np.min(y_true), np.min(pred_before), np.min(pred_after)]))
    hi = float(np.nanmax([np.max(y_true), np.max(pred_before), np.max(pred_after)]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, label="1:1 line")
    ax.set_xlabel("Observed PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    ax.set_title("Observed vs Predicted PM2.5 on Test Set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / f"Before_After_{after_label}_Observed_vs_Predicted.jpg")
    plt.close(fig)


def plot_time_series(dates, y_true, pred_before, pred_after, out_root: Path, after_label: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, y_true, linewidth=1.8, label="Observed")
    ax.plot(dates, pred_before, linewidth=1.4, label="Before")
    ax.plot(dates, pred_after, linewidth=1.4, label=f"After {after_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5")
    ax.set_title(f"Test Time Series: Before vs After {after_label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_root / f"Before_After_{after_label}_Time_Series.jpg")
    plt.close(fig)


def plot_residual_boxplot(y_true, pred_before, pred_after, out_root: Path, after_label: str):
    resid_before = pred_before - y_true
    resid_after = pred_after - y_true
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([resid_before, resid_after], labels=["Before", f"After {after_label}"], showmeans=True)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Residual")
    ax.set_title("Test Residual Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / f"Before_After_{after_label}_Residual_Boxplot.jpg")
    plt.close(fig)


def plot_model_comparison(df: pd.DataFrame, out_root: Path, metric: str, optimizer: str):
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.8), 5))
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df[f"baseline_test_{metric}"], width, label="Original")
    ax.bar(x + width / 2, df[f"optimized_test_{metric}"], width, label=optimizer)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=35, ha="right")
    ax.set_ylabel(f"Test {metric}")
    ax.set_title(f"Model-wise Test {metric}: Baseline vs {optimizer}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / f"Modelwise_Baseline_vs_{optimizer}_Test_{metric}.jpg")
    plt.close(fig)


def plot_optimizer_comparison(final_df: pd.DataFrame, out_root: Path):
    if final_df.empty:
        return
    df = final_df.sort_values("after_validation_R2", ascending=False).copy()




    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["optimizer"], df["after_test_R2"])
    ax.set_ylabel("Test R2")
    ax.set_title("Optimizer-wise Final Ensemble Test R2")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "Optimizerwise_Final_Ensemble_Test_R2.jpg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["optimizer"], df["after_test_RMSE"])
    ax.set_ylabel("Test RMSE")
    ax.set_title("Optimizer-wise Final Ensemble Test RMSE")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "Optimizerwise_Final_Ensemble_Test_RMSE.jpg")
    plt.close(fig)


def plot_optimizer_convergence(all_histories: Dict[Tuple[str, str], pd.DataFrame], out_root: Path):

    fig, ax = plt.subplots(figsize=(10, 5))
    for opt in CFG.OPTIMIZERS:
        candidates = []
        for (optimizer, model), h in all_histories.items():
            if optimizer == opt and h is not None and not h.empty and "best_validation_R2" in h.columns:
                hh = h.copy()
                hh["model"] = model
                candidates.append(hh)
        if not candidates:
            continue

        best_h = max(candidates, key=lambda x: float(x["best_validation_R2"].iloc[-1]))
        ax.plot(best_h["iteration"], best_h["best_validation_R2"], marker="o", linewidth=1.2, label=opt)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Validation R2")
    ax.set_title("Optimizer Convergence for Model Hyperparameter Search")
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "Optimizer_Model_Convergence_Validation_R2.jpg")
    plt.close(fig)


def plot_feature_importance(model, feature_names: List[str], out_root: Path, label: str):
    importances = None
    if hasattr(model, "feature_importances_"):
        try:
            importances = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            importances = None
    if importances is None:
        return

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(out_root / f"{label}_feature_importance.csv", index=False, encoding="utf-8-sig")

    top = df.head(25).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top Feature Importances: {label}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / f"{label}_feature_importance_top25.jpg")
    plt.close(fig)


def _metric_from_row(row: pd.Series, metric: str, prefix: str = "") -> float:
    key = prefix + metric
    return float(row[key]) if key in row.index and pd.notna(row[key]) else np.nan


def make_application_summary(baseline_df: pd.DataFrame,
                                       model_opt_df: pd.DataFrame,
                                       out_root: Path) -> pd.DataFrame:
    """Create the compact application table.

    Required comparison format:
    - default models are named directly as SVR, RandomForest, ExtraTrees, and MLP;
    - SCLPSO-assisted models are named as SCLPSO-SVR, SCLPSO-RandomForest,
      SCLPSO-ExtraTrees, and SCLPSO-MLP;
    - no default ensemble and no SCLPSO-optimized ensemble are included.
    """
    rows: List[Dict[str, Any]] = []

    ok_base = baseline_df[baseline_df["status"] == "ok"].copy()
    if not ok_base.empty:
        ok_base = ok_base.sort_values(["test_R2", "test_RMSE"], ascending=[False, True])
        for _, r in ok_base.iterrows():
            rows.append({
                "method": str(r["model"]),
                "method_group": "Default model",
                "optimizer": "None",
                "model_components": str(r["model"]),
                "validation_R2": _metric_from_row(r, "R2", "validation_"),
                "test_R2": _metric_from_row(r, "R2", "test_"),
                "test_RMSE": _metric_from_row(r, "RMSE", "test_"),
                "test_MAE": _metric_from_row(r, "MAE", "test_"),
                "test_MAPE_percent": _metric_from_row(r, "MAPE_percent", "test_"),
                "test_Bias": _metric_from_row(r, "Bias", "test_"),
                "test_R2_improvement_vs_own_default": 0.0,
                "test_RMSE_reduction_percent_vs_own_default": 0.0,
                "paper_role": "default prediction model",
            })

    ok_opt = model_opt_df[model_opt_df["status"] == "ok"].copy() if model_opt_df is not None and not model_opt_df.empty else pd.DataFrame()
    if not ok_opt.empty:
        ok_opt = ok_opt.sort_values(["optimized_test_R2", "optimized_test_RMSE"], ascending=[False, True])
        for _, r in ok_opt.iterrows():
            optimizer = str(r["optimizer"])
            model = str(r["model"])
            rows.append({
                "method": f"{optimizer}-{model}",
                "method_group": "SCLPSO-optimized model",
                "optimizer": optimizer,
                "model_components": model,
                "validation_R2": float(r.get("optimized_validation_R2", np.nan)),
                "test_R2": float(r.get("optimized_test_R2", np.nan)),
                "test_RMSE": float(r.get("optimized_test_RMSE", np.nan)),
                "test_MAE": float(r.get("optimized_test_MAE", np.nan)),
                "test_MAPE_percent": float(r.get("optimized_test_MAPE_percent", np.nan)),
                "test_Bias": float(r.get("optimized_test_Bias", np.nan)),
                "test_R2_improvement_vs_own_default": float(r.get("test_R2_improvement", np.nan)),
                "test_RMSE_reduction_percent_vs_own_default": float(r.get("test_RMSE_reduction_percent", np.nan)),
                "best_params": str(r.get("best_params", "")),
                "paper_role": "SCLPSO-optimized individual model",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out


    group_order = {"Default model": 0, "SCLPSO-optimized model": 1}
    model_order = {name: i for i, name in enumerate(CFG.MODEL_NAMES)}
    out["_group_order"] = out["method_group"].map(group_order).fillna(9)
    out["_model_order"] = out["model_components"].map(model_order).fillna(99)
    out = out.sort_values(["_group_order", "_model_order", "test_R2"], ascending=[True, True, False])
    out = out.drop(columns=["_group_order", "_model_order"])

    out.to_csv(out_root / "individual_model_application_results.csv", index=False, encoding="utf-8-sig")
    out.to_csv(out_root / "application_results.csv", index=False, encoding="utf-8-sig")


    paper_cols = ["method", "test_R2", "test_RMSE", "test_MAE", "test_MAPE_percent"]
    paper = out[paper_cols].copy()
    paper["test_R2"] = paper["test_R2"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    paper["test_RMSE"] = paper["test_RMSE"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    paper["test_MAE"] = paper["test_MAE"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    paper["test_MAPE_percent"] = paper["test_MAPE_percent"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    paper.to_csv(out_root / "individual_model_application_results_latex_ready.csv", index=False, encoding="utf-8-sig")
    paper.to_csv(out_root / "application_results_latex_ready.csv", index=False, encoding="utf-8-sig")
    return out


def plot_application_results(application_df: pd.DataFrame, out_root: Path):
    """Generate compact application figures similar to application result presentation."""
    if application_df is None or application_df.empty:
        return
    df = application_df.copy()
    df = df.sort_values("test_R2", ascending=False)


    fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(df)), 5.2))
    ax.bar(df["method"], df["test_R2"])
    ax.set_ylabel("Test R2")
    ax.set_title(f"{CFG.CITY_NAME} Individual Model Application Comparison: Test R2")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    add_bar_labels(ax, "{:.3f}")
    fig.tight_layout()
    fig.savefig(out_root / "Individual_Model_Application_Test_R2.jpg")
    plt.close(fig)


    df2 = application_df.copy().sort_values("test_RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(df2)), 5.2))
    ax.bar(df2["method"], df2["test_RMSE"])
    ax.set_ylabel("Test RMSE")
    ax.set_title(f"{CFG.CITY_NAME} Individual Model Application Comparison: Test RMSE")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    add_bar_labels(ax, "{:.2f}")
    fig.tight_layout()
    fig.savefig(out_root / "Individual_Model_Application_Test_RMSE.jpg")
    plt.close(fig)


def _ordered_city_application_rows(application_df: pd.DataFrame) -> pd.DataFrame:
    """Return interleaved default/SCLPSO rows for manuscript tables and figures."""
    if application_df is None or application_df.empty:
        return pd.DataFrame()

    df = application_df.copy()
    model_order = {name: i for i, name in enumerate(CFG.MODEL_NAMES)}
    rows = []
    for model in CFG.MODEL_NAMES:
        default = df[(df["method_group"] == "Default model") & (df["model_components"] == model)]
        optimized = df[(df["method_group"] == "SCLPSO-optimized model") & (df["model_components"] == model)]
        if not default.empty:
            rows.append(default.iloc[0].to_dict())
        if not optimized.empty:
            rows.append(optimized.iloc[0].to_dict())

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["model_order"] = out["model_components"].map(model_order).fillna(99)
    out["row_order"] = out["method_group"].map({"Default model": 0, "SCLPSO-optimized model": 1}).fillna(9)
    out = out.sort_values(["model_order", "row_order"]).drop(columns=["model_order", "row_order"])
    return out.reset_index(drop=True)


def save_city_paper_result_table(application_df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    """Save one city-level manuscript table: default models and their SCLPSO counterparts."""
    ordered = _ordered_city_application_rows(application_df)
    if ordered.empty:
        return ordered

    table = ordered[[
        "method", "test_R2", "test_RMSE", "test_MAE", "test_MAPE_percent",
        "test_R2_improvement_vs_own_default", "test_RMSE_reduction_percent_vs_own_default"
    ]].copy()
    table = table.rename(columns={
        "method": "Method",
        "test_R2": "Test_R2",
        "test_RMSE": "Test_RMSE",
        "test_MAE": "Test_MAE",
        "test_MAPE_percent": "Test_MAPE_percent",
        "test_R2_improvement_vs_own_default": "Delta_R2_vs_default",
        "test_RMSE_reduction_percent_vs_own_default": "RMSE_reduction_percent_vs_default",
    })
    table.to_csv(out_root / f"{CFG.CITY_NAME}_application_table_default_vs_sclpso.csv", index=False, encoding="utf-8-sig")

    latex_rows = []
    for _, r in table.iterrows():
        method = str(r["Method"])
        is_sclpso = method.startswith("SCLPSO-")
        fmt_method = f"\\textbf{{{method}}}" if is_sclpso else method
        fmt_r2 = f"{float(r['Test_R2']):.4f}"
        fmt_rmse = f"{float(r['Test_RMSE']):.3f}"
        fmt_mae = f"{float(r['Test_MAE']):.3f}"
        fmt_mape = f"{float(r['Test_MAPE_percent']):.3f}"
        if is_sclpso:
            fmt_r2 = f"\\textbf{{{fmt_r2}}}"
            fmt_rmse = f"\\textbf{{{fmt_rmse}}}"
            fmt_mae = f"\\textbf{{{fmt_mae}}}"
            fmt_mape = f"\\textbf{{{fmt_mape}}}"
        latex_rows.append(f"{fmt_method} & {fmt_r2} & {fmt_rmse} & {fmt_mae} & {fmt_mape} " + r"\\")

    label_city = re.sub(r"[^A-Za-z0-9]+", "_", CFG.CITY_NAME.lower()).strip("_")
    latex = "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{Prediction performance of default and SCLPSO-optimized models on the {CFG.CITY_NAME} dataset.}}",
        f"\\label{{tab:{label_city}_application_results}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Test $R^2$ & Test RMSE & Test MAE & Test MAPE (\%) \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ])
    (out_root / f"{CFG.CITY_NAME}_application_table_default_vs_sclpso.tex").write_text(latex, encoding="utf-8")
    return table


def _city_method_display_name(method: str) -> str:
    """Short display names used consistently in all city figures."""
    mapping = {
        "SVR": "SVR",
        "RandomForest": "RF",
        "ExtraTrees": "ET",
        "MLP": "MLP",
        "SCLPSO-SVR": "SCLPSO-SVR",
        "SCLPSO-RandomForest": "SCLPSO-RF",
        "SCLPSO-ExtraTrees": "SCLPSO-ET",
        "SCLPSO-MLP": "SCLPSO-MLP",
    }
    return mapping.get(str(method), str(method))


def _city_method_color_map() -> Dict[str, str]:
    """Fixed high-contrast palette shared by Beijing, Shanghai and Hefei.

    Keep this mapping unchanged in the other city scripts so that the same method
    always has the same color across all urban air-quality application figures.
    """
    return {
        "SVR": "#1f77b4",
        "SCLPSO-SVR": "#ff7f0e",
        "RandomForest": "#2ca02c",
        "SCLPSO-RandomForest": "#d62728",
        "ExtraTrees": "#9467bd",
        "SCLPSO-ExtraTrees": "#8c564b",
        "MLP": "#17becf",
        "SCLPSO-MLP": "#e377c2",
    }


def _paired_city_method_order() -> List[str]:
    """Default model and corresponding SCLPSO-optimized model are adjacent."""
    order: List[str] = []
    for model in CFG.MODEL_NAMES:
        order.append(model)
        order.append(f"SCLPSO-{model}")
    return order


def _paired_city_plot_frame(application_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows in the exact plotting order:
    SVR, SCLPSO-SVR, RF, SCLPSO-RF, ET, SCLPSO-ET, MLP, SCLPSO-MLP.
    """
    ordered = _ordered_city_application_rows(application_df)
    if ordered.empty:
        return ordered
    method_order = _paired_city_method_order()
    tmp = ordered.set_index("method", drop=False)
    rows = []
    for method in method_order:
        if method in tmp.index:
            rows.append(tmp.loc[method].to_dict())
    out = pd.DataFrame(rows)
    return out.reset_index(drop=True)


def plot_city_application_four_metrics(application_df: pd.DataFrame, out_root: Path):
    """One city, one manuscript figure: 2x2 subplot layout for R2, RMSE, MAE and MAPE.

    This version is tailored for paper readability:
    1. a 2x2 subplot layout is used so each metric keeps its own y-axis scale;
    2. subplot titles are removed to save space for larger labels and ticks;
    3. typography is enlarged substantially for clear display in the manuscript;
    4. default models and their SCLPSO-optimized counterparts remain pair-adjacent.
    """
    plot_df = _paired_city_plot_frame(application_df)
    if plot_df is None or plot_df.empty:
        return

    method_labels = [_city_method_display_name(m) for m in plot_df["method"].tolist()]
    method_colors = _city_method_color_map()
    colors = [method_colors.get(str(m), "#7f7f7f") for m in plot_df["method"].tolist()]

    pair_gap = 0.28
    group_gap = 1.08
    positions = []
    pos = 0.0
    for i in range(0, len(method_labels), 2):
        positions.append(pos)
        if i + 1 < len(method_labels):
            positions.append(pos + 1.0 + pair_gap)
        pos += 2.0 + pair_gap + group_gap
    x = np.asarray(positions, dtype=float)
    bar_width = 0.82

    metrics = [
        ("test_R2", r"Test $R^2$", "{:.3f}"),
        ("test_RMSE", "Test RMSE", "{:.2f}"),
        ("test_MAE", "Test MAE", "{:.2f}"),
        ("test_MAPE_percent", "Test MAPE (%)", "{:.2f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20.5, 14.8))
    axes = axes.ravel()

    for ax, (col, ylabel, fmt) in zip(axes, metrics):
        vals = plot_df[col].astype(float).values
        bars = ax.bar(
            x, vals, width=bar_width,
            color=colors, edgecolor="black", linewidth=0.75,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=28, ha="right", fontsize=18, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=24, fontweight="bold")
        ax.tick_params(axis="y", labelsize=19)
        ax.grid(axis="y", linestyle="--", linewidth=0.85, alpha=0.40, zorder=0)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        finite_vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if finite_vals.size > 0:
            ymin, ymax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
            if col == "test_R2":
                ax.set_ylim(max(0.0, ymin - 0.10), min(1.03, ymax + 0.10))
            else:
                pad = max(0.12 * ymax, 1.0)
                ax.set_ylim(0, ymax + pad)

        for cut_i in range(1, len(x) // 2):
            cut = (x[2 * cut_i - 1] + x[2 * cut_i]) / 2.0
            ax.axvline(cut, color="0.82", linestyle=":", linewidth=1.15, zorder=1)

        for b in bars:
            height = b.get_height()
            if np.isfinite(height):
                ax.annotate(
                    fmt.format(height),
                    (b.get_x() + b.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=12.5,
                    xytext=(0, 4), textcoords="offset points",
                )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=method_colors[m], edgecolor="black", linewidth=0.75)
        for m in _paired_city_method_order() if m in plot_df["method"].values
    ]
    legend_labels = [_city_method_display_name(m) for m in _paired_city_method_order() if m in plot_df["method"].values]
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=4, frameon=True,
        bbox_to_anchor=(0.5, 0.018), fontsize=19,
        columnspacing=1.6, handlelength=1.6,
    )
    fig.tight_layout(rect=[0.012, 0.105, 1.0, 0.995])


    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_raw_panel.jpg", dpi=CFG.DPI, bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_raw_panel.pdf", bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_paired_raw_panel.jpg", dpi=CFG.DPI, bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_paired_raw_panel.pdf", bbox_inches="tight")



    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_reference_like_raw.jpg", dpi=CFG.DPI, bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_reference_like_raw.pdf", bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_paired_reference_like_raw.jpg", dpi=CFG.DPI, bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_paired_reference_like_raw.pdf", bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_reference_like_not_normalized.jpg", dpi=CFG.DPI, bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_reference_like_not_normalized.pdf", bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_paired_reference_like_not_normalized.jpg", dpi=CFG.DPI, bbox_inches="tight")
    fig.savefig(out_root / f"{CFG.CITY_NAME}_application_four_metrics_paired_reference_like_not_normalized.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. Main experiment
# ============================================================

def evaluate_predictions(y_train, y_val, y_test, pred_train, pred_val, pred_test, prefix: str) -> Dict[str, float]:
    res = {}
    res.update(metrics_dict(y_train, pred_train, prefix + "train_"))
    res.update(metrics_dict(y_val, pred_val, prefix + "validation_"))
    res.update(metrics_dict(y_test, pred_test, prefix + "test_"))
    return res


def main():
    setup_seed(CFG.RANDOM_STATE)
    setup_matplotlib()
    out_root = ensure_dir(CFG.OUT_ROOT)

    with open(out_root / "workflow_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"{CFG.CITY_NAME} PM2.5 individual-model SCLPSO application study")
    print(f"Data:   {CFG.DATA_PATH}")
    print(f"Output: {CFG.OUT_ROOT}")
    print(f"Models: {CFG.MODEL_NAMES}")
    print(f"Optimizer used for individual model tuning: {CFG.OPTIMIZERS}")
    print("=" * 100)

    pack = load_prepare_data(out_root)
    data = make_features(pack["df"], pack["target_col"], out_root)

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    model_names = available_model_names()
    print(f"Feature matrix: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")




    baseline_rows = []
    baseline_records = {}

    print("\n" + "=" * 100)
    print("Training default baseline models")
    print("=" * 100)

    for name in model_names:
        print(f"  Baseline {name}")
        model = default_model(name, CFG.RANDOM_STATE)
        try:
            fitted, p_train, p_val, p_test = fit_predict(model, X_train, y_train, X_val, X_test)
            metrics = evaluate_predictions(y_train, y_val, y_test, p_train, p_val, p_test, "")
            row = {"model": name, "status": "ok", **metrics}
            baseline_rows.append(row)
            baseline_records[name] = {
                "model": fitted,
                "train_pred": p_train,
                "val_pred": p_val,
                "test_pred": p_test,
                "metrics": metrics,
                "params": "default",
            }
        except Exception as e:
            baseline_rows.append({"model": name, "status": "failed", "error": repr(e)})
            print(f"    failed: {e}")

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(out_root / "baseline_default_model_metrics.csv", index=False, encoding="utf-8-sig")

    ok_baseline = baseline_df[baseline_df["status"] == "ok"].copy()
    if ok_baseline.empty:
        raise RuntimeError("No baseline model succeeded.")

    ok_baseline = ok_baseline.sort_values(["validation_R2", "validation_RMSE"], ascending=[False, True])
    best_baseline_name = str(ok_baseline.iloc[0]["model"])
    best_baseline_val_r2 = float(ok_baseline.iloc[0]["validation_R2"])
    print(f"\nBest baseline by validation R2: {best_baseline_name}, val R2={best_baseline_val_r2:.6f}")

    if CFG.OPTIMIZE_TOP_K_MODELS and CFG.OPTIMIZE_TOP_K_MODELS > 0:
        optimize_names = ok_baseline["model"].head(CFG.OPTIMIZE_TOP_K_MODELS).tolist()
    else:
        optimize_names = ok_baseline["model"].tolist()
    print(f"Models selected for SCLPSO optimization: {optimize_names}")





    optimizer_model_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    all_histories: Dict[Tuple[str, str], pd.DataFrame] = {}
    model_opt_rows = []

    print("\n" + "=" * 100)
    print("SCLPSO model hyperparameter optimization")
    print("=" * 100)

    for optimizer in CFG.OPTIMIZERS:
        print("\n" + "-" * 100)
        print(f"Optimizer: {optimizer}")
        print("-" * 100)
        for name in optimize_names:
            baseline_val_r2_for_name = float(baseline_records[name]["metrics"]["validation_R2"])
            try:
                res = sclpso_like_optimize_model(optimizer, name, data, baseline_val_r2_for_name, out_root)
                optimizer_model_results[(optimizer, name)] = res
                all_histories[(optimizer, name)] = res["history"]

                m = evaluate_predictions(
                    y_train, y_val, y_test,
                    res["train_pred"], res["val_pred"], res["test_pred"],
                    "optimized_",
                )
                b = baseline_records[name]["metrics"]
                row = {
                    "optimizer": optimizer,
                    "model": name,
                    "status": "ok",
                    "baseline_validation_R2": b["validation_R2"],
                    "optimized_validation_R2": m["optimized_validation_R2"],
                    "baseline_test_R2": b["test_R2"],
                    "optimized_test_R2": m["optimized_test_R2"],
                    "baseline_test_RMSE": b["test_RMSE"],
                    "optimized_test_RMSE": m["optimized_test_RMSE"],
                    "baseline_test_MAE": b["test_MAE"],
                    "optimized_test_MAE": m["optimized_test_MAE"],
                    "baseline_test_MAPE_percent": b["test_MAPE_percent"],
                    "optimized_test_MAPE_percent": m["optimized_test_MAPE_percent"],
                    "baseline_test_Bias": b["test_Bias"],
                    "optimized_test_Bias": m["optimized_test_Bias"],
                    "test_R2_improvement": m["optimized_test_R2"] - b["test_R2"],
                    "test_RMSE_reduction_percent": improvement_percent(b["test_RMSE"], m["optimized_test_RMSE"], True),
                    "test_MAE_reduction_percent": improvement_percent(b["test_MAE"], m["optimized_test_MAE"], True),
                    "test_MAPE_reduction_percent": improvement_percent(b["test_MAPE_percent"], m["optimized_test_MAPE_percent"], True),
                    "best_params": json.dumps(res["best_params"], ensure_ascii=False, default=str),
                    "fes": res["fes"],
                    "evaluated_unique_params": res["evaluated_unique_params"],
                }
                model_opt_rows.append(row)
            except Exception as e:
                print(f"      failed {optimizer}-{name}: {e}")
                model_opt_rows.append({"optimizer": optimizer, "model": name, "status": "failed", "error": repr(e)})

    model_opt_df = pd.DataFrame(model_opt_rows)
    model_opt_df.to_csv(out_root / "sclpso_modelwise_hyperparameter_results.csv", index=False, encoding="utf-8-sig")
    model_opt_df.to_csv(out_root / "optimizer_modelwise_hyperparameter_results.csv", index=False, encoding="utf-8-sig")
    plot_optimizer_convergence(all_histories, out_root)




    application_df = make_application_summary(baseline_df, model_opt_df, out_root)
    plot_application_results(application_df, out_root)
    save_city_paper_result_table(application_df, out_root)
    plot_city_application_four_metrics(application_df, out_root)

    report_optimizer = CFG.REPORT_OPTIMIZER if CFG.REPORT_OPTIMIZER in CFG.OPTIMIZERS else "SCLPSO"
    ok_opt = model_opt_df[model_opt_df["status"] == "ok"].copy()
    if ok_opt.empty:
        raise RuntimeError("No SCLPSO-optimized model succeeded.")


    ok_opt = ok_opt.sort_values(["optimized_validation_R2", "optimized_test_RMSE"], ascending=[False, True])
    best_opt_row = ok_opt.iloc[0]
    best_optimized_model = str(best_opt_row["model"])
    report_res = optimizer_model_results[(str(best_opt_row["optimizer"]), best_optimized_model)]


    p_test_before = baseline_records[best_baseline_name]["test_pred"]
    p_train_before = baseline_records[best_baseline_name]["train_pred"]
    p_val_before = baseline_records[best_baseline_name]["val_pred"]

    p_test_after = report_res["test_pred"]
    p_train_after = report_res["train_pred"]
    p_val_after = report_res["val_pred"]

    before_metrics = metrics_dict(y_test, p_test_before, "before_test_")
    after_metrics = metrics_dict(y_test, p_test_after, "after_test_")
    summary = {
        "city_name": CFG.CITY_NAME,
        "target_col": pack["target_col"],
        "best_default_model": best_baseline_name,
        "best_sclpso_model": best_optimized_model,
        "report_optimizer": report_optimizer,
        "comparison_type": "individual_model_without_ensemble",
        **before_metrics,
        **after_metrics,
        "test_R2_improvement": after_metrics["after_test_R2"] - before_metrics["before_test_R2"],
        "test_RMSE_reduction_percent": improvement_percent(before_metrics["before_test_RMSE"], after_metrics["after_test_RMSE"], True),
        "test_MAE_reduction_percent": improvement_percent(before_metrics["before_test_MAE"], after_metrics["after_test_MAE"], True),
        "test_MAPE_reduction_percent": improvement_percent(before_metrics["before_test_MAPE_percent"], after_metrics["after_test_MAPE_percent"], True),
    }
    summary.update(metrics_dict(y_train, p_train_before, "before_train_"))
    summary.update(metrics_dict(y_val, p_val_before, "before_validation_"))
    summary.update(metrics_dict(y_train, p_train_after, "after_train_"))
    summary.update(metrics_dict(y_val, p_val_after, "after_validation_"))

    pd.DataFrame([summary]).to_csv(out_root / f"final_individual_model_summary_{report_optimizer}.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "date": data["date_test"],
        "observed_PM25": y_test,
        f"{best_baseline_name}_pred": p_test_before,
        f"{report_optimizer}_{best_optimized_model}_pred": p_test_after,
        f"{best_baseline_name}_residual": p_test_before - y_test,
        f"{report_optimizer}_{best_optimized_model}_residual": p_test_after - y_test,
    })
    pred_df.to_csv(out_root / f"test_predictions_{best_baseline_name}_vs_{report_optimizer}_{best_optimized_model}.csv", index=False, encoding="utf-8-sig")


    report_model_df = model_opt_df[model_opt_df["status"] == "ok"].copy()
    report_model_df.to_csv(out_root / f"modelwise_baseline_vs_{report_optimizer}_individual.csv", index=False, encoding="utf-8-sig")
    if not report_model_df.empty:
        plot_model_comparison(report_model_df, out_root, "R2", report_optimizer)
        plot_model_comparison(report_model_df, out_root, "RMSE", report_optimizer)


    after_label = f"{report_optimizer}-{best_optimized_model}"
    plot_before_after_metric(summary, out_root, "R2", lower_is_better=False, after_label=after_label)
    plot_before_after_metric(summary, out_root, "RMSE", lower_is_better=True, after_label=after_label)
    plot_before_after_metric(summary, out_root, "MAE", lower_is_better=True, after_label=after_label)
    plot_before_after_metric(summary, out_root, "MAPE_percent", lower_is_better=True, after_label=after_label)
    plot_prediction_scatter(y_test, p_test_before, p_test_after, out_root, after_label)
    plot_time_series(data["date_test"], y_test, p_test_before, p_test_after, out_root, after_label)
    plot_residual_boxplot(y_test, p_test_before, p_test_after, out_root, after_label)


    plot_feature_importance(baseline_records[best_baseline_name]["model"], data["feature_names"], out_root, f"{best_baseline_name}")
    plot_feature_importance(report_res["model"], data["feature_names"], out_root, f"{report_optimizer}_{best_optimized_model}")

    print("\n" + "=" * 100)
    print("FINAL INDIVIDUAL-MODEL SUMMARY")
    print("=" * 100)
    print(f"Best default model by validation R2: {best_baseline_name}")
    print(f"Best SCLPSO-optimized model by validation R2: {best_optimized_model}")
    print(f"Before test R2:   {summary['before_test_R2']:.6f}")
    print(f"After  test R2:   {summary['after_test_R2']:.6f}")
    print(f"R2 improvement:   {summary['test_R2_improvement']:+.6f}")
    print(f"Before test RMSE: {summary['before_test_RMSE']:.6f}")
    print(f"After  test RMSE: {summary['after_test_RMSE']:.6f}")
    print(f"RMSE reduction:   {summary['test_RMSE_reduction_percent']:+.2f}%")
    print(f"Output saved to:  {out_root}")
    print("=" * 100)


    return summary


if __name__ == "__main__":
    main()
