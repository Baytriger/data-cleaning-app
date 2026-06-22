import io
import re
import uuid
import warnings

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

app = Flask(__name__)
OUTPUT_STORE: dict = {}

DEFAULT_CONTAMINATION = 0.05


def _estimate_contamination(df: pd.DataFrame, feature_cols: list) -> float:
    """
    Estimate a data-driven contamination rate instead of blindly using a
    fixed 5%. A fixed rate forces Isolation Forest to flag exactly N% of
    rows regardless of whether any of them are actually anomalous -- on a
    clean, well-distributed dataset like a salary range of 50k-120k, this
    means perfectly normal values get flagged just because they sit at the
    edges of a uniform distribution.

    Strategy: for each numeric feature column, count values that fall
    outside 3 standard deviations from the mean (a standard statistical
    definition of an outlier). The estimated contamination is the fraction
    of rows that are extreme in at least one feature, clamped between 1%
    and 15% so the model stays stable.

    If the data has no genuine outliers by this measure, the estimate
    returns a very low rate (1%) so Isolation Forest flags almost nothing
    rather than inventing anomalies to fill a fixed quota.
    """
    if not feature_cols:
        return DEFAULT_CONTAMINATION

    X = df[feature_cols].copy()
    outlier_mask = pd.Series(False, index=df.index)

    for col in feature_cols:
        col_data = X[col].dropna()
        if len(col_data) < 10:
            continue
        mean = col_data.mean()
        std  = col_data.std()
        if std == 0:
            continue
        is_outlier = (X[col] - mean).abs() > 3 * std
        outlier_mask |= is_outlier.fillna(False)

    estimated = float(outlier_mask.sum()) / max(len(df), 1)
    # Cap at DEFAULT_CONTAMINATION (5%): a dataset with many 3-sigma outliers
    # (e.g. laptops ranging from budget to gaming) should not push the rate
    # above what a domain expert would consider a reasonable anomaly ceiling.
    # Floor at 1% so the model always has something to evaluate.
    clamped = max(0.01, min(DEFAULT_CONTAMINATION, estimated))
    return clamped
TEST_SIZE             = 0.20
MIN_TRAIN_ROWS        = 10


def _is_text_col(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

def _is_numeric_col(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def _text_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if _is_text_col(df[c])]

def _real_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if not c.endswith("_original")]

def _missing_count(df: pd.DataFrame) -> int:
    cols = _real_columns(df)
    if not cols:
        return 0
    return int(df[cols].isnull().sum().sum())

def _try_numeric(series: pd.Series) -> tuple:
    cleaned = (
        series.astype(str)
              .str.replace(r"[₹$€£,%\s]", "", regex=True)
              .str.replace(r",", "", regex=False)
              .str.replace(r"\.$", "", regex=True)
              .replace("nan", np.nan)
              .replace("None", np.nan)
              .replace("", np.nan)
    )
    coerced = pd.to_numeric(cleaned, errors="coerce")
    original_non_null = series.notna().sum()
    if original_non_null == 0:
        return series, False
    success_rate = coerced.notna().sum() / original_non_null
    if success_rate >= 0.5:
        return coerced, True
    return series, False

def _extract_leading_number(series: pd.Series) -> tuple:
    def _get(val):
        if pd.isnull(val):
            return np.nan
        m = re.search(r"\d+\.?\d*|\.\d+", str(val))
        if not m:
            return np.nan
        try:
            return float(m.group())
        except ValueError:
            return np.nan
    extracted = series.apply(_get)
    original_non_null = series.notna().sum()
    if original_non_null == 0:
        return series, False
    if extracted.notna().sum() / original_non_null >= 0.5:
        return extracted, True
    return series, False

_MEASUREMENT_PATTERN = re.compile(
    r"^\s*\d+\.?\d*\s*(gb|tb|mb|inch|inches|in|cm|mm|kg|g|lb|hz|ghz|mhz|w|"
    r"hrs?|hours?|mins?|minutes?|x|stars?|%)\b",
    re.IGNORECASE,
)
_PARENTHESIZED_COUNT_PATTERN = re.compile(r"^\s*\(\s*[\d,]+\.?\d*\s*\)\s*$")
_PLUS_COUNT_PATTERN = re.compile(r"^\s*\d+\.?\d*\s*\+")

def _looks_like_measurements(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    matches = (
        non_null.str.match(_MEASUREMENT_PATTERN)
        | non_null.str.match(_PARENTHESIZED_COUNT_PATTERN)
        | non_null.str.match(_PLUS_COUNT_PATTERN)
    )
    return matches.sum() / len(non_null) >= 0.5

def _looks_like_dates(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.astype(str)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().sum() / len(sample) >= 0.8

def _detect_datetime_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if _is_text_col(df[c]) and _looks_like_dates(df[c])]


def impute_datetime(df: pd.DataFrame, log) -> tuple:
    log("Step 3b: Datetime imputation")
    df = df.copy()
    imputed_flag = pd.Series(False, index=df.index)
    date_cols = _detect_datetime_columns(df)

    if not date_cols:
        log("  No datetime columns detected")
        return df, imputed_flag

    for col in date_cols:
        if not df[col].isnull().any():
            continue

        missing_before = int(df[col].isnull().sum())
        was_null = df[col].isnull().copy()

        # Parse to datetime so we can do time-aware filling
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")

        # Strategy 1: forward-fill (last known timestamp carries forward)
        filled = parsed.ffill()

        # Strategy 2: backward-fill for any still-null at the start
        filled = filled.bfill()

        # Strategy 3: median timestamp for any remaining gaps
        # (convert to int64 epoch, take median, convert back)
        still_null = filled.isnull()
        if still_null.any():
            valid_epochs = parsed.dropna().astype(np.int64)
            if len(valid_epochs) > 0:
                median_epoch = int(valid_epochs.median())
                median_ts = pd.Timestamp(median_epoch)
                filled[still_null] = median_ts

        # Write back as ISO strings so the column stays as text (consistent
        # with how the rest of the pipeline treats non-numeric columns)
        newly_filled = was_null & filled.notna()
        df.loc[newly_filled, col] = filled[newly_filled].dt.strftime("%Y-%m-%d %H:%M:%S")
        imputed_flag |= newly_filled

        total_filled = missing_before - int(df[col].isnull().sum())
        log(f"  {col}: {total_filled} missing timestamps filled "
            f"(ffill/bfill/median) | {int(df[col].isnull().sum())} remain")

    log(f"  Total datetime cells imputed: {int(imputed_flag.sum())}")
    return df, imputed_flag


def _is_likely_free_text(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    as_str = non_null.astype(str)
    avg_len = as_str.str.len().mean()
    avg_words = as_str.str.split().apply(len).mean()
    uniqueness = non_null.nunique() / len(non_null)
    looks_like_text_by_shape = avg_len > 30 and avg_words >= 4
    looks_like_text_by_uniqueness = uniqueness > 0.9 and avg_len > 12
    return looks_like_text_by_shape or looks_like_text_by_uniqueness

_MOJIBAKE_HINT = re.compile(r"[ÂÃâãÄäÅåÆæÇçÈ-Ëè-ë][\x80-\xbf\x00-\x1f]")

def _repair_mojibake(text: str) -> str:
    if not isinstance(text, str) or not _MOJIBAKE_HINT.search(text):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    import unicodedata
    cleaned = "".join(c for c in repaired if unicodedata.category(c) != "Cf")
    return cleaned

def _repair_mojibake_in_column(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: _repair_mojibake(v) if isinstance(v, str) else v)

_INVISIBLE_MARK_PATTERN = re.compile(
    r"\\u200[0-9a-fA-F]|[\u200b-\u200f\ufeff]|â€[Žâ€™\u008e\u0099]",
)

def _clean_invisible_marks(text) -> str:
    if not isinstance(text, str):
        return text
    return _INVISIBLE_MARK_PATTERN.sub("", text)

def _clean_text_columns(df: pd.DataFrame, log) -> pd.DataFrame:
    df = df.copy()
    cleaned_any = 0
    for col in _text_columns(df):
        had_marks = df[col].astype(str).str.contains(_INVISIBLE_MARK_PATTERN, na=False, regex=True)
        n = int(had_marks.sum())
        if n > 0:
            df[col] = df[col].apply(_clean_invisible_marks)
            cleaned_any += n
            log(f"  {col}: removed invisible/mojibake formatting marks from {n} value(s)")
    if cleaned_any:
        log(f"  Total values cleaned of invisible formatting marks: {cleaned_any}")
    return df

def preprocess_columns(df: pd.DataFrame, log) -> pd.DataFrame:
    log("Step 1: Column type standardisation")
    df = df.copy()
    converted = 0
    for col in _text_columns(df):
        if _looks_like_dates(df[col]):
            continue
        coerced, ok = _try_numeric(df[col])
        if ok:
            df[f"{col}_original"] = df[col].copy()
            df[col] = coerced
            converted += 1
            log(f"  {col}: converted to numeric ({int(coerced.notna().sum())} values parsed)")
            continue
        if _is_likely_free_text(df[col]):
            continue
        if not _looks_like_measurements(df[col]):
            continue
        extracted, ok = _extract_leading_number(df[col])
        if ok:
            df[f"{col}_original"] = df[col].copy()
            df[col] = extracted
            converted += 1
            log(f"  {col}: extracted numeric prefix ({int(extracted.notna().sum())} values parsed)")
    log(f"  Columns converted to numeric: {converted}")
    return df

def _normalise_string(val) -> str:
    if not isinstance(val, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", val.lower())).strip()

def _is_identifier_column(df: pd.DataFrame, col: str) -> bool:
    """
    True for numeric columns that are identifiers rather than measurements
    -- phone numbers, zip codes, IDs, etc. These should never be fed to
    Isolation Forest because their numeric values carry no mathematical
    meaning; flagging a row because its phone number is "unusually high"
    is nonsense.

    Signals:
    - Column name contains identifier keywords (phone, zip, postal, code,
      id, ssn, ein, fax, mobile, tel)
    - Values are all negative (e.g. phone numbers stored as overflowed ints)
    - Values are large integers with high cardinality (nearly unique per row)
      but no plausible measurement meaning (e.g. a 10-digit phone number)
    """
    name = col.strip().lower()
    id_keywords = {"phone", "zip", "postal", "fax", "mobile", "tel",
                   "ssn", "ein", "account", "creditcard", "debitcard", "pin"}
    # Match whole words only so 'card' doesn't fire on 'graphics_card',
    # 'tel' doesn't fire on 'hotel', etc.
    name_words = set(re.split(r"[_\s\-]+", name))
    if name_words & id_keywords:
        return True
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    non_null = df[col].dropna()
    if len(non_null) == 0:
        return False
    # All negative AND large magnitude: overflowed phone/ID.
    # Small negatives (e.g. discount percentages like -28%, temperatures,
    # coordinates) are real measurements and must not be excluded.
    if (non_null < 0).all() and non_null.abs().median() > 10_000:
        return True
    # Large integers (>6 digits), high cardinality -- phone/ID pattern
    if non_null.abs().median() > 1_000_000 and non_null.nunique() / len(non_null) > 0.8:
        return True
    return False


def _is_low_variance_categorical(series: pd.Series, max_unique: int = 10) -> bool:
    """
    True for numeric columns that are really categorical codes rather than
    continuous measurements -- e.g. Age stored as {25, 30, 35, 40}.
    Flagging "40 is higher than typical 30" in such a column is misleading
    since every value is a legitimate category, not a deviation.
    """
    non_null = series.dropna()
    return 0 < non_null.nunique() <= max_unique


def _looks_like_row_index(df: pd.DataFrame, col: str) -> bool:
    name = col.strip().lower()
    if name in {"unnamed: 0", "id", "index", "row_id", "rowid", "row_index", "_id"}:
        return True
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    non_null = df[col].dropna()
    if len(non_null) < 2:
        return False
    is_int_valued = (non_null == non_null.round()).all()
    is_unique = non_null.nunique() == len(non_null)
    is_monotonic = non_null.is_monotonic_increasing
    return is_int_valued and is_unique and is_monotonic

def _value_columns_disagree(row_a: pd.Series, row_b: pd.Series, value_cols: list, rel_tol: float = 0.02) -> bool:
    for col in value_cols:
        if col not in row_a.index or col not in row_b.index:
            continue
        a, b = row_a[col], row_b[col]
        a_null, b_null = pd.isnull(a), pd.isnull(b)
        if a_null and b_null:
            continue
        if a_null != b_null:
            return True
        try:
            a_f, b_f = float(a), float(b)
        except (TypeError, ValueError):
            if str(a) != str(b):
                return True
            continue
        denom = max(abs(a_f), abs(b_f), 1e-9)
        if abs(a_f - b_f) / denom > rel_tol:
            return True
    return False

def deduplicate(df: pd.DataFrame, log, value_cols: list = None) -> tuple:
    log("Step 2: Deduplication")
    n_before = len(df)

    index_like_cols = [c for c in df.columns if _looks_like_row_index(df, c)]
    compare_cols = [c for c in df.columns if c not in index_like_cols]
    if index_like_cols:
        log(f"  Ignoring index-like column(s) for exact-duplicate comparison: "
            f"{', '.join(index_like_cols)}")

    df = df.drop_duplicates(subset=compare_cols if compare_cols else None).reset_index(drop=True)
    exact_removed = n_before - len(df)
    log(f"  Exact duplicates removed: {exact_removed}")

    if value_cols is None:
        value_cols = [c for c in df.columns if c.lower() in ("price", "mrp")]

    str_cols = [c for c in _text_columns(df) if not c.endswith("_original")]
    key_col = None
    for c in str_cols:
        if df[c].nunique() / max(len(df), 1) > 0.5:
            key_col = c
            break

    if key_col is None:
        log("  Near-duplicate check skipped (no suitable string key column found)")
        df["price_variant_flag"] = 0
        df["near_dup_flag"] = 0
        return df, exact_removed, 0, df.iloc[0:0].copy()

    df["_norm_key"] = df[key_col].apply(_normalise_string)
    near_removed = 0
    flagged_rows = []
    rows_with_pos = []

    for norm_val, group in df.groupby("_norm_key"):
        if norm_val == "" or len(group) == 1:
            for pos, row in group.iterrows():
                rows_with_pos.append((pos, pd.DataFrame([row]).drop(columns=["_norm_key"])))
            continue

        if value_cols:
            disagreement = False
            base_row = group.iloc[0]
            for _, other_row in group.iloc[1:].iterrows():
                if _value_columns_disagree(base_row, other_row, value_cols):
                    disagreement = True
                    break
            if disagreement:
                chunk = group.drop(columns=["_norm_key"])
                flagged_rows.append(chunk)
                for pos, row in group.iterrows():
                    rows_with_pos.append((pos, pd.DataFrame([row]).drop(columns=["_norm_key"])))
                log(f"  Held back from merge (differs on {', '.join(value_cols)}): "
                    f"{len(group)} rows with key '{norm_val[:60]}'", "warn")
                continue

        # Same title, same price -- genuine near-duplicate, flag and keep all rows
        for pos, row in group.iterrows():
            rows_with_pos.append((pos, pd.DataFrame([row]).drop(columns=["_norm_key"])))
        near_removed += len(group) - 1
        log(f"  Near-duplicate flagged (same title+price): {len(group)} rows, key '{norm_val[:60]}'")

    rows_with_pos.sort(key=lambda pair: pair[0])
    merged_rows = [row for _, row in rows_with_pos]

    df = pd.concat(merged_rows, ignore_index=True)
    flagged_df = (
        pd.concat(flagged_rows, ignore_index=True) if flagged_rows
        else df.iloc[0:0].copy()
    )

    if not flagged_df.empty:
        df["price_variant_flag"] = df[key_col].isin(flagged_df[key_col]).astype(int)
    else:
        df["price_variant_flag"] = 0

    # near_dup_flag: rows that are genuine near-dups (same title AND same price)
    # These are kept in the output but flagged for review
    near_dup_key_groups = set()
    for norm_val, group in df.assign(_nk=df[key_col].apply(_normalise_string)).groupby("_nk"):
        if norm_val == "" or len(group) == 1:
            continue
        # Only flag groups that are NOT price variants
        if not df.loc[group.index, "price_variant_flag"].any():
            near_dup_key_groups.add(norm_val)

    temp_norm = df[key_col].apply(_normalise_string)
    df["near_dup_flag"] = temp_norm.isin(near_dup_key_groups).astype(int)

    log(f"  Near-duplicate rows flagged (kept, not merged): {int(df['near_dup_flag'].sum())}")
    if not flagged_df.empty:
        log(f"  Rows held back as price/value variants (not merged): {len(flagged_df)}", "warn")
    log(f"  Rows after deduplication: {len(df)}")
    return df, exact_removed, near_removed, flagged_df


def _group_median(df: pd.DataFrame, col: str, group_col: str) -> pd.Series:
    if group_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    global_med = df[col].median()
    result = df[col].copy()
    mask = result.isnull()
    if not mask.any():
        return result
    group_med = df.groupby(group_col)[col].median()
    for idx in df[mask].index:
        g = df.at[idx, group_col]
        if df[group_col].eq(g).sum() >= 3 and g in group_med and not pd.isnull(group_med[g]):
            result.at[idx] = group_med[g]
        else:
            result.at[idx] = global_med
    return result

def _find_group_col(df: pd.DataFrame) -> str | None:
    candidates = [
        c for c in _text_columns(df)
        if not c.endswith("_original") and 2 <= df[c].nunique() <= 30
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: df[c].nunique())

def impute_categorical(df: pd.DataFrame, log) -> tuple:
    log("Step 3a: Categorical imputation")
    df = df.copy()
    imputed_flag = pd.Series(False, index=df.index)
    backup_cols = [c for c in df.columns if c.endswith("_original")]
    date_cols   = set(_detect_datetime_columns(df))
    cat_targets = [c for c in _text_columns(df) if c not in backup_cols and c not in date_cols]
    group_col = _find_group_col(df)

    for col in cat_targets:
        if not df[col].isnull().any():
            continue
        missing_before = int(df[col].isnull().sum())
        was_null = df[col].isnull().copy()

        if group_col and group_col != col:
            def _group_mode(grp):
                m = grp[col].mode()
                return m.iloc[0] if not m.empty else np.nan
            group_modes = df.groupby(group_col).apply(_group_mode)
            for idx in df[was_null].index:
                g = df.at[idx, group_col]
                if g in group_modes and not pd.isnull(group_modes[g]):
                    df.at[idx, col] = group_modes[g]

        mode_val = df[col].mode()
        mode_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        df.loc[df[col].isnull(), col] = mode_val

        filled = missing_before - int(df[col].isnull().sum())
        imputed_flag |= was_null & df[col].notna()
        log(f"  {col}: {filled} missing values filled")

    log(f"  Total categorical cells imputed: {int(imputed_flag.sum())}")
    return df, imputed_flag

def impute_numeric(df: pd.DataFrame, X_train: pd.DataFrame, log, group_col: str | None = None) -> tuple:
    log("Step 3c: Numeric imputation")
    df = df.copy()
    imputed_flag = pd.Series(False, index=df.index)

    meta_cols = (
        [c for c in df.columns if c.endswith("_original")]
        + ["price_variant_flag", "near_dup_flag"]
    )
    numeric_targets = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in meta_cols
    ]

    for col in numeric_targets:
        if not df[col].isnull().any():
            continue
        missing_before = int(df[col].isnull().sum())
        was_null = df[col].isnull().copy()
        rf_filled = pd.Series(False, index=df.index)

        other_cols = [c for c in numeric_targets if c != col]
        rf_used = False

        if other_cols:
            train_rows = X_train[X_train[col].notna()]
            train_meds = X_train[other_cols].median()

            usable_features = [
                c for c in other_cols
                if train_rows[c].notna().sum() >= MIN_TRAIN_ROWS
            ]

            corr_strength = 0.0
            if usable_features and len(train_rows) >= MIN_TRAIN_ROWS:
                corrs = train_rows[usable_features + [col]].corr()[col].drop(col).abs()
                corr_strength = float(corrs.max()) if not corrs.empty else 0.0

            if len(train_rows) >= MIN_TRAIN_ROWS and corr_strength >= 0.4:
                Xtr = train_rows[usable_features].fillna(train_meds[usable_features])
                ytr = train_rows[col]
                missing_mask = df[col].isnull()
                has_features = ~df.loc[missing_mask, usable_features].isnull().all(axis=1)
                can_impute = missing_mask & missing_mask.index.isin(
                    df[missing_mask][has_features].index
                )
                if can_impute.sum() > 0:
                    X_pred = df.loc[can_impute, usable_features].fillna(train_meds[usable_features])
                    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
                    rf.fit(Xtr, ytr)
                    df.loc[can_impute, col] = rf.predict(X_pred)
                    rf_filled |= can_impute
                    rf_used = True
                    log(f"  {col}: {int(can_impute.sum())} filled by Random Forest "
                        f"(strongest correlation: {corr_strength:.2f})")

        if rf_used:
            imputed_flag |= rf_filled

        if df[col].isnull().any():
            if group_col:
                df[col] = df[col].fillna(_group_median(df, col, group_col))
            global_med = df[col].median()
            if not pd.isnull(global_med):
                df[col] = df[col].fillna(global_med)

        filled_total = missing_before - int(df[col].isnull().sum())
        median_filled = filled_total - int(rf_filled.sum())
        if median_filled > 0:
            imputed_flag |= was_null & df[col].notna() & ~rf_filled
            log(f"  {col}: {median_filled} filled by median | {int(df[col].isnull().sum())} remain")
        elif filled_total == 0 and missing_before > 0:
            log(f"  {col}: {missing_before} could not be imputed (no median available)", "warn")

    log(f"  Total numeric cells imputed: {int(imputed_flag.sum())}")
    log(f"  Missing values remaining: {_missing_count(df)}")
    return df, imputed_flag

def _explain_anomalies(df, anomaly_index, feature_cols, train_stats, top_n=3):
    medians = train_stats.loc["median"]
    mads = train_stats.loc["mad"].replace(0, np.nan)
    # Columns with very few distinct values are categorical-as-numeric (e.g. Age
    # stored as {25,30,35,40}). Citing "40 is higher than typical 30" for such a
    # column is misleading -- every value is a legitimate category, not a deviation.
    # Exclude them from the human-readable explanation (still used by Isolation Forest).
    low_var_cols = {f for f in feature_cols if _is_low_variance_categorical(df[f])}
    explain_cols = [f for f in feature_cols if f not in low_var_cols]
    explanations = {}
    for idx in anomaly_index:
        if idx not in df.index:
            continue
        row = df.loc[idx, explain_cols] if explain_cols else pd.Series(dtype=float)
        z_scores = ((row - medians[explain_cols]) / mads[explain_cols] * 0.6745).abs()
        z_scores = z_scores.dropna().sort_values(ascending=False)
        if z_scores.empty or z_scores.iloc[0] < 1:
            explanations[idx] = (
                "Isolation Forest flagged this row based on its overall combination of "
                "feature values -- no single feature stands out strongly on its own."
            )
            continue
        top_feats = z_scores.head(top_n)
        parts = []
        for feat, z in top_feats.items():
            # Require z > 2 (roughly 2 standard deviations via MAD) before
            # citing a feature in the explanation. A z of 1-2 means the value
            # is modestly different from the median -- well within normal
            # organizational variance (e.g. salary spread across roles).
            # Without this threshold, the explanation cites the feature with
            # the highest z even when that z is trivially small, producing
            # misleading messages like "57k is lower than typical 85k" when
            # 57k is a completely normal salary in the dataset.
            if z < 2:
                continue
            actual = row[feat]
            med = medians[feat]
            direction = "higher" if actual > med else "lower"
            parts.append(f"{feat} ({actual:g} is {direction} than typical {med:g})")
        if parts:
            explanations[idx] = "Flagged due to unusual values in: " + "; ".join(parts) + "."
        else:
            explanations[idx] = (
                "Isolation Forest flagged this row based on its overall combination of "
                "feature values -- no single feature stands out strongly on its own."
            )
    return explanations

def rf_supervised_evaluation(X_train, X_test, y_train, y_test, log) -> dict:
    log("Step 5: Random Forest supervised evaluation")
    train_meds = X_train.median(numeric_only=True)
    Xtr = X_train.select_dtypes(include=[np.number]).fillna(train_meds)
    Xte = X_test.select_dtypes(include=[np.number]).fillna(train_meds)
    is_classification = (
        _is_text_col(y_train)
        or y_train.nunique() <= 20
        or str(y_train.dtype).startswith("int")
    )
    metrics: dict = {}
    if is_classification:
        le = LabelEncoder()
        le.fit(pd.concat([y_train, y_test]).astype(str))
        ytr = le.transform(y_train.astype(str))
        yte = le.transform(y_test.astype(str))
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        y_pred = rf.predict(Xte)
        avg = "binary" if len(le.classes_) == 2 else "weighted"
        metrics = {
            "task":      "classification",
            "accuracy":  round(float(accuracy_score(yte, y_pred)), 4),
            "precision": round(float(precision_score(yte, y_pred, average=avg, zero_division=0)), 4),
            "recall":    round(float(recall_score(yte, y_pred, average=avg, zero_division=0)), 4),
            "f1":        round(float(f1_score(yte, y_pred, average=avg, zero_division=0)), 4),
        }
        log(f"  Task: Classification ({len(le.classes_)} classes) | Accuracy: {metrics['accuracy']:.4f}")
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(Xtr, y_train)
        y_pred = rf.predict(Xte)
        metrics = {
            "task": "regression",
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "r2":   round(float(r2_score(y_test, y_pred)), 4),
        }
        log(f"  Task: Regression | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f}")
    return metrics

def _signed_delta(before: int, after: int) -> str:
    diff = before - after
    return f"(-{diff})" if diff >= 0 else f"(+{-diff})"

def build_validation_report(df_raw, df_after_pre, df_final, n_exact, n_near, log) -> dict:
    total_raw = _missing_count(df_raw)
    total_pre = _missing_count(df_after_pre)
    total_fin = _missing_count(df_final)
    log("Step 6: Validation report")
    log(f"  Missing values (raw input)        : {total_raw}")
    log(f"  Missing values (after pre-process): {total_pre}  {_signed_delta(total_raw, total_pre)}")
    if total_pre > total_raw:
        log(f"    note: pre-processing converted text to numbers; "
            f"{total_pre - total_raw} cell(s) held placeholder text now correctly counted as missing", "warn")
    log(f"  Missing values (after imputation) : {total_fin}  {_signed_delta(total_pre, total_fin)}")
    log(f"  Exact duplicates removed          : {n_exact}")
    log(f"  Near-duplicates flagged (kept)    : {n_near}")
    log(f"  Final shape                       : {df_final.shape[0]} rows x {df_final.shape[1]} cols")
    return {
        "missing_raw":       total_raw,
        "missing_after_pre": total_pre,
        "missing_after_imp": total_fin,
        "exact_dups":        n_exact,
        "near_dups":         n_near,
        "final_shape":       list(df_final.shape),
    }

def _safe_test_size(n_rows: int, target: float = TEST_SIZE) -> float:
    if n_rows < 2:
        return target
    min_frac = 1.0 / n_rows
    max_frac = 1.0 - min_frac
    return min(max(target, min_frac), max_frac)


def run_pipeline(
    df_raw: pd.DataFrame,
    label_col,
    contamination: float = DEFAULT_CONTAMINATION,
    remove_anomalies: bool = False,
) -> tuple:

    logs: list = []
    def log(msg: str, level: str = "info"):
        logs.append({"msg": msg, "level": level})

    log(f"Pipeline started: {len(df_raw)} rows, {df_raw.shape[1]} columns")

    log("Step 0: Text cleanup (invisible/mojibake formatting marks)")
    df_raw = _clean_text_columns(df_raw, log)
    df_snapshot_raw = df_raw.copy()

    df = preprocess_columns(df_raw.copy(), log)
    df_snapshot_pre = df.copy()

    log(f"Missing values at input: {_missing_count(df)}")

    y = None
    if label_col and label_col in df.columns:
        log(f"Label column '{label_col}' set aside for evaluation")

    df, n_exact, n_near, price_variant_df = deduplicate(df, log)
    price_variant_cols = [c for c in df.columns if c.lower() in ("price", "mrp")]

    if len(df) < 2:
        raise ValueError(
            f"Only {len(df)} row(s) remain after removing duplicates; "
            "at least 2 distinct rows are required to run the pipeline."
        )

    if label_col and label_col in df.columns:
        y = df[label_col].copy()
        df = df.drop(columns=[label_col])
        n_missing_label = int(y.isnull().sum())
        if n_missing_label > 0:
            keep = y.notna()
            log(f"  Dropping {n_missing_label} row(s) with missing label value", "warn")
            df = df[keep].reset_index(drop=True)
            y = y[keep].reset_index(drop=True)
        if len(y) < 2 or y.nunique() < 2:
            log("  Label column has fewer than 2 distinct non-null values; skipping supervised evaluation", "warn")
            y = None
            label_col = None

    df, imputed_flag_cat = impute_categorical(df, log)

    df, imputed_flag_dt = impute_datetime(df, log)

    numeric_group_col = _find_group_col(df)
    if numeric_group_col:
        log(f"  Using '{numeric_group_col}' as grouping key for numeric imputation")

    backup_cols = [c for c in df.columns if c.endswith("_original")]
    cat_cols = [c for c in _text_columns(df) if c not in backup_cols]
    original_cat = {col: df[col].copy() for col in cat_cols}
    encoders: dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].dropna().astype(str))
        encoders[col] = le
        df[col] = df[col].apply(
            lambda x, _le=le: _le.transform([str(x)])[0] if pd.notnull(x) else np.nan
        )

    split_size = _safe_test_size(len(df))
    log(f"Train/test split: {int((1 - split_size) * 100)}% / {int(split_size * 100)}%")

    can_stratify = False
    if y is not None and 2 <= y.nunique() <= 20:
        class_counts = y.value_counts()
        can_stratify = class_counts.min() >= 2
        if not can_stratify:
            log("  Stratified split skipped: at least one class has fewer than 2 examples", "warn")

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=split_size, random_state=42,
            stratify=y if can_stratify else None
        )
    else:
        y_train = y_test = None
        X_train, X_test = train_test_split(df, test_size=split_size, random_state=42)

    log(f"  Training rows: {len(X_train)}, Test rows: {len(X_test)}")

    df, imputed_flag_num = impute_numeric(df, X_train, log, group_col=numeric_group_col)
    imputed_flag = (
        imputed_flag_cat.reindex(df.index, fill_value=False)
        | imputed_flag_dt.reindex(df.index, fill_value=False)
        | imputed_flag_num
    )
    df["imputed_flag"] = imputed_flag.astype(int)
    log(f"  Rows with at least one imputed value: {int(imputed_flag.sum())}")

    meta_flag_cols = {"price_variant_flag", "near_dup_flag", "imputed_flag"}
    id_cols = {c for c in df.columns if _is_identifier_column(df, c)}
    if id_cols:
        log(f"  Excluding identifier column(s) from anomaly detection: {', '.join(sorted(id_cols))}")
    numeric_for_iso = [
        c for c in X_train.select_dtypes(include=[np.number]).columns
        if c not in meta_flag_cols and c not in id_cols
    ]
    train_meds = X_train[numeric_for_iso].median()
    X_train_filled = X_train[numeric_for_iso].fillna(train_meds)
    X_all_filled   = df[numeric_for_iso].fillna(train_meds)

    # If the user did not override contamination, estimate it from the data
    # so Isolation Forest flags based on actual statistical extremity rather
    # than a fixed quota that forces anomalies even in clean datasets.
    if contamination == DEFAULT_CONTAMINATION:
        contamination = _estimate_contamination(X_train_filled, numeric_for_iso)
        log(f"Step 4: Anomaly detection (contamination auto-estimated: {contamination:.1%})")
    else:
        log(f"Step 4: Anomaly detection (contamination={contamination:.1%}, user-supplied)")

    iso = IsolationForest(
        n_estimators=100, contamination=contamination,
        random_state=42, n_jobs=-1
    )
    iso.fit(X_train_filled)

    all_preds  = iso.predict(X_all_filled)
    all_scores = iso.decision_function(X_all_filled)
    n_anomalies = int((all_preds == -1).sum())

    log(f"  Anomalies flagged: {n_anomalies}", "warn")
    log(f"  Normal records   : {len(df) - n_anomalies}")

    # Per-row anomaly explanations using robust z-scores (median/MAD)
    # Restrict to genuinely numeric columns -- label-encoded cat codes are
    # arbitrary integers and citing them in explanations would be misleading
    explain_feature_cols = [c for c in numeric_for_iso if c not in cat_cols]
    anomaly_row_index = df.index[all_preds == -1]
    train_stats = pd.DataFrame({
        "median": X_train_filled[explain_feature_cols].median(),
        "mad":    (X_train_filled[explain_feature_cols] - X_train_filled[explain_feature_cols].median()).abs().median(),
    }).T
    anomaly_explanations = _explain_anomalies(
        X_all_filled, anomaly_row_index, explain_feature_cols, train_stats
    )

    iso_metrics = {}
    rf_metrics  = {}

    if y is not None:
        log("Evaluating Isolation Forest against provided labels")
        X_test_filled = X_test[numeric_for_iso].fillna(train_meds)
        test_preds    = iso.predict(X_test_filled)
        y_pred_bin    = (test_preds == -1).astype(int)
        y_true_bin    = y_test.iloc[:len(y_pred_bin)].reset_index(drop=True)
        if set(y_true_bin.unique()).issubset({0, 1}):
            iso_metrics = {
                "precision": round(float(precision_score(y_true_bin, y_pred_bin, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_true_bin, y_pred_bin, zero_division=0)), 4),
                "f1":        round(float(f1_score(y_true_bin, y_pred_bin, zero_division=0)), 4),
            }
        else:
            log("  Label column is not binary; Isolation Forest metrics skipped")
        rf_metrics = rf_supervised_evaluation(X_train, X_test, y_train, y_test, log)

    for col in cat_cols:
        if col in df.columns:
            df[col] = original_cat[col].reindex(df.index).values

    df["anomaly_flag"]  = all_preds
    df["anomaly_score"] = all_scores.round(5)
    if y is not None:
        df[label_col] = y.values

    anomaly_df = df[df["anomaly_flag"] == -1].copy()
    anomaly_df["anomaly_explanation"] = anomaly_df.index.map(
        lambda i: anomaly_explanations.get(i, "Isolation Forest identified this row as statistically unusual.")
    )
    if remove_anomalies:
        df = df[df["anomaly_flag"] != -1].reset_index(drop=True)
        log(f"  Anomalous rows removed from output: {n_anomalies}", "warn")
    else:
        log("  Anomalous rows retained in output with flag column")

    validation = build_validation_report(
        df_snapshot_raw, df_snapshot_pre, df, n_exact, n_near, log
    )

    log("Step 7: Building error report")
    error_rows: list = []

    for idx, row in anomaly_df.iterrows():
        error_rows.append({
            "row_index": int(idx),
            "category":  "A - Anomaly",
            "column":    "Whole row",
            "value":     f"score: {row['anomaly_score']}",
            "reason":    row["anomaly_explanation"],
            "suggestion": "Review manually -- could be a data error or genuine outlier.",
        })

    for idx in df[df.get("price_variant_flag", pd.Series(0, index=df.index)) == 1].index:
        error_rows.append({
            "row_index": int(idx),
            "category":  "B - Price/value variant",
            "column":    ", ".join(price_variant_cols) if price_variant_cols else "price",
            "value":     "see row",
            "reason":    "Same identifying title as another row but differs on price/value -- "
                         "likely the same listing at a different point in time, so kept rather than merged.",
            "suggestion": "Review both rows; treat the more recent price as current record.",
        })

    for idx in df[df.get("near_dup_flag", pd.Series(0, index=df.index)) == 1].index:
        error_rows.append({
            "row_index": int(idx),
            "category":  "D - Near-duplicate",
            "column":    "Whole row",
            "value":     "see row",
            "reason":    "Same title and price as another row -- retained but flagged for manual review.",
            "suggestion": "Verify whether this is a true duplicate or a legitimate separate record.",
        })

    skip_cols = {"anomaly_flag", "anomaly_score", "imputed_flag", "price_variant_flag", "near_dup_flag"} | set(
        c for c in df.columns if c.endswith("_original")
    )
    for col in df.columns:
        if col in skip_cols:
            continue
        for idx in df[df[col].isnull()].index:
            error_rows.append({
                "row_index": int(idx),
                "category":  "C - Residual missing value",
                "column":    col,
                "value":     "NaN",
                "reason":    "Value could not be recovered or imputed.",
                "suggestion": "Fill manually using domain knowledge.",
            })

    cat_a = sum(1 for r in error_rows if r["category"].startswith("A"))
    cat_b = sum(1 for r in error_rows if r["category"].startswith("B"))
    cat_c = sum(1 for r in error_rows if r["category"].startswith("C"))
    cat_d = sum(1 for r in error_rows if r["category"].startswith("D"))
    log(f"  Anomalous rows          : {cat_a}", "warn")
    log(f"  Price/value variants    : {cat_b}", "warn")
    log(f"  Residual missing values : {cat_c}", "warn")
    log(f"  Near-duplicate rows     : {cat_d}", "warn")
    log("Pipeline complete.")

    error_df = (
        pd.DataFrame(error_rows) if error_rows
        else pd.DataFrame(columns=["row_index", "category", "column", "value", "reason", "suggestion"])
    )

    summary = {
        "rows_in":        len(df_raw),
        "rows_out":       len(df),
        "dups_removed":   n_exact,
        "exact_dups":     n_exact,
        "near_dups":      n_near,
        "price_variants": cat_b,
        "near_dup_rows":  cat_d,
        "missing_before": _missing_count(df_snapshot_raw),
        "missing_after":  _missing_count(df),
        "anomalies":      n_anomalies,
        "errors":         len(error_rows),
        "iso_metrics":    iso_metrics,
        "rf_metrics":     rf_metrics,
        "validation":     validation,
    }

    return df, anomaly_df, error_df, logs, summary


# ---------------------------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f                = request.files["file"]
    label_col        = request.form.get("label_col", "").strip() or None
    try:
        contamination = float(request.form.get("contamination", DEFAULT_CONTAMINATION))
    except (TypeError, ValueError):
        contamination = DEFAULT_CONTAMINATION
    remove_anomalies = request.form.get("remove_anomalies", "false").lower() == "true"
    contamination = max(0.01, min(0.5, contamination))

    try:
        df_raw = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    try:
        clean_df, anomaly_df, error_df, logs, summary = run_pipeline(
            df_raw, label_col, contamination, remove_anomalies
        )
    except Exception as e:
        import traceback
        return jsonify({"error": f"Pipeline error: {e}", "trace": traceback.format_exc()}), 500

    token = str(uuid.uuid4())[:8]
    OUTPUT_STORE[token] = {"clean": clean_df, "anomaly": anomaly_df, "errors": error_df}

    preview_cols = [c for c in clean_df.columns if not c.endswith("_original")][:7]
    preview = (
        clean_df[preview_cols].head(10)
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    return jsonify({
        "token":   token,
        "logs":    logs,
        "summary": summary,
        "preview": {"cols": preview_cols, "rows": preview},
    })

@app.route("/download/<token>/<which>")
def download(token, which):
    if token not in OUTPUT_STORE:
        return "Session expired", 404
    name_map = {
        "clean":   "cleaned_dataset.csv",
        "anomaly": "anomaly_report.csv",
        "errors":  "uncorrectable_errors.csv",
    }
    if which not in OUTPUT_STORE[token]:
        return "Unknown file", 404
    buf = io.StringIO()
    OUTPUT_STORE[token][which].to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=name_map[which],
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)