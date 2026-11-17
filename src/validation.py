"""Input checks that run BEFORE any model.predict call (proposal Section 7.2).

The schema (allowed categories + numeric ranges) is read from meta.json. Because
key names can differ between dataset versions, we look in several common places
and fall back to the training data / UCI ranges when nothing is listed."""
import math

import pandas as pd

from .config import FALLBACK_RANGES


def _find(meta: dict, feature: str, kind: str):
    """Search meta.json for allowed values (kind='cat') or a range (kind='num')."""
    candidates = []
    for key, val in meta.items():
        if isinstance(val, dict) and feature in val:
            candidates.append((key.lower(), val[feature]))
    for key, val in candidates:
        if kind == "cat" and isinstance(val, list):
            return [str(v) for v in val]
        if kind == "cat" and isinstance(val, dict):
            for k in ("allowed", "values", "allowed_values", "categories"):
                if isinstance(val.get(k), list):
                    return [str(v) for v in val[k]]
        if kind == "num":
            if isinstance(val, (list, tuple)) and len(val) == 2 and all(
                    isinstance(x, (int, float)) for x in val):
                return float(val[0]), float(val[1])
            if isinstance(val, dict):
                lo = val.get("min", val.get("minimum"))
                hi = val.get("max", val.get("maximum"))
                if lo is not None and hi is not None:
                    return float(lo), float(hi)
    return None


def build_schema(meta: dict, main_df: pd.DataFrame) -> dict:
    """{feature: {"type": "num"|"cat", "range": (lo, hi) | "allowed": [...]}}"""
    num, cat = meta["numeric_features"], meta["categorical_features"]
    cols = list(dict.fromkeys(num + cat + meta.get("cluster_features", [])))
    schema = {}
    for c in cols:
        if c in cat:
            allowed = _find(meta, c, "cat")
            if allowed is None:
                allowed = sorted(main_df[c].dropna().astype(str).unique().tolist())
            schema[c] = {"type": "cat", "allowed": allowed}
        else:
            rng = _find(meta, c, "num") or FALLBACK_RANGES.get(c)
            if rng is None:
                rng = (float(main_df[c].min()), float(main_df[c].max()))
            integer = pd.api.types.is_integer_dtype(main_df[c]) if c in main_df else True
            schema[c] = {"type": "num", "range": (float(rng[0]), float(rng[1])),
                         "integer": bool(integer)}
    return schema


def validate_record(values: dict, schema: dict):
    """Return (clean_record, errors). Prediction is allowed only if errors == []."""
    errors, clean = [], {}
    for col, rule in schema.items():
        v = values.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            errors.append(f"**{col}** is missing. Every field is required (no imputation).")
            continue
        if rule["type"] == "cat":
            if str(v) not in rule["allowed"]:
                errors.append(f"**{col}** = `{v}` is not allowed. Use one of: "
                              f"{', '.join(rule['allowed'])}.")
            else:
                clean[col] = str(v)
        else:
            try:
                x = float(v)
            except (TypeError, ValueError):
                errors.append(f"**{col}** must be a number, got `{v}`.")
                continue
            lo, hi = rule["range"]
            if not lo <= x <= hi:
                errors.append(f"**{col}** = {x:g} is outside the valid range "
                              f"{lo:g} to {hi:g}. Please correct it.")
            elif rule.get("integer") and x != int(x):
                errors.append(f"**{col}** must be a whole number, got {x:g}.")
            else:
                clean[col] = int(x) if rule.get("integer") else x
    return clean, errors


def validate_table(df: pd.DataFrame, schema: dict):
    """Check required columns for a CSV upload. Returns list of errors."""
    missing = [c for c in schema if c not in df.columns]
    return [f"Missing required column(s): {', '.join(missing)}"] if missing else []
