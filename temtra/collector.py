from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np


def _split_cd_axis(arr: np.ndarray, tune: int):
    """
    Split an array assumed to be shaped (C, D, ...), where D = tune + draws.
    Returns (arr_tune, arr_draws).
    """
    a = np.asarray(arr)
    if a.ndim < 2:
        raise ValueError(f"Expected stats array with ndim>=2 (chains, draws, ...), got shape={a.shape}")
    return a[:, :tune, ...], a[:, tune:, ...]


def save_to_npz(
    path: str | Path,
    trace: np.ndarray,
    stats: Dict[str, np.ndarray],
    *,
    draws: int,
    tune: int,
    discard_tuned_samples: bool,
    compress: bool = True,
    meta: Dict[str, Any] | None = None,
) -> Path:
    """
    Writes a single .npz with split tune/draws parts for both trace and stats.

    Keys:
      - trace_tune, trace_draws
      - stats_tune__<name>, stats_draws__<name>
      - meta_json
    """
    if discard_tuned_samples:
        raise ValueError("This split writer requires discard_tuned_samples=False (so tune is present).")

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    trace = np.asarray(trace)
    if trace.dtype == object:
        raise TypeError("trace has dtype=object; convert to numeric dtype before saving")

    # Validate length along draw axis
    if trace.ndim < 2:
        raise ValueError(f"trace must have at least 2 dims, got shape={trace.shape}")

    total_expected = int(draws + tune)
    got_total = int(trace.shape[1])
    if got_total != total_expected:
        raise ValueError(f"trace.shape[1]={got_total}, expected draws+tune={total_expected}")

    trace_tune, trace_draws = _split_cd_axis(trace, tune)

    arrays: Dict[str, np.ndarray] = {
        "trace_tune": trace_tune,
        "trace_draws": trace_draws,
    }

    stats_keys = []
    for k, v in stats.items():
        k = str(k)
        a = np.asarray(v)
        if a.dtype == object:
            raise TypeError(f"stats['{k}'] has dtype=object; convert it before saving")

        # If the stat is per-draw, it should match total length; otherwise store unsplit under stats_misc__.
        if a.ndim >= 2 and a.shape[0] == trace.shape[0] and a.shape[1] == total_expected:
            a_tune, a_draws = _split_cd_axis(a, tune)
            arrays[f"stats_tune__{k}"] = a_tune
            arrays[f"stats_draws__{k}"] = a_draws
        else:
            arrays[f"stats_misc__{k}"] = a

        stats_keys.append(k)

    meta_payload = {
        "format": "littlemcmc_trace_stats_npz_v2_split",
        "draws": int(draws),
        "tune": int(tune),
        "discard_tuned_samples": False,
        "trace_shape_total": list(trace.shape),
        "trace_dtype": str(trace.dtype),
        "stats_keys": stats_keys,
    }
    if meta:
        meta_payload["user_meta"] = meta

    arrays["meta_json"] = np.array(json.dumps(meta_payload, ensure_ascii=False), dtype="U")

    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)

    return path


def load_trace_and_stats_npz_split(
    path: str | Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    Returns:
      trace_parts = {"tune": trace_tune, "draws": trace_draws}
      stats_parts = {"tune": {...}, "draws": {...}, "misc": {...}}
      meta dict
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))

        trace_parts = {
            "tune": z["trace_tune"],
            "draws": z["trace_draws"],
        }

        stats_parts: Dict[str, Dict[str, np.ndarray]] = {"tune": {}, "draws": {}, "misc": {}}
        for name in z.files:
            if name.startswith("stats_tune__"):
                stats_parts["tune"][name[len("stats_tune__"):]] = z[name]
            elif name.startswith("stats_draws__"):
                stats_parts["draws"][name[len("stats_draws__"):]] = z[name]
            elif name.startswith("stats_misc__"):
                stats_parts["misc"][name[len("stats_misc__"):]] = z[name]

    return trace_parts, stats_parts, meta

def concat_traces(traces):
    """List[(C, Dk, ...)] -> (C, sum(Dk), ...)"""
    if not traces:
        raise ValueError("empty traces")
    return np.concatenate([np.asarray(t) for t in traces], axis=1)


def concat_stats(stats_list, *, chains: int, total: int):

    if not stats_list:
        return {}
    keys = set().union(*(s.keys() for s in stats_list))
    out = {}
    for k in keys:
        arrs = []
        for s in stats_list:
            if k not in s:
                arrs = None
                break
            a = np.asarray(s[k])
            if a.ndim < 2 or a.shape[0] != chains:
                arrs = None
                break
            arrs.append(a)
        if not arrs:
            continue
        a = np.concatenate(arrs, axis=1)
        if a.shape[1] != total:
            raise ValueError(f"stats['{k}'] concat D={a.shape[1]} expected {total}")
        out[str(k)] = a
    return out
