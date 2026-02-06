# app.py
# DM04 Calibration app (PDF workflow) + Multi-folder loader + scatter-vs-line plots
#
# Workflow (per PDF):
# - OE: fit CSL parameters (e0, lambda_c, xi) using e(p)=e0 - lambda_c*(p/pa)^xi
# - TMD: calibration (Stage 2/3) because drained epsv is available
# - TMU/TMU-MT-AP: validation overlays only (not in optimization loss)
#
# UI:
# - Multi-folder (Load across all folders) toggle
# - Guidance: what to toggle ON/OFF + how to improve when fit is bad
#
# Run:
#   streamlit run app.py
#
# Folder layout under Root folder:
#   OE-all/*.dat
#   TMD-all/*.dat
#   TMU-all/*.dat
#   TMU-MT-AP-all/*.dat

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


# ============================================================
# Constants / defaults
# ============================================================

PA_KPA = 101.325

FOLDERS = {
    "OE": "OE-all",
    "TMD": "TMD-all",
    "TMU": "TMU-all",
    "TMU-MT-AP": "TMU-MT-AP-all",
}

DEFAULT_PARAMS = {
    # CSL (fit from OE)
    "e0_csl": 1.03,
    "lambda_c": 0.08,
    "xi": 0.33,
    # Critical stress ratio (estimated from TMD tail)
    "M": 1.42,
    # Elasticity
    "nu": 0.20,
    "G0": 50.0,
    # Hardening / dilatancy / fabric
    "h0": 8.0,
    "ch": 1.0,
    "nb": 1.1,
    "Ad": 1.0,
    "nd": 2.2,
    "zmax": 5.0,
    "cz": 600.0,
    # yield radius
    "m": 0.01,
}

BOUNDS_STAGE2 = {
    "h0": (0.1, 50.0),
    "nb": (0.05, 10.0),
    "Ad": (0.01, 10.0),
}
BOUNDS_STAGE3 = {
    "G0": (1.0, 300.0),
    "h0": (0.1, 80.0),
    "ch": (0.05, 5.0),
    "nb": (0.01, 20.0),
    "Ad": (0.001, 20.0),
    "nd": (0.0, 10.0),
    "zmax": (0.0, 30.0),
    "cz": (0.0, 5000.0),
}

TestKind = Literal["OE", "TMD", "TMU"]


# ============================================================
# Data containers
# ============================================================


@dataclass
class TestData:
    name: str
    kind: TestKind
    df: pd.DataFrame

    eps1: np.ndarray

    # triaxial
    epsv: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None

    # OE
    sigma1: Optional[np.ndarray] = None  # ~p
    e: Optional[np.ndarray] = None
    e0: Optional[float] = None

    sigma3_eff_0: Optional[float] = None


@dataclass
class SimResult:
    eps1: np.ndarray
    epsv: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None

    # OE analytic overlay
    p_oe: Optional[np.ndarray] = None
    e_oe: Optional[np.ndarray] = None


# ============================================================
# Parsing helpers
# ============================================================


def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace("ε", "eps")
    c = re.sub(r"\s+", "", c)
    c = c.replace("=", "")
    return c


def _read_dat_table(path: str) -> pd.DataFrame:
    """
    Your .dat format:
      line 1: names (may include 'Void ratio' as 2 tokens)
      line 2: units
      rest: whitespace-separated numeric values

    We parse header ourselves, merge "Void ratio", then use delim_whitespace for body.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip() != ""]

    if len(lines) < 3:
        raise ValueError("File too short (needs header + units + data).")

    hdr_tokens = re.split(r"\s+", lines[0].strip())

    merged = []
    i = 0
    while i < len(hdr_tokens):
        tok = hdr_tokens[i]
        if (
            i + 1 < len(hdr_tokens)
            and tok.lower() == "void"
            and hdr_tokens[i + 1].lower() == "ratio"
        ):
            merged.append("Void ratio")
            i += 2
        else:
            merged.append(tok)
            i += 1

    from io import StringIO

    body = "\n".join(lines[2:])  # skip header + units
    df_body = pd.read_csv(StringIO(body), delim_whitespace=True, header=None, dtype=str)

    n_expected = len(merged)
    if df_body.shape[1] != n_expected:
        if df_body.shape[1] > n_expected:
            df_body = df_body.iloc[:, :n_expected]
        else:
            for _ in range(n_expected - df_body.shape[1]):
                df_body[df_body.shape[1]] = np.nan

    df_body.columns = merged
    return df_body


def _detect_kind_and_map(df: pd.DataFrame) -> Tuple[TestKind, Dict[str, str]]:
    cols = list(df.columns)
    ncols = {_norm_col(c): c for c in cols}

    # OE
    if ("sigma1" in ncols) and ("eps1" in ncols) and ("voidratio" in ncols):
        return "OE", {
            "sigma1": ncols["sigma1"],
            "eps1": ncols["eps1"],
            "e": ncols["voidratio"],
        }

    # TMD
    if ("eps1" in ncols) and ("epsv" in ncols) and ("q" in ncols) and ("p" in ncols):
        mapping = {
            "eps1": ncols["eps1"],
            "epsv": ncols["epsv"],
            "q": ncols["q"],
            "p": ncols["p"],
        }
        for k in ncols:
            if "voidratio" in k:
                mapping["e"] = ncols[k]
                break
        return "TMD", mapping

    # TMU/TMU-MT-AP
    if ("eps1" in ncols) and ("p" in ncols) and ("q" in ncols):
        return "TMU", {"eps1": ncols["eps1"], "p": ncols["p"], "q": ncols["q"]}

    raise ValueError(f"Unknown file format. Columns detected: {cols}")


def load_test(path: str) -> TestData:
    df = _read_dat_table(path)
    kind, mapping = _detect_kind_and_map(df)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("Too few numeric rows after parsing.")

    name = os.path.basename(path)

    if kind == "OE":
        sigma1 = df[mapping["sigma1"]].to_numpy(float)
        eps1 = df[mapping["eps1"]].to_numpy(float) / 100.0
        e = df[mapping["e"]].to_numpy(float)
        e0 = float(e[0]) if len(e) else None
        return TestData(
            name=name, kind="OE", df=df, eps1=eps1, sigma1=sigma1, e=e, e0=e0
        )

    if kind == "TMD":
        eps1 = df[mapping["eps1"]].to_numpy(float) / 100.0
        epsv = df[mapping["epsv"]].to_numpy(float) / 100.0
        q = df[mapping["q"]].to_numpy(float)
        p = df[mapping["p"]].to_numpy(float)

        e = None
        e0 = None
        if "e" in mapping:
            e = df[mapping["e"]].to_numpy(float)
            e0 = float(e[0]) if len(e) else None

        sig3 = float(p[0] - q[0] / 3.0) if len(p) else None
        return TestData(
            name=name,
            kind="TMD",
            df=df,
            eps1=eps1,
            epsv=epsv,
            p=p,
            q=q,
            e=e,
            e0=e0,
            sigma3_eff_0=sig3 if (sig3 is not None and sig3 > 0) else None,
        )

    # TMU
    eps1 = df[mapping["eps1"]].to_numpy(float) / 100.0
    p = df[mapping["p"]].to_numpy(float)
    q = df[mapping["q"]].to_numpy(float)
    epsv = np.zeros_like(eps1, float)
    return TestData(name=name, kind="TMU", df=df, eps1=eps1, epsv=epsv, p=p, q=q)


# ============================================================
# OE CSL fit (PDF)
# ============================================================


def ec_of_p(p_kpa: np.ndarray, e0: float, lam: float, xi: float) -> np.ndarray:
    p = np.maximum(1e-9, p_kpa)
    return e0 - lam * (p / PA_KPA) ** xi


def estimate_csl_from_oe(
    oe_tests: List[TestData],
    p_min_kpa: float = 10.0,
    use_log_weight: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Fits e(p)=e0 - lambda_c*(p/pa)^xi to OE data.

    Practical improvements:
    - ignore very low p (p < p_min_kpa)
    - optional log-weighting to reduce low-p dominance
    """
    if minimize is None or not oe_tests:
        return None

    p_all = []
    e_all = []
    for t in oe_tests:
        if t.sigma1 is None or t.e is None:
            continue
        p = np.asarray(t.sigma1, float)
        e = np.asarray(t.e, float)
        mask = (
            np.isfinite(p) & np.isfinite(e) & (p >= p_min_kpa) & (e > 0.01) & (e < 5.0)
        )
        if np.sum(mask) < 10:
            continue
        p_all.append(p[mask])
        e_all.append(e[mask])

    if not p_all:
        return None

    p = np.concatenate(p_all)
    e = np.concatenate(e_all)

    e0_guess = float(np.percentile(e, 95))
    lam_guess = 0.08
    xi_guess = 0.33

    def obj(x):
        e0, lam, xi = x
        if not np.isfinite(e0 + lam + xi):
            return 1e30
        if lam <= 0 or xi <= 0:
            return 1e30
        pred = ec_of_p(p, e0, lam, xi)
        r = pred - e
        if use_log_weight:
            w = np.log(np.maximum(p, 1.0))
            w = w / np.mean(w)
            return float(np.mean((w * r) ** 2))
        return float(np.mean(r**2))

    bounds = [(0.2, 2.5), (1e-4, 1.0), (0.05, 1.5)]
    res = minimize(
        obj,
        np.array([e0_guess, lam_guess, xi_guess], float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )
    if not hasattr(res, "x"):
        return None

    e0, lam, xi = [float(v) for v in res.x]
    return {
        "e0_csl": e0,
        "lambda_c": lam,
        "xi": xi,
        "csl_fit_mse": float(getattr(res, "fun", np.nan)),
        "p_min_kpa": float(p_min_kpa),
        "log_weight": bool(use_log_weight),
    }


def estimate_M_from_tmd(tmd_tests: List[TestData]) -> Optional[float]:
    if not tmd_tests:
        return None
    etas = []
    for t in tmd_tests:
        if t.p is None or t.q is None:
            continue
        p = np.asarray(t.p, float)
        q = np.asarray(t.q, float)
        mask = np.isfinite(p) & np.isfinite(q) & (p > 1e-6)
        if np.sum(mask) < 20:
            continue
        p = p[mask]
        q = q[mask]
        n = len(p)
        tail = slice(int(0.9 * n), n)
        eta_tail = q[tail] / p[tail]
        eta_tail = eta_tail[np.isfinite(eta_tail)]
        if len(eta_tail):
            etas.append(float(np.median(eta_tail)))
    if not etas:
        return None
    return float(np.median(etas))


# ============================================================
# DM04 core (safe exponentials)
# ============================================================

EXP_CLAMP = 60.0
PSI_CLAMP = 5.0


def safe_exp(x: float) -> float:
    return math.exp(max(-EXP_CLAMP, min(EXP_CLAMP, x)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def double_contract(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.tensordot(A, B, axes=2))


def dev(T: np.ndarray) -> np.ndarray:
    return T - np.eye(3) * np.trace(T) / 3.0


def mean_p(sig: np.ndarray) -> float:
    return float(np.trace(sig) / 3.0)


def q_invariant(sig: np.ndarray) -> float:
    s = dev(sig)
    return float(np.sqrt(1.5 * double_contract(s, s)))


def safe_norm(T: np.ndarray, eps: float = 1e-14) -> float:
    return math.sqrt(max(eps, double_contract(T, T)))


def csl_ec_scalar(p_kpa: float, params: Dict[str, float]) -> float:
    p = max(1e-9, p_kpa)
    return float(params["e0_csl"] - params["lambda_c"] * (p / PA_KPA) ** params["xi"])


def psi_state(e: float, p_kpa: float, params: Dict[str, float]) -> float:
    psi = float(e - csl_ec_scalar(p_kpa, params))
    return clamp(psi, -PSI_CLAMP, PSI_CLAMP)


def dm04_GK(p_kpa: float, e: float, params: Dict[str, float]) -> Tuple[float, float]:
    p = max(1e-9, p_kpa)
    fac = ((2.97 - e) ** 2) / max(1e-9, (1.0 + e))
    G = float(params["G0"] * PA_KPA * fac * math.sqrt(p / PA_KPA))
    nu = clamp(params["nu"], 0.01, 0.49)
    K = float(G * (2.0 * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)))
    return G, K


def Ce_isotropic(G: float, K: float) -> np.ndarray:
    I = np.eye(3)
    I4s = 0.5 * (np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I))
    I4v = (1.0 / 3.0) * np.einsum("ij,kl->ijkl", I, I)
    return 2.0 * G * (I4s - I4v) + 3.0 * K * I4v


def apply_Ce(Ce: np.ndarray, deps: np.ndarray) -> np.ndarray:
    return np.einsum("ijkl,kl->ij", Ce, deps)


def invariants(sig: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    p = mean_p(sig)
    s = dev(sig)
    q = q_invariant(sig)
    r = s / max(1e-12, p)
    return p, q, s, r


def yield_f(sig: np.ndarray, alpha: np.ndarray, m: float) -> float:
    _, _, _, r = invariants(sig)
    x = r - alpha
    return float(safe_norm(x) - m)


def Mb_Md(psi: float, params: Dict[str, float]) -> Tuple[float, float]:
    Mb = float(params["M"] * safe_exp(-params["nb"] * psi))
    Md = float(params["M"] * safe_exp(params["nd"] * psi))
    Mb = clamp(Mb, 1e-6, 50.0)
    Md = clamp(Md, 1e-6, 50.0)
    return Mb, Md


def n_tensor(sig: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    _, _, _, r = invariants(sig)
    x = r - alpha
    return x / safe_norm(x)


def dilatancy_D(
    sig: np.ndarray, alpha: np.ndarray, e: float, params: Dict[str, float]
) -> float:
    p, _, _, r = invariants(sig)
    psi = psi_state(e, p, params)
    _, Md = Mb_Md(psi, params)
    n = n_tensor(sig, alpha)
    D = float(params["Ad"] * (Md - double_contract(r, n)))
    return clamp(D, -50.0, 50.0)


def rb_tensor(sig: np.ndarray, e: float, params: Dict[str, float]) -> np.ndarray:
    p, _, _, r = invariants(sig)
    psi = psi_state(e, p, params)
    Mb, _ = Mb_Md(psi, params)
    rn = safe_norm(r)
    return np.zeros((3, 3), float) if rn < 1e-14 else (Mb / rn) * r


def hardening_modulus(
    sig: np.ndarray, alpha: np.ndarray, e: float, params: Dict[str, float], G: float
) -> Tuple[float, float]:
    p, _, _, r = invariants(sig)
    psi = psi_state(e, p, params)
    mult = safe_exp(-params["ch"] * psi)
    Kp = float(params["h0"] * G * mult)
    Kp = clamp(Kp, 1e-9, 1e9)
    rb = rb_tensor(sig, e, params)
    dist = safe_norm(rb - r)
    h = float(Kp / max(1e-12, dist))
    h = clamp(h, 0.0, 1e9)
    return Kp, h


def flow_direction(
    sig: np.ndarray, alpha: np.ndarray, e: float, z: float, params: Dict[str, float]
) -> np.ndarray:
    D = dilatancy_D(sig, alpha, e, params)
    if D > 0:
        D = float(D * (1.0 + z))
    n = n_tensor(sig, alpha)
    return (D / 3.0) * np.eye(3) + n


def update_void_ratio(e: float, deps_v: float) -> float:
    e_new = float(e - (1.0 + e) * deps_v)
    return clamp(e_new, 0.05, 2.5)


def df_dsigma_numeric(
    sig: np.ndarray, alpha: np.ndarray, m: float, h: float = 1e-6
) -> np.ndarray:
    base = yield_f(sig, alpha, m)
    grad = np.zeros((3, 3), float)
    for i in range(3):
        for j in range(3):
            d = np.zeros((3, 3), float)
            d[i, j] = h
            d[j, i] = h if i != j else h
            fp = yield_f(sig + d, alpha, m)
            grad[i, j] = (fp - base) / h
    return 0.5 * (grad + grad.T)


def dm04_step(sig, alpha, e, z, deps, params):
    p = mean_p(sig)
    if not np.isfinite(p) or p <= 0:
        nan = np.full((3, 3), np.nan)
        return nan, nan, np.nan, np.nan

    G, K = dm04_GK(p, e, params)
    Ce = Ce_isotropic(G, K)

    sig_trial = sig + apply_Ce(Ce, deps)
    if not np.all(np.isfinite(sig_trial)):
        nan = np.full((3, 3), np.nan)
        return nan, nan, np.nan, np.nan

    f_trial = yield_f(sig_trial, alpha, params["m"])
    if f_trial <= 0.0:
        e_new = update_void_ratio(e, float(np.trace(deps)))
        return sig_trial, alpha, e_new, z

    df = df_dsigma_numeric(sig_trial, alpha, params["m"])
    dg = flow_direction(sig, alpha, e, z, params)
    Kp, hmod = hardening_modulus(sig, alpha, e, params, G)

    denom = double_contract(df, apply_Ce(Ce, dg)) + Kp
    denom = denom if abs(denom) > 1e-14 else 1e-14

    dgamma = float(max(0.0, f_trial / denom))
    dgamma = clamp(dgamma, 0.0, 10.0)

    deps_p = dgamma * dg
    sig_new = sig_trial - apply_Ce(Ce, deps_p)

    rb = rb_tensor(sig, e, params)
    alpha_new = alpha + dgamma * (2.0 / 3.0) * hmod * (rb - alpha)

    Dn = dilatancy_D(sig, alpha, e, params)
    if Dn < 0.0:
        z_new = z - dgamma * params["cz"] * (params["zmax"] - z) * Dn
        z_new = float(max(0.0, min(params["zmax"], z_new)))
    else:
        z_new = z

    e_new = update_void_ratio(e, float(np.trace(deps)))

    # projection
    f_after = yield_f(sig_new, alpha_new, params["m"])
    if f_after > 1e-8:
        p_new, _, _, r_new = invariants(sig_new)
        x = r_new - alpha_new
        nx = safe_norm(x)
        if nx > 1e-14 and np.isfinite(p_new) and p_new > 0:
            scale = params["m"] / nx
            r_corr = alpha_new + scale * x
            s_corr = p_new * r_corr
            sig_new = s_corr + p_new * np.eye(3)

    return sig_new, alpha_new, e_new, z_new


# ============================================================
# Simulators
# ============================================================


def simulate_OE_analytical(test: TestData, params: Dict[str, float]) -> SimResult:
    if test.sigma1 is None or test.e is None:
        return SimResult(eps1=test.eps1.copy())
    p = np.asarray(test.sigma1, float)
    e_hat = ec_of_p(p, params["e0_csl"], params["lambda_c"], params["xi"])
    return SimResult(eps1=test.eps1.copy(), p_oe=p.copy(), e_oe=e_hat)


def simulate_TMD_drained(
    test: TestData, params: Dict[str, float], n_substeps: int
) -> SimResult:
    eps1_exp = test.eps1
    if len(eps1_exp) < 3:
        return SimResult(eps1=np.array([]))

    p0 = float(test.p[0])
    q0 = float(test.q[0])
    sig3 = float(
        test.sigma3_eff_0 if test.sigma3_eff_0 is not None else (p0 - q0 / 3.0)
    )
    sig3 = max(1e-6, sig3)
    sig1 = sig3 + q0
    sig = np.diag([sig1, sig3, sig3]).astype(float)

    e = float(test.e0 if test.e0 is not None else params["e0_csl"])
    alpha = np.zeros((3, 3), float)
    z = 0.0

    eps1_model = [float(eps1_exp[0])]
    epsv_model = [0.0]
    p_model = [mean_p(sig)]
    q_model = [q_invariant(sig)]

    eps3_acc = 0.0
    eps1_acc = float(eps1_exp[0])

    for k in range(1, len(eps1_exp)):
        d_eps1_total = float(eps1_exp[k] - eps1_exp[k - 1])
        if not np.isfinite(d_eps1_total):
            continue

        for _ in range(max(1, int(n_substeps))):
            d_eps1 = d_eps1_total / max(1, int(n_substeps))

            def res(d_eps3: float) -> float:
                deps = np.diag([d_eps1, d_eps3, d_eps3])
                sig_try, _, _, _ = dm04_step(sig, alpha, e, z, deps, params)
                return (
                    float(sig_try[1, 1] - sig3) if np.all(np.isfinite(sig_try)) else 1e9
                )

            d_eps3 = -params["nu"] * d_eps1
            for _it in range(12):
                r0 = res(d_eps3)
                if abs(r0) < 1e-4:
                    break
                h = 1e-6 if abs(d_eps3) < 1e-4 else 1e-4 * abs(d_eps3)
                r1 = res(d_eps3 + h)
                dr = (r1 - r0) / h
                if abs(dr) < 1e-10:
                    break
                d_eps3 = float(np.clip(d_eps3 - r0 / dr, -0.2, 0.2))

            deps = np.diag([d_eps1, d_eps3, d_eps3])
            sig, alpha, e, z = dm04_step(sig, alpha, e, z, deps, params)
            if not np.all(np.isfinite(sig)):
                return SimResult(eps1=np.array([]))

            eps1_acc += d_eps1
            eps3_acc += d_eps3

        eps1_model.append(eps1_acc)
        epsv_model.append(float(eps1_acc + 2.0 * eps3_acc))
        p_model.append(mean_p(sig))
        q_model.append(q_invariant(sig))

    return SimResult(
        eps1=np.asarray(eps1_model, float),
        epsv=np.asarray(epsv_model, float),
        p=np.asarray(p_model, float),
        q=np.asarray(q_model, float),
    )


def simulate_TMU_undrained(
    test: TestData, params: Dict[str, float], n_substeps: int
) -> SimResult:
    eps1_exp = test.eps1
    if len(eps1_exp) < 3:
        return SimResult(eps1=np.array([]))

    p0 = float(test.p[0])
    q0 = float(test.q[0])
    sig3 = max(1e-6, float(p0 - q0 / 3.0))
    sig1 = sig3 + q0
    sig = np.diag([sig1, sig3, sig3]).astype(float)

    e = float(params["e0_csl"])
    alpha = np.zeros((3, 3), float)
    z = 0.0

    eps1_model = [float(eps1_exp[0])]
    epsv_model = [0.0]
    p_model = [mean_p(sig)]
    q_model = [q_invariant(sig)]

    eps1_acc = float(eps1_exp[0])
    eps3_acc = 0.0

    for k in range(1, len(eps1_exp)):
        d_eps1_total = float(eps1_exp[k] - eps1_exp[k - 1])
        if not np.isfinite(d_eps1_total):
            continue

        for _ in range(max(1, int(n_substeps))):
            d_eps1 = d_eps1_total / max(1, int(n_substeps))
            d_eps3 = -0.5 * d_eps1  # epsv ~ 0
            deps = np.diag([d_eps1, d_eps3, d_eps3])

            sig, alpha, e, z = dm04_step(sig, alpha, e, z, deps, params)
            if not np.all(np.isfinite(sig)):
                return SimResult(eps1=np.array([]))

            eps1_acc += d_eps1
            eps3_acc += d_eps3

        eps1_model.append(eps1_acc)
        epsv_model.append(float(eps1_acc + 2.0 * eps3_acc))
        p_model.append(mean_p(sig))
        q_model.append(q_invariant(sig))

    return SimResult(
        eps1=np.asarray(eps1_model, float),
        epsv=np.asarray(epsv_model, float),
        p=np.asarray(p_model, float),
        q=np.asarray(q_model, float),
    )


def simulate(test: TestData, params: Dict[str, float], n_substeps: int) -> SimResult:
    if test.kind == "OE":
        return simulate_OE_analytical(test, params)
    if test.kind == "TMD":
        return simulate_TMD_drained(test, params, n_substeps)
    return simulate_TMU_undrained(test, params, n_substeps)


# ============================================================
# Optimization (TMD-only loss)
# ============================================================


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2)) if len(a) else 0.0


def interp_to(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    if len(x_src) < 2:
        return np.full_like(x_tgt, y_src[0] if len(y_src) else np.nan, dtype=float)
    return np.interp(x_tgt, x_src, y_src)


def total_loss_TMD_only(
    params: Dict[str, float],
    tests: List[TestData],
    n_substeps: int,
    w: Dict[str, float],
) -> float:
    if params["nu"] <= 0 or params["nu"] >= 0.5:
        return 1e30
    if params["m"] <= 0:
        return 1e30

    tmd_tests = [t for t in tests if t.kind == "TMD"]
    if not tmd_tests:
        return 1e25

    L = 0.0
    for t in tmd_tests:
        simr = simulate(t, params, n_substeps)
        if len(simr.eps1) < 3 or simr.q is None or simr.p is None or simr.epsv is None:
            return 1e25

        x = t.eps1
        q_hat = interp_to(simr.eps1, simr.q, x)
        p_hat = interp_to(simr.eps1, simr.p, x)
        ev_hat = interp_to(simr.eps1, simr.epsv, x)

        if not (
            np.all(np.isfinite(q_hat))
            and np.all(np.isfinite(p_hat))
            and np.all(np.isfinite(ev_hat))
        ):
            return 1e25

        L += (
            w["q"] * mse(t.q, q_hat)
            + w["p"] * mse(t.p, p_hat)
            + w["ev"] * mse(t.epsv, ev_hat)
        )

    return float(L)


def dict_to_vec(d: Dict[str, float], keys: List[str]) -> np.ndarray:
    return np.array([float(d[k]) for k in keys], dtype=float)


def vec_to_dict(
    v: np.ndarray, keys: List[str], base: Dict[str, float]
) -> Dict[str, float]:
    out = dict(base)
    for i, k in enumerate(keys):
        out[k] = float(v[i])
    return out


def optimize_params(
    params0: Dict[str, float],
    tests: List[TestData],
    stage: int,
    n_substeps: int,
    w: Dict[str, float],
):
    if minimize is None:
        return params0, None

    if stage == 2:
        keys = ["h0", "nb", "Ad"]
        bounds = [BOUNDS_STAGE2[k] for k in keys]
    else:
        keys = ["G0", "h0", "ch", "nb", "Ad", "nd", "zmax", "cz"]
        bounds = [BOUNDS_STAGE3[k] for k in keys]

    x0 = dict_to_vec(params0, keys)

    def obj(x):
        p = vec_to_dict(x, keys, params0)
        return total_loss_TMD_only(p, tests, n_substeps, w)

    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})
    fitted = vec_to_dict(res.x, keys, params0) if hasattr(res, "x") else params0
    return fitted, res


# ============================================================
# Plotting (data scatter, model line)
# ============================================================


def plot_oe(test: TestData, simr: SimResult):
    p = np.asarray(test.sigma1, float)
    e = np.asarray(test.e, float)

    fig = plt.figure()
    plt.scatter(p, e, s=18, label="Data e(p)")
    if simr.p_oe is not None and simr.e_oe is not None:
        order = np.argsort(simr.p_oe)
        plt.plot(simr.p_oe[order], simr.e_oe[order], label="Model e_c(p)")
    plt.xlabel("p ≈ σv [kPa]")
    plt.ylabel("Void ratio e [-]")
    plt.title(f"{test.name} (OE): e vs p")
    plt.legend()
    st.pyplot(fig, clear_figure=True)


def plot_triaxial(test: TestData, simr: SimResult):
    fig = plt.figure()
    plt.scatter(test.eps1, test.q, s=14, label="Data q")
    plt.plot(simr.eps1, simr.q, label="Model q")
    plt.xlabel("ε1 [-]")
    plt.ylabel("q [kPa]")
    plt.title(f"{test.name}: q vs ε1")
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    fig = plt.figure()
    plt.scatter(test.eps1, test.p, s=14, label="Data p")
    plt.plot(simr.eps1, simr.p, label="Model p")
    plt.xlabel("ε1 [-]")
    plt.ylabel("p [kPa]")
    plt.title(f"{test.name}: p vs ε1")
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    if test.kind == "TMD":
        fig = plt.figure()
        plt.scatter(test.eps1, test.epsv, s=14, label="Data εv")
        plt.plot(simr.eps1, simr.epsv, label="Model εv")
        plt.xlabel("ε1 [-]")
        plt.ylabel("εv [-]")
        plt.title(f"{test.name} (TMD): εv vs ε1")
        plt.legend()
        st.pyplot(fig, clear_figure=True)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="DM04 Calibration (Multi-folder)", layout="wide")
st.title("DM04 Calibration – Load across folders + scatter-vs-line plots")

root = os.path.dirname(os.path.abspath(__file__))

with st.sidebar:
    st.header("Data loading")
    data_root = st.text_input(
        "Root folder (contains OE-all, TMD-all, TMU-all, ...)", value=root
    )

    mode = st.radio(
        "Load mode",
        ["Load across all folders (recommended)", "Single-folder (old)"],
        index=0,
        help="Recommended: select OE+TMD+TMU files together without switching folders.",
    )

    st.header("OE fit controls (Stage 1)")
    p_min_kpa = st.slider("Use OE points with p ≥ [kPa]", 0.0, 100.0, 10.0, 1.0)
    use_log_weight = st.checkbox("Log-weight OE fit (downweight low p)", value=True)

    st.header("Simulation")
    n_substeps = st.slider("Substeps per increment (TMD/TMU only)", 1, 10, 3)

    st.header("Stages")
    do_stage1 = st.checkbox(
        "Stage 1: Fit (e0, λc, ξ) from OE + estimate M from TMD", value=True
    )
    run_stage2 = st.checkbox("Stage 2: Optimize (h0, nb, Ad)  [TMD only]", value=True)
    run_stage3 = st.checkbox(
        "Stage 3: Optimize (G0,h0,ch,nb,Ad,nd,zmax,cz)  [TMD only]", value=False
    )

    st.header("TMD loss weights")
    w = {
        "q": float(st.number_input("wq", value=1.0, min_value=0.0, step=0.5)),
        "p": float(st.number_input("wp", value=1.0, min_value=0.0, step=0.5)),
        "ev": float(st.number_input("wεv", value=1.0, min_value=0.0, step=0.5)),
    }

    st.header("Initial parameters")
    params = dict(DEFAULT_PARAMS)

    c1, c2 = st.columns(2)
    with c1:
        params["e0_csl"] = st.number_input(
            "e0", value=float(params["e0_csl"]), step=0.001, format="%.6f"
        )
        params["lambda_c"] = st.number_input(
            "lambda_c", value=float(params["lambda_c"]), step=0.001, format="%.6f"
        )
        params["xi"] = st.number_input(
            "xi", value=float(params["xi"]), step=0.001, format="%.6f"
        )
        params["M"] = st.number_input(
            "M", value=float(params["M"]), step=0.001, format="%.6f"
        )
        params["nu"] = st.number_input(
            "nu", value=float(params["nu"]), step=0.001, format="%.6f"
        )
    with c2:
        params["G0"] = st.number_input(
            "G0", value=float(params["G0"]), step=0.5, format="%.6f"
        )
        params["h0"] = st.number_input(
            "h0", value=float(params["h0"]), step=0.1, format="%.6f"
        )
        params["ch"] = st.number_input(
            "ch", value=float(params["ch"]), step=0.01, format="%.6f"
        )
        params["nb"] = st.number_input(
            "nb", value=float(params["nb"]), step=0.01, format="%.6f"
        )
        params["Ad"] = st.number_input(
            "Ad", value=float(params["Ad"]), step=0.01, format="%.6f"
        )

    with st.expander("More (nd, zmax, cz, m)"):
        params["nd"] = st.number_input(
            "nd", value=float(params["nd"]), step=0.05, format="%.6f"
        )
        params["zmax"] = st.number_input(
            "zmax", value=float(params["zmax"]), step=0.1, format="%.6f"
        )
        params["cz"] = st.number_input(
            "cz", value=float(params["cz"]), step=10.0, format="%.6f"
        )
        params["m"] = st.number_input(
            "m", value=float(params["m"]), step=0.001, format="%.6f"
        )

    run_btn = st.button("Run", type="primary")


def list_dat_files(folder_path: str) -> List[str]:
    if not os.path.isdir(folder_path):
        return []
    return sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".dat")])


# -----------------------------
# File selectors
# -----------------------------
selected_paths: List[str] = []

if mode.startswith("Load across"):
    st.sidebar.header("Select files per dataset")

    for label, folder in FOLDERS.items():
        abs_folder = os.path.join(data_root, folder)
        files = list_dat_files(abs_folder)
        with st.sidebar.expander(f"{label} ({folder})", expanded=(label == "OE")):
            if not os.path.isdir(abs_folder):
                st.error(f"Missing folder: {abs_folder}")
                continue
            if not files:
                st.warning("No .dat files found.")
                continue
            default_pick = files[: min(3, len(files))]
            picks = st.multiselect(
                f"Files in {folder}", files, default=default_pick, key=f"pick_{folder}"
            )
            for f in picks:
                selected_paths.append(os.path.join(abs_folder, f))

else:
    st.sidebar.header("Single-folder selection")
    folder_choice = st.sidebar.selectbox(
        "Dataset folder", list(FOLDERS.values()), index=0
    )
    folder_path = os.path.join(data_root, folder_choice)
    files = list_dat_files(folder_path)
    if not os.path.isdir(folder_path):
        st.sidebar.error(f"Missing folder: {folder_path}")
    elif not files:
        st.sidebar.error("No .dat files found.")
    else:
        picks = st.sidebar.multiselect(
            "Select files", files, default=files[: min(6, len(files))]
        )
        selected_paths = [os.path.join(folder_path, f) for f in picks]


# -----------------------------
# Load selected tests
# -----------------------------
tests: List[TestData] = []
errors = []

for pth in selected_paths:
    try:
        tests.append(load_test(pth))
    except Exception as ex:
        errors.append((os.path.basename(pth), str(ex)))

if errors:
    st.warning("Some files failed to load:")
    st.write(errors)

if not tests:
    st.info("Select at least one .dat file from the sidebar.")
    st.stop()

oe_tests = [t for t in tests if t.kind == "OE"]
tmd_tests = [t for t in tests if t.kind == "TMD"]
tmu_tests = [t for t in tests if t.kind == "TMU"]


# -----------------------------
# Guidance panel
# -----------------------------
with st.expander("What to toggle ON/OFF + How to improve if fit is bad", expanded=True):
    st.markdown("### What you loaded")
    st.write(
        {
            "OE files": len(oe_tests),
            "TMD files": len(tmd_tests),
            "TMU files": len(tmu_tests),
        }
    )

    st.markdown("### Recommended toggles")
    if len(oe_tests) > 0:
        st.write("✅ Stage 1 ON (fits e0, λc, ξ from OE)")
    else:
        st.write("⚠️ No OE selected → Stage 1 can’t fit (e0, λc, ξ).")

    if len(tmd_tests) > 0:
        st.write("✅ TMD selected → Stage 2/3 can run.")
    else:
        st.write("❌ No TMD selected → Stage 2/3 should be OFF (TMD-only calibration).")

    st.markdown("### If OE fit looks off")
    st.write(
        "- Increase **p_min** (10→20 kPa) to ignore low-pressure noise.\n"
        "- Keep **Log-weight** ON.\n"
        "- One CSL curve may not match all OE tests if they start at different densities."
    )

    st.markdown("### If TMD fit is unstable / flat / poor")
    st.write(
        "- Increase **Substeps** (3→6 or 8).\n"
        "- Start with 1–2 TMD files + Stage 2 only, then add more.\n"
        "- If loss stays huge: reduce tests and confirm simulations aren’t failing."
    )

# Auto-disable stages if impossible
if len(tmd_tests) == 0:
    run_stage2 = False
    run_stage3 = False


# -----------------------------
# Run workflow
# -----------------------------
if run_btn:
    st.subheader("Loaded tests")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "file": t.name,
                    "kind": t.kind,
                    "n": len(t.eps1),
                    "eps1_end": float(t.eps1[-1]),
                }
                for t in tests
            ]
        ),
        use_container_width=True,
    )

    if (do_stage1 or run_stage2 or run_stage3) and minimize is None:
        st.error("SciPy not installed. Install: pip install scipy")
        st.stop()

    # Stage 1
    if do_stage1:
        st.subheader("Stage 1 (Direct measurement)")
        if len(oe_tests) > 0:
            csl = estimate_csl_from_oe(
                oe_tests,
                p_min_kpa=float(p_min_kpa),
                use_log_weight=bool(use_log_weight),
            )
            if csl is not None:
                params["e0_csl"] = float(csl["e0_csl"])
                params["lambda_c"] = float(csl["lambda_c"])
                params["xi"] = float(csl["xi"])
                st.write("CSL fit from OE:")
                st.json(csl)
            else:
                st.warning(
                    "OE fit failed. Try lowering p_min or select more OE points/files."
                )
        else:
            st.info("No OE files selected, skipping CSL fit.")

        if len(tmd_tests) > 0:
            M_est = estimate_M_from_tmd(tmd_tests)
            if M_est is not None:
                params["M"] = float(M_est)
                st.write(f"Estimated M from TMD tails: {M_est:.6f}")
            else:
                st.warning("Could not estimate M from selected TMD files.")
        else:
            st.info("No TMD files selected, skipping M estimation.")

    # Stage 2/3
    if run_stage2:
        st.subheader("Stage 2 optimization (TMD only)")
        with st.spinner("Optimizing h0, nb, Ad..."):
            params, res2 = optimize_params(
                params, tests, stage=2, n_substeps=n_substeps, w=w
            )
        st.json({k: params[k] for k in ["h0", "nb", "Ad"]})
        st.write(
            f"success={getattr(res2, 'success', None)} | loss={getattr(res2, 'fun', None)}"
        )

    if run_stage3:
        st.subheader("Stage 3 optimization (TMD only)")
        with st.spinner("Optimizing G0,h0,ch,nb,Ad,nd,zmax,cz..."):
            params, res3 = optimize_params(
                params, tests, stage=3, n_substeps=n_substeps, w=w
            )
        st.json(
            {k: params[k] for k in ["G0", "h0", "ch", "nb", "Ad", "nd", "zmax", "cz"]}
        )
        st.write(
            f"success={getattr(res3, 'success', None)} | loss={getattr(res3, 'fun', None)}"
        )

    # Overlays
    st.subheader("Overlays (Data scatter, Model line)")
    tabs = st.tabs([t.name for t in tests])
    for tab, t in zip(tabs, tests):
        with tab:
            simr = simulate(t, params, n_substeps)
            if t.kind == "OE":
                plot_oe(t, simr)
            else:
                if len(simr.eps1) < 3 or simr.p is None or simr.q is None:
                    st.warning(
                        "Simulation failed. Try increasing substeps or reduce number of tests."
                    )
                else:
                    plot_triaxial(t, simr)

    st.subheader("Final parameters")
    st.json(params)

else:
    st.info("Select files in the sidebar and click **Run**.")
