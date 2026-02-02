import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import least_squares

# ============================================================
# Streamlit config + style
# ============================================================
st.set_page_config(page_title="SoilWorkbench | TU BAF Calibrator", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.ribbon {background:#0b1220; border:1px solid #1f2937; padding:12px 14px; border-radius:14px; color:white;}
.ribbon b {font-size:1.05rem;}
.small {color:#94a3b8; font-size:0.9rem;}
.badge {display:inline-block; padding:2px 8px; border-radius:999px; background:#111827;
        border:1px solid #1f2937; color:#e5e7eb; font-size:0.82rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ribbon"><b>SoilWorkbench</b> &nbsp;|&nbsp; TU BAF / KFSDB Sand Calibration — OE (λ,κ,N,pc) + Triax (M)</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================
def safe_std(x):
    s = float(np.std(x))
    return s if s > 1e-9 else float(np.max(np.abs(x)) + 1.0)


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def as_float_array(x):
    return np.asarray(x, dtype=float)


# ============================================================
# Dataset discovery (root or direct folder)
# ============================================================
KNOWN_DATASET_DIRNAMES = {"OE-all", "TMD-all", "TMU-all", "TMU-MT-AP-all"}

st.markdown("## Dataset Root")
default_root = str(Path(__file__).parent)

dataset_path_str = st.text_input(
    "Paste ROOT folder containing OE-all / TMD-all / TMU-all / TMU-MT-AP-all (or paste a direct dataset folder path)",
    value=default_root,
)

RAW_PATH = Path(dataset_path_str).expanduser()
if RAW_PATH.name in KNOWN_DATASET_DIRNAMES:
    DATA_ROOT = RAW_PATH.parent.resolve()
else:
    DATA_ROOT = RAW_PATH.resolve()

CANDIDATE_ROOTS = [DATA_ROOT]
if DATA_ROOT.exists():
    if not (DATA_ROOT / "OE-all").exists() and not (DATA_ROOT / "TMU-all").exists():
        for sub in DATA_ROOT.iterdir():
            if sub.is_dir():
                CANDIDATE_ROOTS.append(sub.resolve())


def _find_dataset_dirs(candidate_roots):
    found = {}
    for root in candidate_roots:
        for name in KNOWN_DATASET_DIRNAMES:
            p = root / name
            if p.exists() and p.is_dir():
                found[name] = p
    return found


FOUND_DIRS = _find_dataset_dirs(CANDIDATE_ROOTS)

with st.expander("Debug: dataset discovery"):
    st.write("You pasted:", str(RAW_PATH))
    st.write("Resolved DATA_ROOT:", str(DATA_ROOT))
    st.write("Candidate roots searched:", [str(p) for p in CANDIDATE_ROOTS])
    st.write("Found dataset folders:", {k: str(v) for k, v in FOUND_DIRS.items()})
    for k, v in FOUND_DIRS.items():
        st.write(f"{k}: {len(list(v.rglob('*.dat')))} *.dat files")

DATASET_FOLDERS: Dict[str, Path] = {}
if "OE-all" in FOUND_DIRS:
    DATASET_FOLDERS["OE-all (Oedometer)"] = FOUND_DIRS["OE-all"]
if "TMD-all" in FOUND_DIRS:
    DATASET_FOLDERS["TMD-all (Drained triaxial)"] = FOUND_DIRS["TMD-all"]
if "TMU-all" in FOUND_DIRS:
    DATASET_FOLDERS["TMU-all (Undrained triaxial)"] = FOUND_DIRS["TMU-all"]
if "TMU-MT-AP-all" in FOUND_DIRS:
    DATASET_FOLDERS["TMU-MT-AP-all (Undrained AP)"] = FOUND_DIRS["TMU-MT-AP-all"]


# ============================================================
# Robust .dat reader
# ============================================================
def _find_numeric_start(lines: List[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if re.match(r"^[-+]?\d", s):
            return i
    return None


def read_dat_with_header(path: Path) -> Tuple[str, pd.DataFrame]:
    lines = path.read_text(errors="ignore").splitlines()
    start = _find_numeric_start(lines)
    if start is None:
        raise ValueError(f"No numeric rows found: {path.name}")

    header = "\n".join(lines[:start]).strip().lower()
    data = "\n".join(lines[start:])
    df = pd.read_csv(
        pd.io.common.StringIO(data), sep=r"\s+", header=None, engine="python"
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return header, df


# ============================================================
# Parse loaders
# ============================================================
def infer_tmu_layout(header_lower: str, df: pd.DataFrame) -> str:
    # Layout A: eps1, u, sigma3, sigma3', sigma1, sigma1', p, q
    # Layout B: eps1, sigma3, sigma3', sigma1, sigma1', u, p, q
    if "u" in header_lower:
        if re.search(r"eps1.*u.*sigma3", header_lower):
            return "A"
        if re.search(r"sigma1'.*u.*p", header_lower) or re.search(
            r"sigma1.*sigma1'.*u", header_lower
        ):
            return "B"
    if df.shape[1] < 8:
        return "A"
    c1 = df.iloc[:20, 1].astype(float).values
    c5 = df.iloc[:20, 5].astype(float).values
    return "A" if np.median(np.abs(c1)) > np.median(np.abs(c5)) else "B"


def load_all_tests_from_folders(selected_dirs: List[Path]):
    oe_tests: List[pd.DataFrame] = []
    triax_tests: List[Dict[str, Any]] = []
    file_log = []

    for folder in selected_dirs:
        folder = Path(folder)
        dat_files = sorted(folder.rglob("*.dat"))
        file_log.append((folder.name, len(dat_files)))

        for path in dat_files:
            name = path.name
            header, df = read_dat_with_header(path)

            # OE: sigma1 eps1 void ratio
            if name.startswith("OE"):
                if df.shape[1] < 3:
                    continue
                d = df.iloc[:, :3].copy()
                d.columns = ["sigma1", "eps1_pct", "e"]
                d = d[d["sigma1"] > 0].copy()
                d["p"] = d["sigma1"].astype(float)
                d["lnp"] = np.log(d["p"].values)
                d["e_obs"] = d["e"].astype(float)
                d["source"] = name
                d = (
                    d[["p", "lnp", "e_obs", "source"]]
                    .sort_values("p")
                    .reset_index(drop=True)
                )
                oe_tests.append(d)
                continue

            # TMD: eps1 epsv eps3 epsq voidratio q p eta
            if name.startswith("TMD"):
                if df.shape[1] < 8:
                    continue
                d = df.iloc[:, :8].copy()
                d.columns = [
                    "eps1_pct",
                    "epsv_pct",
                    "eps3_pct",
                    "epsq_pct",
                    "voidratio",
                    "q",
                    "p",
                    "eta",
                ]
                eps1 = d["eps1_pct"].astype(float).values / 100.0
                p_obs = d["p"].astype(float).values
                q_obs = d["q"].astype(float).values
                p0 = float(p_obs[0])
                sigma3_target = float(p_obs[0] - q_obs[0] / 3.0)

                triax_tests.append(
                    {
                        "test_id": name.replace(".dat", ""),
                        "source": name,
                        "family": "TMD (drained)",
                        "undrained": False,
                        "eps1": eps1,
                        "p_obs": p_obs,
                        "q_obs": q_obs,
                        "p0": p0,
                        "sigma3_target": sigma3_target,
                    }
                )
                continue

            # TMU / TMU-AP
            if name.startswith("TMU"):
                if df.shape[1] < 8:
                    continue
                layout = infer_tmu_layout(header, df)
                d = df.iloc[:, :8].copy()
                if layout == "A":
                    d.columns = [
                        "eps1_pct",
                        "u",
                        "sigma3",
                        "sigma3_eff",
                        "sigma1",
                        "sigma1_eff",
                        "p",
                        "q",
                    ]
                else:
                    d.columns = [
                        "eps1_pct",
                        "sigma3",
                        "sigma3_eff",
                        "sigma1",
                        "sigma1_eff",
                        "u",
                        "p",
                        "q",
                    ]

                eps1 = d["eps1_pct"].astype(float).values / 100.0
                p_obs = d["p"].astype(float).values
                q_obs = d["q"].astype(float).values
                p0 = (
                    float(d["sigma3_eff"].iloc[0])
                    if float(d["sigma3_eff"].iloc[0]) > 0
                    else float(p_obs[0])
                )

                fam = (
                    "TMU (undrained)"
                    if "TMU-all" in str(folder)
                    else "TMU-AP (undrained)"
                )
                triax_tests.append(
                    {
                        "test_id": name.replace(".dat", ""),
                        "source": name,
                        "family": fam,
                        "undrained": True,
                        "eps1": eps1,
                        "p_obs": p_obs,
                        "q_obs": q_obs,
                        "p0": p0,
                        "sigma3_target": None,
                        "tmu_layout": layout,
                    }
                )
                continue

    return oe_tests, triax_tests, file_log


# ============================================================
# Material model pieces (document-consistent calibration targets)
# ============================================================


# --- OE model (calibrate lambda, kappa, pc0, N per test) ---
def predict_e_piecewise(p, lam, kap, pc0, N):
    """
    e = N - lam ln(p) on NC line (p >= pc0)
    e = e(pc0) - kap ln(p/pc0) on swelling line (p < pc0)
    """
    p = np.asarray(p, dtype=float)
    pc0 = max(float(pc0), 1e-8)
    e_pc = N - lam * np.log(pc0)
    return np.where(p >= pc0, N - lam * np.log(p), e_pc - kap * np.log(p / pc0))


# --- Triax proxy using document target: critical state q = M p ---
# We need a prediction "path" in eps1 to compare q(eps1) and optionally p(eps1).
# For NOW: a stable monotone surrogate path that respects the CSL scaling via M*p0.
def triax_surrogate_predict(eps1, p0, M, undrained=False):
    """
    Stable monotone surrogate that respects q ~ M p-scale.
    - q increases towards M*p0
    - p changes mildly; in undrained it tends to deviate more (proxy)
    """
    eps1 = np.asarray(eps1, dtype=float)
    q = (M * p0) * (1.0 - np.exp(-15.0 * np.clip(eps1, 0.0, None)))
    if undrained:
        p = p0 * (1.0 - 0.18 * (1.0 - np.exp(-9.0 * np.clip(eps1, 0.0, None))))
    else:
        p = p0 * (1.0 - 0.08 * (1.0 - np.exp(-9.0 * np.clip(eps1, 0.0, None))))
    return p, q


# ============================================================
# Per-test calibration
# ============================================================
def fit_oe_test(d: pd.DataFrame, st_state: Dict[str, Any], max_nfev=80):
    p = as_float_array(d["p"].values)
    e = as_float_array(d["e_obs"].values)

    lam0 = float(st_state.get("lam0", 0.18))
    kap0 = float(st_state.get("kap0", 0.05))
    pc00 = float(np.clip(np.median(p), 1e-2, 1e9))
    N0 = float(st_state.get("N0", float(np.median(e) + 0.15)))

    x0 = np.array([lam0, kap0, pc00, N0], dtype=float)
    lb = np.array([0.01, 0.001, np.min(p) * 0.5, 0.2], dtype=float)
    ub = np.array([0.9, 0.5, np.max(p) * 3.0, 4.0], dtype=float)

    def res(x):
        lam, kap, pc0, N = map(float, x)
        if kap >= lam:
            return 1e3 * np.ones_like(e)
        ehat = predict_e_piecewise(p, lam, kap, pc0, N)
        return (ehat - e) / (safe_std(e) + 1e-6)

    out = least_squares(res, x0=x0, bounds=(lb, ub), max_nfev=int(max_nfev), verbose=0)
    lam, kap, pc0, N = out.x
    ehat = predict_e_piecewise(p, lam, kap, pc0, N)

    return {
        "source": d["source"].iloc[0],
        "lam": float(lam),
        "kap": float(kap),
        "pc0": float(pc0),
        "N": float(N),
        "rmse_e": rmse(e, ehat),
        "r2_e": r2(e, ehat),
        "nfev": int(out.nfev),
        "cost": float(out.cost),
    }


def fit_triax_test(t: Dict[str, Any], st_state: Dict[str, Any], max_nfev=80):
    eps1 = as_float_array(t["eps1"])
    p_obs = as_float_array(t["p_obs"])
    q_obs = as_float_array(t["q_obs"])
    p0 = float(t["p0"])
    undrained = bool(t["undrained"])

    M0 = float(st_state.get("M0", 1.2))
    x0 = np.array([M0], dtype=float)
    lb = np.array([0.2], dtype=float)
    ub = np.array([3.5], dtype=float)

    w_q = float(st_state.get("w_q", 1.0))
    w_p = float(st_state.get("w_p", 0.5))

    def res(x):
        M = float(x[0])
        p_hat, q_hat = triax_surrogate_predict(eps1, p0, M, undrained=undrained)
        r = [(q_hat - q_obs) * (w_q / safe_std(q_obs))]
        if undrained:
            r.append((p_hat - p_obs) * (w_p / safe_std(p_obs)))
        return np.concatenate(r)

    out = least_squares(res, x0=x0, bounds=(lb, ub), max_nfev=int(max_nfev), verbose=0)
    M = float(out.x[0])
    p_hat, q_hat = triax_surrogate_predict(eps1, p0, M, undrained=undrained)

    return {
        "test_id": t["test_id"],
        "source": t["source"],
        "family": t["family"],
        "undrained": undrained,
        "M": float(M),
        "rmse_q": rmse(q_obs, q_hat),
        "r2_q": r2(q_obs, q_hat),
        "rmse_p": rmse(p_obs, p_hat) if undrained else np.nan,
        "r2_p": r2(p_obs, p_hat) if undrained else np.nan,
        "nfev": int(out.nfev),
        "cost": float(out.cost),
        "p0": float(p0),
    }


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.markdown("## Dataset selection")
    selected_dirs: List[Path] = []

    if not DATASET_FOLDERS:
        st.error("No dataset folders found. Fix dataset root path above.")
    else:
        for label, folder in DATASET_FOLDERS.items():
            key = f"ds_{hash(str(Path(folder).resolve()))}"
            if st.checkbox(label, value=True, key=key):
                selected_dirs.append(Path(folder))

    st.divider()
    st.markdown("## Calibration controls (document targets)")
    st.caption("OE fits: λ, κ, pc0, N per test.  Triax fits: M per test.")

    M0 = st.number_input("Initial M", value=1.2)
    lam0 = st.number_input("Initial λ", value=0.18)
    kap0 = st.number_input("Initial κ", value=0.05)
    N0 = st.number_input("Initial N", value=1.1)

    w_q = st.slider("Weight q residual", 0.0, 5.0, 1.0, 0.1)
    w_p = st.slider("Weight p residual (undrained)", 0.0, 5.0, 0.5, 0.1)
    max_nfev = st.slider("Max optimizer steps (per test)", 20, 300, 80, 10)

    fit_all_btn = st.button(
        "▶ Fit ALL loaded tests", type="primary", use_container_width=True
    )

fit_defaults = {"M0": M0, "lam0": lam0, "kap0": kap0, "N0": N0, "w_q": w_q, "w_p": w_p}

# ============================================================
# Load datasets
# ============================================================
oe_tests, triax_tests, file_log = load_all_tests_from_folders(selected_dirs)

st.markdown("## Dataset discovery report")
st.write("Selected folders:", [d.name for d in selected_dirs])
st.write("Files found (recursive *.dat):", file_log)
st.write(f"OE tests parsed: {len(oe_tests)}")
st.write(f"Triax tests parsed: {len(triax_tests)}")

if len(oe_tests) == 0 and len(triax_tests) == 0:
    st.error("No datasets parsed. Check dataset root + selected folders.")
    st.stop()

# ============================================================
# Session state
# ============================================================
if "oe_fit_table" not in st.session_state:
    st.session_state["oe_fit_table"] = pd.DataFrame()
if "triax_fit_table" not in st.session_state:
    st.session_state["triax_fit_table"] = pd.DataFrame()

# ============================================================
# Fit all tests
# ============================================================
if fit_all_btn:
    oe_rows = [fit_oe_test(d, fit_defaults, max_nfev=max_nfev) for d in oe_tests]
    st.session_state["oe_fit_table"] = pd.DataFrame(oe_rows).sort_values("source")

    tr_rows = [fit_triax_test(t, fit_defaults, max_nfev=max_nfev) for t in triax_tests]
    st.session_state["triax_fit_table"] = pd.DataFrame(tr_rows).sort_values(
        ["family", "test_id"]
    )

    st.success("Per-test fitting complete (document targets).")

# ============================================================
# Results tables + downloads
# ============================================================
st.markdown("## Per-test fitted parameters")

cA, cB = st.columns(2)

with cA:
    st.markdown("### OE fits (λ, κ, pc0, N) per dataset")
    if not st.session_state["oe_fit_table"].empty:
        st.dataframe(st.session_state["oe_fit_table"], use_container_width=True)
        st.download_button(
            "Download OE fit table (CSV)",
            st.session_state["oe_fit_table"].to_csv(index=False).encode("utf-8"),
            "oe_fits.csv",
            "text/csv",
        )
    else:
        st.info(
            "No OE fits yet. Use **Fit ALL loaded tests** or fit one test in the viewer below."
        )

with cB:
    st.markdown("### Triax fits (M) per dataset")
    if not st.session_state["triax_fit_table"].empty:
        st.dataframe(st.session_state["triax_fit_table"], use_container_width=True)
        st.download_button(
            "Download triax fit table (CSV)",
            st.session_state["triax_fit_table"].to_csv(index=False).encode("utf-8"),
            "triax_fits.csv",
            "text/csv",
        )
    else:
        st.info(
            "No triax fits yet. Use **Fit ALL loaded tests** or fit one test in the viewer below."
        )

# ============================================================
# Viewer tabs
# ============================================================
st.markdown("## Test Viewer")
tab1, tab2 = st.tabs(["OE Viewer (e–ln(p))", "Triax Viewer (q–ε1 and q–p)"])

# ---------------- OE viewer ----------------
with tab1:
    if not oe_tests:
        st.warning("No OE tests loaded.")
    else:
        oe_names = [d["source"].iloc[0] for d in oe_tests]
        oe_map = {name: d for name, d in zip(oe_names, oe_tests)}

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            selected = st.selectbox(
                "Select OE dataset", oe_names, key="oe_select_single"
            )
        with c2:
            fit_this = st.button("Fit selected OE", key="fit_one_oe")
        with c3:
            show_all = st.checkbox(
                "Overlay all OE datasets (scatter)", value=False, key="oe_overlay_all"
            )

        d = oe_map[selected]

        if fit_this:
            row = fit_oe_test(d, fit_defaults, max_nfev=max_nfev)
            df_old = st.session_state["oe_fit_table"]
            df_new = pd.DataFrame([row])
            if df_old.empty:
                st.session_state["oe_fit_table"] = df_new
            else:
                df_old = df_old[df_old["source"] != row["source"]]
                st.session_state["oe_fit_table"] = pd.concat(
                    [df_old, df_new], ignore_index=True
                ).sort_values("source")
            st.success(f"Fitted {selected}")

        df_fit = st.session_state["oe_fit_table"]
        if not df_fit.empty and (df_fit["source"] == selected).any():
            r = df_fit[df_fit["source"] == selected].iloc[0]
            lam, kap, pc0, N = (
                float(r["lam"]),
                float(r["kap"]),
                float(r["pc0"]),
                float(r["N"]),
            )
        else:
            rtmp = fit_oe_test(d, fit_defaults, max_nfev=max(40, max_nfev // 2))
            lam, kap, pc0, N = (
                float(rtmp["lam"]),
                float(rtmp["kap"]),
                float(rtmp["pc0"]),
                float(rtmp["N"]),
            )

        p = as_float_array(d["p"].values)
        e = as_float_array(d["e_obs"].values)

        p_grid = np.geomspace(max(1e-3, p.min()), p.max(), 300)
        e_grid = predict_e_piecewise(p_grid, lam, kap, pc0, N)
        e_hat = predict_e_piecewise(p, lam, kap, pc0, N)

        fig = plt.figure(figsize=(8.6, 5.0))
        if show_all:
            for name in oe_names:
                dd = oe_map[name].sort_values("p").reset_index(drop=True)
                plt.scatter(dd["lnp"], dd["e_obs"], s=18, alpha=0.30)
        plt.scatter(d["lnp"], d["e_obs"], s=42, alpha=0.9, label=f"{selected} obs")
        plt.plot(np.log(p_grid), e_grid, "k--", linewidth=3.0, label="fit")
        plt.xlabel("ln(p)")
        plt.ylabel("void ratio e")
        plt.title(f"OE: {selected} | per-test fit (λ, κ, pc0, N)")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig, clear_figure=True)
        st.caption(
            f"{selected}  RMSE(e)={rmse(e, e_hat):.5f} | R²={r2(e, e_hat):.4f}  |  "
            f"λ={lam:.4f}, κ={kap:.4f}, pc0={pc0:.2f}, N={N:.4f}"
        )

# ---------------- Triax viewer ----------------
with tab2:
    if not triax_tests:
        st.warning("No triax tests loaded.")
    else:
        tr_names = [t["test_id"] for t in triax_tests]
        tr_map = {
            t["test_id"]: t for t in tr_names for t in []
        }  # placeholder to keep lint calm
        tr_map = {t["test_id"]: t for t in triax_tests}

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            selected = st.selectbox(
                "Select triax dataset", tr_names, key="triax_select_single"
            )
        with c2:
            fit_this = st.button("Fit selected triax (M)", key="fit_one_tr")
        with c3:
            overlay_family = st.checkbox(
                "Overlay same family", value=False, key="triax_overlay_family"
            )

        t = tr_map[selected]
        fam = t["family"]
        undr = bool(t["undrained"])

        if fit_this:
            row = fit_triax_test(t, fit_defaults, max_nfev=max_nfev)
            df_old = st.session_state["triax_fit_table"]
            df_new = pd.DataFrame([row])
            if df_old.empty:
                st.session_state["triax_fit_table"] = df_new
            else:
                df_old = df_old[df_old["test_id"] != row["test_id"]]
                st.session_state["triax_fit_table"] = pd.concat(
                    [df_old, df_new], ignore_index=True
                ).sort_values(["family", "test_id"])
            st.success(f"Fitted {selected}")

        df_fit = st.session_state["triax_fit_table"]
        if not df_fit.empty and (df_fit["test_id"] == selected).any():
            r = df_fit[df_fit["test_id"] == selected].iloc[0]
            M = float(r["M"])
        else:
            rtmp = fit_triax_test(t, fit_defaults, max_nfev=max(40, max_nfev // 2))
            M = float(rtmp["M"])

        eps1 = as_float_array(t["eps1"])
        p_obs = as_float_array(t["p_obs"])
        q_obs = as_float_array(t["q_obs"])
        p_hat, q_hat = triax_surrogate_predict(eps1, float(t["p0"]), M, undrained=undr)

        # Plot 1: q vs eps1
        fig1 = plt.figure(figsize=(8.6, 4.9))
        if overlay_family:
            for tt in triax_tests:
                if tt["family"] == fam:
                    plt.plot(tt["eps1"], tt["q_obs"], linewidth=1.0, alpha=0.18)
        plt.plot(eps1, q_obs, linewidth=2.2, label=f"{selected} obs")
        plt.plot(eps1, q_hat, "k--", linewidth=3.0, label=f"fit (M={M:.3f})")
        plt.xlabel("ε1 [-]")
        plt.ylabel("q [kPa]")
        plt.title(f"Triax: {selected} | q–ε1  ({fam})")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig1, clear_figure=True)

        # Plot 2: q vs p
        fig2 = plt.figure(figsize=(8.6, 4.9))
        if overlay_family:
            for tt in triax_tests:
                if tt["family"] == fam:
                    plt.plot(tt["p_obs"], tt["q_obs"], linewidth=1.0, alpha=0.18)
        plt.plot(p_obs, q_obs, linewidth=2.2, label=f"{selected} obs")
        plt.plot(p_hat, q_hat, "k--", linewidth=3.0, label="fit path (proxy)")
        plt.xlabel("p [kPa]")
        plt.ylabel("q [kPa]")
        plt.title(f"Triax: {selected} | q–p  ({'undrained' if undr else 'drained'})")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig2, clear_figure=True)

        cap = (
            f"{selected} | family={fam} | undrained={undr}  |  "
            f"RMSE(q)={rmse(q_obs,q_hat):.3f}, R²(q)={r2(q_obs,q_hat):.4f}"
        )
        if undr:
            cap += f"  |  RMSE(p)={rmse(p_obs,p_hat):.3f}, R²(p)={r2(p_obs,p_hat):.4f}"
        st.caption(cap)
