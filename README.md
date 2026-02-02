# SoilWorkbench  
### Calibration of Sand Using TU Braunschweig (KFSDB) Experimental Data

This repository provides a **Streamlit-based calibration workbench**
for sand, following the **TU Braunschweig / KFSDB calibration methodology**.

The objective is to **reproduce the parameter calibration exactly as described
in the reference document**, while providing a clean, extensible research tool.

---

## 1. Experimental Data

The following experimental datasets are supported:

| Test Type | Folder | Description |
|---------|--------|-------------|
| Oedometer | `OE-all` | Isotropic compression: void ratio vs pressure |
| Triaxial Drained | `TMD-all` | Shear with drained conditions |
| Triaxial Undrained | `TMU-all` | Shear with constant volume |
| Triaxial Undrained (AP) | `TMU-MT-AP-all` | Advanced pore-pressure tests |

Each dataset is **calibrated independently**, exactly as in the TU BAF procedure.

---

## 2. Stress Variables

Mean and deviatoric stress are defined as:

\[
p = \frac{1}{3}(\sigma_1 + 2\sigma_3), \qquad
q = \sigma_1 - \sigma_3
\]

---

## 3. Oedometer Calibration (OE)

The oedometer test is used to calibrate **compressibility parameters**.

### 3.1 Compression Law

\[
e(p) =
\begin{cases}
N - \lambda \ln(p) & p \ge p_c \\
e(p_c) - \kappa \ln\left(\dfrac{p}{p_c}\right) & p < p_c
\end{cases}
\]

### 3.2 Calibrated Parameters (per OE test)

- \( \lambda \) — slope of normal compression line  
- \( \kappa \) — slope of swelling line  
- \( p_c \) — preconsolidation pressure  
- \( N \) — void ratio at \( \ln(p) = 0 \)

Each OE dataset is fitted **independently**.

---

## 4. Triaxial Calibration (TMD, TMU, TMU-AP)

Triaxial tests are used to calibrate the **critical state parameter**.

### 4.1 Critical State Relation

\[
q = M \, p
\]

where:
- \( M \) controls shear strength
- \( M \) is calibrated **per triaxial test**

---

### 4.2 Drainage Conditions

| Test Type | Constraint |
|---------|-----------|
| Drained | \( \dot{\varepsilon}_v \neq 0 \) |
| Undrained | \( \dot{\varepsilon}_v = 0 \Rightarrow \dot{p} \neq 0 \) |

In this implementation:
- Drained tests fit **q–ε₁**
- Undrained tests fit **q–ε₁ and p–ε₁**

---

## 5. Calibration Strategy

Each dataset is calibrated using **nonlinear least squares**:

\[
\min_{\theta} \| r(\theta) \|_2^2
\]

- No global parameter sharing
- No artificial regularization
- Direct correspondence with experimental procedure

---

## 6. Streamlit Application

### Features
- Automatic dataset discovery
- Per-test calibration
- OE: \( e \)–\( \ln(p) \)
- Triaxial: \( q \)–\( \varepsilon_1 \), \( q \)–\( p \)
- Drainage-aware residuals
- Export of calibrated parameters (CSV)

Run locally:

```bash
streamlit run app.py


⸻

7. Model Scope (Important)

This repository does not yet implement:
	•	Plastic flow rules
	•	Hardening laws
	•	Return-mapping algorithms
	•	Consistent tangents

These are intentionally deferred to the next development phase.

⸻

8. References
	•	TU Braunschweig – Institute of Geotechnical Engineering
	•	Torsten Wichtmann – KFSDB Sand Database
	•	Schofield & Wroth (1968) – Critical State Soil Mechanics

⸻

9. Disclaimer

This tool is intended for research and educational use only.
It is not a certified geotechnical design code.

---

# 🧠 `app.py` (document-consistent calibration)

What this version **does**:
- Fits **exactly the parameters fitted in the document**
- Per-test, drainage-aware
- No invented physics
- Keeps your UI & extensibility

What it **does NOT** do (yet):
- Full MCC plasticity
- Hardening evolution
- Consistent tangent

👉 This is **correct engineering sequencing**.


---

