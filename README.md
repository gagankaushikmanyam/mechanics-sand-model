# SoilWorkbench  
### Calibration of Sand Using TU Braunschweig (KFSDB) Experimental Data

SoilWorkbench is a **Streamlit-based calibration workbench** for sand, aligned with the **TU Braunschweig / KFSDB experimental calibration workflow**.

The goal is to **reproduce the parameter identification steps exactly as described in the reference document**, while keeping the codebase clean and extensible for later upgrades (plasticity, hardening, return mapping).

---

## 1) Project Structure


SoilWorkbench/
├─ app.py
├─ README.md
├─ OE-all/
│  ├─ OE1.dat
│  └─ ...
├─ TMD-all/
│  ├─ TMD1.dat
│  └─ ...
├─ TMU-all/
│  ├─ TMU1.dat
│  └─ ...
└─ TMU-MT-AP-all/
   ├─ TMU_AP1.dat
   └─ ...


The app expects the dataset folders to exist under a common Root folder (selected in the Streamlit sidebar).

⸻

2) Experimental Datasets Supported

Test Type	Folder	Typical Observables	Use in this Phase
Oedometer (OE)	OE-all	(e), (p) (or (\sigma_v))	Compressibility parameters
Triaxial Drained (TMD)	TMD-all	(q(\varepsilon_1)), (p(\varepsilon_1)), (\varepsilon_v(\varepsilon_1))	Strength parameter (M) (per test)
Triaxial Undrained (TMU)	TMU-all	(q(\varepsilon_1)), (p(\varepsilon_1))	(M) (per test) validation
Triaxial Undrained (AP)	TMU-MT-AP-all	(q(\varepsilon_1)), (p(\varepsilon_1))	(M) (per test) validation

Important: In this phase, each test is treated independently, matching the document’s identification procedure (no global parameter sharing across tests).

⸻

3) Stress Variables and Notation

Mean and deviatoric stress are defined as:

[
p = \frac{1}{3}(\sigma_1 + 2\sigma_3),
\qquad
q = \sigma_1 - \sigma_3
]

Strains are used as engineering strains; input files are typically in percent and are converted to unit strain in the code.

⸻

4) Oedometer Calibration (OE)

Oedometer tests are used to identify compressibility parameters based on the void ratio–pressure relationship.

4.1 Compression Law (Piecewise)

The document-calibrated form is:

[
e(p) =
\begin{cases}
N - \lambda \ln(p) & p \ge p_c \
e(p_c) - \kappa \ln\left(\dfrac{p}{p_c}\right) & p < p_c
\end{cases}
]

Where:
	•	( \lambda ): slope of the normal compression line
	•	( \kappa ): slope of the swelling/recompression line
	•	( p_c ): preconsolidation pressure
	•	( N ): void ratio intercept at ( \ln(p)=0 )

4.2 Per-Test Fit (No Pooling)

Each OE file is calibrated independently:

[
\min_{\theta}\ |e_{\text{model}}(p;\theta) - e_{\text{data}}(p)|_2^2,
\qquad
\theta = (N,\lambda,\kappa,p_c)
]

Outputs are stored per-file and can be exported to CSV from the UI.

⸻

5) Triaxial Calibration (TMD, TMU, TMU-AP)

Triaxial tests are used to calibrate the critical state strength parameter per test.

5.1 Critical State Relation

[
q = M,p
]

Where (M) controls shear strength.

5.2 Drainage Conditions and Residuals

Test Type	Constraint	Plots/Residuals in this Phase
Drained (TMD)	(\dot{\varepsilon}_v \neq 0)	Fit (q-\varepsilon_1); show (q-p)
Undrained (TMU, AP)	(\dot{\varepsilon}_v = 0 \Rightarrow \dot{p}\neq 0)	Fit (q-\varepsilon_1) and (p-\varepsilon_1); show (q-p)

In this phase (document-consistent), the app focuses on direct parameter identification rather than constitutive integration.

⸻

6) Calibration Strategy

All calibrations are nonlinear least squares:

[
\min_{\theta}\ |r(\theta)|_2^2
]

Design principles (matching the document):
	•	Per-test calibration only
	•	No global parameter sharing
	•	No artificial regularization
	•	Direct correspondence with experimental procedure

⸻

7) Streamlit Application

7.1 Features
	•	Automatic dataset discovery from the Root folder
	•	Multi-folder selection (OE + TMD + TMU + AP in one run)
	•	Per-test calibration:
	•	OE: (e)–(\ln(p))
	•	Triaxial: (q)–(\varepsilon_1), (p)–(\varepsilon_1), (q)–(p)
	•	Drainage-aware fitting logic
	•	Scatter for data, line for model
	•	Export calibrated parameters to CSV

7.2 Run locally

pip install -r requirements.txt
streamlit run app.py


⸻

8) Recommended Workflow (Practical)

OE-only check
	•	Select OE files
	•	Run OE calibration only
	•	Inspect (e) vs (\ln(p)) and parameters (N,\lambda,\kappa,p_c)

TMD calibration
	•	Select 1–2 TMD files first
	•	Fit (M) per file
	•	Add more files after the first fits look stable

TMU / AP validation
	•	Select TMU/TMU-MT-AP files
	•	Fit/overlay (q(\varepsilon_1)) and (p(\varepsilon_1))
	•	Validate consistency of per-test (M)

⸻

9) Model Scope (Important)

This repository does not yet implement:
	•	Plastic flow rules
	•	Hardening evolution laws
	•	Return-mapping algorithms
	•	Consistent tangent operators

These are intentionally deferred to the next development phase to keep calibration steps:
	•	transparent,
	•	document-consistent,
	•	and easy to validate.

⸻

10) References
	•	TU Braunschweig – Institute of Geotechnical Engineering
	•	Torsten Wichtmann – KFSDB Sand Database
	•	Schofield & Wroth (1968) – Critical State Soil Mechanics

⸻

11) Disclaimer

This tool is intended for research and educational use only.
It is not a certified geotechnical design code.

⸻

🧠 app.py (document-consistent calibration)

What this version does:
	•	Fits exactly the parameters fitted in the reference procedure
	•	Calibrates per test, drainage-aware
	•	Uses scatter for data and line for model
	•	Keeps the UI and structure extensible for future constitutive upgrades

What it does NOT do (yet):
	•	Full MCC / DM04 plasticity
	•	Hardening evolution
	•	Return mapping
	•	Consistent tangent



