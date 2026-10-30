"""Calibrate the constitutive constants used by model.py from measured uniaxial data.

Input: paper_tension/fig1_membrane/results/unloading_stress_stretch.csv
(engineering stress = force / (t0 w0), unloading branch, cycles 2-4, mean).

Strain energy per reference volume (incompressible):
    W = C10 (I1-3) + C20 (I1-3)^2 + C30 (I1-3)^3                      matrix (Yeoh)
      + sum_d [ k1_d (I4_d-1)^2 + k2_d (I4_d-1)^3 ]  for I4_d > 1       fibre families (tension only)
In a uniaxial test along fibre direction d the transverse fibres are compressed (inactive), so the
lateral stretch is lambda^-1/2 and the engineering stress is
    P = 2 (lam - lam^-2) W1 + 2 lam W4,   W1 = dW/dI1, W4 = dW/dI4 at I4 = lam^2
which is linear in the constants -> non-negative least squares (keeps W convex in the fitted range).
Run: python calibrate.py  (prints the dict pasted into model.py)
"""
import numpy as np, pandas as pd
from scipy.optimize import nnls

CSV = r"C:/Users/stefk/Documents/paper_tension/fig1_membrane/results/unloading_stress_stretch.csv"
d = pd.read_csv(CSV)
lam = d["stretch"].to_numpy()


def basis_matrix(l):
    I1 = l ** 2 + 2 / l
    k = 2 * (l - l ** -2)
    return np.c_[k, k * 2 * (I1 - 3), k * 3 * (I1 - 3) ** 2]


def basis_fibre(l):
    e = l ** 2 - 1
    return np.c_[2 * l * 2 * e, 2 * l * 3 * e ** 2]


LAM_MAX = 1.22   # chordwise curve locks up above ~1.22 (PET lock-up, one point at 105 kPa): fit range limited


def clean(col, lmin=1.0):
    y = d[col].to_numpy() * 1e3
    m = np.isfinite(y) & (lam >= lmin) & (lam <= LAM_MAX)
    return lam[m], y[m]


out = {}
# isotropic materials: Yeoh only
for name, col, lmin in [("P6760", "Wacker P6760 mean_kPa", 1.03), ("DS20", "DragonSkin 20 mean_kPa", 1.0)]:
    l, y = clean(col, lmin)
    c, _ = nnls(basis_matrix(l), y)
    r = basis_matrix(l) @ c - y
    out[name] = dict(C=c.tolist(), rms_kPa=float(np.sqrt(np.mean(r ** 2)) / 1e3), lam_max=float(l.max()))

# anisotropic membrane: matrix = matrix volume fraction (0.577, fig1 optimised_volume_fractions.csv) x P6760 fit,
# so the matrix is the measured silicone; chordwise and spanwise fibre families fitted to the remaining stress.
V_M = 0.577
Cm = np.array(out["P6760"]["C"]) * V_M
res = dict(C=Cm.tolist())
for key, col in [("k_chord", "Membrane, chordwise mean_kPa"), ("k_span", "Membrane, spanwise mean_kPa")]:
    l, y = clean(col)
    k, _ = nnls(basis_fibre(l), y - basis_matrix(l) @ Cm)
    r = basis_fibre(l) @ k + basis_matrix(l) @ Cm - y
    res[key] = k.tolist(); res["rms_kPa_" + key[2:]] = float(np.sqrt(np.mean(r ** 2)) / 1e3)
res["lam_max"] = LAM_MAX
out["aniso"] = res
for k, v in out.items():
    print(k, v)
