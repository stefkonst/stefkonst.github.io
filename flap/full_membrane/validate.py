"""Verification of membrane.py against analytical solutions, the older FE (../plagiopatagiales/model.py) and an independent solver.
Run: python validate.py -> validation/V_checks.png, validation/validation.json"""
import json, math, os, sys
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
OUT = os.path.join(HERE, "validation"); os.makedirs(OUT, exist_ok=True)
rep = {}
c1, t = M.YEOH[0], 70e-6


def square(L=0.04, h=2e-3, lam=1.03, lam_n=None, **kw):
    return M.Wing(panels=(0,), A=(0, 0), D=(0, L), C=(L, L), E=(2 * L, 0), T=(2 * L, L), theta_deg=0.0, h=h, lam_d=(lam, lam),
                  lam_n=(lam if lam_n is None else lam_n, None), w_s=(0.0, 0.0), t=t, **kw)


# V1: forces are the gradient of the relaxed energy (taut, wrinkled, with and without Maxwell)
def energy(w, x, Pe):
    xe = x[w.tri]
    F0, F1 = np.sum(xe * w.G[:, :, 0:1], 1), np.sum(xe * w.G[:, :, 1:2], 1)
    return float(np.sum(w.A0 * M.sheet_energy(np.sum(F0 * F0, 1), np.sum(F1 * F1, 1), np.sum(F0 * F1, 1), t, Pe)))


rng = np.random.default_rng(1)
errs = {}
for name, lam_n, V in (("taut, 0 V", 1.03, 0.0), ("taut, 800 V", 1.03, 800.0), ("wrinkled, 0 V", 0.96, 0.0), ("wrinkled, 1600 V", 0.96, 1600.0)):
    w = square(h=8e-3, lam_n=lam_n)
    x = w.x0 + rng.normal(0, 2e-5, w.x0.shape)
    Pe = w.P_e(V)
    f = w.membrane(x, Pe)
    d = rng.normal(0, 1, x.shape); d /= np.linalg.norm(d)
    e = 1e-8
    fd = (energy(w, x + e * d, Pe) - energy(w, x - e * d, Pe)) / (2 * e)
    errs[name] = abs(fd - np.sum(f * d)) / abs(fd)
rep["V1_force_vs_energy_relerr"] = errs

# V2: homogeneous states against closed forms
lams = np.linspace(1.005, 1.2, 30)
un, bi, un_a, bi_a = [], [], [], []
for l in lams:
    un.append(M.sheet_stress(np.array([l ** 2]), np.array([l ** -1.0]), np.array([0.0]), t, np.array(0.0))[4][0])
    bi.append(M.sheet_stress(np.array([l ** 2]), np.array([l ** 2]), np.array([0.0]), t, np.array(0.0))[4][0])
    un_a.append(2 * t * M.W1_of(l ** 2 + 2 / l) * (l ** 2 - 1 / l) * l ** -0.5)   # Cauchy resultant: thickness t l^-1/2
    bi_a.append(2 * t * M.W1_of(2 * l ** 2 + l ** -4) * (l ** 2 - l ** -4) * l ** -2)
rep["V2_uniaxial_maxerr"] = float(np.max(np.abs(np.array(un) / np.array(un_a) - 1)))
rep["V2_biaxial_maxerr"] = float(np.max(np.abs(np.array(bi) / np.array(bi_a) - 1)))
# Maxwell drop. Taut: the resultant falls by P lam_x lam_y (thinning). Wrinkled: against an independent root solve of n_y = 0,
# and P/2 in the small-strain, small-load limit (sigma_x = 3 G e_x - p/2).
from scipy.optimize import brentq
ev = lambda a, b, q: M.sheet_stress(np.array([a]), np.array([b]), np.array([0.0]), t, np.array(q))[4][0]
tt, wr_err = [], []
for l in (1.02, 1.05, 1.10):
    for P in (0.1, 0.3, 0.7):
        if M.sheet_stress(np.array([l ** 2]), np.array([l ** 2]), np.array([0.0]), t, np.array(P))[3][0] == 0:   # only while still taut
            tt.append(abs((ev(l ** 2, l ** 2, 0.0) - ev(l ** 2, l ** 2, P)) / (P * l ** 2) - 1))
        l1 = l ** 2
        I = lambda y: l1 + y + 1 / (l1 * y)
        y = brentq(lambda y: 2 * t * M.W1_of(I(y)) * (1 - 1 / (l1 * y ** 2)) - P * l1, 0.5, 3.0, xtol=1e-14)
        S1 = 2 * t * M.W1_of(I(y)) * (1 - 1 / (l1 ** 2 * y)) - P * y
        if S1 > 0:
            wr_err.append(abs(ev(l1, 0.8, P) / (S1 * l1 / math.sqrt(l1 * y)) - 1))
rep["V2_taut_drop_relerr"] = float(max(tt)); rep["V2_wrinkled_vs_root_solve_relerr"] = float(max(wr_err))
l = 1.0005; Psmall = 1e-4
rep["V2_drop_over_P_small_limit_expect_0.5"] = float((ev(l ** 2, l ** -1.0, 0.0) - ev(l ** 2, l ** -1.0, Psmall)) / Psmall)


# V3: uniform pressure on a pre-stressed square: Fourier series of T lap w = -p, mesh refinement, older FE
def fourier(L, T, p, n=99):
    s = 0.0
    for i in range(1, n, 2):
        for j in range(1, n, 2):
            s += (-1) ** ((i + j) // 2 - 1) / (i * j * (i * i + j * j))
    return 16 * p * L * L / (math.pi ** 4 * T) * s


L, lam, p_ = 0.04, 1.03, 2.0
Tn = ev(lam ** 2, lam ** 2, 0.0)
hs, wm = [5e-3, 2.5e-3, 1.25e-3], []
for h in hs:
    w = square(L=L, h=h, lam=lam, fix_te=True)
    r = w.relax(0.0, uniform_p=p_, tol=2e-9, max_iter=150000)
    wm.append(float(np.abs(r["x"][:, 2]).max()))
wf = fourier(L, Tn, p_)
rep["V3_fourier_w_mm"] = wf * 1e3; rep["V3_fe_w_mm"] = [q * 1e3 for q in wm]; rep["V3_err_finest"] = abs(wm[-1] / wf - 1)
import model as OLD
o = OLD.Wing(span=L, chord=L, le_mode="rigid", clamp_LE=True, clamp_TE=True, clamp_root=True, clamp_tip=True, material="P7670", t_mem=t, pre_span=lam,
             pre_chord=lam, n_dea_chord=0, n_dea_span=0, te_cord=False, p_uniform=p_, nx=32, ny=32, max_iter=120000, tol=1e-7, ramp=50)
ro = o.solve(0.0)
wo = float(np.abs(ro[0][:, 2]).max())
rep["V3_older_FE_w_mm"] = wo * 1e3; rep["V3_vs_older_FE"] = abs(wm[-1] / wo - 1)

# V4a: cable alone under a known line load: parabola, sag = q S^2 / (8 F_x). The sheet is made negligible.
S, sag_fe, sag_an = 0.04, [], []
qs = [0.5, 1.0, 2.0, 4.0]
for q_ in qs:
    w = M.Wing(panels=(0,), A=(0, 0), D=(0, S), C=(S, S), E=(2 * S, 0), T=(2 * S, S), theta_deg=0.0, h=2.5e-3, lam_d=(1.02, 1.02), lam_n=(1.02, None),
               w_s=(0.0, 0.0), t=t, cord_EA=(2.0, 2.0), cord_pre=(0.1, 0.1))
    ids = w.fib[0]
    fx = np.ones(w.x0.shape[0], bool); fx[ids[1:-1]] = False
    w.f_ext = np.zeros_like(w.x0)
    hx = np.gradient(w.x0[ids, 0])
    w.f_ext[ids, 1] = -q_ * hx
    w.A0[:] = 0.0                                              # no sheet at all, so the cable is tested alone
    r = w.relax(0.0, fixed=fx, tol=1e-8, max_iter=100000)
    f = w.fibre_report(r["x"])[0]
    xc = r["x"][ids]
    Fx = 2.0 * (np.linalg.norm(np.diff(xc, axis=0), axis=1)[0] / w.bL0[0] / 1.02 - 1 + 0.1) * (np.diff(xc[:2, 0])[0] / np.linalg.norm(np.diff(xc[:2], axis=0)))
    sag_fe.append(f["sag"]); sag_an.append(q_ * S ** 2 / (8 * Fx))
rep["V4a_cable_sag_mm_fe"] = [q * 1e3 for q in sag_fe]; rep["V4a_cable_sag_mm_exact"] = [q * 1e3 for q in sag_an]
rep["V4a_maxerr"] = float(np.max(np.abs(np.array(sag_fe) / np.array(sag_an) - 1)))
# V4b: sheet on a stiff cable: sag tends to S^2 N / (8 F) as F grows; a softer cable sags less than this because the sheet relaxes
N0 = ev(1.02 ** 2, 1.02 ** 2, 0.0)
Fs, ratio = [0.1, 0.2, 0.4, 0.8], []   # above about 1 N the explicit solver goes unstable (stated on the page)
for F in Fs:
    w = M.Wing(panels=(0,), A=(0, 0), D=(0, S), C=(S, S), E=(2 * S, 0), T=(2 * S, S), theta_deg=0.0, h=1.25e-3, lam_d=(1.02, 1.02), lam_n=(1.02, None),
               w_s=(0.0, 0.0), t=t, cord_EA=(5.0, 5.0), cord_pre=(F / 5.0, F / 5.0))
    r = w.relax(0.0, tol=5e-8, max_iter=150000)
    f = w.fibre_report(r["x"])[0]
    ratio.append(f["sag"] / (S ** 2 * N0 / (8 * f["F"])))
rep["V4b_F_N"] = Fs; rep["V4b_sag_over_arc"] = ratio

# V5: independent solver: L-BFGS on the total potential energy, large deflection; dynamic relaxation must match
w = square(L=L, h=4e-3, lam=1.005, fix_te=True)
pp = 20.0
free = ~w.fixed
Pe = w.P_e(0.0)
x0 = w.x0.copy()


def tot(u):
    x = x0.copy(); x[free] = u.reshape(-1, 3)
    xe = x[w.tri]
    n = np.cross(xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0]); sg = np.sign(n[:, 2] + 1e-30)
    vol = np.sum(sg * np.einsum("ij,ij->i", xe[:, 0], np.cross(xe[:, 1], xe[:, 2]))) / 6
    g = w._rhs(x, Pe, 0.0, np.full(len(w.tri), pp))
    return energy(w, x, Pe) - pp * vol, -g[free].ravel()


u0 = x0[free].copy(); u0[:, 2] = 1e-4 * np.sin(np.pi * u0[:, 0] / L) * np.sin(np.pi * u0[:, 1] / L)
res = minimize(tot, u0.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=60000, maxfun=80000, ftol=1e-16, gtol=1e-9))
xb = x0.copy(); xb[free] = res.x.reshape(-1, 3)
rd = w.relax(0.0, uniform_p=pp, tol=1e-8, max_iter=150000)
rep["V5_w_DR_mm"] = float(np.abs(rd["x"][:, 2]).max() * 1e3); rep["V5_w_LBFGS_mm"] = float(np.abs(xb[:, 2]).max() * 1e3)
rep["V5_relerr"] = abs(rep["V5_w_DR_mm"] / rep["V5_w_LBFGS_mm"] - 1)

# V6: vortex lattice: rectangular wings (Helmbold) and a parabolic arc (thin airfoil, finite-wing factor)
ARs, cla, hel = [1, 2, 4, 8, 16, 64], [], []
for AR in ARs:
    c = 0.05; b = AR * c / 2
    xs = b * (1 - np.cos(np.linspace(0, math.pi / 2, 41)))
    v = M.VLM(xs, np.zeros(41), np.full(41, c), 16, True)
    cla.append(v.CL(v.solve(1.0, 0.01)) / 0.01)
    hel.append(2 * math.pi * AR / (2 + math.sqrt(4 + AR ** 2)))
rep["V6_CLa_vlm"] = cla; rep["V6_CLa_helmbold"] = hel
rep["V6_maxerr"] = float(np.max(np.abs(np.array(cla) / np.array(hel) - 1)))
c = 0.05; AR = 40; b = AR * c / 2
xs = b * (1 - np.cos(np.linspace(0, math.pi / 2, 41)))
v = M.VLM(xs, np.zeros(41), np.full(41, c), 16, True)
hc = 0.03
sl = 4 * hc * (1 - 2 * v.yc / c)
rep["V6_camber_CL_vlm"] = float(v.CL(v.solve(1.0, 0.0, sl))); rep["V6_camber_CL_thin_airfoil_finite"] = 4 * math.pi * hc * AR / (AR + 2)

# V7: frame against the closed form (tip load on the LE rod)
w = M.Wing(h=2.5e-3)
fk, nodes, le = M.frame_K(w)
f = np.zeros_like(w.x0); f[le[-1], 2] = 0.01
u = fk(f)
Lrod = w.x0[le[-1], 0] - w.x0[le[0], 0]
EI = 120e9 * math.pi * (3e-3 ** 4 - 2e-3 ** 4) / 64
rep["V7_tip_mm"] = float(u[le[-1], 2] * 1e3); rep["V7_tip_analytic_mm"] = 0.01 * Lrod ** 3 / (3 * EI) * 1e3
rep["V7_relerr"] = abs(rep["V7_tip_mm"] / rep["V7_tip_analytic_mm"] - 1)

json.dump(rep, open(os.path.join(OUT, "validation.json"), "w"), indent=1, default=float)
for k, v_ in rep.items():
    print(k, v_)
fig, ax = plt.subplots(2, 3, figsize=(11, 6.4))
ax[0, 0].plot(lams, un_a, "k-", label="uniaxial, closed form"); ax[0, 0].plot(lams[::3], un[::3], "o", label="code")
ax[0, 0].plot(lams, bi_a, "k--", label="equibiaxial, closed form"); ax[0, 0].plot(lams[::3], bi[::3], "s", label="code")
ax[0, 0].set(xlabel=r"Stretch $\lambda$", ylabel=r"Resultant $N$ in N/m"); ax[0, 0].legend()
ax[0, 1].semilogx(hs, np.array(wm) * 1e3, "o-", label="this model"); ax[0, 1].axhline(wf * 1e3, color="k", label="Fourier series"); ax[0, 1].axhline(wo * 1e3, color="k", ls=":", label="older FE, 32 x 32")
ax[0, 1].set(xlabel="Mesh size $h$ in m", ylabel=r"Peak deflection $w$ in mm"); ax[0, 1].legend()
ax[0, 2].plot(np.array(sag_an) * 1e3, np.array(sag_fe) * 1e3, "o", label="Cable alone"); m_ = max(sag_an) * 1e3; ax[0, 2].plot([0, m_], [0, m_], "k-", label="Exact")
ax[0, 2].set(xlabel=r"$qS^2/8F_x$ in mm", ylabel="Edge sag in the FE, in mm"); ax[0, 2].legend()
ax[1, 0].plot(xb[:, 0] * 1e3, rd["x"][:, 2] * 1e3, ".", ms=3, label="dynamic relaxation"); ax[1, 0].plot(xb[:, 0] * 1e3, xb[:, 2] * 1e3, "+", ms=4, label="L-BFGS on the energy")
ax[1, 0].set(xlabel="$x$ in mm", ylabel=r"$w$ in mm"); ax[1, 0].legend()
ax[1, 1].plot(ARs, hel, "k-", label="Helmbold"); ax[1, 1].plot(ARs, cla, "o", label="vortex lattice")
ax[1, 1].set(xlabel="Aspect ratio", ylabel=r"$C_{L\alpha}$ per rad"); ax[1, 1].legend()
ks = list(errs); ax[1, 2].bar(range(4), [max(errs[k], 1e-12) for k in ks], color=PS.MATLAB[0]); ax[1, 2].set_yscale("log"); ax[1, 2].set_xticks(range(4)); ax[1, 2].set_xticklabels(ks, rotation=20, fontsize=7)
ax[1, 2].set(ylabel="Force vs energy gradient, relative error")
PS.finish(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "V_checks.png")); plt.close(fig)
