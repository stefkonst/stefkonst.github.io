"""Validation of model.py against measurements and analytical solutions. Writes validation/*.png and validation.json.

V1  constitutive fit vs measured uniaxial curves (fig1 data)
V2  FE patch test: homogeneous biaxial pre-stretch -> exact analytical membrane tensions
V3  DEA force drop in the FE vs analytical Maxwell expression, and vs bench blocked force (fig2, 5 fibres)
V4  pre-stressed isotropic membrane under small uniform pressure vs Fourier series (Poisson equation), mesh study
V5  DEA authority: orthotropic membrane + 4 chordwise DEAs under uniform pressure vs smeared-tension series (on/off)
V6  aerodynamics: Glauert numerics (flat plate, parabolic camber) and FE aeroelastic chord vs independent 1D string
V7  global equilibrium (reactions = applied load) and mesh convergence of lift and authority (rigid and rod LE)
V8  leading-edge rod: discrete bending operator vs Euler-Bernoulli cantilever (uniform and tapered)
Run: python validate.py
"""
import json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M
import plotstyle as PS
import matplotlib.pyplot as plt

OUT = os.path.join(HERE, "validation")
os.makedirs(OUT, exist_ok=True)
FIG1 = r"C:/Users/stefk/Documents/paper_tension/fig1_membrane/results/unloading_stress_stretch.csv"
FIG2 = r"C:/Users/stefk/Documents/paper_tension/fig2_dea_actuator/results/dea_isometric_summary.csv"
PS.apply()
INK, C1, C2, C3 = "k", PS.MATLAB[0], PS.MATLAB[1], PS.MATLAB[4]
RIGID = dict(le_mode="rigid", clamp_tip=True)                         # rigid frame for the analytical checks
report = {}


def save(fig, name):
    PS.finish(fig.axes)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


def W1(I1, C):
    e = I1 - 3
    return C[0] + 2 * C[1] * e + 3 * C[2] * e ** 2


def W4(I4, Ef):
    r = np.sqrt(I4)
    return np.where(r > 1, Ef * (r - 1) / (2 * r), 0.0)


def Cm(mat):
    return tuple(mat["v_m"] * c for c in mat["C"])


# ---------------- V1 ----------------
d = pd.read_csv(FIG1)
lam = d["stretch"].to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
rows = []
for name, col, key, k_attr, colr in [("Membrane, chordwise", "Membrane, chordwise mean_kPa", "aniso", "Ef_chord", C2),
                                     ("Membrane, spanwise", "Membrane, spanwise mean_kPa", "aniso", "Ef_span", C1),
                                     ("Wacker P6760", "Wacker P6760 mean_kPa", "P6760", "Ef_chord", INK),
                                     ("DragonSkin 20", "DragonSkin 20 mean_kPa", "DS20", "Ef_chord", C3)]:
    mat = M.MATERIALS[key]
    l = np.linspace(1, 1.3, 200)
    P = 2 * (l - l ** -2) * W1(l ** 2 + 2 / l, Cm(mat)) + 2 * l * W4(l ** 2, mat[k_attr])
    y = d[col].to_numpy()
    m = np.isfinite(y) & (lam <= M.LAM_CAL)
    Pm = np.interp(lam[m], l, P) / 1e3
    rms = float(np.sqrt(np.mean((Pm - y[m]) ** 2)))
    rows.append(dict(material=name, rms_kPa=rms))
    a = ax[0] if "Membrane" in name else ax[1]
    a.plot(lam, y, "o", ms=2.5, color=colr, alpha=0.5)
    a.plot(l, P / 1e3, "-", color=colr, label=f"{name} (rms {rms:.2f} kPa)")
for a in ax:
    a.axvline(M.LAM_CAL, color="0.6", lw=0.8, ls="--")
    a.set_xlabel(r"Stretch $\lambda$"); a.set_ylabel(r"Nominal stress $s$ in kPa"); a.legend(fontsize=7)
save(fig, "V1_constitutive.png")
report["V1"] = rows

# ---------------- V2 ----------------
res = []
for ls, lc, mat in [(1.03, 1.03, "aniso"), (1.10, 1.02, "aniso"), (1.05, 1.15, "aniso"), (1.08, 1.08, "P6760"), (1.15, 1.05, "DS20")]:
    w = M.Wing(material=mat, pre_span=ls, pre_chord=lc, n_dea_chord=0, te_cord=False, clamp_TE=True, U=0.0, nx=8, ny=4, **RIGID)
    x, info = w.solve(0.0)
    r = w.evaluate(x, 0.0, info)
    m = M.MATERIALS[mat]
    i1 = ls ** 2 + lc ** 2 + 1 / (ls * lc) ** 2
    t = 180e-6 / (ls * lc)
    Ts = (2 * W1(i1, Cm(m)) * (ls ** 2 - 1 / (ls * lc) ** 2) + 2 * W4(ls ** 2, m["Ef_span"]) * ls ** 2) * t
    Tc = (2 * W1(i1, Cm(m)) * (lc ** 2 - 1 / (ls * lc) ** 2) + 2 * W4(lc ** 2, m["Ef_chord"]) * lc ** 2) * t
    res.append(dict(material=mat, pre_span=ls, pre_chord=lc, T_span_FE=r["T_span"], T_span_exact=Ts,
                    T_chord_FE=r["T_chord"], T_chord_exact=Tc,
                    err_pct=100 * max(abs(r["T_span"] / Ts - 1), abs(r["T_chord"] / Tc - 1))))
report["V2"] = res

# ---------------- V3 ----------------
Vs = np.arange(0, 1801, 200.0)
fe_drop = []
base = dict(n_dea_chord=4, U=0.0, clamp_TE=True, te_cord=False, nx=12, ny=6, **RIGID)
w0 = M.Wing(V=0, **base)
x0, i0 = w0.solve(0.0)
F0 = w0.evaluate(x0, 0.0, i0)["dea_force_mean_mN"]
for V in Vs:
    x, i = w0.solve(V, x_init=x0, ramp_steps=0)
    fe_drop.append(F0 - w0.evaluate(x, V, i)["dea_force_mean_mN"])
p = M.DEFAULTS
A0 = p["dea_width"] * (p["t_act"] + p["t_pass"])
vact = p["t_act"] / (p["t_act"] + p["t_pass"])
ana = p["eta"] * vact * M.EPS0 * p["eps_r"] * (Vs / p["t_act"]) ** 2 * A0 * 1e3
bench = pd.read_csv(FIG2)
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
ax[0].plot(Vs, ana, "-", color=INK, label=r"analytical $\eta\chi\varepsilon_0\varepsilon_r (V/H)^2 A_0$")
ax[0].plot(Vs, fe_drop, "o", color=C2, ms=4, mfc="none", label="FE (membrane clamped on all edges)")
ax[0].set_xlabel(r"Voltage $V$ in V"); ax[0].set_ylabel(r"Force drop per DEA $\Delta F$ in mN"); ax[0].legend(fontsize=7)
for clamp, mk in [(0, "o"), (38, "s")]:
    b = bench[bench.clamp_mm == clamp]
    ax[1].plot(b.V, b.dF_pp_mN, mk, color=C1, ms=4, mfc="none", label=f"bench, 5 rolled DEAs (clamp {clamp} mm)")
ax[1].plot(Vs, 5 * ana, "-", color=INK, label=r"model, 5 DEAs, $\eta$ = 0.40")
ax[1].set_xlabel(r"Voltage $V$ in V"); ax[1].set_ylabel(r"Blocked force change in mN"); ax[1].legend(fontsize=7)
save(fig, "V3_dea.png")
report["V3"] = dict(max_err_pct=float(np.max(np.abs(np.array(fe_drop[1:]) / ana[1:] - 1)) * 100),
                    dF_1600V_mN=float(ana[Vs == 1600][0]))


# ---------------- V4 ----------------
def series(Tx, Ty, pr, b, c, X, Y, nmax=101):
    X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
    w = np.zeros(X.shape)
    for m_ in range(1, nmax, 2):
        for n_ in range(1, nmax, 2):
            w += 16 * pr / (math.pi ** 2 * m_ * n_ * (Tx * (m_ * math.pi / b) ** 2 + Ty * (n_ * math.pi / c) ** 2)) * \
                 np.sin(m_ * math.pi * X / b) * np.sin(n_ * math.pi * Y / c)
    return w


pr = 0.3
mesh = []
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
for nx, ny in [(8, 4), (12, 6), (20, 10), (32, 16)]:
    w = M.Wing(material="P6760", pre_span=1.05, pre_chord=1.05, n_dea_chord=0, te_cord=False, clamp_TE=True,
               p_uniform=pr, nx=nx, ny=ny, tol=1e-7, max_iter=80000, **RIGID)
    x, info = w.solve(0.0)
    r = w.evaluate(x, 0.0, info)
    T = 0.5 * (r["T_span"] + r["T_chord"])
    z = np.array(r["z_mm"]) / 1e3
    i = int(np.argmin(np.abs(w.xs - 0.035)))
    wex = series(T, T, pr, 0.07, 0.035, w.xs[i], w.ys)
    mesh.append(dict(mesh=f"{nx}x{ny}", w_centre_FE_um=float(z[i, ny // 2] * 1e6), w_centre_exact_um=float(wex[ny // 2] * 1e6)))
    ax[0].plot(w.ys * 1e3, z[i] * 1e6, "o-", ms=3, lw=0.8, label=f"FE {nx}x{ny}")
yy = np.linspace(0, 0.035, 200)
ax[0].plot(yy * 1e3, series(T, T, pr, 0.07, 0.035, 0.035, yy) * 1e6, "k-", lw=1.2, label="Fourier series")
ax[0].set_xlabel(r"Chord $y$ in mm"); ax[0].set_ylabel(r"Deflection $w$ at mid-span in µm"); ax[0].legend(fontsize=7)
ax[1].plot([m_["mesh"] for m_ in mesh], [100 * (m_["w_centre_FE_um"] / m_["w_centre_exact_um"] - 1) for m_ in mesh], "o-", color=C2)
ax[1].axhline(0, color="0.6", lw=0.8); ax[1].set_xlabel("Mesh"); ax[1].set_ylabel(r"Centre deflection error in %")
save(fig, "V4_membrane_pressure.png")
report["V4"] = mesh

# ---------------- V5 ----------------
def ritz_tendons(Tx, Ty, F, xk, pr, b, c, X, Y, M=61, Nn=61):
    """Independent semi-analytical solution: orthotropic membrane (Tx, Ty) + discrete chordwise tendons of tension F
    at span positions xk, all edges w = 0, uniform pressure. Energy: Tx w_x^2 + Ty w_y^2 + sum_k F w_y^2 delta(x-xk)
    with w = sum a_mn sin(m pi x/b) sin(n pi y/c); modes decouple in n."""
    X, Y = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float))
    w = np.zeros(X.shape)
    ms = np.arange(1, M + 1)
    Sk = np.sin(np.outer(ms, np.asarray(xk)) * math.pi / b)          # (M, K)
    for n in range(1, Nn + 1, 2):
        kn = n * math.pi / c
        K = np.diag(b * c / 4 * (Tx * (ms * math.pi / b) ** 2 + Ty * kn ** 2)) + F * kn ** 2 * (c / 2) * Sk @ Sk.T
        f = pr * (b / (ms * math.pi)) * (1 - np.cos(ms * math.pi)) * (c / (n * math.pi)) * 2
        a = np.linalg.solve(K, f)
        w += np.sin(np.multiply.outer(X, ms) * math.pi / b) @ a * np.sin(kn * Y)
    return w


pr = 0.3
cfg = dict(material="aniso", n_dea_chord=4, clamp_TE=True, te_cord=False, p_uniform=pr, nx=24, ny=12, V=1600.0, tol=1e-7, max_iter=80000, **RIGID)
w = M.Wing(**cfg)
xo, io = w.solve(0.0); ro = w.evaluate(xo, 0.0, io)
xn, inn = w.solve(1600.0, x_init=xo, ramp_steps=0); rn = w.evaluate(xn, 1600.0, inn)
zo, zn = np.array(ro["z_mm"]) / 1e3, np.array(rn["z_mm"]) / 1e3
XX, YY = np.meshgrid(w.xs, w.ys, indexing="ij")
# all analytical inputs: membrane tensions from the exact biaxial state (V2), DEA force from the Yeoh bar at its pre-stretch
m = dict(M.MATERIALS["aniso"]); m["C"] = Cm(m); ls = lc = 1.03
i1 = ls ** 2 + lc ** 2 + 1 / (ls * lc) ** 2; tcur = 180e-6 / (ls * lc)
Tx = (2 * W1(i1, m["C"]) * (ls ** 2 - 1 / (ls * lc) ** 2) + 2 * W4(ls ** 2, m["Ef_span"]) * ls ** 2) * tcur
Ty = (2 * W1(i1, m["C"]) * (lc ** 2 - 1 / (ls * lc) ** 2) + 2 * W4(lc ** 2, m["Ef_chord"]) * lc ** 2) * tcur
ld = M.DEFAULTS["pre_dea"]
F_off = A0 * 2 * (ld - ld ** -2) * W1(ld ** 2 + 2 / ld, M.DEA_YEOH)
dF1 = float(ana[Vs == 1600][0]) / 1e3
F_on = F_off - dF1
xk = [w.xs[k] for k in w.dea_lines["chord"]]
an = {"off": ritz_tendons(Tx, Ty, F_off, xk, pr, 0.07, 0.035, XX, YY),
      "on": ritz_tendons(Tx, Ty, F_on, xk, pr, 0.07, 0.035, XX, YY)}
nk = len(xk)
sm = {"off": series(Tx, Ty + nk * F_off / 0.07, pr, 0.07, 0.035, XX, YY, nmax=81),
      "on": series(Tx, Ty + nk * F_on / 0.07, pr, 0.07, 0.035, XX, YY, nmax=81)}
mean = lambda f: float(np.mean(f))
report["V5"] = dict(
    T_membrane_span_analytic=Tx, T_membrane_chord_analytic=Ty, F_dea_off_analytic_mN=F_off * 1e3, dF_dea_mN=dF1 * 1e3,
    F_dea_off_FE_mN=ro["dea_force_mean_mN"], F_dea_on_FE_mN=rn["dea_force_mean_mN"],
    mean_w_FE_off_um=mean(zo) * 1e6, mean_w_FE_on_um=mean(zn) * 1e6,
    mean_w_ritz_off_um=mean(an["off"]) * 1e6, mean_w_ritz_on_um=mean(an["on"]) * 1e6,
    rel_change_FE_pct=100 * (mean(zn) / mean(zo) - 1), rel_change_ritz_pct=100 * (mean(an["on"]) / mean(an["off"]) - 1),
    rel_change_smeared_pct=100 * (mean(sm["on"]) / mean(sm["off"]) - 1))
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
j = w.ny // 2
ax[0].plot(w.xs * 1e3, zo[:, j] * 1e6, "o", color=C1, ms=3, label="FE, DEA off")
ax[0].plot(w.xs * 1e3, zn[:, j] * 1e6, "o", color=C2, ms=3, label="FE, DEA on (1.6 kV)")
xf = np.linspace(0, 0.07, 400)
for tag, colr in (("off", C1), ("on", C2)):
    ax[0].plot(xf * 1e3, ritz_tendons(Tx, Ty, F_off if tag == "off" else F_on, xk, pr, 0.07, 0.035, xf, 0.0175) * 1e6, "-", color=colr,
               label=f"analytical, discrete tendons, {tag}")
    ax[0].plot(xf * 1e3, series(Tx, Ty + nk * (F_off if tag == "off" else F_on) / 0.07, pr, 0.07, 0.035, xf, 0.0175) * 1e6, ":", color=colr,
               label=f"smeared tension, {tag}")
for k in xk:
    ax[0].axvline(k * 1e3, color="0.85", lw=0.8)
ax[0].set_xlabel(r"Span $x$ in mm (grey: DEA lines)"); ax[0].set_ylabel(r"Deflection $w$ at mid-chord in µm"); ax[0].legend(fontsize=6)
i = int(np.argmin(np.abs(w.xs - (xk[1] + xk[2]) / 2)))
ax[1].plot(w.ys * 1e3, zo[i] * 1e6, "o", color=C1, ms=3, label="FE off")
ax[1].plot(w.ys * 1e3, zn[i] * 1e6, "o", color=C2, ms=3, label="FE on")
yf = np.linspace(0, 0.035, 200)
ax[1].plot(yf * 1e3, ritz_tendons(Tx, Ty, F_off, xk, pr, 0.07, 0.035, w.xs[i], yf) * 1e6, "-", color=C1, label="analytical off")
ax[1].plot(yf * 1e3, ritz_tendons(Tx, Ty, F_on, xk, pr, 0.07, 0.035, w.xs[i], yf) * 1e6, "-", color=C2, label="analytical on")
ax[1].set_xlabel(r"Chord $y$ in mm"); ax[1].set_ylabel(r"Deflection $w$ between DEAs in µm"); ax[1].legend(fontsize=7)
save(fig, "V5_authority.png")

# ---------------- V6 ----------------
w = M.Wing(nx=8, ny=40, n_dea_chord=0, te_cord=False, **RIGID)
fs = dict(alpha=np.full(w.nx + 1, math.radians(5)), q=np.ones(w.nx + 1), load=np.ones(w.nx + 1))
x = w.x0.copy()
A, cam, cl, _ = w.glauert(x, fs)
CL_flat = math.pi * (2 * A[0, 0] + A[0, 1])
h = 0.02
x = w.x0.copy(); xi = x[:, 1] / 0.035; x[:, 2] = 4 * h * 0.035 * xi * (1 - xi)
fs0 = dict(alpha=np.zeros(w.nx + 1), q=np.ones(w.nx + 1), load=np.ones(w.nx + 1))
A, cam, cl, _ = w.glauert(x, fs0)
CL_camb = math.pi * (2 * A[0, 0] + A[0, 1])
dcp = w.cell_dcp(A)[0]
CL_int = float(np.sum(dcp * np.diff(w.ys / 0.035)))
xi_c = 0.5 * (w.ys[:-1] + w.ys[1:]) / 0.035
report["V6_glauert"] = dict(CL_flat_5deg=CL_flat, CL_flat_exact=2 * math.pi * math.radians(5),
                            CL_parabolic_2pct=CL_camb, CL_parabolic_exact=4 * math.pi * h, CL_from_cell_integration=CL_int)

# aeroelastic chord: membrane strip, LE/TE clamped, root/tip free, chordwise pre-stretch only, 2D aerodynamics
U, al = 4.0, 5.0
cfg = dict(material="DS20", pre_span=1.0, pre_chord=1.05, clamp_root=False, clamp_tip=False, clamp_TE=True, te_cord=False, le_mode="rigid",
           n_dea_chord=0, U=U, alpha_deg=al, three_d=False, elliptic=False, nx=8, ny=24, tol=1e-7, max_iter=80000)
w = M.Wing(**cfg)
x, info = w.solve(0.0)
r = w.evaluate(x, 0.0, info)
i = int(np.argmin(np.abs(w.xs - 0.035)))
z_fe = np.array(r["z_mm"])[i] / 1e3
T = r["T_chord"]


def string_1d(T, q, c, alpha, n=400, it=200):
    """Independent 1D solver: T w'' = -q dCp(w) with w(0)=w(c)=0, thin-airfoil load, fixed-point iteration."""
    th = np.linspace(1e-6, math.pi - 1e-6, n)
    xs = 0.5 * c * (1 - np.cos(th))
    w = np.zeros(n)
    for _ in range(it):
        s = np.gradient(w, xs)
        A0 = alpha - np.trapezoid(s, th) / math.pi
        An = [2 / math.pi * np.trapezoid(s * np.cos(k * th), th) for k in range(1, 30)]
        dcp = 4 * (A0 * (1 + np.cos(th)) / np.sin(th) + sum(a * np.sin((k + 1) * th) for k, a in enumerate(An)))
        load = q * dcp
        # w(x) = (1/T) int_0^c G(x,s) load(s) ds,  G = Green's function of -d2/dx2 with zero ends
        Gm = np.where(xs[:, None] <= xs[None, :], xs[:, None] * (c - xs[None, :]), xs[None, :] * (c - xs[:, None])) / c
        wn = Gm @ (load * np.gradient(xs)) / T
        if np.max(np.abs(wn - w)) < 1e-12:
            w = wn; break
        w = 0.5 * w + 0.5 * wn
    return xs, w


xs1, w1 = string_1d(T, 0.5 * 1.225 * U ** 2, 0.035, math.radians(al))
report["V6_aeroelastic"] = dict(T_chord=T, max_w_FE_um=float(z_fe.max() * 1e6), max_w_1D_um=float(w1.max() * 1e6),
                                err_pct=float(100 * (z_fe.max() / w1.max() - 1)))
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
ax[0].plot(w.ys * 1e3, z_fe * 1e6, "o", color=C2, ms=3, label="FE mid-span (root/tip free)")
ax[0].plot(xs1 * 1e3, w1 * 1e6, "-", color=INK, label="independent 1D string + thin airfoil")
ax[0].set_xlabel(r"Chord $y$ in mm"); ax[0].set_ylabel(r"Deflection $w$ in µm"); ax[0].legend(fontsize=7)
ax[1].plot(xi_c, dcp, "o", ms=3, color=C1, label="model cell means (2 % parabolic camber)")
tt = np.linspace(0.01, math.pi - 0.01, 300)
ax[1].plot(0.5 * (1 - np.cos(tt)), 4 * 4 * h * np.sin(tt), "k-", label="exact 16 (h/c) sin(theta)")
ax[1].set_xlabel(r"Chord position $\xi = y/c$"); ax[1].set_ylabel(r"Pressure coefficient $\Delta C_p$"); ax[1].legend(fontsize=7)
save(fig, "V6_aero.png")

# ---------------- V7 ----------------
w = M.Wing()
x, info = w.solve(0.0)
fs = info["fs"]
fext = w.pressure_forces(x, fs, 1.0)
fint = w.membrane(x) + w.rod_bending(x) + w.bar_forces(x, 0, 0)[0]
reac = fint - fext
report["V7_equilibrium"] = dict(applied_z_mN=float(fext[:, 2].sum() * 1e3), reactions_z_mN=float(reac[w.fixed, 2].sum() * 1e3),
                                free_node_residual_mN=float(np.abs(reac[~w.fixed]).max() * 1e3))
conv = []
CONFIGS = (("bat armwing (default)", {}), ("rigid frame", dict(le_mode="rigid", clamp_tip=True)))
for tag, kw in CONFIGS:
    for nx, ny in [(12, 6), (16, 8), (24, 12)]:
        r = M.run(dict(nx=nx, ny=ny, tol=1e-5, max_iter=80000, **kw))
        conv.append(dict(config=tag, mesh=f"{nx}x{ny}", L_mN=r["off"]["L_mN"], dL_mN=r["d"]["L_mN"],
                         converged=bool(r["off"]["converged"] and r["on"]["converged"]),
                         camber_mean_pct=r["off"]["camber_mean_pct"], dT_chord_total_pct=r["d"]["T_chord_total_pct"]))
report["V7_convergence"] = conv
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
for (tag, _), colr in zip(CONFIGS, (C1, C2)):
    cc = [c_ for c_ in conv if c_["config"] == tag]
    ax[0].plot([c_["mesh"] for c_ in cc], [c_["L_mN"] for c_ in cc], "o-", color=colr, label=tag)
    ax[1].plot([c_["mesh"] for c_ in cc], [c_["dL_mN"] for c_ in cc], "o-", color=colr, label=tag)
ax[0].set_ylabel(r"Lift $L$ (DEA off) in mN"); ax[1].set_ylabel(r"Lift change $\Delta L$ at 1.6 kV in mN")
for a in ax:
    a.set_xlabel("Mesh"); a.legend(fontsize=7); a.set_ylim(bottom=0)
save(fig, "V7_convergence.png")

# ---------------- V8 ----------------
# The rod bending operator is linear in x: assemble K by unit probes on a bare rod (root clamped) and compare the tip
# deflection under a tip load P with Euler-Bernoulli: delta = int_0^L P (L - s)^2 / (E I(s)) ds  (= P L^3 / 3EI if uniform).
trap = getattr(np, "trapezoid", None) or np.trapz
P_tip, Lr = 1e-3, 0.07
v8 = []
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
for (dr, dt), colr in (((1e-3, 1e-3), C1), ((1e-3, 0.5e-3), C2)):
    s_ = np.linspace(0, Lr, 4001)
    EI = 120e9 * math.pi * (dr + (dt - dr) * s_ / Lr) ** 4 / 64
    ana_tip = float(trap(P_tip * (Lr - s_) ** 2 / EI, s_))
    for nx in (8, 16, 32):
        w = M.Wing(nx=nx, ny=2, n_dea_chord=0, te_cord=False, digits="", rod_d_root=dr, rod_d_tip=dt, U=0.0)
        free = [w.nid(i, 0) for i in range(1, w.nx + 1)]
        n = len(free)
        K = np.zeros((n, n))
        x0 = w.x0.copy()
        f0 = w.rod_bending(x0)[free, 2]
        for k, nd in enumerate(free):
            xp = x0.copy(); xp[nd, 2] += 1e-6
            K[:, k] = (w.rod_bending(xp)[free, 2] - f0) / 1e-6
        F = np.zeros(n); F[-1] = P_tip
        u = np.linalg.solve(K, F)
        v8.append(dict(d_root_mm=dr * 1e3, d_tip_mm=dt * 1e3, segments=w.nx, tip_FE_mm=float(u[-1] * 1e3), tip_exact_mm=ana_tip * 1e3,
                       err_pct=float(100 * (u[-1] / ana_tip - 1))))
        if nx == 32:
            ax[0].plot(w.xs * 1e3, np.r_[0, u] * 1e3, "o", ms=3, mfc="none", color=colr, label=f"discrete rod, d = {dr*1e3:g} to {dt*1e3:g} mm")
    xs_ = np.linspace(0, Lr, 120)
    wa = [float(trap(P_tip * (Lr - s_[s_ <= xx]) * (xx - s_[s_ <= xx]) / EI[s_ <= xx], s_[s_ <= xx])) if xx > 0 else 0.0 for xx in xs_]
    ax[0].plot(xs_ * 1e3, np.array(wa) * 1e3, "-", color=colr, label="Euler-Bernoulli")
ax[0].set_xlabel(r"Span $x$ in mm"); ax[0].set_ylabel(r"Deflection in mm (tip load 1 mN)"); ax[0].legend(fontsize=6)
for (dr, dt), colr in (((1e-3, 1e-3), C1), ((1e-3, 0.5e-3), C2)):
    vv = [v for v in v8 if v["d_tip_mm"] == dt * 1e3]
    ax[1].plot([v["segments"] for v in vv], [v["err_pct"] for v in vv], "o-", color=colr, label=f"d = {dr*1e3:g} to {dt*1e3:g} mm")
ax[1].axhline(0, color="0.6", lw=0.8); ax[1].set_xlabel("Rod segments"); ax[1].set_ylabel(r"Tip deflection error in %"); ax[1].legend(fontsize=7)
save(fig, "V8_rod.png")
report["V8_rod"] = v8
cap = []
for c_ in (50.0, 100.0, 400.0):
    r = M.run(dict(rod_EA_cap=c_, tol=1e-5))
    cap.append(dict(rod_EA_cap=c_, L_mN=r["off"]["L_mN"], dL_mN=r["d"]["L_mN"], converged=bool(r["off"]["converged"] and r["on"]["converged"])))
report["V8_EA_cap"] = cap

with open(os.path.join(OUT, "validation.json"), "w") as f:
    json.dump(report, f, indent=1, default=float)
print(json.dumps(report, indent=1, default=float))
