"""Reduced-order active membrane airfoil (2D section) -- the trustable core of the wing model.

Physics (per unit span; chord c between a leading-edge and a trailing-edge support, both at z = 0):
  Membrane      T z'' = -q dCp(x)                 (string equilibrium, small slope, T uniform along the chord)
  Aerodynamics  thin-airfoil theory, Glauert series dCp = 4[A0 cot(theta/2) + sum An sin(n theta)],
                A0 = alpha - (1/pi) int z' dtheta,  An = (2/pi) int z' cos(n theta) dtheta,  CL = 2 pi (A0 + A1/2)
  Solution      z'(x) = (q/T) [ int_x^c dCp dxi - (1/c) int_0^c xi dCp dxi ]   (string Green's function, exact)
                -> linear system (I - (q c/T) R) A = alpha e0 for the Glauert coefficients (R evaluated in closed form)
  Tension       T = T_mem(lambda_c) + (n/b) max(F_pass(lambda_d) - dF(V), 0)
                lambda_c = pre_chord S/c, lambda_d = pre_dea S/c, S = int sqrt(1 + z'^2) dx (extensibility)
  Membrane      incompressible Yeoh + fibre term (her SMS 2023 form), plane strain in span (lambda_s = pre_span):
                N = t lambda_c S11 / lambda_s,  S11 = 2 W1 (1 - lambda_c^-4 lambda_s^-2) + 2 W4
  DEA           measured: F_pass from the bundle Yeoh fit, dF(V) = dF_1600 (V/1600)^2 from the isometric pulses
                (stretch independent, her CST equation of state); tension only.
Divergence      T must exceed T_crit = q c max eig(R) (static membrane divergence).
Run: python section.py   -> validation/S_*.png, validation/section.json
"""
import json, math, os, sys
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M

NG = 24                       # Glauert coefficients
MQ = 3000                     # quadrature points in theta
TH = (np.arange(MQ) + 0.5) * np.pi / MQ
DTH = np.pi / MQ


def _Fk(k, th):
    """int_0^theta of the k-th basis of dCp sin(theta)/4: (1 + cos) for k = 0, sin(k th) sin(th) otherwise."""
    if k == 0:
        return th + np.sin(th)
    if k == 1:
        return 0.5 * (th - np.sin(2 * th) / 2)
    return 0.5 * (np.sin((k - 1) * th) / (k - 1) - np.sin((k + 1) * th) / (k + 1))


def _R():
    """z' = (q c / T) sum_k A_k g_k(theta);  R[n, k] = Glauert projection of g_k."""
    G = np.zeros((NG, MQ))
    for k in range(NG):
        tot = 2 * math.pi * (k == 0) + math.pi * (k == 1)                       # int_0^c dCp dxi / c, per A_k
        mom = (math.pi / 2) * ((k == 0) + (k == 1) - 0.5 * (k == 2))           # int_0^c xi dCp dxi / c^2
        G[k] = tot - 2 * _Fk(k, TH) - mom
    P = np.zeros((NG, MQ))
    P[0] = -DTH / math.pi
    for n in range(1, NG):
        P[n] = 2 / math.pi * np.cos(n * TH) * DTH
    return P @ G.T, G


RMAT, GBAS = _R()
EIG_MAX = float(np.max(np.linalg.eigvals(RMAT).real))

DEA_A0 = 3e-6 / 5             # per-DEA reference section used in the bundle fit (product C_i A0 is what is measured)
DF1600_ISO = 0.04314 / 5      # N per DEA, isometric pulses at lambda = 1.12, 1.6 kV (isometric_force_drop.csv)
DF1600_CYC = DF1600_ISO * 0.15 / 0.40   # quasi-static cycles (eta 0.15 vs 0.40)

SEC = dict(c=0.05, U=4.0, alpha_deg=6.0, rho=1.225, material="aniso", t_mem=180e-6, pre_chord=1.03, pre_span=1.03,
           spacing=0.012, pre_dea=1.10, V=1600.0, dF1600=DF1600_ISO, anchor=1.0)


def T_mem(lc, p):
    m = M.MATERIALS[p["material"]]
    C = [m["v_m"] * ci for ci in m["C"]]
    ls = p["pre_span"]
    I1 = lc ** 2 + ls ** 2 + 1 / (lc * ls) ** 2
    e = I1 - 3
    W1 = C[0] + 2 * C[1] * e + 3 * C[2] * e ** 2
    W4 = m["Ef_chord"] * (lc - 1) / (2 * lc) if lc > 1 else 0.0
    S11 = 2 * W1 * (1 - lc ** -4 * ls ** -2) + 2 * W4
    return max(p["t_mem"] * lc * S11 / ls, 0.0) if lc > 1 else 0.0


def F_dea(ld, V, p):
    I1 = ld ** 2 + 2 / ld
    C = M.DEA_YEOH
    Fp = DEA_A0 * 2 * (ld - ld ** -2) * (C[0] + 2 * C[1] * (I1 - 3) + 3 * C[2] * (I1 - 3) ** 2)
    return max(Fp - p["anchor"] * p["dF1600"] * (V / 1600) ** 2, 0.0) if ld > 1 else 0.0


def T_total(S, V, p):
    lc, ld = p["pre_chord"] * S / p["c"], p["pre_dea"] * S / p["c"]
    n_per_m = 1.0 / p["spacing"] if p["spacing"] else 0.0
    return T_mem(lc, p) + n_per_m * F_dea(ld, V, p)


def shape(T, p):
    q = 0.5 * p["rho"] * p["U"] ** 2
    al = math.radians(p["alpha_deg"])
    A = np.linalg.solve(np.eye(NG) - q * p["c"] / T * RMAT, al * np.eye(NG)[0])
    zp = q * p["c"] / T * (A @ GBAS)                                     # z'(theta)
    dx = p["c"] / 2 * np.sin(TH) * DTH
    S = float(np.sum(np.sqrt(1 + zp ** 2) * dx))
    z = np.cumsum(zp * dx)
    return dict(A=A, CL=2 * math.pi * (A[0] + A[1] / 2), zp=zp, z=z, x=p["c"] / 2 * (1 - np.cos(TH)), S=S,
                camber_pct=float(np.max(np.abs(z)) / p["c"] * 100) * np.sign(z[np.argmax(np.abs(z))]))


def solve(p=None, V=None):
    p = dict(SEC, **(p or {}))
    V = p["V"] if V is None else V
    q = 0.5 * p["rho"] * p["U"] ** 2
    Tc = q * p["c"] * EIG_MAX
    h = lambda T: T - T_total(shape(T, p)["S"], V, p)
    lo, hi = Tc * (1 + 1e-9), max(10 * Tc, 2 * T_total(p["c"] * 1.0, V, p) + Tc, 1e-3)
    while h(hi) < 0:
        hi *= 2
    if h(lo) > 0:
        return None
    T = brentq(h, lo, hi, xtol=1e-12, rtol=1e-10)
    r = shape(T, p)
    r.update(T=T, C_T=T / (q * p["c"]), T_crit=Tc, q=q)
    return r


def authority(p=None):
    p = dict(SEC, **(p or {}))
    a, b = solve(p, 0.0), solve(p, p["V"])
    if a is None or b is None:
        return dict(dCL=float("nan"), CL0=float("nan"), C_T0=float("nan"), dcam=float("nan"), cam0=float("nan"))
    return dict(dCL=b["CL"] - a["CL"], CL0=a["CL"], C_T0=a["C_T"], C_T1=b["C_T"], dcam=b["camber_pct"] - a["camber_pct"],
                cam0=a["camber_pct"], T0=a["T"], T1=b["T"])


# ------------------------------------------------------------------ independent check: lumped-vortex + finite differences
def vortex_fd(T, p, N=200):
    """Membrane on N cosine-spaced panels, lumped vortices (1/4 panel) and control points (3/4 panel); z on panel ends."""
    p = dict(SEC, **p)
    c, U = p["c"], p["U"]
    q, al = 0.5 * p["rho"] * U ** 2, math.radians(p["alpha_deg"])
    xe = c / 2 * (1 - np.cos(np.linspace(0, np.pi, N + 1)))
    dx = np.diff(xe)
    xv, xc = xe[:-1] + 0.25 * dx, xe[:-1] + 0.75 * dx
    K = 1 / (2 * np.pi * (xc[:, None] - xv[None, :]))                   # induced downwash per unit Gamma
    # slope at control point from the panel's end nodes: z'_i = (z_{i+1} - z_i)/dx_i
    Ds = np.zeros((N, N + 1)); Ds[np.arange(N), np.arange(N)] = -1 / dx; Ds[np.arange(N), np.arange(N) + 1] = 1 / dx
    # tangency: U (alpha - z'_i) = sum K_ij Gamma_j ; loading dp_j = rho U Gamma_j / dx_j acting on panel j
    # membrane at interior node k: T (z'_k+ - z'_k-) = -(dp_{k-1} dx_{k-1} + dp_k dx_k)/2
    nZ, nG = N + 1, N
    Amat = np.zeros((nZ + nG, nZ + nG)); rhs = np.zeros(nZ + nG)
    Amat[:N, :nZ] = U * Ds; Amat[:N, nZ:] = K; rhs[:N] = U * al
    for k in range(1, N):
        r = N + k - 1
        Amat[r, k + 1] += T / dx[k]; Amat[r, k] -= T / dx[k] + T / dx[k - 1]; Amat[r, k - 1] += T / dx[k - 1]
        Amat[r, nZ + k - 1] += p["rho"] * U / 2; Amat[r, nZ + k] += p["rho"] * U / 2
    Amat[N + N - 1, 0] = 1; Amat[N + N, N] = 1
    sol = np.linalg.solve(Amat, rhs)
    z, G = sol[:nZ], sol[nZ:]
    return dict(CL=2 * G.sum() / (U * c), camber_pct=float(np.max(np.abs(z)) / c * 100), x=xe, z=z)


# ------------------------------------------------------------------ load path and penalties (Eqs. S10-S11)
K_DEA = 6 * M.DEA_YEOH[0] * DEA_A0 / SEC["c"]              # DEA axial stiffness EA/L at lambda ~ 1 (N/m)


def anchor_fraction(k_a):
    """Eq. S10: fraction of the DEA force change (and pre-tension) delivered through an anchor of stiffness k_a."""
    return k_a / (k_a + K_DEA)


ANCHORS = {   # k_a in N/m (Eq. S10); gliding (6 cm) and FAAV (12 cm) wings, carbon LE rod
    "free TE, no edge tension": 0.02 * K_DEA,
    "TE tendon 0.1 N": 8 * 0.1 * 0.012 / 0.06 ** 2,
    "TE tendon 0.5 N": 8 * 0.5 * 0.012 / 0.06 ** 2,
    "LE rod 1 mm, 6 cm, 5 DEAs": 3 * 120e9 * math.pi * 1e-12 / 64 / 0.06 ** 3 / 5,
    "LE rod 1 mm, 12 cm, 12 DEAs": 3 * 120e9 * math.pi * 1e-12 / 64 / 0.12 ** 3 / 12,
    "LE rod 2 mm, 12 cm, 12 DEAs": 3 * 120e9 * math.pi * 16e-12 / 64 / 0.12 ** 3 / 12,
}
ROLL = dict(L=50e-3, w=12e-3, margin=1e-3, offset=1.5e-3, H=35e-6, eps_r=2.8, turns=7)


def dF_ideal(V, n_films, H=ROLL["H"], eps_r=ROLL["eps_r"], g=0.0):
    """Eq. S11: ideal force drop of a roll with n_films active films (electroded width w - 2 margin), air gap g per film."""
    we = ROLL["w"] - 2 * ROLL["margin"]
    return M.EPS0 * eps_r * V ** 2 * n_films * we * H / (H + eps_r * g) ** 2


if __name__ == "__main__":
    import plotstyle as PS
    import matplotlib.pyplot as plt
    PS.apply()
    OUT = os.path.join(HERE, "validation")
    rep = {}
    q4 = 0.5 * SEC["rho"] * SEC["U"] ** 2

    # ---------------- verification S1-S3 ----------------
    sh = shape(1e9, dict(SEC))
    rep["S1_rigid"] = dict(CL=sh["CL"], exact=2 * math.pi * math.radians(SEC["alpha_deg"]))
    s2 = []
    for CT in (2.0, 3.0, 6.0, 20.0):
        T = CT * q4 * SEC["c"]
        a, b = shape(T, SEC), vortex_fd(T, SEC)
        s2.append(dict(C_T=CT, CL_glauert=a["CL"], CL_vortex=b["CL"], cam_glauert=abs(a["camber_pct"]), cam_vortex=b["camber_pct"]))
    rep["S2_independent"] = s2
    rep["C_T_crit"] = EIG_MAX
    s3 = []
    for nd, sp in ((0, 0.0), (16, 0.2 / 16)):
        fe = M.run(dict(span=0.20, chord=0.05, nx=32, ny=10, le_mode="rigid", clamp_tip=True, clamp_TE=True, te_cord=False,
                        three_d=False, elliptic=False, n_dea_chord=nd, U=4.0, alpha_deg=6.0))
        sec = authority(dict(spacing=sp))
        s3.append(dict(n_dea=nd, FE_CL_off=fe["off"]["CL"], FE_CL_on=fe["on"]["CL"], FE_cam_off=fe["off"]["camber_mid_pct"],
                       FE_cam_on=fe["on"]["camber_mid_pct"], FE_conv=bool(fe["off"]["converged"] and fe["on"]["converged"]),
                       sec_CL_off=sec["CL0"], sec_CL_on=sec["CL0"] + sec["dCL"], sec_cam_off=sec["cam0"], sec_cam_on=sec["cam0"] + sec["dcam"]))
    rep["S3_FE"] = s3
    rep["S3_dCL_ratio_section_over_FE"] = (s3[1]["sec_CL_on"] - s3[1]["sec_CL_off"]) / (s3[1]["FE_CL_on"] - s3[1]["FE_CL_off"])

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].plot([r["C_T"] for r in s2], [r["CL_glauert"] for r in s2], "o-", label="Glauert + Green, Eqs. S1-S5")
    ax[0].plot([r["C_T"] for r in s2], [r["CL_vortex"] for r in s2], "s--", label="lumped vortex + FD (independent)")
    ax[0].axhline(rep["S1_rigid"]["exact"], color="k", ls=":", lw=0.8, label=r"rigid $2\pi\alpha$")
    ax[0].set(xlabel=r"Tension coefficient $C_T=T/(qc)$", ylabel=r"$C_L$ ($\alpha$ = 6°)", xscale="log"); ax[0].legend(fontsize=7)
    T = 3.0 * q4 * SEC["c"]
    a, b = shape(T, SEC), vortex_fd(T, SEC)
    ax[1].plot(a["x"] / SEC["c"], a["z"] / SEC["c"] * 100, "-", label="Glauert + Green")
    ax[1].plot(b["x"] / SEC["c"], b["z"] / SEC["c"] * 100, "--", label="lumped vortex + FD")
    ax[1].set(xlabel=r"$x/c$", ylabel=r"Camber $z/c$ in % ($C_T$ = 3)"); ax[1].legend(fontsize=7)
    lab = ["no DEA\n0 V", "no DEA\n1.6 kV", "16 DEAs\n0 V", "16 DEAs\n1.6 kV"]
    xx = np.arange(4)
    ax[2].bar(xx - 0.2, [s3[0]["FE_CL_off"], s3[0]["FE_CL_on"], s3[1]["FE_CL_off"], s3[1]["FE_CL_on"]], 0.4, label="3D FE, wide frame, mid-span")
    ax[2].bar(xx + 0.2, [s3[0]["sec_CL_off"], s3[0]["sec_CL_on"], s3[1]["sec_CL_off"], s3[1]["sec_CL_on"]], 0.4, label="section model")
    ax[2].set_xticks(xx); ax[2].set_xticklabels(lab, fontsize=7); ax[2].set(ylabel=r"$C_L$", ylim=(0, 1.6)); ax[2].legend(fontsize=7)
    PS.finish(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "S_verification.png")); plt.close(fig)

    # ---------------- predictions: framed section trends ----------------
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.6))
    pcs = np.linspace(0.97, 1.10, 40)
    for U, col in ((3.0, PS.MATLAB[0]), (4.0, PS.MATLAB[1]), (6.0, PS.MATLAB[2])):
        for dF, ls, lab_ in ((DF1600_ISO, "-", "isometric"), (DF1600_CYC, "--", "cycles")):
            r = [authority(dict(U=U, pre_chord=pc, dF1600=dF)) for pc in pcs]
            ax[0, 0].plot(pcs, [x["dCL"] for x in r], ls, color=col, label=rf"$U$ = {U:g} m/s, $\Delta F$ {lab_}")
            if dF == DF1600_ISO:
                ax[0, 1].plot([x["C_T0"] for x in r], [x["dCL"] / x["CL0"] * 100 for x in r], "-", color=col, label=f"$U$ = {U:g} m/s")
    ax[0, 0].set(xlabel=r"Membrane pre-stretch $\lambda_{c,0}$", ylabel=r"$\Delta C_L$ at 1.6 kV"); ax[0, 0].legend(fontsize=6)
    ax[0, 1].axvline(EIG_MAX, color="k", lw=0.8, ls=":"); ax[0, 1].text(EIG_MAX * 1.04, 5, "divergence\n(Eq. S5)", fontsize=7)
    ax[0, 1].set(xlabel=r"$C_T=T/(qc)$ at 0 V", ylabel=r"$\Delta C_L / C_{L,0}$ in %", xscale="log"); ax[0, 1].legend(fontsize=7)
    sps = np.array([4, 6, 8, 12, 16, 24, 40, 60]) * 1e-3
    for pc, col in ((1.00, PS.MATLAB[0]), (1.03, PS.MATLAB[1]), (1.06, PS.MATLAB[2])):
        ax[0, 2].plot(sps * 1e3, [authority(dict(spacing=s, pre_chord=pc))["dCL"] for s in sps], "o-", color=col, label=rf"$\lambda_{{c,0}}$ = {pc:.2f}")
    ax[0, 2].set(xlabel=r"DEA spacing $s$ in mm", ylabel=r"$\Delta C_L$ at 1.6 kV", xscale="log"); ax[0, 2].legend(fontsize=7)
    pds = np.linspace(1.0, 1.25, 30)
    for pc, col in ((1.00, PS.MATLAB[0]), (1.03, PS.MATLAB[1]), (1.06, PS.MATLAB[2])):
        ax[1, 0].plot(pds, [authority(dict(pre_dea=pd_, pre_chord=pc))["dCL"] for pd_ in pds], "-", color=col, label=rf"$\lambda_{{c,0}}$ = {pc:.2f}")
    ax[1, 0].set(xlabel=r"DEA pre-stretch $\lambda_{d,0}$", ylabel=r"$\Delta C_L$ at 1.6 kV"); ax[1, 0].legend(fontsize=7)
    als = np.linspace(-6, 14, 21)
    cl = lambda rr: np.array([x["CL"] if x else np.nan for x in rr])
    E3 = {}
    for U, col in ((3.0, PS.MATLAB[0]), (4.0, PS.MATLAB[1]), (6.0, PS.MATLAB[2])):
        c0 = cl([solve(dict(U=U, alpha_deg=a_, pre_chord=1.01), 0.0) for a_ in als])
        c1 = cl([solve(dict(U=U, alpha_deg=a_, pre_chord=1.01), 1600.0) for a_ in als])
        ax[1, 1].plot(als, c0, "-", color=col, label=f"$U$ = {U:g} m/s, 0 V")
        ax[1, 1].plot(als, c1, "--", color=col, label=f"$U$ = {U:g} m/s, 1.6 kV")
        ax[1, 2].plot(als, c1 - c0, "o-", color=col, ms=3, label=f"$U$ = {U:g} m/s")
        E3[U] = dict(alpha=als.tolist(), CL0=c0.tolist(), CL1=c1.tolist())
    ax[1, 1].plot(als, 2 * np.pi * np.radians(als), "k:", lw=0.8, label=r"rigid $2\pi\alpha$")
    ax[1, 1].set(xlabel=r"Angle of attack $\alpha$ in °", ylabel=r"$C_L$ ($\lambda_{c,0}$ = 1.01)"); ax[1, 1].legend(fontsize=5.5, ncol=2)
    ax[1, 2].set(xlabel=r"Angle of attack $\alpha$ in °", ylabel=r"$\Delta C_L$ at 1.6 kV ($\lambda_{c,0}$ = 1.01)"); ax[1, 2].legend(fontsize=7)
    PS.finish(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "S_trends.png")); plt.close(fig)
    rep["E3_prediction"] = E3

    # ---------------- load path and penalties ----------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    geo = (ROLL["L"] - 2 * ROLL["offset"]) / ROLL["L"] * (1 - 1 / (2 * ROLL["turns"]))
    bars = [("ideal,\n2 films", dF_ideal(1600, 2)), ("ideal,\n1 film", dF_ideal(1600, 1)),
            ("2 films x\ngeometry", dF_ideal(1600, 2) * geo), ("measured\nisometric", DF1600_ISO), ("measured\ncycles", DF1600_CYC)]
    ax[0].bar(range(5), [b_[1] * 1e3 for b_ in bars], color=[PS.MATLAB[0]] * 3 + [PS.MATLAB[1]] * 2)
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels([b_[0] for b_ in bars], fontsize=7)
    ax[0].set(ylabel=r"$\Delta F$ per DEA at 1.6 kV in mN")
    gs = np.linspace(0, 30e-6, 60)
    for H, col in ((25e-6, PS.MATLAB[0]), (35e-6, PS.MATLAB[1]), (50e-6, PS.MATLAB[2])):
        ax[1].plot(gs * 1e6, [dF_ideal(1600, 2, H=H, g=g) * 1e3 for g in gs], color=col, label=rf"$H$ = {H*1e6:.0f} µm, 2 films")
    ax[1].axhline(DF1600_ISO * 1e3, color="k", ls="--", lw=0.8, label="measured, isometric")
    ax[1].axhline(DF1600_CYC * 1e3, color="k", ls=":", lw=0.8, label="measured, cycles")
    ax[1].set(xlabel=r"Air gap per film $g$ in µm", ylabel=r"Ideal $\Delta F$ at 1.6 kV in mN", yscale="log"); ax[1].legend(fontsize=6.5)
    ka = np.logspace(-2.3, 2, 100) * K_DEA
    ax[2].semilogx(ka / K_DEA, anchor_fraction(ka), "k-")
    for k_, (lab_, v) in enumerate(ANCHORS.items()):
        ax[2].plot(v / K_DEA, anchor_fraction(v), "o", color=PS.MATLAB[k_ % 7], label=lab_)
    ax[2].set(xlabel=r"Anchor / DEA stiffness $k_a/k_d$", ylabel=r"Retained fraction $r=k_a/(k_a+k_d)$"); ax[2].legend(fontsize=6)
    PS.finish(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "S_loadpath.png")); plt.close(fig)
    rep["penalties"] = dict(dF_ideal_2films_mN=dF_ideal(1600, 2) * 1e3, dF_ideal_1film_mN=dF_ideal(1600, 1) * 1e3, geometric_factor=geo,
                            dF_iso_mN=DF1600_ISO * 1e3, dF_cyc_mN=DF1600_CYC * 1e3,
                            eta_2films=DF1600_ISO / dF_ideal(1600, 2), eta_1film=DF1600_ISO / dF_ideal(1600, 1),
                            unexplained_2films=DF1600_ISO / (dF_ideal(1600, 2) * geo),
                            gap_um_explaining_iso_2films=float(brentq(lambda g: dF_ideal(1600, 2, g=g) * geo - DF1600_ISO, 0, 1e-4) * 1e6))
    rep["anchor"] = dict(k_dea_N_per_m=K_DEA, **{k: anchor_fraction(v) for k, v in ANCHORS.items()})

    # ---------------- flapping: effective-angle envelope and quasi-steady reconstruction ----------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    Us = np.linspace(1.5, 8, 60); R = 0.15; Phi0 = math.radians(30)
    for f, col in ((1.0, PS.MATLAB[0]), (2.0, PS.MATLAB[1]), (5.0, PS.MATLAB[2])):
        ax[0].plot(Us, np.degrees(np.arctan(Phi0 * 2 * math.pi * f * R / Us)), color=col, label=f"$f$ = {f:g} Hz, tip")
        ax[0].plot(Us, np.degrees(np.arctan(Phi0 * 2 * math.pi * f * 0.75 * R / Us)), "--", color=col, lw=0.8)
        ax[1].plot(Us, math.pi * f * 0.07 / Us, color=col, label=f"$f$ = {f:g} Hz")
    ax[0].axhline(12, color="k", ls=":", lw=0.8); ax[0].text(6, 12.8, "attached-flow limit", fontsize=7)
    ax[0].set(xlabel=r"Flow speed $U$ in m/s", ylabel=r"$\max|\Delta\alpha_{eff}|$ in ° (Eq. S12)"); ax[0].legend(fontsize=7)
    ax[1].axhline(0.1, color="k", ls=":", lw=0.8); ax[1].text(4.5, 0.11, "quasi-steady below ~0.1", fontsize=7)
    ax[1].set(xlabel=r"Flow speed $U$ in m/s", ylabel=r"Reduced frequency $k=\pi fc/U$", yscale="log"); ax[1].legend(fontsize=7)
    ph = np.linspace(0, 360, 37)
    for U, col in ((3.0, PS.MATLAB[0]), (4.0, PS.MATLAB[1]), (6.0, PS.MATLAB[2])):
        dl = []
        for p_ in ph:
            Om = -Phi0 * 2 * math.pi * 1.0 * math.sin(math.radians(p_)); r_ = 0.75 * R
            Ue = math.hypot(U, Om * r_); ae = 6.0 - math.degrees(math.atan(Om * r_ / U))
            a0, a1 = solve(dict(U=Ue, alpha_deg=ae, pre_chord=1.01), 0.0), solve(dict(U=Ue, alpha_deg=ae, pre_chord=1.01), 1600.0)
            dl.append((a1["CL"] - a0["CL"]) * Ue ** 2 / U ** 2 if a0 and a1 else np.nan)
        ax[2].plot(ph, dl, color=col, label=f"$U$ = {U:g} m/s")
    ax[2].set(xlabel=r"Flap phase $\varphi$ in °", ylabel=r"$\Delta C_L\,(U_{eff}/U)^2$ at 0.75$R$"); ax[2].legend(fontsize=7)
    PS.finish(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "S_flapping.png")); plt.close(fig)

    with open(os.path.join(OUT, "section.json"), "w") as f:
        json.dump(json.loads(json.dumps(rep, default=float).replace("NaN", "null")), f, indent=1)
    print(json.dumps({k: rep[k] for k in ("S1_rigid", "S2_independent", "S3_FE", "S3_dCL_ratio_section_over_FE", "penalties", "anchor")}, indent=1, default=float))
