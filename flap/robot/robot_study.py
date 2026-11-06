"""The whole robot, from the measured parts: inertia, resonance, edge mode, thickness window, forward-flight power and the ranges.
Every equation is printed on the robot page next to its plot. Run: python robot_study.py -> data/robot_study.json"""
import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import params as P
import edge_theory as ET
CAL = json.load(open(os.path.join(HERE, "data", "edge_calib.json")))
rng = np.random.default_rng(7)
OUT = {}
R2 = lambda x: float(np.round(x, 4))

# ---------------- 1. wing inertia about the flapping hinge ----------------
G = json.load(open(os.path.join(HERE, "..", "first", "data", "geometry.json")))["points_mm_along_LE"]
H0 = 5e-3                                   # A  hinge 5 mm inboard of the leading-edge root (body half width + flexure)
x_te0, x_tip = G["TE_root"][0] * 1e-3, G["tip"][0] * 1e-3


def chord(x):
    """Local chord from the outline, m (LE along x, TE line from root to tip)."""
    y_te = np.interp(x, [G["TE_root"][0] * 1e-3, G["TE_tip"][0] * 1e-3], [G["TE_root"][1] * 1e-3, G["TE_tip"][1] * 1e-3])
    return np.clip(y_te, 0, None)


def inertia(t_film=P.FILM_T, roll_w=P.ROLL_W, extras=None, le=P.LE_MASS, h0=H0):
    """Mass (kg) and moment of inertia (kg m2) of one wing. extras = cables, glue, contacts near the root."""
    x = np.linspace(x_te0, x_tip, 400)
    c = chord(x)
    dx = x[1] - x[0]
    skin = P.RHO_F * t_film * c * dx
    xl = np.linspace(0, x_tip, 200)
    m_le = np.full_like(xl, le / len(xl))
    roll_len = P.EDGE_L
    m_roll_tot = P.RHO_F * roll_w * t_film * P.ROLL_CUT
    m_roll = np.full_like(x, m_roll_tot / len(x))
    if extras is None:
        extras = P.WING_MASS - (skin.sum() + le + m_roll_tot) if t_film == P.FILM_T else EXTRAS
    I = np.sum(skin * (x + h0) ** 2) + np.sum(m_le * (xl + h0) ** 2) + np.sum(m_roll * (x + h0) ** 2) + extras * (h0 + 3e-3) ** 2
    m = skin.sum() + le + m_roll_tot + extras
    r_lift = math.sqrt(np.sum(c * (x + h0) ** 2 * dx) / np.sum(c * dx))     # second moment radius of area
    S = np.sum(c * dx)
    I_air = P.RHO_AIR * math.pi / 4 * np.sum(c ** 2 * (x + h0) ** 2 * dx)  # added mass of air, plate normal to itself
    return dict(m=m, I=I, I_air=I_air, skin=skin.sum(), roll=m_roll_tot, extras=extras, r2=r_lift, S=S)


W0 = inertia()
EXTRAS = W0["extras"]
OUT["wing"] = {k: R2(v * (1e6 if k in ("m", "skin", "roll", "extras") else 1)) for k, v in W0.items()}
print("wing: %.0f mg (skin %.0f, roll %.0f, LE 23, rest %.0f), I %.2e kg m2, air %.2e, area %.1f cm2, r2 %.1f mm" % (
    W0["m"] * 1e6, W0["skin"] * 1e6, W0["roll"] * 1e6, W0["extras"] * 1e6, W0["I"], W0["I_air"], W0["S"] * 1e4, W0["r2"] * 1e3))

# ---------------- 2. flapping resonance with the two cores ----------------
L_CORE = 20e-3                                # A  core length, as in the POC drawings


def core_k(m_core, lam=1.10):
    """Axial stiffness of a rolled core, N/m: Yeoh tangent E_t A / L, with A = m / (rho L)."""
    A = m_core / (P.RHO_F * L_CORE)
    dN = 2 * P.C1 * (1 + 2 * lam ** -3)       # d(sigma)/d(lam) for Yeoh C1, first term
    return dN * A / L_CORE


k_c = core_k(P.CORE_MASS)
r_lev = np.linspace(1.5e-3, 6e-3, 46)
res = {}
for name, I in (("wing in air", W0["I"] + W0["I_air"]), ("wing, no air", W0["I"])):
    res[name] = [R2(math.sqrt(2 * k_c * r ** 2 / I) / (2 * math.pi)) for r in r_lev]
OUT["resonance"] = dict(r_mm=[R2(r * 1e3) for r in r_lev], f=res, k_core=R2(k_c))
f_r = lambda r, I=W0["I"] + W0["I_air"], k=k_c: math.sqrt(2 * k * r ** 2 / I) / (2 * math.pi)
r_for = lambda f, I=W0["I"] + W0["I_air"], k=k_c: 2 * math.pi * f * math.sqrt(I / (2 * k))
print("core k %.0f N/m each; resonance at r 3 / 3.5 / 4 mm: %.1f / %.1f / %.1f Hz; r for 20 Hz %.2f mm, 25 Hz %.2f mm" % (k_c, f_r(3e-3), f_r(3.5e-3), f_r(4e-3), r_for(20) * 1e3, r_for(25) * 1e3))

# quality factor from air power: Q = 2 pi E_kin / E_air per cycle
def Q_factor(f, Phi, P_air, I=W0["I"] + W0["I_air"]):
    w = 2 * math.pi * f
    return 2 * math.pi * 0.5 * I * (w * Phi) ** 2 / (P_air / f)


Qs = {"%g mW" % pa: [R2(Q_factor(f, math.radians(20), pa * 1e-3)) for f in (15, 20, 25, 30)] for pa in (10, 20, 30)}
OUT["Q"] = dict(f=[15, 20, 25, 30], Q=Qs)
print("Q at 25 Hz, +-20 deg, 20 mW per wing: %.1f" % Q_factor(25, math.radians(20), 0.02))

# core strain and energy at the stroke
def core_strain(Phi_deg, r):
    return math.radians(Phi_deg) * r / L_CORE


OUT["core_strain"] = {"r %.1f mm" % (r * 1e3): [R2(core_strain(ph, r)) for ph in (10, 15, 20, 25, 30)] for r in (3e-3, 3.5e-3, 4e-3)}

# ---------------- 3. edge mode with voltage, and what the stroke does to the edge ----------------
N0c, cc = CAL["N0"], CAL["c"]
mu_roll = P.RHO_F * P.ROLL_W * P.FILM_T / 1.2
Vs = np.linspace(0, 1800, 37)
modes = {}
for nm, kw in (("roll, eta_f 0.4, 43 mN", dict(T0=0.043, eta_f=0.4)), ("roll, eta_f 0.8, 43 mN", dict(T0=0.043, eta_f=0.8)), ("roll, eta_f 0.4, 70 mN", dict(T0=0.07, eta_f=0.4)), ("passive edge, 43 mN", dict(T0=0.043, eta_f=0.0))):
    fs = []
    for V in Vs:
        s = ET.solve(V, N0=N0c, c=cc, **kw)
        fs.append(R2(ET.edge_mode(s["T"] * 1e-3, s["N"], mu_roll, c=P.CHORD_ROOT * 0.66)))
    modes[nm] = fs
OUT["edge_mode"] = dict(V=Vs.tolist(), f=modes)
print("edge mode, eta_f 0.4, 43 mN: 0 V %.0f Hz, 1400 V %.0f, 1800 V %.0f" % (modes["roll, eta_f 0.4, 43 mN"][0], modes["roll, eta_f 0.4, 43 mN"][28], modes["roll, eta_f 0.4, 43 mN"][-1]))


def te_ratio(f, fe, zeta):
    """Edge motion relative to the leading edge, per unit of leading-edge motion (base-excited oscillator), and phase lag (deg)."""
    r = f / fe
    amp = r ** 2 / math.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)
    ph = math.degrees(math.atan2(2 * zeta * r, 1 - r ** 2))
    return amp, ph


fq = np.linspace(1, 40, 157)
tr = {}
for lab, fe, z in (("0 V, bench", modes["roll, eta_f 0.4, 43 mN"][0], 0.03), ("0 V, in flow", modes["roll, eta_f 0.4, 43 mN"][0], 0.15),
                   ("1800 V, bench", modes["roll, eta_f 0.4, 43 mN"][-1], 0.03), ("1800 V, in flow", modes["roll, eta_f 0.4, 43 mN"][-1], 0.15)):
    tr[lab] = dict(fe=fe, zeta=z, amp=[R2(te_ratio(f, fe, z)[0]) for f in fq], ph=[R2(te_ratio(f, fe, z)[1]) for f in fq])
OUT["te_transfer"] = dict(f=fq.tolist(), cases=tr)
# twist the stroke gives the edge at 25 Hz, +-20 deg: LE moves Phi r, the edge adds amp * Phi r over the chord
for lab in tr:
    a, ph = te_ratio(25, tr[lab]["fe"], tr[lab]["zeta"])
    tr[lab]["twist_deg_25Hz"] = R2(math.degrees(math.atan(a * math.radians(20) * W0["r2"] / 0.016)))
print("twist from the edge at 25 Hz, +-20 deg:", {k: v["twist_deg_25Hz"] for k, v in tr.items()})

# ---------------- 4. thickness window ----------------
tt = np.linspace(20e-6, 200e-6, 91)
E_f = (46e6, 60e6)
win = dict(t_um=[R2(t * 1e6) for t in tt])
for n_e in (2, 4):                              # electrodes: n_e - 1 active layers in the skin, 2 or 4 in the roll
    n_s = n_e - 1
    n_f = 2 if n_e == 2 else 4
    H_l = tt / (n_s + 1)                        # stack 0.5D + E + D + ... + 0.5D: total = (n_s + 1) D
    for E in E_f:
        key = "%d electrodes, %d V/um" % (n_e, E / 1e6)
        Vd = E * H_l
        dF = 0.4 * n_f * P.EPS * E ** 2 * H_l * P.ROLL_WA                 # eta_f 0.4
        dN = 0.15 * n_s * P.EPS * E ** 2 * H_l                             # eta_s 0.15
        win[key] = dict(V=[R2(v) for v in Vd], dF_mN=[R2(v * 1e3) for v in dF], dN=[R2(v) for v in dN])
ms, fr, lw = [], [], []
for t in tt:
    w = inertia(t_film=t, extras=EXTRAS)
    ms.append(R2(w["m"] * 1e6))
    fr.append(R2(f_r(3.5e-3, I=w["I"] + w["I_air"])))
win["wing_mg"] = ms
win["f_res_35"] = fr
# edge tension a flight edge needs: drop at 0 V under lift below 0.5 mm, q_e = p c / 2 at p = 8 Pa
q_fl = 8.0 * 0.016 / 2
win["T_needed_mN"] = R2(q_fl * P.EDGE_L ** 2 / (8 * 0.5e-3) * 1e3)
OUT["thickness"] = win
print("edge tension a flight edge needs: %.0f mN; wing mass at 60 / 90 / 120 um: %s mg" % (win["T_needed_mN"], [ms[int(np.argmin(np.abs(tt - t)))] for t in (60e-6, 90e-6, 120e-6)]))

# ---------------- 5. forward flight ----------------
SPAN_TOT = 2 * 0.0566 + 0.012                 # tip to tip, m (outline 56.6 mm per wing incl. the root slant, body 12 mm)
S_W = 2 * W0["S"]


def p_required(U, W, f=25.0, Phi=math.radians(20), CD0=0.08, e=0.7, SbCd=1e-4, eta_p=0.3):
    """Power to fly level at speed U (W): (parasite + induced + flapping profile drag) U / eta_p. Lift = W."""
    q = 0.5 * P.RHO_AIR * U ** 2
    Di = W ** 2 / (q * math.pi * SPAN_TOT ** 2 * e)
    th = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    Vf = 2 * math.pi * f * Phi * W0["r2"] * np.cos(th)          # flapping speed at the area radius
    kap = np.mean((1 + (Vf / U) ** 2) ** 1.5)
    Dp = q * S_W * CD0 * kap
    Db = q * SbCd
    return (Di + Dp + Db) * U / eta_p, dict(Di=Di, Dp=Dp, Db=Db, kap=kap)


def u_min(W, CLmax=1.1):
    return math.sqrt(2 * W / (P.RHO_AIR * S_W * CLmax))


m_frame, m_trans = 0.15e-3, 0.05e-3          # A
M0 = 2 * P.WING_MASS + 2 * P.CORE_MASS + m_frame + m_trans
W = M0 * P.G
U = np.linspace(2.0, 8.0, 61)
curve = {lab: [R2(p_required(u, W, **kw)[0] * 1e3) for u in U] for lab, kw in (("model", {}), ("optimistic", dict(CD0=0.05, e=0.9, eta_p=0.4)), ("pessimistic", dict(CD0=0.12, e=0.5, eta_p=0.2)))}
avail = {k: R2(2 * P.CORE_MASS * e * 25 * 0.8 * 1e3) for k, e in P.E_CYCLE.items()}
ok_u = U >= u_min(W)
OUT["flight"] = dict(U=U.tolist(), P_req_mW=curve, P_min_mW=R2(min(np.array(curve["model"])[ok_u])), U_at_min=R2(U[ok_u][int(np.argmin(np.array(curve["model"])[ok_u]))]), P_avail_mW_25Hz=avail, mass_g=R2(M0 * 1e3), W_mN=R2(W * 1e3), U_min=R2(u_min(W)), CL_at4=R2(W / (0.5 * P.RHO_AIR * 16 * S_W)),
                     St_25Hz_4ms=R2(25 * 2 * W0["r2"] * 1.6 * math.sin(math.radians(20)) / 4.0))
i4 = int(np.argmin(np.abs(U - 4)))
print("robot %.2f g, weight %.1f mN, U_min %.1f m/s, CL at 4 m/s %.2f, P_req at 4 m/s %.0f mW (min %.0f at %.1f m/s), available at 25 Hz %s mW" % (
    M0 * 1e3, W * 1e3, u_min(W), W / (0.5 * P.RHO_AIR * 16 * S_W), curve["model"][i4], OUT["flight"]["P_min_mW"], OUT["flight"]["U_at_min"], avail))
_, parts = p_required(4.0, W)
print("drag parts at 4 m/s (mN): induced %.2f, profile %.2f (x%.2f from flapping), body %.2f" % (parts["Di"] * 1e3, parts["Dp"] * 1e3, parts["kap"], parts["Db"] * 1e3))

# Monte Carlo over the unknowns: what fraction can hold level flight, and the margin
n = 20000
draw = dict(e_cyc=rng.uniform(0.5, 2.1, n), eta_tr=rng.uniform(0.6, 0.9, n), f=rng.uniform(18, 28, n), CD0=rng.uniform(0.05, 0.12, n), e=rng.uniform(0.5, 0.9, n),
            eta_p=rng.uniform(0.2, 0.4, n), SbCd=rng.uniform(0.5e-4, 1.5e-4, n), CLmax=rng.uniform(0.9, 1.4, n), m_frame=rng.uniform(0.1e-3, 0.25e-3, n),
            m_w=rng.uniform(0.25e-3, 0.30e-3, n), m_c=rng.uniform(0.5e-3, 0.6e-3, n), Phi=np.radians(rng.uniform(15, 30, n)))
margin, ubest = np.zeros(n), np.zeros(n)
for i in range(n):
    d = {k: v[i] for k, v in draw.items()}
    Wt = (2 * d["m_w"] + 2 * d["m_c"] + d["m_frame"] + 0.05e-3) * P.G
    um = u_min(Wt, d["CLmax"])
    uu = np.linspace(um, 8, 25)
    pr = [p_required(u, Wt, f=d["f"], Phi=d["Phi"], CD0=d["CD0"], e=d["e"], SbCd=d["SbCd"], eta_p=d["eta_p"])[0] for u in uu]
    j = int(np.argmin(pr))
    pa = 2 * d["m_c"] * d["e_cyc"] * d["f"] * d["eta_tr"]
    margin[i], ubest[i] = pa / pr[j], uu[j]
corr = {k: R2(np.corrcoef(v, np.log(margin))[0, 1]) for k, v in draw.items()}
OUT["mc"] = dict(n=n, margin_pct=[R2(np.percentile(margin, q)) for q in (5, 25, 50, 75, 95)], p_fly=R2(np.mean(margin >= 1)), corr=corr,
                 U_best_pct=[R2(np.percentile(ubest, q)) for q in (5, 50, 95)], hist=np.histogram(margin, bins=40, range=(0, 2))[0].tolist(),
                 e_cyc=draw["e_cyc"][:3000].tolist(), margin=margin[:3000].tolist())
print("Monte Carlo: power margin 5/25/50/75/95 %%: %s; share >= 1: %.2f; best speed %s m/s" % (OUT["mc"]["margin_pct"], OUT["mc"]["p_fly"], OUT["mc"]["U_best_pct"]))
print("correlations with log margin:", corr)

# what energy per cycle it needs, against core share of mass
fr_c = np.linspace(0.3, 0.8, 26)
need = []
for fc in fr_c:
    # robot of mass M with cores fc M; holding the rest at 0.75 g (wings 0.55, frame and transmission 0.2)
    Mr = 0.75e-3 / (1 - fc)
    Wt = Mr * P.G
    um = u_min(Wt)
    pr = min(p_required(u, Wt)[0] for u in np.linspace(um, 8, 25))
    need.append(R2(pr / (fc * Mr * 25 * 0.8)))
OUT["need"] = dict(core_frac=fr_c.tolist(), e_needed=need)
print("energy per cycle needed at 25 Hz, core share 0.6: %.1f J/kg" % need[int(np.argmin(np.abs(fr_c - 0.6)))])

json.dump(OUT, open(os.path.join(HERE, "data", "robot_study.json"), "w"))
