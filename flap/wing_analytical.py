"""Active membrane wing: one analytical model for both wings. Change the parameters below and run:  python wing_analytical.py
Equations A1-A9 are written out, in the same notation, at https://stefkonst.github.io/flap/model/
Notation as in Konstantinidi et al., Smart Mater. Struct. 2023 and Compos. Sci. Technol. 2024: V, H, H_p, chi = H / (H + H_p), lambda_pre, C_i, A, F_c.

A1  Shape        w = A sin(pi x / a) sin(pi y / b),  amplitude A
A2  Energy       U(A) = (T - D) A^2 / 2 + K A^4 / 4 - q Phi a_L alpha A     (same form as U = U_s A^2 + U_b + U_f of the buckling model)
A3  Balance      dU/dA = 0:  (T - D) A + K A^3 = q Phi a_L alpha
A4  Stiffness    T = n_x X + n_y Y,  X = pi^2 / 4 a^2,  Y = pi^2 / 4 b^2,  Phi = 4 / pi^2
A5  Softening    D = q Phi a_L k_cam c_cam / b.  T = D is static divergence, the analogue of F = F_c
A6  Stretching   K = gamma 2 G (H + H_p) (X^2 + X Y + Y^2),  G = 2 C_1.  With no load, past divergence: A = sqrt((D - T) / K)
A7  Lift         C_L = a_L (alpha + k_cam c_cam A / b)
A8  Full sheet   n_x = 2 G (H + H_p) (2 e_x + e_y) - P,  n_y = 2 G (H + H_p) (e_x + 2 e_y) - P,  P = eta chi eps0 epsr (V / H)^2 (H + H_p).  At n_y = 0: n_x = 3 G (H + H_p) e_x - P / 2
    Discrete     n_y = n_y,membrane + r max(F_D - dF (V / 1.6 kV)^2, 0) / s.  r = k_a / (k_a + k_d),  k_a = 8 T_c s / L_s^2.  Free trailing edge: T_c = 0, r = 0 (Eq. S13)
A9  Active fibre F = F_0 - dF_f,  F_0 = 2 C_1 A_0 (lambda - lambda^-2),  dF_f = eta_f eps0 epsr (V / H)^2 A_0 w_e / w_s.  Stays in tension if F_0 > dF_f (near lambda - 1 = eta_f eps0 epsr (V / H)^2 (w_e / w_s) / (6 C_1))
"""
import math
import numpy as np
from scipy.optimize import brentq

EPS0 = 8.854e-12

COMMON = dict(
    U=4.0, alpha_deg=6.0, rho=1.225, V=1600.0,      # flow and drive
    area=33.3e-4,                                   # one wing (m2)
    lift_slope=3.71,                                # a_L (per rad): flat wing of this planform, from a vortex lattice
    gamma=0.7,                                      # A6: the one fitted number. One bulge shape over-counts stretching (check_analytical.py)
    Phi0_deg=30.0, f=2.0,                           # stroke amplitude and frequency, for cycle()
)

FULL = dict(
    C1=42842.0, H=35e-6, eps_r=2.8, eta=0.50,       # P7670, dielectric per film, flat-sheet eta (estimate: no rolled air gaps; NOT measured)
    n_layers=(2, 2),                                # active layers per panel. Stack 0.5D + E + (D + E) x n + 0.5D: H_m = (n + 1) H, chi = n / (n + 1)
    lam_d=(1.08, 1.08), lam_n=(1.002, 1.002),       # stretch along and across the fibre, per panel (necking + 0.04)
    roll_width=(13e-3, 13e-3), roll_margin=2e-3, eta_f=0.4,   # rolled strip as cut, electrode-free width, eta of the roll (bench, isometric)
    edge=(0.50, 0.20),                              # mean n_y / free n_y once the edge has relaxed (from the FE)
    a=(0.040, 0.068), b=(0.046, 0.031), share=(0.54, 0.46), radius=(0.03, 0.075),   # span, chord, area share, distance from the stroke axis
    length_fibre=(0.040, 0.068),
)

ROTATED = dict(                                     # full membrane, cut unstretched and tensioned by bringing the root edge and batten to vertical
    C1=42842.0, H=35e-6, eps_r=2.8, eta=0.50, n_layers=(2, 2), lam_n=1.04,
    theta_b=5.0, theta_r=8.85,                      # tilt of the batten and root edge as cut, in degrees. theta_r > theta_b so the batten balances (balance_study.py)
    x_root=3.8e-3, x_batten=60e-3, x_tip=108e-3, chord=41.1e-3, tip_drop=3.1e-3, fibre_2_angle=38.0,
    edge=(0.5, 0.2), edge_rot=(0.79, 1.0),          # edge_rot: mean tension / uniform tension in the rotation state (from the FE)
    a=(0.056, 0.061), b=(0.041, 0.016), share=(0.70, 0.30), radius=(0.03, 0.08), lift_slope=4.39,   # flat wing with a vertical root edge
)

DISCRETE = dict(
    C1=3252.0, H_m=180e-6, lam_x=1.03, lam_y=1.03,  # membrane: isotropic part (v_m C_1), thickness, pre-stretch
    E_x=68770.0, E_y=143990.0,                      # composite stiffness along span and chord (Pa)
    dF=8.6e-3, F_D=14.1e-3, s=0.012, k_d=3.1,       # DEA: force drop at 1.6 kV, force at its pre-stretch (N), spacing (m), stiffness (N/m, Eq. S10)
    T_c=0.5,                                        # trailing-edge tension (N). 0 = free trailing edge
    L_s=0.1042,                                     # its span (m)
    a=(0.1042, 0.1042), b=(0.046, 0.031), share=(0.54, 0.46), radius=(0.03, 0.075),
)

# k_cam = 1.476 makes A3 reproduce the exact 2D string limit C_T,crit = 1.727 (Eq. S5) when a_L = 2 pi. c_cam from thin-airfoil theory on the chord shape, times the span mean 2 / pi.
_th = np.linspace(0, math.pi, 4001)
CAMBER = float(-np.trapezoid(math.pi * np.cos(math.pi * 0.5 * (1 - np.cos(_th))) * (1 - np.cos(_th)), _th) / math.pi * 2 / math.pi)
K_CAM = 1.727 * (math.pi ** 2 / 4) / ((4 / math.pi ** 2) * 2 * math.pi * CAMBER)


def load_path(w):
    """A8: fraction of the DEA force that reaches the membrane. Free trailing edge: 0, from Eq. S13."""
    if w["T_c"] <= 0:
        return 0.0
    k_a = 8 * w["T_c"] * w["s"] / w["L_s"] ** 2
    return k_a / (k_a + w["k_d"])


def fibre_force(w, i, V):
    """A9: force in the active rolled fibre (N), and the stretch it needs to stay in tension at V."""
    Hm = (w["n_layers"][i] + 1) * w["H"]
    A0 = w["roll_width"][i] * Hm
    frac = (w["roll_width"][i] - w["roll_margin"]) / w["roll_width"][i]
    s_max = w["eta_f"] * EPS0 * w["eps_r"] * (V / w["H"]) ** 2 * frac
    lam = w["lam_d"][i]
    limit = brentq(lambda l: 2 * (l - l ** -2) * w["C1"] - s_max, 1.0, 2.0)
    return A0 * 2 * (lam - lam ** -2) * w["C1"] - s_max * A0, limit


def full_tension(w, i, V):
    n = w["n_layers"][i]
    Hm, G = (n + 1) * w["H"], 2 * w["C1"]
    P = w["eta"] * n * EPS0 * w["eps_r"] * V ** 2 / w["H"]             # A7, using chi H_m = n H
    neck = 0.5 * (w["lam_d"][i] - 1)
    ex, ey = w["lam_d"][i] - 1, w["edge"][i] * (w["lam_n"][i] - 1 + neck) - neck
    nx, ny = 2 * G * Hm * (2 * ex + ey), 2 * G * Hm * (ex + 2 * ey)
    if ny - P >= 0:
        return nx - P, ny - P, Hm, G
    return 3 * G * Hm * ex - P / 2, 0.0, Hm, G


def rotation_geometry(w):
    """A10: the stretch of each fibre when the root edge (tilt theta_r) and batten (tilt theta_b) rotate about the LE to vertical. All lengths in m."""
    b = w["chord"]
    tb, tr = math.tan(math.radians(w["theta_b"])), math.tan(math.radians(w["theta_r"]))
    L1 = w["x_batten"] - w["x_root"]
    dy = w["chord"] - w["tip_drop"]
    L2_0 = math.hypot(w["x_tip"] - (w["x_batten"] + b * tb), dy)
    L2 = math.hypot(w["x_tip"] - w["x_batten"], dy)
    return L1 / (L1 + b * (tb - tr)), L2 / L2_0, 0.5 * (tb + tr)


def rotation_tension(w, i, V):
    """A10: sheet cut unstretched. Panel 1 shears as it goes vertical; panel 2 is pulled along its fibre. Both are taut along one direction, with a small chord stretch lam_n across it."""
    n = w["n_layers"][i]
    Hm, G = (n + 1) * w["H"], 2 * w["C1"]
    P = w["eta"] * n * EPS0 * w["eps_r"] * V ** 2 / w["H"]
    lam1, lam2, gam = rotation_geometry(w)
    if i == 0:
        g = lam1 * gam
        tr_, det = lam1 ** 2 + g ** 2 + 1.0, lam1 ** 2
        l2 = 0.5 * tr_ + math.sqrt(0.25 * tr_ ** 2 - det)                      # largest eigenvalue of F F^T
        phi = 0.5 * math.atan2(2 * g, lam1 ** 2 + g ** 2 - 1.0)
    else:
        l2, phi = lam2 ** 2, math.radians(w["fibre_2_angle"])
    l = math.sqrt(l2)
    N1 = max(2 * Hm * w["C1"] * (l2 - 1 / l) / math.sqrt(l) * w["edge_rot"][i] - P / 2, 0.0)
    ey = w["lam_n"] - 1
    return N1 * math.cos(phi) ** 2, N1 * math.sin(phi) ** 2 + max(2 * G * Hm * 2 * ey * w["edge"][i] - P, 0.0), Hm, G


def discrete_tension(w, i, V):
    Hm, G = w["H_m"], 2 * w["C1"]
    ex, ey = w["lam_x"] - 1, w["lam_y"] - 1
    r = load_path(w)
    nx = 2 * G * Hm * (2 * ex + ey) + w["E_x"] * Hm * ex
    ny = 2 * G * Hm * (ex + 2 * ey) + w["E_y"] * Hm * ey + r * max(w["F_D"] - w["dF"] * (V / 1600.0) ** 2, 0.0) / w["s"]
    return nx, ny, Hm, G + 0.5 * (w["E_x"] + w["E_y"]) / 6


def panel(w, i, V, q, alpha, c, tension, h_prev=None):
    """A3: the cubic. Returns amplitude A (m), C_L of the panel, and the terms. Takes a stable root: on the side of the load, or nearest h_prev so a stroke follows its branch."""
    nx, ny, Hm, G = tension(w, i, V)
    a, b = w["a"][i], w["b"][i]
    X, Y, Phi = math.pi ** 2 / (4 * a * a), math.pi ** 2 / (4 * b * b), 4 / math.pi ** 2
    k_cam = K_CAM * CAMBER / b
    T = nx * X + ny * Y
    D = q * Phi * c["lift_slope"] * k_cam
    K = c["gamma"] * 2 * G * Hm * (X * X + X * Y + Y * Y)
    r = np.roots([K, 0.0, T - D, -q * Phi * c["lift_slope"] * alpha])
    r = r[np.abs(r.imag) < 1e-9].real
    r = r[3 * K * r * r + T - D > 0]
    h = float(r[np.argmax(r * (np.sign(alpha) if alpha != 0 else 1.0))]) if h_prev is None else float(r[np.argmin(np.abs(r - h_prev))])
    return h, c["lift_slope"] * (alpha + k_cam * h), dict(T=T, D=D, K=K, nx=nx, ny=ny)


def solve(wing, tension, V=None, **flow):
    c = dict(COMMON, **flow)
    c["lift_slope"] = wing.get("lift_slope", c["lift_slope"])
    q, V = 0.5 * c["rho"] * c["U"] ** 2, c["V"] if V is None else V
    al = math.radians(c["alpha_deg"])
    out = [panel(wing, i, V, q, al, c, tension) for i in range(len(wing["a"]))]
    CL = sum(s * o[1] for s, o in zip(wing["share"], out))
    return dict(CL=CL, L_mN=CL * q * c["area"] * 1e3, h_mm=[o[0] * 1e3 for o in out], info=[o[2] for o in out])


def cycle(wing, tension, V=None, **flow):
    """Lift over a stroke, quasi-steady (Eq. S12). Each panel has alpha and q at its own radius."""
    c = dict(COMMON, **flow)
    c["lift_slope"] = wing.get("lift_slope", c["lift_slope"])
    V = c["V"] if V is None else V
    A, om = math.radians(c["Phi0_deg"]), 2 * math.pi * c["f"]
    phases = np.arange(0.0, 360.0, 15.0)
    h = [None] * len(wing["a"])
    for _ in (0, 1):                                                   # the first stroke settles the branch
        L = []
        for p_ in phases:
            Om = -A * om * math.sin(math.radians(p_))
            tot = 0.0
            for i in range(len(wing["a"])):
                ve = Om * wing["radius"][i]
                q = 0.5 * c["rho"] * (c["U"] ** 2 + ve ** 2)
                al = math.radians(c["alpha_deg"]) - math.atan(ve / c["U"])
                h[i], cl, _i = panel(wing, i, V, q, al, c, tension, h[i])
                tot += wing["share"][i] * cl * q
            L.append(tot * c["area"] * 1e3)
    return phases, np.array(L)


def delta_cycle(wing, tension, **flow):
    return cycle(wing, tension, **flow)[1] - cycle(wing, tension, V=0.0, **flow)[1]


def mean_delta(wing, tension, **flow):
    """Cycle-mean lift change, leaving out 240-300 deg: there the outer panel's angle crosses zero, and the model and the FE disagree on which state flips."""
    ph = cycle(wing, tension, **flow)[0]
    keep = (ph < 240) | (ph > 300)
    return float(delta_cycle(wing, tension, **flow)[keep].mean())


def report(name, wing, tension):
    print(name)
    for U in (3.0, 4.0, 6.0):
        a, b = solve(wing, tension, V=0.0, U=U), solve(wing, tension, U=U)
        print(f"  U {U:g} m/s:  C_L {a['CL']:.2f} to {b['CL']:.2f},  L {a['L_mN']:5.1f} mN,  dL {b['L_mN'] - a['L_mN']:5.1f} mN")
    print(f"  mean dL over a stroke, 4 m/s, 2 Hz, leaving out 240-300 deg: {mean_delta(wing, tension):.1f} mN")


if __name__ == "__main__":
    report("Full membrane", FULL, full_tension)
    F, lam = fibre_force(FULL, 0, COMMON["V"])
    print(f"  active fibre: {F * 1e3:.1f} mN at {COMMON['V']:.0f} V; stays in tension above lam_d = {lam:.3f}")
    report(f"Discrete DEAs, trailing-edge tension {DISCRETE['T_c']} N, r = {load_path(DISCRETE):.2f}", DISCRETE, discrete_tension)
    report("Discrete DEAs, free trailing edge, r = 0", dict(DISCRETE, T_c=0.0), discrete_tension)
    report(f"Full membrane, cut tilted {ROTATED['theta_b']:g} and {ROTATED['theta_r']:g} deg, no pre-stretch", ROTATED, rotation_tension)
    print("  fibre stretch from the rotation: %.3f, %.3f" % rotation_geometry(ROTATED)[:2])
