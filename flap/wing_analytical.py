"""Active membrane wing: analytical model, one file, both wings. Change the parameters below and run:  python wing_analytical.py

Each panel is a rectangle a x b (a between supports, b chord), all four edges held: LE and sides by the frame, TE by the fibre, which is assumed to move little.
It bulges in one shape:  w = h sin(pi x / a) sin(pi y / b),  so the whole problem is one cubic equation for the bulge height h:

        (T - D) h + K h^3 = q Phi a_L alpha

  T = n_x X + n_y Y           stiffness from the tension in the sheet
  D = q Phi a_L c_cam / b     aerodynamic softening: more camber, more lift, more load. T = D is static divergence
  K = 2 G t (X^2 + X Y + Y^2) stiffening: the bulge stretches the sheet. If T - D < 0 the wing sits near h = sqrt((D - T) / K), a buckled, cambered state
  X = pi^2 / 4 a^2,  Y = pi^2 / 4 b^2,  Phi = 4 / pi^2,  G = 2 C1,  q = rho U^2 / 2
  c_cam (= CAMBER below) converts the bulge to an extra incidence, thin-airfoil theory on the chord shape, times a gain

Tension, incompressible, small strain, frame fixed:  n_x = 2 G t (2 e_x + e_y),  n_y = 2 G t (e_x + 2 e_y)
Voltage, full membrane:  P = eta n eps0 epsr V^2 / H;  taut: n_x - P, n_y - P;  once n_y reaches 0 the sheet wrinkles: n_x = 3 G t e_x - P / 2, n_y = 0
Voltage, discrete DEAs:  the chord tension is n_y + r max(F_D - dF (V / 1.6 kV)^2, 0) / s, with r the load-path factor
Lift:  C_L = a_L (alpha + c_cam h / b), each panel with its own alpha and q when flapping.
"""
import math
import numpy as np

EPS0 = 8.854e-12

COMMON = dict(
    U=4.0, alpha_deg=6.0, rho=1.225, V=1600.0,      # flow and drive
    area=33.3e-4,                                   # one wing (m2)
    lift_slope=3.71,                                # a_L (per rad): flat wing, from the vortex lattice
    camber_gain=1.0,                                # multiplies the thin-airfoil camber angle (1 = theory)
    stretch_gain=0.25,                              # multiplies K. The one fitted number: a real bulge is not uniform, so one shape over-counts stretching. See check_analytical.py
    Phi0_deg=30.0, f=2.0,                           # stroke amplitude and frequency, for cycle()
)

FULL = dict(
    C1=42842.0, H=35e-6, eps_r=2.8, eta=0.40,       # P7670; dielectric per film; eta of a flat sheet is NOT measured
    n_layers=(1, 1),                                # active layers per panel. Stack: 0.5D + E + (D + E) x n + 0.5D, so t = (n + 1) H
    lam_d=(1.05, 1.05), lam_n=(1.016, 1.016),       # stretch along and across the fibre, per panel
    roll_width=(13e-3, 13e-3),                      # rolled strip as cut (m): the trailing-edge cable
    edge=(0.50, 0.20),                              # how much chord tension survives the edge relaxing: mean n_y / free n_y (from the FE)
    a=(0.040, 0.068), b=(0.046, 0.031), share=(0.54, 0.46), radius=(0.03, 0.075),   # span, chord, area share, distance from the stroke axis
)

DISCRETE = dict(
    C1=3252.0, t=180e-6, lam_x=1.03, lam_y=1.03,    # membrane: isotropic part (v_m C1), thickness, pre-stretch
    E_x=68770.0, E_y=143990.0,                      # composite stiffness along span and chord (Pa)
    dF=8.6e-3, F_D=14.1e-3, s=0.012,                # DEA: force drop at 1.6 kV, force at its pre-stretch (N), spacing (m)
    r=1.0,                                          # load path: 1 rigid TE, 0.81 tendon of 0.5 N, 0.02 free TE. Sets how much of the DEA force reaches the membrane
    a=(0.1042, 0.1042), b=(0.046, 0.031), share=(0.54, 0.46), radius=(0.03, 0.075),   # no batten: the span between supports is the whole wing
)

# Thin-airfoil incidence of the chord shape g = sin(pi y / b), per unit h / b, times the span mean of sin(pi x / a) = 2 / pi
_th = np.linspace(0, math.pi, 4001)
_y = 0.5 * (1 - np.cos(_th))
CAMBER = float(-np.trapezoid(math.pi * np.cos(math.pi * _y) * (1 - np.cos(_th)), _th) / math.pi * 2 / math.pi)


def full_tension(w, i, V):
    t, G = (w["n_layers"][i] + 1) * w["H"], 2 * w["C1"]
    P = w["eta"] * w["n_layers"][i] * EPS0 * w["eps_r"] * V ** 2 / w["H"]
    neck = 0.5 * (w["lam_d"][i] - 1)                                   # natural contraction
    ex, ey = w["lam_d"][i] - 1, w["edge"][i] * (w["lam_n"][i] - 1 + neck) - neck
    nx, ny = 2 * G * t * (2 * ex + ey), 2 * G * t * (ex + 2 * ey)
    if ny - P >= 0:
        nx, ny = nx - P, ny - P
    else:
        nx, ny = 3 * G * t * ex - P / 2, 0.0
    F = w["roll_width"][i] * t * 2 * (w["lam_d"][i] - w["lam_d"][i] ** -2) * w["C1"]
    return nx, ny, F, t, G


def discrete_tension(w, i, V):
    t, G = w["t"], 2 * w["C1"]
    ex, ey = w["lam_x"] - 1, w["lam_y"] - 1
    nx = 2 * G * t * (2 * ex + ey) + w["E_x"] * t * ex
    ny = 2 * G * t * (ex + 2 * ey) + w["E_y"] * t * ey + w["r"] * max(w["F_D"] - w["dF"] * (V / 1600.0) ** 2, 0.0) / w["s"]
    return nx, ny, 0.0, t, G + 0.5 * (w["E_x"] + w["E_y"]) / 6


def panel(w, i, V, q, alpha, c, tension, h_prev=None):
    """The cubic. Returns bulge height h (m), C_L of the panel, and the terms.
    Picks a stable root: on the side of the load or, if h_prev is given, the one nearest to it, so that a stroke follows its branch."""
    nx, ny, F, t, G = tension(w, i, V)
    a, b = w["a"][i], w["b"][i]
    X, Y, Phi = math.pi ** 2 / (4 * a * a), math.pi ** 2 / (4 * b * b), 4 / math.pi ** 2
    k_cam = c["camber_gain"] * CAMBER / b
    T = nx * X + ny * Y
    D = q * Phi * c["lift_slope"] * k_cam
    K = c["stretch_gain"] * 2 * G * t * (X * X + X * Y + Y * Y)
    rhs = q * Phi * c["lift_slope"] * alpha
    r = np.roots([K, 0.0, T - D, -rhs])
    r = r[np.abs(r.imag) < 1e-9].real
    r = r[3 * K * r * r + T - D > 0]                                   # stable roots only
    h = float(r[np.argmax(r * (np.sign(alpha) if alpha != 0 else 1.0))]) if h_prev is None else float(r[np.argmin(np.abs(r - h_prev))])
    return h, c["lift_slope"] * (alpha + k_cam * h), dict(T=T, D=D, K=K, nx=nx, ny=ny, F=F)


def solve(wing, tension, V=None, **flow):
    c = dict(COMMON, **flow)
    q, V = 0.5 * c["rho"] * c["U"] ** 2, c["V"] if V is None else V
    al = math.radians(c["alpha_deg"])
    out = [panel(wing, i, V, q, al, c, tension) for i in range(len(wing["a"]))]
    CL = sum(s * o[1] for s, o in zip(wing["share"], out))
    return dict(CL=CL, L_mN=CL * q * c["area"] * 1e3, h_mm=[o[0] * 1e3 for o in out], info=[o[2] for o in out])


def cycle(wing, tension, V=None, **flow):
    """Lift over a stroke, quasi-steady. Each panel sees alpha and q at its own radius, so the outer panel can cross zero while the inner one does not."""
    c = dict(COMMON, **flow)
    V = c["V"] if V is None else V
    A, om = math.radians(c["Phi0_deg"]), 2 * math.pi * c["f"]
    phases = np.arange(0.0, 360.0, 30.0)
    h = [None] * len(wing["a"])
    for pass_ in (0, 1):                                               # the first stroke settles the branch
        L = []
        for p_ in phases:
            Om = -A * om * math.sin(math.radians(p_))
            tot = 0.0
            for i in range(len(wing["a"])):
                ve = Om * wing["radius"][i]
                q = 0.5 * c["rho"] * (c["U"] ** 2 + ve ** 2)
                al = math.radians(c["alpha_deg"]) - math.atan(ve / c["U"])
                h[i], cl, _ = panel(wing, i, V, q, al, c, tension, h[i])
                tot += wing["share"][i] * cl * q
            L.append(tot * c["area"] * 1e3)
    return phases, np.array(L)


def report(name, wing, tension):
    print(name)
    for U in (3.0, 4.0, 6.0):
        a, b = solve(wing, tension, V=0.0, U=U), solve(wing, tension, U=U)
        print(f"  U {U:g} m/s:  C_L {a['CL']:.2f} to {b['CL']:.2f},  L {a['L_mN']:5.1f} mN,  dL {b['L_mN'] - a['L_mN']:5.1f} mN,  h 0 V {np.round(a['h_mm'], 1).tolist()} mm,  h on {np.round(b['h_mm'], 1).tolist()} mm")
    ph, L0 = cycle(wing, tension, V=0.0)
    ph, L1 = cycle(wing, tension)
    print("  cycle (4 m/s, 2 Hz) dL by phase:", np.round(L1 - L0, 1).tolist(), f"mean {np.mean(L1 - L0):.1f} mN")


if __name__ == "__main__":
    report("Full membrane", FULL, full_tension)
    report("Discrete DEAs, ideal rigid TE", DISCRETE, discrete_tension)
