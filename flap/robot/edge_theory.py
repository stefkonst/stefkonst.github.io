"""Edge equation: the trailing edge as a string of tension T held by the skin behind it.

  drop of the edge, out of plane   d = q_e / (8 T / L^2 + N / c)        two springs in parallel: edge string, skin strip
  bow of the edge, in plane        s = N L^2 / (8 T)
  edge tension                     T = T0 - dF_eff(V) + k_e (8 / 3L) (d^2 + s^2 - d0^2 - s0^2)     (>= 0)
  skin tension across              N = N0 - dN_s(V) + (E t / c) (8 / 3c) (h^2 - h0^2)           (>= 0)
  skin bulge                       h = p_s c^2 / (8 N)
  active force lost by the edge    dF = eta_f n_f eps (V/H)^2 H w_a ;  in series with a passive part: dF_eff = dF k_p / (k_a + k_p)
  skin tension lost                dN_s = eta_s n eps (V/H)^2 H
q_e = load per length the edge carries: its own weight plus half the chord's net load. When T reaches 0, only the skin holds the edge:
d -> q_e c / N, which is what makes the motion jump. Run: python edge_theory.py -> data/edge_theory.json"""
import json, math, os
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))
import params as P

L, C = P.EDGE_L, 0.016            # edge length; mean chord behind it (outline: 24 mm at the root, 8 at the tip)
T_ = 60e-6                         # skin, all layers
ET = 6 * P.C1 * 1.5 * T_           # skin E t, N/m (Yeoh small-strain modulus, stiffened by the stretch)


def cord(name):
    c = P.CORDS[name]
    return dict(EA=c["E"] * c["A"], mu=c["rho"] * c["A"])


def solve(V, T0=0.07, N0=0.2, eta_f=0.4, eta_s=0.15, p=0.0, edge="roll", a=1.0, passive="roll", H=P.H, n_f=2, n_s=1, t=T_, mu=None, c=C, Le=L):
    """Edge and skin state at voltage V (V), lift pressure p (Pa, up). Returns dict in mm and mN.
    edge: 'roll' active roll over the inner fraction a, the rest 'passive' ('roll' unpowered, or a cord name); or a cord name (no roll)."""
    roll = dict(EA=6 * P.C1 * 2.16 * P.ROLL_W * t, mu=P.RHO_F * P.ROLL_W * t / 1.2)     # tangent EA near lam 1.2; mass per stretched length
    if edge == "roll":
        if a >= 1 or passive == "roll":
            k_a, k_p = roll["EA"] / (a * Le), roll["EA"] / max((1 - a) * Le, 1e-9)
            mu_e = roll["mu"]
        else:
            cp = cord(passive)
            k_a, k_p = roll["EA"] / (a * Le), cp["EA"] / ((1 - a) * Le)
            mu_e = a * roll["mu"] + (1 - a) * cp["mu"]
        k_anchor = 2000.0                                   # thread to the body, stiff
        k_e = 1 / (1 / k_a + 1 / k_p + 1 / k_anchor) if a < 1 else 1 / (1 / k_a + 1 / k_anchor)
        dF = P.roll_dF(V, eta_f, n_f, P.ROLL_WA, H)
        dF_eff = dF * (1 / (1 / k_p + 1 / k_anchor)) / (k_a + 1 / (1 / k_p + 1 / k_anchor)) if a < 1 else dF * k_anchor / (k_a + k_anchor)
    else:
        cp = cord(edge)
        k_e, dF_eff, mu_e = cp["EA"] / Le, 0.0, cp["mu"]
    mu_e = mu if mu is not None else mu_e
    dN = eta_s * n_s * P.EPS * (V / H) ** 2 * H
    w_film = P.RHO_F * t * P.G
    q_e = mu_e * P.G + (w_film - p) * c / 2               # + down
    p_s = p - w_film
    Et = 6 * P.C1 * 1.5 * t

    def state(T, N):
        s = N * Le ** 2 / (8 * max(T, 1e-9))
        d = q_e / (8 * T / Le ** 2 + max(N, 1e-9) / c)
        h = p_s * c ** 2 / (8 * max(N, 1e-9))
        return s, d, h

    # 0 V reference
    s0, d0, h0 = state(T0, N0)

    def skin_N(Nt):
        h = abs(p_s) * c ** 2 / (8 * max(Nt, 1e-9))
        return N0 - dN + Et / c * 8 / (3 * c) * (h ** 2 - h0 ** 2) * c - Nt     # E t (8/3)(h/c)^2 strain

    N = brentq(skin_N, 1e-7, 50.0) if skin_N(1e-7) > 0 else 1e-7

    def edge_T(Tt):
        s, d, _ = state(Tt, N)
        return T0 - dF_eff + k_e * 8 / (3 * Le) * (d ** 2 + s ** 2 - d0 ** 2 - s0 ** 2) - Tt

    T = brentq(edge_T, 1e-7, 10.0) if edge_T(1e-7) > 0 else 1e-7
    s, d, h = state(T, N)
    return dict(V=V, T=T * 1e3, N=N, d=d * 1e3, s=s * 1e3, h=h * 1e3, d_edge_only=q_e * Le ** 2 / (8 * T0) * 1e3, k_e=k_e, dF=dF_eff * 1e3, q_e=q_e)


def edge_mode(T, N, mu_e, t=T_, c=C, Le=L):
    """First out-of-plane mode of the edge with the skin behind it, Hz: string T (pi/L)^2 plus skin N/c, over edge mass + a third of the skin strip."""
    m = mu_e + P.RHO_F * t * c / 3
    return math.sqrt((T * (math.pi / Le) ** 2 + N / c) / m) / (2 * math.pi)


if __name__ == "__main__":
    Vs = np.linspace(0, 1800, 37)
    out = {}
    cases = {"roll, eta_f 0.4": dict(edge="roll", eta_f=0.4), "roll, eta_f 0.8": dict(edge="roll", eta_f=0.8, T0=0.083), "roll, passive": dict(edge="roll", eta_f=0.0),
             "nylon 0.10 mm": dict(edge="nylon 0.10 mm"), "TPU 1 x 0.1 mm": dict(edge="TPU 1 x 0.1 mm"),
             "inner half active, outer roll": dict(edge="roll", a=0.5, eta_f=0.4), "inner half active, outer TPU": dict(edge="roll", a=0.5, passive="TPU 1 x 0.1 mm", eta_f=0.4),
             "inner half active, outer nylon": dict(edge="roll", a=0.5, passive="nylon 0.10 mm", eta_f=0.4)}
    for p in (0.0, 8.0):
        for nm, kw in cases.items():
            out["%s | p %g" % (nm, p)] = [solve(V, p=p, **kw) for V in Vs]
    for k, v in out.items():
        print("%-45s" % k, " ".join("%4.0fV d %+.2f T %4.1f" % (r["V"], r["d"], r["T"]) for r in v[::9]))
    json.dump(out, open(os.path.join(HERE, "data", "edge_theory.json"), "w"), indent=0)
    rmu = P.RHO_F * P.ROLL_W * T_ / 1.2
    print("edge mode 0 V, T 70 mN: %.0f Hz; T 5 mN %.0f Hz; T 0, N 0.15: %.0f Hz" % (edge_mode(.07, .2, rmu), edge_mode(.005, .2, rmu), edge_mode(0, .15, rmu)))
