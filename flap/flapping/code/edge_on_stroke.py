"""One number for the wing's state: the stiffness kappa of a one-sided (taut-or-slack) trailing edge on the stroke.

Start from the powered robot, which harmonic balance describes with one real quadratic term (beta_p, the same at 7 and
8 Hz and at 1400 V) on top of the one-mode fit (f_n, zeta, tau, G from the sweep). Add the edge as a spring that pulls
only on one side of the stroke:
    x'' + 2 zeta w x' + w^2 x + beta_p x^2 + kappa w^2 * min(x - x_s, 0) = G u(t)
kappa = edge stiffness on its taut side / stroke stiffness; x_s = angle where it goes slack (0: at the mean stroke).
Powered = kappa 0 (the roll slack at voltage). Sweep kappa and x_s, and ask whether one value gives BOTH the measured
size ratio (unpowered/powered fundamental 0.73) and the measured asymmetry of the unpowered run (harmonic balance).
Run: python edge_on_stroke.py -> ../data/edge_on_stroke.json"""
import os, json, numpy as np
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
D2R = np.pi / 180
SW = json.load(open(os.path.join(DATA, "sweepfit.json"))); P = SW["fits"]["zeta 0.07 (decay)"]
HB = json.load(open(os.path.join(DATA, "harmonic_balance.json")))
R = {r["run"]: r for r in SW["runs"]}
PW, UN = "full_DEA_core_wing_8hz_1200V_side", "only_DEA_core_8hz_1200V_side"
beta_p = HB[PW]["cases"]["fn 8.38 (sweep fit)"]["beta_re"]
w = 2 * np.pi * P["fn"]; zeta = P["zeta"]; tau = P["tau"]


@njit(cache=True)
def sim(w, zeta, beta, kappa, xs, G, tau, f, T, dt):
    n = int(T / dt); x = 0.0; v = 0.0; u = 0.0; out = np.empty(n)
    for k in range(n):
        t = k * dt
        s = np.sin(np.pi * f * t) ** 4
        u += dt * (s - u) / tau
        d = x - xs
        edge = kappa * w * w * d if d < 0 else 0.0
        a = G * u - 2 * zeta * w * v - w * w * x - beta * x * x - edge
        v += dt * a; x += dt * v
        out[k] = x
    return out


def harm(kappa, xs, G, f=8.0, beta=beta_p):
    dt = 1 / 4000; th = sim(w, zeta, beta, kappa, xs, G, tau, f, 4.0, dt); t = np.arange(len(th)) * dt
    m = t > 2.5; y = th[m]; tt = t[m]; W = 2 * np.pi * f
    X1 = 2 * np.mean(y * np.exp(-1j * W * tt)); X2 = 2 * np.mean(y * np.exp(-2j * W * tt))
    return X1, X2, y


# G so that the powered model (kappa 0) gives the measured powered fundamental
from scipy.optimize import brentq
A1p = R[PW]["A"][0] * D2R
G = brentq(lambda g: abs(harm(0.0, 0.0, g)[0]) - A1p, 10, 5000)
X1p, X2p, _ = harm(0.0, 0.0, G)
print("G %.0f rad/s2; powered model: A1 %.2f deg, A2 %.2f deg (data %.2f, %.2f)" % (G, abs(X1p) / D2R, abs(X2p) / D2R, R[PW]["A"][0], R[PW]["A"][1]))
target_ratio = R[UN]["A"][0] / R[PW]["A"][0]; target_A2 = R[UN]["A"][1]
kap = np.round(np.arange(0.0, 2.01, 0.05), 3); xsv = np.round(np.arange(-6, 6.01, 1.0) * D2R, 4)
grid = []
for xs in xsv:
    for k in kap:
        X1, X2, _ = harm(k, xs, G)
        grid.append((float(k), float(xs / D2R), float(abs(X1) / D2R), float(abs(X2) / D2R), float(abs(X1) / A1p)))
grid = np.array(grid)
err = ((grid[:, 4] - target_ratio) / 0.03) ** 2 + ((grid[:, 3] - target_A2) / 0.17) ** 2
i = int(np.argmin(err)); kb, xb = grid[i, 0], grid[i, 1]
print("best: kappa %.2f, slack angle %+.0f deg -> A1 ratio %.2f (data %.2f), A2 %.2f deg (data %.2f), misfit %.2f" % (kb, xb, grid[i, 4], target_ratio, grid[i, 3], target_A2, err[i]))
X1u, X2u, yu = harm(kb, xb * D2R, G)
# same-sign control: the edge on the other side of the stroke
gridm = []
for xs in xsv:
    for k in kap:
        dt = 1 / 4000
        th = sim(w, zeta, beta_p, -k, xs, G, tau, 8.0, 4.0, dt)          # negative kappa term flips the side via min()
        t = np.arange(len(th)) * dt; m = t > 2.5; W = 2 * np.pi * 8
        X1 = 2 * np.mean(th[m] * np.exp(-1j * W * t[m])); X2 = 2 * np.mean(th[m] * np.exp(-2j * W * t[m]))
        gridm.append((k, xs / D2R, abs(X1) / D2R, abs(X2) / D2R, abs(X1) / A1p))
gridm = np.array(gridm)
out = dict(G=G, beta_p=beta_p, fn=P["fn"], zeta=zeta, tau=tau, target_ratio=target_ratio, target_A2=target_A2,
           best=dict(kappa=kb, slack_deg=xb, ratio=float(grid[i, 4]), A2=float(grid[i, 3]), misfit=float(err[i])),
           grid=grid.tolist(), kappa=kap.tolist(), slack=(xsv / D2R).tolist(),
           cycles=dict(powered=list(map(float, harm(0.0, 0.0, G)[2][-500:] / D2R)), unpowered=list(map(float, yu[-500:] / D2R))))
json.dump(out, open(os.path.join(DATA, "edge_on_stroke.json"), "w"))
for xs in (-2.0, 0.0, 2.0):
    rows = grid[np.isclose(grid[:, 1], xs)]
    print("slack %+.0f deg:" % xs, " ".join("k%.2f:r%.2f/A2 %.2f" % (r[0], r[4], r[3]) for r in rows[::4]))
