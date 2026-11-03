"""Validation of uvlm.py. V1 steady flat wing against Helmbold's slope and induced drag.
V2 lift amplitude against Theodorsen, thrust and power against Garrick, plunge at high aspect ratio.
V3 rigid against free wake, on the same plunge."""
import json, os, time
import numpy as np
from scipy.special import hankel2
from uvlm import UVLM

HERE = os.path.dirname(os.path.abspath(__file__))
RHO, U, C = 1.225, 4.0, 0.025


def rect(b, c, ns, nc, cosine=True):
    yy = b * np.sin(np.linspace(0, np.pi / 2, ns + 1)) if cosine else np.linspace(0, b, ns + 1)
    X = np.zeros((ns + 1, nc + 1, 3))
    X[..., 0], X[..., 1] = np.linspace(0, c, nc + 1)[None, :], yy[:, None]
    return X


def C_theo(k):
    H1, H0 = hankel2(1, k), hankel2(0, k)
    return H1 / (H1 + 1j * H0)


def steady(ar, a_deg=5.0, ns=12, nc=6, free=False):
    """ar counts both wings."""
    b = C * ar / 2
    X0 = rect(b, C, ns, nc)
    a = np.radians(a_deg)
    T = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    X = X0 @ T.T
    s = UVLM(lambda t: X, ns, nc, C / nc / U, U=U, free=free)
    for i in range(int(12 * nc * max(1, ar / 4))):
        r = s.step()
    F = r["F"].sum((0, 1))
    S, q = b * C, 0.5 * RHO * U ** 2
    L, D = F[2], F[0]               # the wing is rotated, the wind stays along x
    return L / q / S / a, D / q / S, (L / q / S) ** 2 / (np.pi * ar)


def plunge(k, h_c, ar=16, ns=10, nc=6, cycles=4, free=False):
    b = C * ar / 2
    X0 = rect(b, C, ns, nc, cosine=False)
    w, h0 = 2 * k * U / C, h_c * C
    Tp = 2 * np.pi / w
    n = int(max(40, round(Tp * U / (C / nc))))
    dt = Tp / n

    def grid(t):
        X = X0.copy()
        X[..., 2] += h0 * np.sin(w * t)
        return X

    s = UVLM(grid, ns, nc, dt, U=U, free=free, wake_max=int(n * (cycles + 1)))
    rows, mid = [], ns // 2
    for i in range(n * cycles):
        t = s.t
        f = s.step()["F"][mid].sum(0)
        rows.append((t, f[0], f[2], f[2] * h0 * w * np.cos(w * t)))
    rows = np.array(rows[-n:])
    db, q = b / ns, 0.5 * RHO * U ** 2
    # first harmonic; the grid moves by backward difference, half a step behind t
    amp = abs(2 * np.mean(rows[:, 2] * np.exp(-1j * w * rows[:, 0])))
    Ck = C_theo(k)
    bs = C / 2
    a_cos = -2 * np.pi * RHO * U * bs * h0 * w * Ck.real
    a_sin = np.pi * RHO * bs ** 2 * h0 * w ** 2 + 2 * np.pi * RHO * U * bs * h0 * w * Ck.imag
    base = np.pi * (w * h0 / U) ** 2
    return dict(k=k, h_c=h_c, free=free, L_amp=amp / db, L_amp_th=np.hypot(a_cos, a_sin),
                CT=-rows[:, 1].mean() / (q * C * db), CT_th=base * abs(Ck) ** 2,
                CP=-rows[:, 3].mean() / (q * C * db * U), CP_th=base * Ck.real)


if __name__ == "__main__":
    t0 = time.time()
    out = dict(steady=[], resolution=[], plunge=[], wake=[])
    print("V1 steady, flat, 5 deg, rigid wake", flush=True)
    for ar in (2, 4, 8):
        a, cd, cdi = steady(ar)
        h = 2 * np.pi / (np.sqrt(1 + (2 / ar) ** 2) + 2 / ar)
        print(f" AR {ar}: slope {a:.3f}/rad Helmbold {h:.3f}; CD {cd:.5f}, CL^2/(pi AR) {cdi:.5f}  {time.time()-t0:.0f}s", flush=True)
        out["steady"].append(dict(AR=ar, slope=a, helmbold=h, CD=cd, CDi=cdi))
    for ns, nc in ((8, 4), (12, 6), (16, 8), (24, 8)):
        a, cd, cdi = steady(4, ns=ns, nc=nc)
        print(f" res {ns}x{nc}: {a:.3f}, CD {cd:.5f}  {time.time()-t0:.0f}s", flush=True)
        out["resolution"].append(dict(ns=ns, nc=nc, slope=a, CD=cd))
    print("V2 plunge at AR 16, rigid wake", flush=True)
    for k in (0.1, 0.2, 0.4, 0.8, 1.5):
        r = plunge(k, 0.25 if k < 0.3 else 0.1)
        print(f" k {k}: L amp {r['L_amp']*1e3:.2f} th {r['L_amp_th']*1e3:.2f} mN/m; CT {r['CT']:.4f} th {r['CT_th']:.4f}; CP {r['CP']:.4f} th {r['CP_th']:.4f}  {time.time()-t0:.0f}s", flush=True)
        out["plunge"].append(r)
    print("V3 free wake, k 0.4", flush=True)
    r = plunge(0.4, 0.1, ar=8, ns=8, nc=4, cycles=3, free=False)
    f = plunge(0.4, 0.1, ar=8, ns=8, nc=4, cycles=3, free=True)
    print(f" rigid CT {r['CT']:.4f} CP {r['CP']:.4f}; free CT {f['CT']:.4f} CP {f['CP']:.4f}  {time.time()-t0:.0f}s", flush=True)
    out["wake"] = [r, f]
    json.dump(out, open(os.path.join(HERE, "data", "uvlm_check.json"), "w"), indent=1, default=float)
