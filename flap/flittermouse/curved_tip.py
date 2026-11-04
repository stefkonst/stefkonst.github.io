"""Curved tip: straight LE to x_c, then a quarter ellipse aft to a short tip edge; straight rolled fibre. Optional diagonal rib from the LE at the start of the curve to the fibre.
Rigid carbon in the limit. free=True leaves the curved edge unsupported past the start of the curve. 3 layers, 30 um, 1370 V, 4 % span stretch, no chord stretch.
The FE wing is unchanged; only its patch map and its lattice LE are swapped here. Run: python curved_tip.py -> data/curved_tip.jsonl"""
import json, math, multiprocessing as mp, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wing_shape as W
M = W.M
_coons, _VLM = M.coons, M.VLM
CUR = {}


def y_le(x):
    """LE as a function of x, m."""
    x = np.asarray(x, float)
    xa, xc, xt, c, ct = W.AX, CUR["xc"], W.AX + 0.05, CUR["c"], CUR["ct"]
    t = np.clip((x - xc) / (xt - xc), 0.0, 1.0)
    return W.AY + (c - ct) * (1 - np.sqrt(np.maximum(1 - t ** 2, 0.0)))


def coons2(a, b, c, d, ni, nj):
    s = np.linspace(0, 1, ni + 1)
    u = np.linspace(0, 1, nj + 1)
    xa, xt = W.AX, W.AX + 0.05
    xc = CUR["xc"]
    s0 = (xc - xa) / (xt - xa)
    x = np.where(s <= s0, xa + (xt - xa) * s, xc + (xt - xc) * np.sin(np.clip((s - s0) / (1 - s0), 0, 1) * math.pi / 2))
    yl = y_le(x)
    yt = W.AY + CUR["c"]
    # the root edge and tip edge may be tilted in the patch: keep them vertical here (straight cut)
    G = np.zeros((ni + 1, nj + 1, 2))
    G[..., 0] = x[:, None]
    G[..., 1] = yl[:, None] + (yt - yl)[:, None] * u[None, :]
    return G


class VLM2(_VLM):
    def __init__(self, xs, ly, ty, ny, cosine=False):
        le = y_le(xs)
        super().__init__(xs, le, np.maximum(ty, le + 1.5e-3), ny, cosine)


def s_of(x_mm):
    return (x_mm * 1e-3) / 0.05


def run(job):
    CUR.update(job["curve"])
    M.coons, M.VLM = coons2, VLM2
    k = {q: v for q, v in job.items() if q != "curve"}
    r = W.run(k)
    r["curve"] = job["curve"]
    return r


BASE = dict(N=3, V=1370.0, theta=0.0, stretch=0.0, lam_d=1.04, lam_n=1.0)


def case(name, xc, c, ct=3.0, rib_to=None, free=False, **over):
    s0 = s_of(xc)
    q = dict(BASE, name=name, curve=dict(xc=W.AX + xc * 1e-3, c=c * 1e-3, ct=ct * 1e-3), cr=c * 1e-3, ct=ct * 1e-3, **over)
    if rib_to is not None:
        q["veins"] = (((s0, 0.0), (rib_to, 1.0)),)
    if free:
        q["free_from"] = s0
    return q


C = [case("curve from 40 mm", 40, 26), case("curve from 35 mm", 35, 27), case("curve from 25 mm", 25, 28),
     case("curve from 35 mm, tip 6 mm", 35, 27, ct=6.0),
     case("curve from 35 mm, rib to 85 %", 35, 27, rib_to=0.85), case("curve from 35 mm, rib to 100 %", 35, 27, rib_to=1.0), case("curve from 35 mm, rib straight back", 35, 27, rib_to=0.7),
     case("curve from 25 mm, rib to 75 %", 25, 28, rib_to=0.75),
     case("curve from 35 mm, soft edge, rib to 85 %", 35, 27, rib_to=0.85, free=True), case("curve from 35 mm, soft edge, rib to 100 %", 35, 27, rib_to=1.0, free=True),
     case("curve from 35 mm, soft edge, no rib", 35, 27, free=True),
     case("curve from 35 mm, stretch 6 %", 35, 27, lam_d=1.06), case("curve from 35 mm, rib to 85 %, mesh 1.5", 35, 27, rib_to=0.85, h=1.5e-3), case("curve from 35 mm, mesh 1.5", 35, 27, h=1.5e-3)]

if __name__ == "__main__":
    jobs = [dict(c, alpha=a) for c in C for a in (6.0, 10.0)]
    OUT = os.path.join(HERE, "data", "curved_tip.jsonl")
    Q = 0.5 * 1.225 * 16 * 1e3
    with mp.Pool(12) as pool, open(OUT, "w") as fh:
        for r in pool.imap_unordered(run, jobs):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            ok = r.get("ok0") and r.get("ok1") and "V1" in r
            print("%-44s a %2.0f %s%s %s area %s L0 %s L1 %s dL %s F %s z %s->%s wall %.0f %s" % (r["name"], r["alpha"], "Y" if r.get("ok0") else "n", "Y" if r.get("ok1") else "n", "", round(r.get("area_cm2", 0), 2), round(r.get("V0", {}).get("CL", 0) * Q * r.get("area_cm2", 0) * 1e-4, 1), round(r.get("V1", {}).get("CL", 0) * Q * r.get("area_cm2", 0) * 1e-4, 1), round((r["V1"]["CL"] - r["V0"]["CL"]) * Q * r["area_cm2"] * 1e-4, 2) if ok else "-", round(r.get("V0", {}).get("F_mN", 0)), round(r.get("V0", {}).get("z_max_mm", 0), 1), round(r.get("V1", {}).get("z_max_mm", 0), 1), r["wall"], r.get("error", "")), flush=True)
    print("DONE", flush=True)
