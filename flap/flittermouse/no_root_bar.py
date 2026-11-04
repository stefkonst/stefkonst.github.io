"""No root bar: the root edge is free and the root end of the rolled trailing edge is pulled by a line. 3 layers, 30 um, 1370 V, LE bonded with no span stretch.
Line: nearly constant force, along the fibre or toward the leading edge root. Run: python no_root_bar.py -> data/no_root_bar.jsonl"""
import itertools, json, multiprocessing as mp, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wing_shape as W
import curved_tip as CT
M = W.M
Q = 0.5 * 1.225 * 16 * 1e3
BASE = dict(N=3, V=1370.0, theta=0.0, stretch=0.0, lam_d=1.0, root_free=True)


def line(anchor, F):
    return dict(at="D", anchor=anchor, kind="line", k=5.0, F0=F)


def run(job):
    if "curve" in job:
        return CT.run(job)
    return W.run(job)


def cases():
    out = []
    for F, ln, (an, ax_) in itertools.product((0.02, 0.05, 0.1, 0.2, 0.4), (1.0, 1.06), (("along fibre", None), ("toward LE root", "le"))):
        for shape in ("rectangle", "curved"):
            c = 0.025 if shape == "rectangle" else 0.027
            anchor = (W.AX - 0.02, W.AY + c) if ax_ is None else (W.AX - 0.02, W.AY)
            q = dict(BASE, name=f"{shape}, {an}, {F * 1e3:.0f} mN, chord {ln:.2f}", shape=shape, F=F, lam_n=ln, anchor_name=an, cr=c, tensioners=(line(anchor, F),))
            if shape == "curved":
                q.update(curve=dict(xc=W.AX + 0.035, c=c, ct=0.003), ct=0.003)
            else:
                q.update(ct=c)
            out.append(q)
    return [dict(q, alpha=a) for q in out for a in (6.0,)]


def sweep():
    """Actual fibre force set by a stiffer line, so force and travel are read directly; along the fibre only."""
    out = []
    for shape, c in (("rectangle", 0.025), ("curved", 0.027)):
        for F, k_, h, N, a in [(F, 5.0, 2e-3, 3, 6.0) for F in (0.06, 0.08, 0.10, 0.12, 0.14, 0.16)] + [(F, 5.0, 2e-3, 3, 10.0) for F in (0.08, 0.10, 0.12)] + [(F, 5.0, 1.5e-3, 3, 6.0) for F in (0.08, 0.10, 0.12)] + [(F, 50.0, 2e-3, 3, 6.0) for F in (0.03, 0.05, 0.07, 0.09)] + [(F, 5.0, 2e-3, 2, 6.0) for F in (0.08, 0.10)]:
            anchor = (W.AX - 0.02, W.AY + c)
            q = dict(BASE, name=f"{shape}, line {F * 1e3:.0f} mN, k {k_:.0f}, h {h * 1e3:.1f}, N {N}, a {a:.0f}", N=N, h=h, lam_n=1.0, cr=c, alpha=a, tensioners=(dict(at="D", anchor=anchor, kind="line", k=k_, F0=F),))
            if shape == "curved":
                q.update(curve=dict(xc=W.AX + 0.035, c=c, ct=0.003), ct=0.003)
            else:
                q.update(ct=c)
            out.append(q)
    return out


if __name__ == "__main__":
    OUT = os.path.join(HERE, "data", "no_root_bar.jsonl")
    if len(sys.argv) > 1:
        OUT = OUT.replace(".jsonl", "_sweep.jsonl")
    with mp.Pool(13) as pool, open(OUT, "w") as fh:
        for r in pool.imap_unordered(run, sweep() if len(sys.argv) > 1 else cases()):
            r.pop("tensioners", None)
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            ok = r.get("ok0") and r.get("ok1") and "V1" in r
            a = r.get("area_cm2", 0)
            print("%-48s %s%s L0 %5.1f L1 %5.1f dL %5s Ffib %4.0f z0 %.1f z1 %.1f zmin0 %.1f wall %.0f %s" % (r["name"], "Y" if r.get("ok0") else "n", "Y" if r.get("ok1") else "n", r.get("V0", {}).get("CL", 0) * Q * a * 1e-4, r.get("V1", {}).get("CL", 0) * Q * a * 1e-4, "%.2f" % ((r["V1"]["CL"] - r["V0"]["CL"]) * Q * a * 1e-4) if ok else "-", r.get("V0", {}).get("F_mN", 0), r.get("V0", {}).get("z_max_mm", 0), r.get("V1", {}).get("z_max_mm", 0), r.get("V0", {}).get("z_min_mm", 0), r["wall"], r.get("error", "")[:30] + " move " + str(np.round(r.get("root_end", {}).get("flow_mm", [0, 0, 0])[:2], 1)) + " wr %.0f/%.0f" % (100 * r.get("wrinkled0", 0), 100 * r.get("wrinkled1", 0))), flush=True)
    print("DONE", flush=True)
