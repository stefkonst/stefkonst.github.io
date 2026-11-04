"""No root bar, roll end held by a stiff thread of fixed length, set by travel. 3 layers, 30 um, 1370 V. Mesh check and 10 deg pitch.
Run: python no_root_thread.py -> data/no_root_thread.jsonl"""
import json, multiprocessing as mp, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wing_shape as W
import no_root_bar as N
Q = N.Q


def cases():
    out = []
    for shape, c in (("rectangle", 0.025), ("curved", 0.027)):
        for tr_mm in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            for k_, h, a in ((500.0, 2e-3, 6.0), (500.0, 1.5e-3, 6.0), (500.0, 2e-3, 10.0), (500.0, 1.5e-3, 10.0)):
                F = k_ * tr_mm * 1e-3 + 0.04                  # about 40 mN left when the corner has moved by tr_mm
                q = dict(N.BASE, name=f"{shape}, thread {tr_mm:.0f} mm, h {h * 1e3:.1f}, a {a:.0f}", shape=shape, travel=tr_mm, N=3, h=h, lam_n=1.0, cr=c, alpha=a,
                         tensioners=(dict(at="D", anchor=(W.AX - 0.02, W.AY + c), kind="line", k=k_, F0=F),))
                if shape == "curved":
                    q.update(curve=dict(xc=W.AX + 0.035, c=c, ct=0.003), ct=0.003)
                else:
                    q.update(ct=c)
                out.append(q)
    return out


if __name__ == "__main__":
    with mp.Pool(13) as pool, open(os.path.join(HERE, "data", "no_root_thread.jsonl"), "w") as fh:
        for r in pool.imap_unordered(N.run, cases()):
            r.pop("tensioners", None)
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            ok = r.get("ok0") and r.get("ok1") and "V1" in r
            a = r.get("area_cm2", 0)
            print("%-42s %s%s L0 %5.1f L1 %5.1f dL %5s Ffib %4.0f->%4.0f z %.1f->%.1f zmin %.1f move %s wr %.0f/%.0f wall %.0f %s" % (r["name"], "Y" if r.get("ok0") else "n", "Y" if r.get("ok1") else "n", r.get("V0", {}).get("CL", 0) * Q * a * 1e-4, r.get("V1", {}).get("CL", 0) * Q * a * 1e-4, "%.2f" % ((r["V1"]["CL"] - r["V0"]["CL"]) * Q * a * 1e-4) if ok else "-", r.get("V0", {}).get("F_mN", 0), r.get("V1", {}).get("F_mN", 0), r.get("V0", {}).get("z_max_mm", 0), r.get("V1", {}).get("z_max_mm", 0), r.get("V0", {}).get("z_min_mm", 0), np.round(r.get("root_end", {}).get("flow_mm", [0, 0, 0])[:2], 1), 100 * r.get("wrinkled0", 0), 100 * r.get("wrinkled1", 0), r["wall"], r.get("error", "")[:25]), flush=True)
    print("DONE", flush=True)
