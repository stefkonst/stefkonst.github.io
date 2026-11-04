"""No root bar, thread at the roll's root end, curved tip with a soft edge and a diagonal rib; and with a stiff edge and the rib. Run: python no_root_rib.py -> data/no_root_rib.jsonl"""
import json, multiprocessing as mp, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wing_shape as W
import no_root_bar as N
Q = N.Q
S0 = 35.0 / 50.0


def cases():
    out = []
    c = 0.027
    for tr in (3.0, 4.0, 5.0):
        for nm, free in (("stiff edge, rib", False), ("soft edge, rib", True)):
            q = dict(N.BASE, name=f"{nm}, thread {tr:.0f} mm", N=3, lam_n=1.0, cr=c, ct=0.003, alpha=6.0, curve=dict(xc=W.AX + 0.035, c=c, ct=0.003), veins=(((S0, 0.0), (0.85, 1.0)),),
                     tensioners=(dict(at="D", anchor=(W.AX - 0.02, W.AY + c), kind="line", k=500.0, F0=500.0 * tr * 1e-3 + 0.04),))
            if free:
                q["free_from"] = S0
            out.append(q)
    return out


if __name__ == "__main__":
    with mp.Pool(6) as pool, open(os.path.join(HERE, "data", "no_root_rib.jsonl"), "w") as fh:
        for r in pool.imap_unordered(N.run, cases()):
            r.pop("tensioners", None)
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            a = r.get("area_cm2", 0)
            print("%-34s %s%s L0 %5.1f L1 %5.1f Ffib %3.0f z %.1f->%.1f wr %.0f/%.0f wall %.0f %s" % (r["name"], "Y" if r.get("ok0") else "n", "Y" if r.get("ok1") else "n", r.get("V0", {}).get("CL", 0) * Q * a * 1e-4, r.get("V1", {}).get("CL", 0) * Q * a * 1e-4, r.get("V0", {}).get("F_mN", 0), r.get("V0", {}).get("z_max_mm", 0), r.get("V1", {}).get("z_max_mm", 0), 100 * r.get("wrinkled0", 0), 100 * r.get("wrinkled1", 0), r["wall"], r.get("error", "")[:25]), flush=True)
    print("DONE", flush=True)
