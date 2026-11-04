"""From the POC lift number to this wing: same solver, measured outline, passive roll, skin eta stepped down. Run: python lift_bridge.py -> data/lift_bridge.jsonl"""
import json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_wing as FW
import fit_cases as FC


def jobs():
    return [dict(FW.BASE, name=f"eta {e}, N {N}, {V:.0f} V, pull {p}", kind="lift", N=N, pull=p, eta=e, eta_f=0.0, V=V, alpha=6.0) for e in (0.5, 0.4, 0.3, 0.15, 0.05) for N, V in ((2, 1500.0), (2, 1370.0), (3, 1370.0), (4, 1500.0)) for p in (3.2, 4.0, 5.0)]


if __name__ == "__main__":
    with mp.Pool(13) as pool, open(os.path.join(HERE, "data", "lift_bridge.jsonl"), "w") as fh:
        for r in pool.imap(FC.lift, jobs()):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            print(r["name"], r.get("error", ""), "dL %+.2f L0 %.1f area %.2f %s z %.1f %.1f" % (r.get("L1", 0) - r.get("L0", 0), r.get("L0", 0), r.get("area_cm2", 0), r.get("ok"), r.get("z0", 0), r.get("z1", 0)), flush=True)
