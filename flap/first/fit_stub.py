"""Stiff stub: the root corner barely moves in the video (10 um along the edge), against 170-240 um for an active roll on a soft stub. Run: python fit_stub.py -> data/fit_stub.jsonl"""
import json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_wing as FW
import fit_cases as FC


def jobs():
    out = []
    for stub in (1.0, 3.0, 10.0, 100.0):
        for p in (1.5, 2.0, 2.5, 3.0, 3.5):
            for nm in ("band 5 mm, eta .05", "band 5 mm, eta .15", "whole roll .15, eta .15", "passive roll, eta .15"):
                e, f, wa = FC.SETS[nm]
                out.append(dict(FW.BASE, name=nm, kind="stub", stub=stub, pull=p, eta=e, eta_f=f, wa=wa, Vs=(1500.0,)))
    return out


if __name__ == "__main__":
    with mp.Pool(13) as pool, open(os.path.join(HERE, "data", "fit_stub.jsonl"), "w") as fh:
        for r in pool.imap(FW.run, jobs()):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["error"]); continue
            a, b = r["res"][0], r["res"][-1]
            print("stub %3.0f pull %.1f %-26s sag %.2f %+.2f T %.0f %.0f corner %+.3f %+.3f" % (r["stub"], r["pull"], r["name"], max(a["sag"]), max(b["sag"]) - max(a["sag"]), a["T_mN"], b["T_mN"], b["rootpt"][0] - a["rootpt"][0], b["rootpt"][1] - a["rootpt"][1]), flush=True)
    print("DONE", flush=True)
