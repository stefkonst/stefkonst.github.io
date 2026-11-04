"""What could make the rolled edge bow more at voltage? Sweep of skin eta, roll eta and active width of the roll, with thread stiff or soft.
Run: python fit_sign.py -> data/fit_sign.jsonl"""
import itertools, json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_wing as FW


def jobs():
    out = []
    for pull, (eta, ef, wa) in itertools.product((3.0, 4.0), [(0.4, 0, 5e-3), (0.15, 0, 5e-3), (0.05, 0, 5e-3), (0.4, 0.4, 5e-3), (0.15, 0.4, 5e-3), (0.05, 0.4, 5e-3), (0, 0.4, 5e-3), (0.4, 0.4, 19e-3), (0.15, 0.4, 19e-3), (0.05, 0.4, 19e-3), (0, 0.4, 19e-3), (0.15, 0.15, 19e-3), (0.05, 0.15, 19e-3), (0, 0.15, 19e-3), (0.05, 0.4, 10e-3), (0, 0.4, 10e-3)]):
        out.append(dict(FW.BASE, pull=pull, eta=eta, eta_f=ef, wa=wa, Vs=(1500.0,)))
    return out


if __name__ == "__main__":
    with mp.Pool(13) as pool, open(os.path.join(HERE, "data", "fit_sign.jsonl"), "w") as fh:
        rows = []
        for r in pool.imap(FW.run, jobs()):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["error"]); continue
            a, b = r["res"][0], r["res"][-1]
            print("pull %.0f eta %.2f eta_f %.2f wa %2.0f | sag %.2f -> %.2f (%+.2f) T %.0f -> %.0f F %.0f -> %.0f wr %.2f -> %.2f sl %.2f | %s %s" % (r["pull"], r["eta"], r["eta_f"], r["wa"] * 1e3, max(a["sag"]), max(b["sag"]), max(b["sag"]) - max(a["sag"]), a["T_mN"], b["T_mN"], a["F_mN"], b["F_mN"], a["wr"], b["wr"], b["slack"], a["ok"], b["ok"]), flush=True)
    print("DONE", flush=True)
