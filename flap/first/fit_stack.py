"""The stack as built: 2 electrodes, so 1 active layer in the skin (60 um of film) and 2 in the roll, where top and bottom overlap from turn to turn.
4 electrodes: 3 in the skin, 4 in the roll. Scan of skin eta, roll eta and active width against the 0.18 mm swing, then pull sweep. Run: python fit_stack.py -> data/fit_stack.jsonl"""
import itertools, json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_wing as FW

B = dict(FW.BASE, N=1, n_f=2)


def jobs():
    out = []
    for eta, ef, wa in itertools.product((0.05, 0.1, 0.15, 0.25, 0.4), (0.0, 0.05, 0.1, 0.15, 0.25, 0.4), (2e-3, 5e-3, 10e-3, 19e-3)):
        if ef == 0 and wa != 5e-3:
            continue
        out.append(dict(B, kind="scan", pull=3.2, eta=eta, eta_f=ef, wa=wa))
    for p in (1.0, 2.0, 3.2, 5.0, 8.0):
        for nm, eta, ef, wa in (("passive", 0.15, 0.0, 5e-3),):
            for N, nf in ((1, 2), (3, 4)):
                out.append(dict(B, kind="pull", name=nm, N=N, n_f=nf, pull=p, eta=eta, eta_f=ef, wa=wa))
    return out


if __name__ == "__main__":
    with mp.Pool(13) as pool, open(os.path.join(HERE, "data", "fit_stack.jsonl"), "w") as fh:
        for r in pool.imap_unordered(FW.run, jobs()):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["error"]); continue
            a, b = r["res"][0], r["res"][-1]
            print("%s N %d nf %d pull %.1f eta %.2f ef %.2f wa %2.0f | sag %.2f %+.3f T %.0f %.0f corner %+.3f %+.3f %s %s" % (r["kind"], r["N"], r["n_f"], r["pull"], r["eta"], r["eta_f"], r["wa"] * 1e3, max(a["sag"]), max(b["sag"]) - max(a["sag"]), a["T_mN"], b["T_mN"], b["rootpt"][0] - a["rootpt"][0], b["rootpt"][1] - a["rootpt"][1], a["ok"], b["ok"]), flush=True)
    print("DONE", flush=True)
