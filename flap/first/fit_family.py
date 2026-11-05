"""Every skin eta needs its own roll force to give the +0.18 mm swing: eta_f w_a = (0.18 + 0.73 eta) / 0.155 mm, from fit_stack.py. Voltage series, pull sweep and lift along that family, as built (1 active in the skin, 2 in the roll) and for 4 electrodes (3, 4).
Run: python fit_family.py -> data/fit_family.jsonl"""
import json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_wing as FW
import fit_cases as FC

W = 10e-3
ETAS = (0.05, 0.15, 0.25, 0.4)
ef = lambda e: (0.18 + 0.73 * e) / 0.155 * 1e-3 / W
B = dict(FW.BASE, N=1, n_f=2, wa=W)


def dispatch(j):
    return FC.lift(j) if j["kind"] == "lift" else FW.run(j)


def jobs():
    out = []
    for e in ETAS:
        out.append(dict(B, kind="series", pull=3.2, eta=e, eta_f=ef(e), Vs=(375.0, 750.0, 1125.0, 1500.0)))
        out.append(dict(B, kind="thin", pull=3.2, eta=e, eta_f=ef(e) * W / 5e-3, wa=5e-3))           # same force, narrower band
        for N, nf in ((1, 2), (3, 4)):
            for p in (1.0, 2.0, 3.2, 5.0, 8.0):
                out.append(dict(B, kind="pull", N=N, n_f=nf, pull=p, eta=e, eta_f=ef(e)))
            for p in (3.2, 4.0, 5.0):
                out.append(dict(B, kind="lift", N=N, n_f=nf, pull=p, eta=e, eta_f=ef(e), V=1500.0, alpha=6.0))
                out.append(dict(B, kind="lift", N=N, n_f=nf, pull=p, eta=e, eta_f=0.0, V=1500.0, alpha=6.0))
    return out


if __name__ == "__main__":
    with mp.Pool(13) as pool, open(os.path.join(HERE, "data", "fit_family.jsonl"), "w") as fh:
        for r in pool.imap_unordered(dispatch, jobs()):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["kind"], r["N"], r["pull"], r["eta"], r["error"]); continue
            if r["kind"] == "lift":
                print("LIFT N %d pull %.1f eta %.2f ef %.3f L %.1f dL %+.2f %s wall %.0f" % (r["N"], r["pull"], r["eta"], r["eta_f"], r["L0"], r["L1"] - r["L0"], r["ok"], r["wall"]), flush=True)
            else:
                q = r["res"]
                print("%s N %d pull %.1f eta %.2f ef %.3f wa %.0f sag %s T %s" % (r["kind"], r["N"], r["pull"], r["eta"], r["eta_f"], r["wa"] * 1e3, " ".join("%.3f" % max(x["sag"]) for x in q), " ".join("%.1f" % x["T_mN"] for x in q)), flush=True)
    print("DONE", flush=True)
