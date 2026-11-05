"""The roll as built: 23 mm strip, 20 mm active, thread pull about 2 mm. Stage scan: sag and swing against pull, skin eta and roll eta. Stage family: for each skin eta, the roll eta that gives the measured swing (read from the scan, one per pull), then voltage series, pull sweep and lift for 2 and 4 electrodes.
Run: python fit_roll.py scan -> data/fit_roll_scan.jsonl; python fit_roll.py family -> data/fit_roll_family.jsonl"""
import itertools, json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import fit_wing as FW
import fit_cases as FC

B = dict(FW.BASE, N=1, n_f=2, ws=23e-3, wa=20e-3)
ETAS = (0.05, 0.15, 0.25, 0.4)
SWING = 0.18


def scan_jobs():
    return [dict(B, kind="scan", pull=p, eta=e, eta_f=f) for p, e, f in itertools.product((1.5, 2.0, 2.5, 3.2), (0.05, 0.15, 0.25, 0.4), (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2))]


def fit_eta_f(rows, pull, eta):
    """Smallest roll eta where the swing first reaches the measured value."""
    q = sorted((r for r in rows if r["pull"] == pull and r["eta"] == eta), key=lambda r: r["eta_f"])
    f = [r["eta_f"] for r in q]
    d = [max(r["res"][-1]["sag"]) - max(r["res"][0]["sag"]) for r in q]
    for i in range(len(d) - 1):
        if d[i] < SWING <= d[i + 1]:
            return f[i] + (SWING - d[i]) / (d[i + 1] - d[i]) * (f[i + 1] - f[i])
    return None


def family_jobs():
    rows = [r for r in map(json.loads, open(os.path.join(HERE, "data", "fit_roll_scan.jsonl"))) if "error" not in r]
    out = []
    for e in ETAS:
        for fit_pull in (2.0, 3.2):
            f = fit_eta_f(rows, fit_pull, e)
            print("fit pull", fit_pull, "eta", e, "eta_f", f, flush=True)
            if f is None:
                continue
            tag = dict(B, fit_pull=fit_pull, eta=e, eta_f=f)
            out.append(dict(tag, kind="series", pull=fit_pull, Vs=(375.0, 750.0, 1125.0, 1500.0)))
            for N, nf in ((1, 2), (3, 4)):
                for p in (1.5, 2.0, 2.5, 3.2, 4.0, 5.0, 8.0):
                    out.append(dict(tag, kind="pull", N=N, n_f=nf, pull=p))
                for p in (2.0, 2.5, 3.2, 4.0, 5.0):
                    out.append(dict(tag, kind="lift", N=N, n_f=nf, pull=p, V=1500.0, alpha=6.0))
    for e in ETAS:
        for N, nf in ((1, 2), (3, 4)):
            for p in (2.0, 2.5, 3.2, 4.0, 5.0):
                out.append(dict(B, fit_pull=0.0, kind="lift", N=N, n_f=nf, pull=p, eta=e, eta_f=0.0, V=1500.0, alpha=6.0))
    return out


def dispatch(j):
    return FC.lift(j) if j["kind"] == "lift" else FW.run(j)


if __name__ == "__main__":
    stage = sys.argv[1]
    js = scan_jobs() if stage == "scan" else family_jobs()
    with mp.Pool(13) as pool, open(os.path.join(HERE, "data", "fit_roll_%s.jsonl" % stage), "w") as fh:
        for r in pool.imap_unordered(dispatch, js):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["kind"], r["N"], r["pull"], r["eta"], r["eta_f"], r["error"][:90], flush=True); continue
            if r["kind"] == "lift":
                print("LIFT fit %.1f N %d pull %.1f eta %.2f ef %.3f L %.2f dL %+.2f %s" % (r["fit_pull"], r["N"], r["pull"], r["eta"], r["eta_f"], r["L0"], r["L1"] - r["L0"], r["ok"]), flush=True)
            else:
                q = r["res"]
                print("%s fit %.1f N %d pull %.1f eta %.2f ef %.3f sag %s T %s corner %+.3f %+.3f ok %s" % (r["kind"], r.get("fit_pull", 0), r["N"], r["pull"], r["eta"], r["eta_f"], " ".join("%.3f" % max(x["sag"]) for x in q), " ".join("%.1f" % x["T_mN"] for x in q), q[-1]["rootpt"][0] - q[0]["rootpt"][0], q[-1]["rootpt"][1] - q[0]["rootpt"][1], q[-1]["ok"]), flush=True)
    print("DONE", flush=True)
