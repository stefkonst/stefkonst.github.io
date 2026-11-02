"""The chosen tilted design, batten 5 deg and root edge 8.85 deg: free root edge against a rib, edges short of or past vertical, and the line force with no flow.
Run: python tilt_extra.py -> validation/tilt_extra.json"""
import json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tilt_study as TS
import balance_study as BS

BASE = dict(theta=5.0, theta_r=8.85, N=2, xb=0.060, yc=0.0475, rot=1.0, root="rib", lam_n=1.04, eta=0.5)


def job(j):
    if j[0] == "line":
        return dict(tag="line", **BS.run(j[1]))
    return TS.run(j)


if __name__ == "__main__":
    J = [("free", dict(BASE, root="free")), ("free", dict(BASE, root="free", N=1))]
    J += [("rot", dict(BASE, rot=r)) for r in (0.7, 0.85, 1.15)]
    J += [("line", dict(theta_b=5.0, theta_r=8.85, N=N, lam_n=l)) for N in (1, 2) for l in (1.0, 1.04)]
    out = []
    with mp.Pool(7) as p:
        for r in p.imap_unordered(job, J):
            out.append(r)
            json.dump(out, open(os.path.join(HERE, "validation", "tilt_extra.json"), "w"), indent=0, default=float)
            print(r["tag"], r.get("root"), r.get("rot"), r.get("N"), r.get("lam_n"), "dL", r.get("dL_mN"), r.get("ok0"), r.get("ok1"), "line", r.get("F_line_mN"), "Mb", r.get("M_batten_mNm"), "F", r.get("F"), r.get("error", "")[:90], flush=True)
    print("done", flush=True)
