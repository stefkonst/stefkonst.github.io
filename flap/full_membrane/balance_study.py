"""Does a line on the trailing edge bring the root edge and the batten to vertical? The sheet is cut unstretched with the batten tilted by theta_b and the root edge by theta_r.
With both edges vertical, the moment on the batten about its LE hinge is computed, no flow. Zero moment means the state is in equilibrium with the batten free to rotate.
Run: python balance_study.py -> validation/balance_study.json"""
import json, math, multiprocessing as mp, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
import tilt_study as TS

AX, AY = 3.8e-3, 6.4e-3


def state(theta_b, theta_r, N=1, xb=0.060, yc=0.0475, lam_n=1.0):
    tb, tr = math.radians(theta_b), math.radians(theta_r)
    C = (xb + (yc - AY) * math.tan(tb), yc)
    D = (AX + (yc - AY) * math.tan(tr), yc)
    Hm = (N + 1) * 35e-6
    w = M.Wing(C=C, D=D, theta_deg=theta_b, lam_d=(1.0, 1.0), lam_n=(lam_n, lam_n), t=(Hm, Hm), n_a=(N, N), w_s=(13e-3, 13e-3), h=2.5e-3, max_iter=40000, tol=1e-6)
    x = w.x0.copy()
    seam, rn = w.ids[0][-1, :], w.ids[0][0, :]
    B, A = x[seam[0], :2].copy(), x[rn[0], :2].copy()
    x[seam, :2] = TS.rotate(x[seam, :2], B, tb)
    x[rn, :2] = TS.rotate(x[rn, :2], A, tr)
    r = w.relax(0.0, x_init=x, fixed=w.fixed, fixed_x=x)
    xr = r["x"]
    R_ = w.reactions(xr)
    arm = xr[seam, :2] - xr[seam[0], :2]
    Mb = float(np.sum(arm[:, 0] * R_[seam, 1] - arm[:, 1] * R_[seam, 0]))     # moment on the batten about its LE hinge
    arm_r = xr[rn, :2] - xr[rn[0], :2]
    Mr = float(np.sum(arm_r[:, 0] * R_[rn, 1] - arm_r[:, 1] * R_[rn, 0]))
    f = w.fibre_report(xr)
    s = w.summary(xr)
    D_len = abs(xr[rn[-1], 1] - xr[rn[0], 1])
    F_line = abs(Mr) / D_len                                                  # line along the span at the rib tip, balancing the rib about its hinge
    return dict(theta_b=theta_b, theta_r=theta_r, N=N, xb=xb, yc=yc, lam_n=lam_n, M_batten_mNm=Mb * 1e3, M_root_mNm=Mr * 1e3, F_line_mN=F_line * 1e3, F=[f[i]["F"] * 1e3 for i in (0, 1)], lam_f=[f[i]["lam"] for i in (0, 1)], sag=[f[i]["sag"] * 1e3 for i in (0, 1)],
                n1=[s[i]["N1"] for i in (0, 1)], n2=[s[i]["N2"] for i in (0, 1)], wrinkled=[s[i]["wrinkled"] for i in (0, 1)], slack=[s[i]["slack"] for i in (0, 1)], ok=bool(r["ok"]))


def run(a):
    try:
        return state(**a)
    except Exception as e:
        return dict(a, error=repr(e))


if __name__ == "__main__":
    J = [dict(theta_b=b, theta_r=r) for b in (7.0, 10.0, 14.0) for r in (0.0, 5.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0)] + [dict(theta_b=b, theta_r=r, xb=x) for b in (10.0, 14.0) for r in (14.0, 22.0, 30.0) for x in (0.045, 0.075)]
    out = []
    with mp.Pool(8) as p:
        for q in p.imap_unordered(run, J):
            out.append(q)
            json.dump(out, open(os.path.join(HERE, "validation", "balance_study.json"), "w"), indent=0, default=float)
            print(q.get("theta_b"), q.get("theta_r"), q.get("xb"), "Mb", round(q.get("M_batten_mNm", 0), 2), "Fline", round(q.get("F_line_mN", 0)), "F", q.get("F"), "lam", q.get("lam_f"), q.get("error", "")[:70], flush=True)
