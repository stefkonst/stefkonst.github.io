"""Tensioning by rotation: the sheet is cut flat and tilted, bonded to the LE unstretched, then the batten (and root rib) rotate about the LE to vertical.
No pre-stretch. Panel 1 shears; panel 2 stretches along its fibre, because the batten tip moves away from the wing tip.
Run: python tilt_study.py -> validation/tilt_study.json"""
import json, math, multiprocessing as mp, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M

AX, AY = 3.8e-3, 6.4e-3
OUT = os.path.join(HERE, "validation", "tilt_study.json")


def rotate(p, hinge, ang):
    """Rotate about the hinge by ang (positive turns the tip inboard)."""
    d = p - hinge
    c, s = math.cos(ang), math.sin(ang)
    return hinge + np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]]


def run(job):
    t0 = time.time()
    tag, k = job
    theta, N, xb, yc, rot, root, U = k["theta"], k["N"], k["xb"], k["yc"], k["rot"], k["root"], k.get("U", 4.0)
    th = math.radians(theta)
    th_r = math.radians(k.get("theta_r", theta))               # root edge tilt; more than the batten so that the batten balances (balance_study.py)
    C = (xb + (yc - AY) * math.tan(th), yc)                    # tip of the tilted batten, as cut
    D = (AX + (yc - AY) * math.tan(th_r), yc)                  # root end of the fibre, as cut
    Hm = (N + 1) * 35e-6
    ln = k.get("lam_n", 1.0)                                   # small chord stretch, set when the sheet is bonded, before the rotation
    par = dict(C=C, D=D, theta_deg=theta, lam_d=(1.0, 1.0), lam_n=(ln, ln), t=(Hm, Hm), n_a=(N, N), w_s=(13e-3, 13e-3), h=k.get("h", 2e-3), max_iter=k.get("iters", 30000), tol=5e-7, U=U, eta_s=k.get("eta", 0.4), active_fibre=k.get("active", False), n_f=N + 1, eta_f=0.4,
               no_batten=False, root_free=root != "rib")
    try:
        w = M.Wing(**par)
        x = w.x0.copy()
        g0, g1 = w.ids[0], w.ids[1]
        fx = w.fixed.copy()
        ang = th * rot
        seam = g0[-1, :]
        B = x[seam[0], :2].copy()
        x[seam, :2] = rotate(x[seam, :2], B, ang)
        if root == "rib":
            rn = g0[0, :]
            x[rn, :2] = rotate(x[rn, :2], x[rn[0], :2].copy(), th_r * rot)
        else:                                                  # free root edge; the fibre's root end is held by the line at its vertical position
            n = int(g0[0, -1])
            a = x[g0[0, 0], :2].copy()
            x[n:n + 1, :2] = rotate(x[n:n + 1, :2], a, th_r * rot)
            fx[n] = True
        r = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)
        xr = r["x"]
        s = w.summary(xr)
        f = w.fibre_report(xr)
        R_ = w.reactions(xr)
        out = dict(tag=tag, **k, ok=bool(r["ok"]), n1=[s[i]["N1"] for i in (0, 1)], n2=[s[i]["N2"] for i in (0, 1)], taut=[s[i]["taut"] for i in (0, 1)], wrinkled=[s[i]["wrinkled"] for i in (0, 1)], slack=[s[i]["slack"] for i in (0, 1)],
                   cv=[s[i]["N1_cv"] for i in (0, 1)], F=[f[i]["F"] * 1e3 for i in (0, 1)], lam_f=[f[i]["lam"] for i in (0, 1)], sag=[f[i]["sag"] * 1e3 for i in (0, 1)],
                   batten_x_mN=float(R_[seam, 0].sum() * 1e3), batten_y_mN=float(R_[seam, 1].sum() * 1e3), root_x_mN=float(R_[g0[0, :], 0].sum() * 1e3), root_y_mN=float(R_[g0[0, :], 1].sum() * 1e3), C_ref=C, D_ref=D, C_final=xr[seam[-1], :2].tolist())
        w.build_aero(xr)
        out["area_cm2"] = float(w.vlm.area * 1e4)
        out["CL_flat"] = float(w.vlm.CL(w.vlm.solve(U, math.radians(6.0))))
        a0 = w.relax(0.0, x_init=xr, fixed=fx, fixed_x=xr, aero=True)
        out.update(CL0=a0["CL"], ok0=bool(a0["ok"]), div0=bool(a0.get("diverged")))
        if not a0.get("diverged"):
            a1 = w.relax(1600.0, x_init=a0["x"], fixed=fx, fixed_x=xr, aero=True)
            q = 0.5 * 1.225 * U ** 2
            out.update(CL1=a1["CL"], ok1=bool(a1["ok"]), div1=bool(a1.get("diverged")), dCL=a1["CL"] - a0["CL"], dL_mN=(a1["CL"] - a0["CL"]) * q * w.vlm.area * 1e3, L0_mN=a0["CL"] * q * w.vlm.area * 1e3)
            if not a1.get("diverged"):
                s1 = w.summary(a1["x"], 1600.0)
                out["wrinkled1"] = [s1[i]["wrinkled"] for i in (0, 1)]
                out["F1"] = [w.fibre_report(a1["x"], 1600.0)[i]["F"] * 1e3 for i in (0, 1)]
        out["wall"] = time.time() - t0
    except Exception as e:
        out = dict(tag=tag, **k, error=repr(e))
    return out


def jobs_balanced():
    J = []
    for th, tr in ((7.0, 12.5), (10.0, 18.0), (14.0, 25.0)):
        for N in (1, 2):
            for ln in (1.0, 1.02, 1.04):
                J.append(("bal", dict(theta=th, theta_r=tr, N=N, xb=0.060, yc=0.0475, rot=1.0, root="rib", lam_n=ln)))
    for ln in (1.02, 1.04):
        J.append(("bal_active", dict(theta=10.0, theta_r=18.0, N=2, xb=0.060, yc=0.0475, rot=1.0, root="rib", lam_n=ln, active=True)))
    J.append(("long", dict(theta=10.0, theta_r=18.0, N=2, xb=0.060, yc=0.0475, rot=1.0, root="rib", lam_n=1.0, iters=120000)))
    J.append(("long", dict(theta=10.0, theta_r=18.0, N=1, xb=0.060, yc=0.0475, rot=1.0, root="rib", lam_n=1.0, iters=120000)))
    return J


def jobs_fine():
    J = []
    for th in (4.0, 5.0, 6.0, 7.0):
        for N in (1, 2):
            for act in (False, True):
                J.append(("fine", dict(theta=th, theta_r=round(1.77 * th, 2), N=N, xb=0.060, yc=0.0475, rot=1.0, root="rib", lam_n=1.04, active=act, eta=0.5)))
    base = dict(theta=7.0, theta_r=12.4, N=2, xb=0.060, yc=0.0475, rot=1.0, root="rib", lam_n=1.04, active=True, eta=0.5)
    for U in (3.0, 6.0):
        J.append(("best", dict(base, U=U)))
        J.append(("best", dict(base, U=U, N=1)))
    for xb in (0.050, 0.070):
        J.append(("best", dict(base, xb=xb)))
    for e in (0.15, 0.3, 0.7):
        J.append(("best", dict(base, eta=e)))
    return J


def jobs_final():
    base = dict(theta=5.0, theta_r=8.85, N=2, xb=0.060, yc=0.0475, rot=1.0, root="rib", lam_n=1.04, eta=0.5)
    J = [("final", dict(base, U=U, N=N)) for U in (3.0, 6.0) for N in (1, 2)]
    J += [("final", dict(base, xb=x, N=N)) for x in (0.050, 0.070) for N in (2,)]
    J += [("final", dict(base, eta=e)) for e in (0.15, 0.3, 0.4, 0.7)]
    J += [("final", dict(base, lam_n=l)) for l in (1.0, 1.02, 1.06)]
    J += [("mesh", dict(base, h=h)) for h in (0.003, 0.0015)]
    return J


def jobs_hybrid():
    J = [("long", dict(theta=14.0, N=2, xb=0.060, yc=0.0475, rot=1.0, root="rib", iters=150000)), ("long", dict(theta=14.0, N=1, xb=0.060, yc=0.0475, rot=1.0, root="rib", iters=150000))]
    for theta in (5.0, 7.0, 10.0, 14.0):
        for N in (1, 2):
            for ln in (1.01, 1.02, 1.04):
                J.append(("hybrid", dict(theta=theta, N=N, xb=0.060, yc=0.0475, rot=1.0, root="rib", lam_n=ln)))
    return J


def jobs():
    J = []
    for theta in (7.0, 10.0, 14.0, 18.0, 21.0):
        for N in (1, 2):
            J.append(("rib", dict(theta=theta, N=N, xb=0.060, yc=0.0475, rot=1.0, root="rib")))
            J.append(("free", dict(theta=theta, N=N, xb=0.060, yc=0.0475, rot=1.0, root="free")))
    for xb in (0.050, 0.070):
        for N in (1, 2):
            J.append(("xb", dict(theta=14.0, N=N, xb=xb, yc=0.0475, rot=1.0, root="rib")))
    for rot in (0.25, 0.5, 0.75):
        for N in (1, 2):
            J.append(("partial", dict(theta=14.0, N=N, xb=0.060, yc=0.0475, rot=rot, root="rib")))
    for U in (3.0, 6.0):
        J.append(("speed", dict(theta=14.0, N=2, xb=0.060, yc=0.0475, rot=1.0, root="rib", U=U)))
    J.append(("active", dict(theta=14.0, N=2, xb=0.060, yc=0.0475, rot=1.0, root="rib", active=True)))
    J.append(("active", dict(theta=14.0, N=1, xb=0.060, yc=0.0475, rot=1.0, root="rib", active=True)))
    return J


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    J = {"balanced": jobs_balanced, "fine": jobs_fine, "final": jobs_final}.get(mode, jobs)()
    if mode:
        OUT = OUT.replace("tilt_study", "tilt_" + mode)
    print(len(J), "jobs", flush=True)
    out = []
    with mp.Pool(13) as pool:
        for r in pool.imap_unordered(run, J):
            out.append(r)
            json.dump(out, open(OUT, "w"), indent=0, default=float)
            print(len(out), r["tag"], r.get("theta"), r.get("N"), r.get("root"), "dL", r.get("dL_mN"), r.get("ok0"), r.get("ok1"), "wr", r.get("wrinkled"), r.get("error", "")[:90], flush=True)
    print("done", flush=True)
