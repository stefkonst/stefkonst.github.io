"""Active rolled fibre on the tilted cut. The sheet is cut unstretched with the batten at theta and the root edge at theta_r, both then brought to vertical.
The fibre's force is F = F_0 - dF: F_0 grows with the strip width w_s and the stretch, dF with the electrode width w_e only. So w_e / w_s sets how close the fibre comes to slack.
Run: python active_study.py <stage> [workers] -> validation/active_<stage>.jsonl, one line per case, safe to stop and rerun."""
import json, math, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
from tilt_study import rotate, AX, AY
RHO = 1100.0                                                  # kg/m3, film
KEYS = ("err_b", "err_r", "f", "rod_od", "rod_id", "batten_d", "H", "ramp", "shape", "replan", "na_x", "na_y")
BASE = dict(theta=5.0, theta_r=8.85, N=2, xb=0.060, yc=0.0475, lam_n=1.04, ws=(0.020, 0.020), we=(0.0104, 0.0104), eta=0.5, eta_f=0.4, active=True, U=4.0, alpha=6.0, h=2e-3, rot=1.0, V=1600.0, iters=80000, aero_relax=0.2, aero_every=5)


def te_split(w, x):
    """z = (hinge about the LE up to the TE height) + (bulge on top). Returns the shape without the hinge part, and the mean TE height."""
    te = np.concatenate([x[w.fib[k], :] for k in w.panels])
    te = te[np.argsort(te[:, 0])]
    yle = w.pts["A"][1]
    yt, zt = np.interp(x[:, 0], te[:, 0], te[:, 1]), np.interp(x[:, 0], te[:, 0], te[:, 2])
    y = x.copy()
    y[:, 2] -= np.clip((x[:, 1] - yle) / np.maximum(yt - yle, 1e-4), 0, 1) * zt
    return y, float(te[:, 2].mean())


def state(w, x, V):
    s, f = w.summary(x, V), w.fibre_report(x, V)
    nr, zte = te_split(w, x)
    out = dict(CL=float(w.aero(x)[1]), CL_no_te=float(w.aero(nr)[1]), zte_mm=zte * 1e3, zmax_mm=float(x[:, 2].max() * 1e3), bulge_mm=float(nr[:, 2].max() * 1e3))
    for k in (0, 1):
        out[f"p{k}"] = dict(F_mN=f[k]["F"] * 1e3, lam=f[k]["lam"], bow_mm=f[k]["sag"] * 1e3, zedge_mm=f[k]["w_edge"] * 1e3, N1=s[k]["N1"], N2=s[k]["N2"], wr=s[k]["wrinkled"], slack=s[k]["slack"])
    return out


def build(k):
    th, tr, N = math.radians(k["theta"]), math.radians(k["theta_r"]), k["N"]
    C = (k["xb"] + (k["yc"] - AY) * math.tan(th), k["yc"])
    D = (AX + (k["yc"] - AY) * math.tan(tr), k["yc"])
    H_ = k.get("H", 35e-6)
    Hm = (N + 1) * H_
    ws, we = tuple(k["ws"]), tuple(k["we"])
    extra = {q: k[q] for q in ("na_x", "na_y") if q in k}
    w = M.Wing(C=C, D=D, theta_deg=k["theta"], lam_d=(1.0, 1.0), lam_n=(k["lam_n"], k["lam_n"]), H=H_, t=(Hm, Hm), n_a=(N, N), w_s=ws, w_a_frac=np.array(we) / np.array(ws), h=k["h"], max_iter=k["iters"], tol=5e-7, U=k["U"], alpha_deg=k["alpha"], eta_s=k["eta"], active_fibre=k["active"], n_f=N + 1, eta_f=k["eta_f"], no_batten=False, root_free=False, **extra)
    w.p["aero_relax"], w.p["aero_every"] = k["aero_relax"], k["aero_every"]
    x = w.x0.copy()
    g0 = w.ids[0]
    seam, rn = g0[-1, :], g0[0, :]
    x[seam, :2] = rotate(x[seam, :2], x[seam[0], :2].copy(), th * k["rot"] + math.radians(k.get("err_b", 0.0)))
    x[rn, :2] = rotate(x[rn, :2], x[rn[0], :2].copy(), tr * k["rot"] + math.radians(k.get("err_r", 0.0)))
    return w, x, seam, rn


def moments(w, x, seam, rn, V=0.0, pe=None):
    R_ = w.reactions(x, V, pe)
    arm = x[seam, :2] - x[seam[0], :2]
    Mb = float(np.sum(arm[:, 0] * R_[seam, 1] - arm[:, 1] * R_[seam, 0]))
    arm = x[rn, :2] - x[rn[0], :2]
    Mr = float(np.sum(arm[:, 0] * R_[rn, 1] - arm[:, 1] * R_[rn, 0]))
    return Mb * 1e3, Mr * 1e3, abs(Mr) / abs(x[rn[-1], 1] - x[rn[0], 1]) * 1e3


def run(job):
    t0 = time.time()
    tag, over = job
    k = dict(BASE, **over)
    try:
        w, x, seam, rn = build(k)
        fx = w.fixed.copy()
        r = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)
        xr = r["x"]
        Mb, Mr, Fl = moments(w, xr, seam, rn)
        w.build_aero(xr)
        q = 0.5 * 1.225 * k["U"] ** 2
        S_ = float(w.vlm.area)
        L_f = [np.linalg.norm(xr[w.fib[j][-1], :2] - xr[w.fib[j][0], :2]) for j in (0, 1)]
        mass = float(sum(RHO * (k["N"] + 1) * 35e-6 * w.A0[w.tp == j].sum() for j in (0, 1)) + sum(RHO * (k["N"] + 1) * 35e-6 * k["ws"][j] * L_f[j] for j in (0, 1))) * 1e6
        out = dict(tag=tag, **over, ok_ns=bool(r["ok"]), M_batten_mNm=Mb, M_root_mNm=Mr, F_line_mN=Fl, area_cm2=S_ * 1e4, mass_mg=mass, CL_flat=float(w.vlm.CL(w.vlm.solve(k["U"], math.radians(k["alpha"])))))
        out["ns"] = state(w, xr, 0.0)
        a0 = w.relax(0.0, x_init=xr, fixed=fx, fixed_x=xr, aero=True)
        out.update(ok0=bool(a0["ok"]), div0=bool(a0.get("diverged")))
        if not a0.get("diverged"):
            out["V0"] = state(w, a0["x"], 0.0)
            out["M_batten_flow0"] = moments(w, a0["x"], seam, rn, 0.0, a0["pe"])[0]
            a1 = w.relax(k["V"], x_init=a0["x"], fixed=fx, fixed_x=xr, aero=True)
            out.update(ok1=bool(a1["ok"]), div1=bool(a1.get("diverged")), rmax1=float(a1["rmax"]))
            if not a1.get("diverged"):
                out["V1"] = state(w, a1["x"], k["V"])
                out["M_batten_flow1"] = moments(w, a1["x"], seam, rn, k["V"], a1["pe"])[0]
                out["dL_mN"] = (a1["CL"] - a0["CL"]) * q * S_ * 1e3
                out["L0_mN"] = a0["CL"] * q * S_ * 1e3
                if k.get("ramp"):                             # up and down in steps: same state both ways means no hysteresis
                    up, dn, xc = [], [], a0["x"]
                    for v_ in np.linspace(0, k["V"], 9)[1:]:
                        c_ = w.relax(float(v_), x_init=xc, fixed=fx, fixed_x=xr, aero=True); xc = c_["x"]
                        up.append(dict(V=float(v_), CL=float(c_["CL"]), ok=bool(c_["ok"]), F=[w.fibre_report(xc, float(v_))[j]["F"] * 1e3 for j in (0, 1)], z=float(xc[:, 2].max() * 1e3)))
                    for v_ in np.linspace(0, k["V"], 9)[-2::-1]:
                        c_ = w.relax(float(v_), x_init=xc, fixed=fx, fixed_x=xr, aero=True); xc = c_["x"]
                        dn.append(dict(V=float(v_), CL=float(c_["CL"]), ok=bool(c_["ok"]), F=[w.fibre_report(xc, float(v_))[j]["F"] * 1e3 for j in (0, 1)], z=float(xc[:, 2].max() * 1e3)))
                    out["ramp_up"], out["ramp_down"] = up, dn
                if k.get("shape"):                            # the deformed sheet, for maps and sections
                    out["shape"] = dict(tri=w.tri.tolist(), x0=a0["x"].tolist(), x1=a1["x"].tolist())
                if k.get("replan"):                           # the lattice follows the fibre as it moves at voltage, instead of staying on the 0 V planform
                    w.build_aero(a1["x"])
                    a2 = w.relax(k["V"], x_init=a1["x"], fixed=fx, fixed_x=xr, aero=True)
                    out["CL1_replan"] = float(a2["CL"]); out["ok_replan"] = bool(a2["ok"]); out["area1_cm2"] = float(w.vlm.area * 1e4)
                    out["dL_replan_mN"] = (a2["CL"] * w.vlm.area - a0["CL"] * S_) * q * 1e3
    except Exception as e:
        out = dict(tag=tag, **over, error=repr(e))
    out["wall"] = time.time() - t0
    return out


def run_elastic(job):
    """Real rods: the LE tube and batten bend under the sheet's reactions. The root rod stays held by the line."""
    t0 = time.time()
    tag, over = job
    k = dict(BASE, **over)
    try:
        w, x, seam, rn = build(k)
        w.p.update(rod_od=k["rod_od"], rod_id=k["rod_id"], batten_d=k["batten_d"])
        fx = w.fixed.copy()
        xr = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)["x"]
        w.x0 = xr.copy()
        w.build_aero(xr)
        q = 0.5 * 1.225 * k["U"] ** 2
        a0 = M.solve_elastic(w, 0.0, x_init=xr, aero=True)
        a1 = M.solve_elastic(w, k["V"], x_init=a0["x"], aero=True)
        tip = lambda r: float(np.abs(r["x"][w.le_nodes, :] - xr[w.le_nodes, :]).max() * 1e3)
        bat = lambda r: float(np.abs(r["x"][seam, :] - xr[seam, :]).max() * 1e3)
        out = dict(tag=tag, **over, ok0=bool(a0["ok"]), ok1=bool(a1["ok"]), V0=state(w, a0["x"], 0.0), V1=state(w, a1["x"], k["V"]), dL_mN=(a1["CL"] - a0["CL"]) * q * w.vlm.area * 1e3, le_mm=[tip(a0), tip(a1)], batten_mm=[bat(a0), bat(a1)], err=[a0.get("frame_err"), a1.get("frame_err")])
    except Exception as e:
        out = dict(tag=tag, **over, error=repr(e))
    out["wall"] = time.time() - t0
    return out


def run_flap(job):
    """Quasi-steady stroke at phases 0, 15, ... 345 deg, each state starting from the last. The first pass only settles the path."""
    t0 = time.time()
    tag, over = job
    k = dict(BASE, **over)
    try:
        w, x, seam, rn = build(k)
        fx = w.fixed.copy()
        xr = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)["x"]
        w.build_aero(xr)
        q = 0.5 * 1.225 * k["U"] ** 2
        A, om = math.radians(30.0), 2 * math.pi * k["f"]
        ph = np.arange(0.0, 360.0, 15.0)
        res = dict(L0=[], L1=[], ok0=[], ok1=[], F1=[])
        x0, x1 = xr, None
        for p_ in range(2):
            for f_ in ph:
                w.set_flap(-A * om * math.sin(math.radians(f_)), -A * om * om * math.cos(math.radians(f_)))
                a0 = w.relax(0.0, x_init=x0, fixed=fx, fixed_x=xr, aero=True, max_iter=30000); x0 = a0["x"]
                a1 = w.relax(k["V"], x_init=x0 if x1 is None else x1, fixed=fx, fixed_x=xr, aero=True, max_iter=30000); x1 = a1["x"]
                if p_ == 1:
                    res["L0"].append(float(a0["CL"] * q * w.vlm.area * 1e3)); res["L1"].append(float(a1["CL"] * q * w.vlm.area * 1e3))
                    res["ok0"].append(bool(a0["ok"])); res["ok1"].append(bool(a1["ok"]))
                    res["F1"].append([w.fibre_report(x1, k["V"])[j]["F"] * 1e3 for j in (0, 1)])
        out = dict(tag=tag, **over, phases=ph.tolist(), **res)
    except Exception as e:
        out = dict(tag=tag, **over, error=repr(e))
    out["wall"] = time.time() - t0
    return out


def go(job):
    return dict(elastic=run_elastic, flap=run_flap).get(job[0], run)(job)


def jobs(stage):
    J = []
    if stage == "sweep":
        for N in (1, 2, 3, 4):
            for ws, wes in ((13, (2.5, 5, 7.5, 10.4)), (16, (5, 7.5, 10.4, 13)), (20, (5, 7.5, 10.4, 14)), (26, (7.5, 10.4, 14, 20))):
                for we in wes:
                    J.append(("wswe", dict(N=N, ws=(ws * 1e-3, ws * 1e-3), we=(we * 1e-3, we * 1e-3))))
        for N in (1, 2, 3, 4):
            for ws in (13, 20, 26):
                J.append(("passive", dict(N=N, ws=(ws * 1e-3, ws * 1e-3), active=False)))
        for th in (3.0, 4.0, 6.0, 7.0, 8.0):
            for ws in (13, 20):
                J.append(("tilt", dict(theta=th, theta_r=round(1.77 * th, 2), ws=(ws * 1e-3, ws * 1e-3))))
        for xb in (50, 55, 65, 70):
            J.append(("xb", dict(xb=xb * 1e-3)))
        for ln in (1.02, 1.03, 1.05, 1.06, 1.08):
            J.append(("lam_n", dict(lam_n=ln)))
    if stage == "explore":
        w2 = lambda a, b=None: (a * 1e-3, (a if b is None else b) * 1e-3)
        for N in (2, 3):
            d = dict(N=N, ws=w2(20), we=w2(5))
            J += [("onepanel", dict(d, we=w2(5, 0))), ("onepanel", dict(d, we=w2(0, 5)))]
            J += [("perpanel", dict(d, ws=w2(16, 26), we=w2(5, 7.5))), ("perpanel", dict(d, ws=w2(20, 26), we=w2(5, 7.5))), ("perpanel", dict(d, ws=w2(20, 13), we=w2(5, 2.5)))]
            J += [("we", dict(d, we=w2(2.5))), ("we", dict(d, we=w2(3.5))), ("we", dict(d, we=w2(6)))]
            J += [("tilt", dict(d, theta=4.0, theta_r=7.08)), ("tilt", dict(d, theta=4.5, theta_r=7.97)), ("tilt", dict(d, theta=5.5, theta_r=9.74)), ("tilt", dict(d, theta=6.0, theta_r=10.62))]
            J += [("balance", dict(d, theta_r=r)) for r in (7.5, 8.0, 8.4, 9.3)]
            J += [("xb", dict(d, xb=x * 1e-3)) for x in (50, 55, 65, 70)]
            J += [("chord", dict(d, yc=y * 1e-3)) for y in (42.5, 45.0, 50.0)]
            J += [("lam_n", dict(d, lam_n=l)) for l in (1.02, 1.03, 1.05, 1.06, 1.08)]
    D = dict(N=2, theta=4.5, theta_r=7.5, lam_n=1.05, ws=(.02, .02), we=(.005, .005))
    W2 = lambda a: (a * 1e-3, a * 1e-3)
    if stage == "check":
        J.append(("design", dict(D, ramp=True, shape=True, replan=True)))
        J.append(("design3", dict(D, N=3, ramp=True, replan=True)))
        J += [("mesh", dict(D, h=h * 1e-3)) for h in (3.0, 2.5, 1.75, 1.5, 1.25)]
        J += [("lattice", dict(D, na_x=a, na_y=b)) for a, b in ((24, 12), (48, 24), (72, 32))]
        J += [("rot", dict(D, rot=r)) for r in (0.7, 0.85, 0.95, 1.05, 1.15)]
        J += [("err", dict(D, err_b=e)) for e in (-2.0, -1.0, 1.0, 2.0)] + [("err", dict(D, err_r=e)) for e in (-2.0, -1.0, 1.0, 2.0)]
        for N in (1, 3, 4):
            J.append(("layers", dict(D, N=N)))
        for N in (1, 2, 3, 4):
            J.append(("layers", dict(D, N=N, active=False)))
        J += [("eta", dict(D, eta=e)) for e in (0.2, 0.3, 0.4, 0.6, 0.7)] + [("eta", dict(D, N=3, eta=e)) for e in (0.3, 0.4, 0.6)]
        J += [("eta_f", dict(D, eta_f=e)) for e in (0.2, 0.3, 0.5, 0.6, 0.8)]
        J += [("U", dict(D, U=u)) for u in (2.0, 3.0, 5.0, 6.0, 7.0)]
        J += [("alpha", dict(D, alpha=a)) for a in (2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 12.0)]
        J += [("V", dict(D, V=v)) for v in (800.0, 1000.0, 1200.0, 1400.0, 1800.0, 2000.0)]
        J += [("H", dict(D, H=h * 1e-6)) for h in (30.0, 32.5, 37.5, 40.0)]
        J += [("ws", dict(D, ws=W2(w_))) for w_ in (12, 14, 16, 18, 22)]
        J += [("we", dict(D, we=W2(w_))) for w_ in (2.5, 3.5, 6.5, 8.0)]
        J += [("elastic", dict(D, rod_od=a, rod_id=b, batten_d=c)) for a, b, c in ((3e-3, 2e-3, 0.8e-3), (2e-3, 0.0, 0.8e-3), (3e-3, 2e-3, 0.5e-3), (3e-3, 2e-3, 1.0e-3))]
        J += [("flap", dict(D, U=u, f=f_, alpha=a)) for u, f_, a in ((4.0, 2.0, 6.0), (4.0, 2.0, 8.0), (4.0, 1.0, 6.0), (3.0, 2.0, 8.0), (6.0, 2.0, 6.0))]
    if stage == "lowEta":                                     # the largest unknown: does the design still work if the sheet's eta is low?
        for e in (0.2, 0.3, 0.4):
            J.append(("lowEta", dict(D, eta=e, iters=200000)))
            J.append(("lowEta", dict(D, eta=e, active=False, iters=200000)))
            J.append(("lowEta", dict(D, eta=e, we=W2(2.5), iters=200000)))
            J.append(("lowEta", dict(D, eta=e, N=3, iters=200000)))
    if stage == "pick":
        for N in (2, 3):
            for tr in (7.0, 7.5, 8.0):
                for ln in (1.04, 1.05, 1.06):
                    J.append(("pick", dict(N=N, theta=4.5, theta_r=tr, lam_n=ln, ws=(.02, .02), we=(.005, .005))))
    return J


if __name__ == "__main__":
    stage = sys.argv[1]
    out = os.path.join(HERE, "validation", f"active_{stage}.jsonl")
    key = lambda tag, d: json.dumps([tag, {a: b for a, b in d.items() if a in BASE or a in KEYS}], sort_keys=True)
    done = {key(q["tag"], q) for q in map(json.loads, open(out))} if os.path.exists(out) else set()
    todo = [j for j in jobs(stage) if key(*j) not in done]
    print(len(todo), "cases", flush=True)
    with mp.Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 12) as p, open(out, "a") as f:
        for r in p.imap_unordered(go, todo, chunksize=1):
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
            print(r["tag"], {a: b for a, b in r.items() if a in BASE}, "ok", r.get("ok0"), r.get("ok1"), "dL", round(r.get("dL_mN", float("nan")), 2), "F1", [round(r["V1"][q]["F_mN"], 1) for q in ("p0", "p1")] if "V1" in r else None, "Mb", round(r.get("M_batten_mNm", 0), 2), r.get("error", "")[:80], int(r["wall"]), flush=True)
