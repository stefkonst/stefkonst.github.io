"""Design study: active layers per panel, stretch, roll width, panel shape, tensioning layout. Rigid LE unless stated; 6 deg; 1.6 kV.
Stack for N active layers: 0.5D + E + (D + E) x N + 0.5D, so H_m = (N + 1) H and chi = N / (N + 1).
Lift change at fixed alpha stands in for the cycle mean (static 8.4 mN against cycle 9.2 mN at 4 m/s, 2 Hz).
Run: python optimise.py <stage> -> validation/opt_<stage>.json"""
import json, math, multiprocessing as mp, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
import design as D

OUT = os.path.join(HERE, "validation")
H = 35e-6
RHO = 1100.0
COATER = 60.0
A0X, A0Y = 3.8e-3, 6.4e-3
TILT = math.tan(math.radians(13.7))          # root edge


def stack(n):
    return (n + 1) * H, n


def geometry(xc=55e-3, yc=52.5e-3):
    return dict(C=(xc, yc), D=(A0X + (yc - A0Y) * TILT, yc))


def mask_widths(w, ws):
    """Width of each mask across its fibre, strip included (mm): the number that must fit the coater."""
    out = []
    for k in (0, 1):
        q = w.mask_panel(k)
        nn = q["n"]
        c = q["sheet"]
        out.append(float(np.ptp(np.r_[c @ nn, c[2] @ nn + q["w"]])))
    return out


def run(job):
    t0 = time.time()
    tag, kw = job
    kw = dict(kw)
    na = kw.pop("na", (1, 1))
    ld = kw.pop("lam_d", (1.05, 1.05)); dn = kw.pop("dn", (0.04, 0.04))
    ws = kw.pop("ws", (13e-3, 13e-3))
    ld = (ld, ld) if np.isscalar(ld) else ld
    dn = (dn, dn) if np.isscalar(dn) else dn
    t = tuple((n + 1) * H for n in na)
    lamn = tuple(l ** -0.5 + d for l, d in zip(ld, dn))
    par = dict(lam_d=ld, lam_n=lamn, t=t, n_a=na, w_s=ws, h=2e-3, max_iter=15000)   # unconverged states are flagged in ok0, ok1
    par.update(kw)
    try:
        q = D.evaluate(**par)
        w = M.Wing(**par)
        mass = [float(RHO * t[k] * (w.A0[w.tp == k].sum() + 0)) * 1e6 for k in (0, 1)]
        fibre = [float(np.sum(w.bA[w.bp == k] * w.bL0[w.bp == k]) * RHO * 1e6) for k in (0, 1)]
        xn = w.x0[w.tri][:, :, 0].mean(1) - A0X
        mom = float(np.sum(RHO * w.te * w.A0 * xn) + np.sum(RHO * w.bA * w.bL0 * 0.5 * (w.x0[w.b1, 0] + w.x0[w.b2, 0] - 2 * A0X))) * 1e6   # mg m
        U = par.get("U", 4.0)
        qd = 0.5 * 1.225 * U ** 2
        area = q.get("planform_cm2", 0) * 1e-4
        r = dict(tag=tag, na=na, lam_d=ld, dn=dn, ws=ws, U=U, eta=par.get("eta_s", 0.4), CL_flat=q["CL_flat"], CL0=q.get("CL0"), CL1=q.get("CL1"), dCL=q.get("dCL"), ok0=q.get("ok0"), ok1=q.get("ok1"), div1=q.get("div1"), area_cm2=area * 1e4,
                 L0_mN=None if q.get("CL0") is None else float(q["CL0"] * qd * area * 1e3), dL_mN=None if q.get("dCL") is None else float(q["dCL"] * qd * area * 1e3), mass_mg=sum(mass) + sum(fibre), mass_panels_mg=mass, mass_fibres_mg=fibre, moment_mg_m=mom,
                 mask_width_mm=mask_widths(w, ws), xc=par.get("C", (55e-3,))[0], yc=par.get("C", (0, 52.5e-3))[1], theta=par.get("theta_deg", 14.0), n1=[q["s0"][k]["N1"] for k in "01"] if q.get("s0") else None, n2=[q["s0"][k]["N2"] for k in "01"] if q.get("s0") else None,
                 wr1=[q["s1"][k]["wrinkled"] for k in "01"] if q.get("s1") else None, sl1=[q["s1"][k]["slack"] for k in "01"] if q.get("s1") else None, wmax1=[q["s1"][k]["wmax"] * 1e3 for k in "01"] if q.get("s1") else None, sag=[q["sag"][k]["sag"] * 1e3 for k in "01"], F0=[q["f0"][k]["F"] * 1e3 for k in "01"] if q.get("f0") else None, wall=time.time() - t0)
    except Exception as e:
        r = dict(tag=tag, error=repr(e))
    return r


def jobs(stage):
    if stage == "check":
        return [("uniform", dict(na=(1, 1)))]
    if stage == "layers":                    # same layers on both panels; stretch re-chosen for each stack
        return [("layers", dict(na=(n, n), lam_d=l, dn=d, U=U)) for n in (1, 2, 3, 4, 5, 6) for l in (1.03, 1.05, 1.08, 1.12) for d in (0.02, 0.04, 0.08) for U in (4.0,)]
    if stage == "speeds":                    # the best stretches for 1-4 layers at the other speeds
        return [("speeds", dict(na=(n, n), lam_d=l, dn=0.04, U=U)) for n in (1, 2, 3, 4) for l in (1.03, 1.05) for U in (3.0, 6.0)]
    if stage == "panels":                    # different layers per panel, at the stretch given in the argument
        l, d = float(sys.argv[2]), float(sys.argv[3])
        return [("panels", dict(na=(a, b), lam_d=l, dn=d, U=U)) for a in (1, 2, 3, 4, 5) for b in (1, 2, 3, 4, 5) for U in (4.0,)]
    if stage in ("roll", "shape", "shape2", "active", "attach", "eta"):
        n, l, d = (int(sys.argv[2]), int(sys.argv[3])), float(sys.argv[4]), float(sys.argv[5])
        base = dict(na=(n[0], n[1]), lam_d=l, dn=d)
        if stage == "roll":
            return [("roll", dict(base, ws=(w * 1e-3, w * 1e-3), U=U)) for w in (3, 6, 9, 13, 18, 25) for U in (3.0, 4.0, 6.0)]
        if stage == "eta":
            return [("eta", dict(base, eta_s=e, U=U)) for e in (0.15, 0.24, 0.4, 0.7) for U in (4.0,)]
        if stage == "active":                # the rolled fibre active too, as built: every dielectric in the roll is active, so n_f = t / H
            ws = float(sys.argv[6]) * 1e-3
            G = geometry(70e-3, 47.5e-3)
            return [("active", dict(na=(N, N), lam_d=l, dn=0.04, ws=(ws, ws), theta_deg=14.0, U=U, active_fibre=True, n_f=N + 1, eta_f=e, **G)) for N in (1, 2) for l in (1.05, 1.06, 1.08, 1.10, 1.12) for e in (0.3,) for U in (4.0,)] + [("active", dict(na=(2, 2), lam_d=l, dn=0.04, ws=(ws, ws), theta_deg=14.0, U=U, active_fibre=True, n_f=3, eta_f=0.4, **G)) for l in (1.08, 1.10, 1.12) for U in (3.0, 4.0, 6.0)]
        if stage == "shape2":                # further outboard, and the best shapes at other speeds and with one layer
            ws = float(sys.argv[6]) * 1e-3
            b2 = dict(base, ws=(ws, ws))
            out = [("shape2", dict(b2, theta_deg=14.0, **geometry(x * 1e-3, y * 1e-3))) for x in (70, 80, 90, 100) for y in (47.5, 52.5)]
            out += [("shape2", dict(b2, theta_deg=14.0, U=U, **geometry(x * 1e-3, 47.5e-3))) for x in (55, 70, 80) for U in (3.0, 6.0)]
            out += [("shape2", dict(b2, na=(1, 1), theta_deg=14.0, U=U, **geometry(x * 1e-3, 47.5e-3))) for x in (55, 70, 80) for U in (3.0, 4.0, 6.0)]
            return out
        if stage == "shape":
            ws = float(sys.argv[6]) * 1e-3
            b2 = dict(base, ws=(ws, ws))
            return [("shape", dict(b2, theta_deg=th, **geometry(x * 1e-3, y * 1e-3))) for x in (40, 47.5, 55, 62.5, 70) for y in (42.5, 47.5, 52.5) for th in (0.0, 14.0, 21.0)]
        if stage == "attach":
            ws = float(sys.argv[6]) * 1e-3
            b2 = dict(base, ws=(ws, ws))
            G = geometry()
            Dx, Dy = G["D"]
            k = 50.0
            axis = (A0X, 0.10)
            body = (Dx - 0.037, Dy)
            line = lambda at, anc, F: dict(at=at, anchor=anc, kind="line", k=k, F0=F)
            dea = lambda at, anc, F: dict(at=at, anchor=anc, kind="dea", w=20.6e-3, F0=F, eta=0.3)
            cases = [("batten + rib", dict()),
                     ("no batten, rib, C on stroke-axis line 20 mN", dict(no_batten=True, tensioners=(line("C", axis, 0.02),))),
                     ("no batten, rib, C on stroke-axis line 40 mN", dict(no_batten=True, tensioners=(line("C", axis, 0.04),))),
                     ("no batten, rib, C on stroke-axis line 80 mN", dict(no_batten=True, tensioners=(line("C", axis, 0.08),))),
                     ("batten, free root, D on body line 30 mN", dict(root_free=True, tensioners=(line("D", body, 0.03),))),
                     ("batten, free root, D on body line 60 mN", dict(root_free=True, tensioners=(line("D", body, 0.06),))),
                     ("batten, free root, D on body DEA fibre 30 mN", dict(root_free=True, tensioners=(dea("D", body, 0.03),))),
                     ("batten, free root, D on body DEA fibre 60 mN", dict(root_free=True, tensioners=(dea("D", body, 0.06),))),
                     ("no batten, free root, C and D on lines 40 mN", dict(no_batten=True, root_free=True, tensioners=(line("C", axis, 0.04), line("D", body, 0.04)))),
                     ("no batten, rib, C on stroke-axis DEA fibre 40 mN", dict(no_batten=True, tensioners=(dea("C", axis, 0.04),)))]
            return [(c, dict(b2, U=U, **kw_)) for c, kw_ in cases for U in (4.0,)]
    raise SystemExit(stage)


if __name__ == "__main__":
    stage = sys.argv[1]
    J = jobs(stage)
    fn = os.path.join(OUT, f"opt_{stage}.json")
    out = json.load(open(fn)) if os.path.exists(fn) else []          # keep finished runs when restarting
    key = lambda n, l, d, U: (tuple(n), round(l, 4), round(d, 4), U)
    done = {key(r["na"], r["lam_d"][0], r["dn"][0], r["U"]) for r in out if "error" not in r and r.get("tag") == stage == "layers"}
    J = [j for j in J if stage != "layers" or key(j[1]["na"], j[1]["lam_d"], j[1]["dn"], j[1]["U"]) not in done]
    print(stage, len(J), "jobs", len(out), "kept", flush=True)
    t0 = time.time()
    with mp.Pool(13) as pool:
        for r in pool.imap_unordered(run, J):
            out.append(r)
            json.dump(out, open(os.path.join(OUT, f"opt_{stage}.json"), "w"), indent=0, default=float)
            if len(out) % 10 == 0 or "error" in r:
                print(len(out), "of", len(J), round(time.time() - t0), r.get("error", ""), flush=True)
    print("done", round(time.time() - t0), flush=True)
