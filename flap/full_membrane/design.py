"""Design studies for the two-panel active membrane wing (membrane.py). Each study caches its numbers in validation/.
Run: python design.py tension | sweep | final   (figures: python figures.py)"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M

OUT = os.path.join(HERE, "validation"); os.makedirs(OUT, exist_ok=True)


def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name + ".json"), "w"), indent=1, default=float)


def stretch_along(w, x, d):
    """Element stretch along the unit vector d, from the current in-plane tensor F F^T (d . b d)^(1/2)."""
    fd = w.field(x)
    F0, F1 = fd["F0"][:, :2], fd["F1"][:, :2]
    return np.sqrt((F0 @ d) ** 2 + (F1 @ d) ** 2)


# ---------------- tensioning procedures: forward simulation from the flat, as-cut sheet ----------------
def procedure(k, how, lam=1.05, theta=14.0, h=1.5e-3):
    """Pull panel k along its fibre by S (lam - 1). Reference = planform at lam = 1; z frozen by symmetry (flat sheet)."""
    w = M.Wing(panels=(k,), lam_d=(1.0, 1.0), lam_n=(1.0, 1.0), theta_deg=theta, h=h, max_iter=120000, tol=1e-6)
    g = w.ids[k]
    fib = w.fib[k]
    d = w.dirs[k] / np.linalg.norm(w.dirs[k])
    S = np.linalg.norm(w.x0[fib[-1]] - w.x0[fib[0]])
    u = d * S * (lam - 1)
    fixed = np.zeros(w.x0.shape[0], bool)
    x = w.x0.copy()
    if how == "tips":                                          # only the two fibre ends are held
        fixed[[fib[0], fib[-1]]] = True
        x[fib[-1], :2] += u
    elif how == "edges":                                       # rigid bar grips on the bonded edges
        if k == 0:
            fixed[g[0, :]] = True; fixed[g[-1, :]] = True
            x[g[-1, :], :2] += u
        else:                                                  # batten held, LE edge and tip dragged along the fibre
            fixed[g[0, :]] = True; fixed[g[:, 0]] = True; fixed[g[-1, :]] = True
            mov = np.r_[g[1:, 0], g[-1, :]]
            x[mov, :2] += u
    elif how == "both":                                        # bar grips as above, the LE left free, plus the fibre tips
        fixed[g[0, :]] = True; fixed[g[-1, :]] = True
        x[g[-1, :], :2] += u
    # affine start for the free nodes, so the relaxation begins near the answer
    r = (w.x0[:, :2] - w.x0[fib[0], :2]) @ d / S
    start = w.x0.copy()
    start[:, :2] += np.outer(np.clip(r, 0, 1), u)
    start[fixed] = x[fixed]
    v = w.relax(0.0, x_init=start, fixed=fixed, fixed_x=start)
    xv = v["x"]
    fd = w.field(xv)
    A = w.A0
    ld = stretch_along(w, xv, d)
    N1 = fd["N1"]
    S_ = w.fibre_report(xv)[k]
    return dict(ok=bool(v["ok"]), it=v["it"], lam_mean=float(np.sum(A * ld) / A.sum()), lam_p5=float(np.percentile(ld, 5)), lam_p95=float(np.percentile(ld, 95)),
                N_mean=float(np.sum(A * N1) / A.sum()), N_cv=float(np.std(N1) / abs(N1.mean())), N_p5=float(np.percentile(N1, 5)), N_p95=float(np.percentile(N1, 95)),
                wrinkled=float(A[fd["state"] == 1].sum() / A.sum()), slack=float(A[fd["state"] == 2].sum() / A.sum()), fibre_lam=S_["lam"], fibre_F=S_["F"],
                x=w.x0[:, :2].tolist(), tri=w.tri.tolist(), Ne=N1.tolist())


def external_strip(lam=1.05, W=0.060, L=0.114, h=1.5e-3, footprint=None):
    """External grips: a flat strip W x L clamped at both short ends, pulled along L. The panel is bonded inside the uniform zone."""
    w = M.Wing(panels=(0,), A=(0, 0), D=(0, W), C=(L, W), E=(2 * L, 0), T=(2 * L, W), theta_deg=0.0, h=h, lam_d=(1.0, 1.0), lam_n=(1.0, 1.0), w_s=(0.0165, 0.0), max_iter=120000, tol=1e-6)
    g = w.ids[0]
    fixed = np.zeros(w.x0.shape[0], bool)
    fixed[g[0, :]] = True; fixed[g[-1, :]] = True
    start = w.x0.copy()
    start[:, 0] *= lam
    r = w.relax(0.0, x_init=start, fixed=fixed, fixed_x=start)
    fd = w.field(r["x"])
    cen = r["x"][w.tri][:, :, :2].mean(1)
    out = {}
    for clear in (5e-3, 10e-3, 20e-3, 30e-3):
        e = (cen[:, 0] > clear) & (cen[:, 0] < L * lam - clear)
        out[f"{clear * 1e3:.0f}"] = dict(N_mean=float(fd["N1"][e].mean()), N_cv=float(fd["N1"][e].std() / fd["N1"][e].mean()), N2_max=float(fd["N2"][e].max()), usable_mm=float((L * lam - 2 * clear) * 1e3))
    return out, bool(r["ok"])


def tension_study():
    rep = {}
    res = {}
    for k in (0, 1):
        for how in ("tips", "edges", "both") if k == 0 else ("tips", "edges"):
            t = time.time()
            res[f"{k}_{how}"] = procedure(k, how)
            q = res[f"{k}_{how}"]
            print(k, how, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in q.items() if a not in ("x", "tri", "Ne")}, round(time.time() - t, 1), flush=True)
    rep["proc"] = {k: {a: b for a, b in v.items() if a not in ("x", "tri", "Ne")} for k, v in res.items()}
    json.dump({k: dict(x=v["x"], tri=v["tri"], Ne=v["Ne"]) for k, v in res.items()}, open(os.path.join(OUT, "tension_fields.json"), "w"))
    rep["strip"], ok = external_strip()
    rep["strip_ok"] = ok
    print("strip", rep["strip"], ok, flush=True)
    # batten tilt against rigid edge grips and against the mask width limit
    th = []
    for theta in (0.0, 7.0, 14.0, 21.0, 28.0):
        row = dict(theta=theta)
        for k in (0, 1):
            q = procedure(k, "edges", theta=theta, h=2e-3)
            row[f"cv{k}"], row[f"wr{k}"], row[f"ok{k}"] = q["N_cv"], q["wrinkled"] + q["slack"], q["ok"]
        print(row, flush=True)
        th.append(row)
    rep["theta"] = th
    save("tension", rep)


def chord_step_study():
    """Recipe: strip pulled along the fibre, necked and flat; batten, root rib and tip bonded; then the LE is moved away from the fibre
    by an extra chord stretch dn and bonded. Starts from the uniform necked state; fibre edge free, as built."""
    rows = []
    for ld in (1.03, 1.05):
        for dn in (0.0, 0.01, 0.02, 0.04, 0.06):
            ln0 = ld ** -0.5
            w = M.Wing(lam_d=(ld, ld), lam_n=(ln0, ln0), h=2.5e-3, max_iter=120000, tol=5e-7)
            # displace only the LE: the shift along the chord is the extra stretch over the LE-to-fibre distance
            chord = w.pts["C"][1] - w.pts["A"][1]
            x0 = w.x0.copy()
            le = np.array(sorted(set(int(q) for q in w.le_nodes)))
            x0[le, 1] -= dn * chord
            r = w.relax(0.0, x_init=x0, fixed_x=x0)
            s, f = w.summary(r["x"]), w.fibre_report(r["x"])
            rows.append(dict(lam_d=ld, dn=dn, shift_mm=dn * chord * 1e3, ok=bool(r["ok"]), s={str(k): v for k, v in s.items()}, f={str(k): v for k, v in f.items()}))
            print(ld, dn, r["ok"], {k: (round(v["N1"], 3), round(v["N2"], 3), round(v["N1_cv"], 3), round(v["wrinkled"], 2)) for k, v in s.items()}, "sag", [round(f[k]["sag"] * 1e3, 2) for k in f], flush=True)
    save("chord_step", rows)


# ---------------- aerodynamic sweeps ----------------
def evaluate(V=1600.0, **kw):
    """Planform at 0 V, then the aeroelastic states at 0 V and V. Reference is the flat wing on the same planform."""
    w = M.Wing(**kw)
    r0 = w.relax(0.0)
    w.build_aero(r0["x"])
    p = w.p
    CL_flat = w.vlm.CL(w.vlm.solve(p["U"], math.radians(p["alpha_deg"])))
    a0 = w.relax(0.0, x_init=r0["x"], aero=True)
    out = dict(ok0=bool(a0["ok"]), div0=bool(a0.get("diverged")), CL_flat=float(CL_flat), CL0=a0["CL"], planform_cm2=float(w.vlm.area * 1e4))
    out["sag"] = {str(k): v for k, v in w.fibre_report(r0["x"]).items()}
    out["no_flow"] = {str(k): v for k, v in w.summary(r0["x"]).items()}
    if a0.get("diverged"):
        return out
    out["s0"] = {str(k): v for k, v in w.summary(a0["x"]).items()}
    out["f0"] = {str(k): v for k, v in w.fibre_report(a0["x"]).items()}
    a1 = w.relax(V, x_init=a0["x"], aero=True)
    out.update(ok1=bool(a1["ok"]), div1=bool(a1.get("diverged")), CL1=a1["CL"])
    if a1.get("diverged"):
        return out
    out["s1"] = {str(k): v for k, v in w.summary(a1["x"], V).items()}
    out["f1"] = {str(k): v for k, v in w.fibre_report(a1["x"], V).items()}
    out["dCL"] = a1["CL"] - a0["CL"]
    return out


def sweep_study():
    rows = []
    for U in (3.0, 4.0, 6.0):
        for ld in (1.02, 1.03, 1.05, 1.08):
            for dn in (0.0, 0.01, 0.02, 0.04):                  # stretch above natural necking across the fibre
                ln = ld ** -0.5 + dn
                t = time.time()
                q = evaluate(U=U, lam_d=(ld, ld), lam_n=(ln, ln), h=2e-3)
                q.update(U=U, lam_d=ld, dn=dn, lam_n=ln, wall=time.time() - t)
                rows.append(q)
                print(U, ld, dn, "CL0", q.get("CL0"), "dCL", q.get("dCL"), "ok", q.get("ok0"), q.get("ok1"), "div", q.get("div0"), q.get("div1"), "sag", round(q["sag"]["1"]["sag_frac"], 3), round(q["wall"]), flush=True)
                save("sweep", rows)


# ---------------- chosen design: responses, sensitivities, frame ----------------
def base(**kw):
    ld, dn = DESIGN["lam_d"], DESIGN["dn"]
    ln = ld ** -0.5 + dn
    q = dict(lam_d=(ld, ld), lam_n=(ln, ln), h=DESIGN["h"], w_s=(DESIGN["w_s"], DESIGN["w_s"]))
    q.update(kw)
    return q


DESIGN = dict(lam_d=1.05, dn=0.04, h=2e-3, w_s=13e-3)


def compact(q, **extra):
    r = dict(CL_flat=q["CL_flat"], CL0=q.get("CL0"), CL1=q.get("CL1"), dCL=q.get("dCL"), ok0=q.get("ok0"), ok1=q.get("ok1"), div0=q.get("div0"), div1=q.get("div1"),
             s0=q.get("s0"), s1=q.get("s1"), f0=q.get("f0"), f1=q.get("f1"), sag=q.get("sag"), area=q.get("planform_cm2"))
    r.update(extra)
    return r


def final_study():
    rep = dict(design=DESIGN)
    # voltage response from one wing, state to state
    w = M.Wing(**base(U=4.0))
    r0 = w.relax(0.0)
    w.build_aero(r0["x"])
    x = r0["x"]
    vr = []
    for V in (0, 400, 800, 1000, 1200, 1400, 1600, 1800):
        a = w.relax(float(V), x_init=x, aero=True)
        x = a["x"]
        s = w.summary(x, float(V))
        vr.append(dict(V=V, CL=a["CL"], ok=bool(a["ok"]), s={str(k): v for k, v in s.items()}))
        print("V", V, a["CL"], a["ok"], flush=True)
    rep["V"] = vr
    save("final", rep)
    # angle of attack and speed
    al = []
    for U in (3.0, 4.0, 6.0):
        for a_ in (-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0):
            q = evaluate(U=U, alpha_deg=a_, **base())
            al.append(compact(q, U=U, alpha=a_))
            print("alpha", U, a_, q.get("CL0"), q.get("dCL"), q.get("ok0"), q.get("ok1"), flush=True)
            rep["alpha"] = al
            save("final", rep)
    # DEA stack and eta (sheet thickness follows the stack, H = 35 um per film)
    st = []
    for lbl, t_, na in (("E/D/E, 1 active film, 35 um", 35e-6, 1), ("E/D/E/D, 1 active of 2 films, 70 um", 70e-6, 1), ("3 active of 4 films, 140 um", 140e-6, 3)):
        for eta in (0.24, 0.4, 0.7, 1.0):
            q = evaluate(t=t_, n_a=na, eta_s=eta, **base())
            st.append(compact(q, label=lbl, eta=eta, t=t_, n_a=na))
            print(lbl, eta, q.get("CL0"), q.get("dCL"), q.get("ok1"), flush=True)
            rep["stack"] = st
            save("final", rep)
    # tolerance on the stretches
    tol = []
    ld, dn = DESIGN["lam_d"], DESIGN["dn"]
    for dld in (-0.01, -0.005, 0.0, 0.005, 0.01):
        for ddn in (-0.005, 0.0, 0.005):
            l_ = ld + dld
            ln = ld ** -0.5 + dn + ddn
            q = evaluate(**base(lam_d=(l_, l_), lam_n=(ln, ln)))
            tol.append(compact(q, dld=dld, ddn=ddn))
            print("tol", dld, ddn, q.get("CL0"), q.get("dCL"), flush=True)
            rep["tol"] = tol
            save("final", rep)
    # strip width as cut (coater limit)
    ws = []
    for w_ in (6e-3, 10e-3, 13e-3, 16.5e-3):
        q = evaluate(**base(w_s=(w_, w_)))
        ws.append(compact(q, w_s=w_))
        print("w_s", w_, q.get("CL0"), q.get("dCL"), q["sag"]["1"]["sag"], flush=True)
    rep["w_s"] = ws
    save("final", rep)
    # batten tilt with the homogeneous state
    th = []
    for t_ in (0.0, 7.0, 14.0, 21.0, 28.0):
        q = evaluate(**base(theta_deg=t_))
        th.append(compact(q, theta=t_))
        print("theta", t_, q.get("CL0"), q.get("dCL"), flush=True)
    rep["theta"] = th
    save("final", rep)
    # panels driven separately
    ind = []
    for V_ in ((1600.0, 0.0), (0.0, 1600.0), (1600.0, 1600.0)):
        q = evaluate(V=V_, **base())
        ind.append(compact(q, V=V_))
        print("panels", V_, q.get("dCL"), flush=True)
    rep["independent"] = ind
    save("final", rep)
    # active fibre, for the record
    act = compact(evaluate(**base(active_fibre=True)))
    rep["active_fibre"] = act
    print("active fibre", act.get("CL1"), act.get("ok1"), flush=True)
    save("final", rep)
    # elastic frame
    rods = []
    for lbl, kw in (("rigid", None), ("1 mm solid", dict(rod_od=1e-3, rod_id=0.0)), ("2 mm solid", dict(rod_od=2e-3, rod_id=0.0)), ("3/2 mm tube", dict(rod_od=3e-3, rod_id=2e-3)),
                    ("taper 3/2 to 1.5/1 mm", dict(rod_od=3e-3, rod_id=2e-3, rod_od_tip=1.5e-3, rod_id_tip=1e-3))):
        if kw is None:
            q = evaluate(**base())
            rods.append(dict(label=lbl, CL0=q.get("CL0"), CL1=q.get("CL1"), dCL=q.get("dCL")))
            continue
        w = M.Wing(**base(**kw))
        r0 = w.relax(0.0)
        w.build_aero(r0["x"])
        a0 = M.solve_elastic(w, 0.0, x_init=r0["x"], aero=True)
        row = dict(label=lbl, CL0=a0.get("CL"), div0=bool(a0.get("diverged")), err0=a0.get("frame_err"))
        if not a0.get("diverged"):
            tipn = max(w.le_nodes, key=lambda n: w.x0[n, 0])
            row["tip0_mm"] = (a0["frame_u"][tipn] * 1e3).tolist()
            a1 = M.solve_elastic(w, 1600.0, x_init=a0["x"], aero=True)
            row.update(CL1=a1.get("CL"), dCL=(a1.get("CL") - a0.get("CL")) if a1.get("CL") is not None else None, err1=a1.get("frame_err"), div1=bool(a1.get("diverged")))
            if not a1.get("diverged"):
                row["tip1_mm"] = (a1["frame_u"][tipn] * 1e3).tolist()
        rods.append(row)
        print(row, flush=True)
        rep["rods"] = rods
        save("final", rep)
    # forces on the frame, 0 V no flow and with flow
    w = M.Wing(**base())
    r0 = w.relax(0.0)
    w.build_aero(r0["x"])
    a0 = w.relax(0.0, x_init=r0["x"], aero=True)
    loads = {}
    for name, x_, pe in (("no flow", r0["x"], None), ("4 m/s", a0["x"], a0["pe"])):
        f = w.reactions(x_, 0.0, pe)
        sm = np.array(sorted(set(int(n) for n in w.seam)))
        C = w.pts["C"]; B = w.pts["B"]
        u = (B - C) / np.linalg.norm(B - C)
        nrm = np.array([-u[1], u[0]])
        Fb = f[sm, :2].sum(0)
        loads[name] = dict(batten_axial_mN=float(Fb @ u * 1e3), batten_lateral_mN=float(Fb @ nrm * 1e3), batten_z_mN=float(f[sm, 2].sum() * 1e3),
                           root_x_mN=float(f[np.array(w.root), 0].sum() * 1e3), root_y_mN=float(f[np.array(w.root), 1].sum() * 1e3),
                           le_y_mN=float(f[np.array(sorted(set(int(n) for n in w.le_nodes))), 1].sum() * 1e3))
    rep["loads"] = loads
    print(loads, flush=True)
    save("final", rep)
    # shape fields for the figures
    a1 = w.relax(1600.0, x_init=a0["x"], aero=True)
    json.dump(dict(x=r0["x"].tolist(), tri=w.tri.tolist(), z0=a0["x"][:, 2].tolist(), z1=a1["x"][:, 2].tolist(), x0=a0["x"][:, :2].tolist(), x1=a1["x"][:, :2].tolist(),
                   N10=w.field(a0["x"])["N1"].tolist(), N11=w.field(a1["x"], 1600.0)["N1"].tolist(), st1=w.field(a1["x"], 1600.0)["state"].tolist(), pts={k: v.tolist() for k, v in w.pts.items()}),
              open(os.path.join(OUT, "shape.json"), "w"))
    # mesh convergence, last because it is the slowest
    mc = []
    for h in (3e-3, 2e-3, 1.5e-3, 1e-3):
        q = evaluate(**base(h=h))
        mc.append(compact(q, h=h))
        print("mesh", h, q.get("CL0"), q.get("dCL"), q.get("ok0"), q.get("ok1"), flush=True)
        rep["mesh"] = mc
        save("final", rep)
    for n_ in ((24, 12), (36, 16), (48, 24)):
        q = evaluate(**base(na_x=n_[0], na_y=n_[1]))
        mc.append(compact(q, lattice=n_))
        print("lattice", n_, q.get("CL0"), q.get("dCL"), flush=True)
        rep["mesh"] = mc
        save("final", rep)


def alpha_fine():
    """Fills the small angles between 0 and 3 degrees at each speed."""
    rep = json.load(open(os.path.join(OUT, "final.json")))
    for U in (3.0, 4.0, 6.0):
        for a_ in (0.5, 1.0, 1.5, 2.0):
            for sg in (1.0,):
                q = evaluate(U=U, alpha_deg=sg * a_, **base())
                rep["alpha"].append(compact(q, U=U, alpha=sg * a_))
                print(U, a_, q.get("CL0"), q.get("dCL"), q.get("ok0"), q.get("ok1"), flush=True)
    save("final", rep)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        DESIGN.update(json.loads(sys.argv[2]))
    {"tension": tension_study, "chord": chord_step_study, "sweep": sweep_study, "final": final_study, "alpha_fine": alpha_fine}[sys.argv[1]]()
