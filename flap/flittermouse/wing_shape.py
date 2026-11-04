"""Wing shape and reinforcement, steady finite elements (full_membrane/membrane.py, unchanged), H = 30 um.
All cases: 5 cm span, 12.5 cm^2, 2 active layers, rolled fibre 20 mm with 5 mm band, 1.6 kV, 4 m/s.
Veins are rigid lines bonded after tensioning: an upper limit on what a stiff vein does to the skin.
Run: python wing_shape.py [workers] -> data/wing_shape.jsonl, one line per case, safe to stop and rerun."""
import json, math, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "full_membrane"), os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
from tilt_study import rotate, AX, AY
from fe_reference import state

OUT = os.path.join(HERE, "data", "wing_shape.jsonl")
BASE = dict(t_extra=0.0, cord_EA=0.0, cord_pre=0.0, lam_d=1.0, H=30e-6, N=2, span=0.050, cr=0.025, ct=0.025, theta=5.0, stretch=1.7e-3, lam_n=1.06, ws=0.020, we=0.005, eta=0.5, eta_f=0.4, U=4.0, alpha=6.0, V=1600.0, h=2e-3, free_tip=False, veins=())
RHO, RHO_CF, LE_MG = 1070.0, 1550.0, 43.0
VEIN_MG_MM = 0.5e-3 * 0.1e-3 * RHO_CF * 1e3            # carbon strip 0.5 x 0.1 mm, mg per mm


def build(k):
    """Cut tilted at both edges, then the root and tip edges are turned to vertical. stretch = the fibre stretch that gives, in m."""
    th = math.radians(k["theta"])
    tr = math.atan((k["stretch"] + k["ct"] * math.tan(th)) / k["cr"])
    C = (AX + k["span"] + k["ct"] * math.tan(th), AY + k["ct"])
    D = (AX + k["cr"] * math.tan(tr), AY + k["cr"])
    Hm = (k["N"] + 1) * k["H"] + k["t_extra"]        # t_extra: dead film, no electrodes
    tip = AX + k["span"]
    w = M.Wing(panels=(0,), C=C, D=D, E=(tip, AY), T=(tip, AY + 1e-3), theta_deg=k["theta"], lam_d=(k["lam_d"], k["lam_d"]), lam_n=(k["lam_n"], k["lam_n"]), H=k["H"], t=(Hm, Hm), n_a=(k["N"], k["N"]),
               w_s=(k["ws"], k["ws"]), w_a_frac=(k["we"] / k["ws"],) * 2, n_f=k["N"], eta_s=k["eta"], eta_f=k["eta_f"], active_fibre=True,
               cord_EA=(k["cord_EA"],) * 2, cord_pre=(k["cord_pre"],) * 2, U=k["U"], alpha_deg=k["alpha"], h=k["h"], na_x=24, na_y=12)
    x = w.x0.copy()
    g0 = w.ids[0]
    seam, rn = g0[-1, :], g0[0, :]
    x[seam, :2] = rotate(x[seam, :2], x[seam[0], :2].copy(), th)
    x[rn, :2] = rotate(x[rn, :2], x[rn[0], :2].copy(), tr)
    return w, x, math.degrees(tr)


def vein_nodes(w, x, veins):
    """Nodes on straight veins. A vein is ((s1, c1), (s2, c2)): s along the span from the root, c along the chord from the LE, both 0 to 1 of the local grid."""
    g = w.ids[0]
    ni, nj = g.shape[0] - 1, g.shape[1] - 1
    out = set()
    for (s1, c1), (s2, c2) in veins:
        n = 2 * max(ni, nj)
        for u in np.linspace(0, 1, n + 1):
            i, j = int(round((s1 + (s2 - s1) * u) * ni)), int(round((c1 + (c2 - c1) * u) * nj))
            out.add(int(g[i, j]))
    return sorted(out)


def vein_mm(w, x, veins):
    g = w.ids[0]
    ni, nj = g.shape[0] - 1, g.shape[1] - 1
    L = 0.0
    for (s1, c1), (s2, c2) in veins:
        a, b = x[g[int(round(s1 * ni)), int(round(c1 * nj))], :2], x[g[int(round(s2 * ni)), int(round(c2 * nj))], :2]
        L += np.linalg.norm(a - b) * 1e3
    return L


def mass_inertia(w, k):
    """Skin + fibre + LE + veins, mg; second moment about the stroke axis, mg mm^2. Sheet and fibre from the mesh."""
    r = (w.x0[w.tri].mean(1)[:, 0] - AX) * 1e3
    te = (k["N"] + 1) * k["H"] + k["t_extra"]
    ms = RHO * te * w.A0 * 1e6
    rb = (0.5 * (w.x0[w.b1, 0] + w.x0[w.b2, 0]) - AX) * 1e3
    mb = RHO * w.bA * w.bL0 * 1e6
    Lmm = k["span"] * 1e3
    return dict(m_skin=float(ms.sum()), m_fibre=float(mb.sum()), m_le=LE_MG, I_mgmm2=float((ms * r ** 2).sum() + (mb * rb ** 2).sum() + LE_MG * Lmm ** 2 / 3))


def run(over):
    t0 = time.time()
    k = dict(BASE, **over)
    out = dict(over)
    try:
        w, x, tr = build(k)
        fx = w.fixed.copy()
        out["tr_deg"] = tr
        r = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)
        xr = r["x"]
        w.build_aero(xr)
        out.update(ok_ns=bool(r["ok"]), ns=state(w, xr, 0.0), **mass_inertia(w, k))
        fa = fx.copy()
        if k["free_tip"]:                      # no tip edge: only the LE corner stays held
            fa[w.ids[0][-1, 1:]] = False
        if k["veins"]:
            fa[vein_nodes(w, xr, k["veins"])] = True
            out["m_veins"] = vein_mm(w, xr, k["veins"]) * VEIN_MG_MM
        out.update(area_cm2=float(w.vlm.area * 1e4), CL_flat=float(w.vlm.CL(w.vlm.solve(k["U"], math.radians(k["alpha"])))))
        a0 = w.relax(0.0, x_init=xr, fixed=fa, fixed_x=xr, aero=True)
        out.update(ok0=bool(a0["ok"]), div0=bool(a0.get("diverged")))
        if not a0.get("diverged"):
            out["V0"] = state(w, a0["x"], 0.0)
            a1 = w.relax(k["V"], x_init=a0["x"], fixed=fa, fixed_x=xr, aero=True)
            out.update(ok1=bool(a1["ok"]), div1=bool(a1.get("diverged")))
            if not a1.get("diverged"):
                out["V1"] = state(w, a1["x"], k["V"])
    except Exception as e:
        out["error"] = repr(e)
    out["wall"] = time.time() - t0
    return out


# name, overrides. s along the span, c along the chord
RIB = lambda s: ((s, 0.0), (s, 1.0))
FAN = lambda *te: tuple(((0.0, 0.0), (s, 1.0)) for s in te)
CONCEPTS = {
    "tensioned, tip batten": dict(),
    "tensioned, tip batten, 35 um": dict(H=35e-6),
    "tensioned, tip batten, 1370 V (same field as 1.6 kV at 35 um)": dict(V=1370.0),
    "taper 35-15, short tip edge": dict(cr=0.035, ct=0.015),
    "taper 40-10, short tip edge": dict(cr=0.040, ct=0.010),
    "taper 45-5, almost no tip edge": dict(cr=0.045, ct=0.005),
    "taper 40-10, more stretch": dict(cr=0.040, ct=0.010, stretch=3.0e-3),
    "taper 45-5, more stretch": dict(cr=0.045, ct=0.005, stretch=3.0e-3),
    "free tip, no veins": dict(free_tip=True),
    "one chord rib at mid span (digit)": dict(veins=(RIB(0.5),)),
    "two chord ribs": dict(veins=(RIB(1 / 3), RIB(2 / 3))),
    "short rib, half chord, mid span": dict(veins=(((0.5, 0.0), (0.5, 0.5)),)),
    "fan of 2 veins from the root (insect)": dict(veins=FAN(1 / 3, 2 / 3)),
    "fan of 3 veins from the root (insect)": dict(veins=FAN(0.25, 0.5, 0.75)),
    "one diagonal from the root, tip batten": dict(veins=FAN(0.6)),
    "one diagonal from the root, free tip (ornithopter)": dict(veins=FAN(0.6), free_tip=True),
    "two diagonals from the root, free tip (ornithopter)": dict(veins=FAN(0.35, 0.75), free_tip=True),
}
# Same electric field as the 35 um work (1370 V at 30 um), equal fibre tension across the tapers, and mesh checks
SAME_FIELD = dict(V=1370.0)
for n_ in ("taper 35-15, short tip edge", "taper 40-10, short tip edge", "taper 40-10, more stretch", "taper 45-5, more stretch", "one chord rib at mid span (digit)", "two chord ribs",
           "short rib, half chord, mid span", "fan of 2 veins from the root (insect)", "fan of 3 veins from the root (insect)", "one diagonal from the root, tip batten"):
    CONCEPTS[n_ + ", 1370 V"] = dict(CONCEPTS[n_], **SAME_FIELD)
CONCEPTS.update({
    "taper 30-20": dict(cr=0.030, ct=0.020),
    "taper 30-20, 1370 V": dict(cr=0.030, ct=0.020, **SAME_FIELD),
    "taper 35-15, more stretch": dict(cr=0.035, ct=0.015, stretch=2.6e-3),
    "taper 35-15, more stretch, 1370 V": dict(cr=0.035, ct=0.015, stretch=2.6e-3, **SAME_FIELD),
    "mesh 1.5: tensioned, tip batten, 1370 V": dict(h=1.5e-3, **SAME_FIELD),
    "mesh 1.5: taper 35-15, more stretch, 1370 V": dict(cr=0.035, ct=0.015, stretch=2.6e-3, h=1.5e-3, **SAME_FIELD),
    "mesh 1.5: one chord rib, 1370 V": dict(veins=(RIB(0.5),), h=1.5e-3, **SAME_FIELD),
    "mesh 1.5: taper 35-15, 1.6 kV": dict(cr=0.035, ct=0.015, h=1.5e-3),
})


def cases():
    return [dict(v, name=n, alpha=a) for n, v in CONCEPTS.items() for a in (6.0, 10.0)]


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    done = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    seen = {(d["name"], d["alpha"]) for d in done}
    todo = [c for c in cases() if (c["name"], c["alpha"]) not in seen]
    print(len(todo), "cases", flush=True)
    with mp.Pool(workers) as pool, open(OUT, "a") as fh:
        for r in pool.imap_unordered(run, todo):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r["name"], r["alpha"], r.get("ok0"), r.get("ok1"), r.get("error"), "CL", r.get("V0", {}).get("CL"), r.get("V1", {}).get("CL"), "wall %.0f" % r["wall"], flush=True)
    print("DONE", flush=True)
