"""Coupled wing at H = 30 um, 1370 V (same field as 1.6 kV at 35 um), tensions from the settled finite elements. Stages:
  r  reinforcement: chord ribs against power, lift and thrust
  f  timing flipped: stiff on the downstroke (voltage on during the upstroke), with stall limits that rise or fall with softness, and lag
  b  bat kinematics with the compliant skin: fold and tilt, 5 x 2.5 cm and 7 x 3 cm
Run: python coupled_study.py <stage> [workers] -> data/coupled_<stage>.jsonl, safe to stop and rerun."""
import itertools, json, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skin

# data/wing_shape.jsonl: tensioned, tip batten, 1370 V, H 30 um
BASE = dict(H=30e-6, V=1370.0, n=2, w_s=0.020, w_e=0.005, N_s0=1.84, N_c0=0.085, F0=0.019, ns_te=0.67, cycles=4, tip_fixed=True)
KIN = [dict(f=25.0, Phi_deg=30.0, alpha_deg=8.0), dict(f=25.0, Phi_deg=20.0, alpha_deg=12.0), dict(f=25.0, Phi_deg=15.0, alpha_deg=16.0),
       dict(f=20.0, Phi_deg=30.0, alpha_deg=8.0), dict(f=15.0, Phi_deg=30.0, alpha_deg=12.0), dict(U=3.0, f=25.0, Phi_deg=20.0, alpha_deg=12.0)]


def run(job):
    t0 = time.time()
    kw = dict(BASE)
    for k, v in job.items():
        if k in ("Phi_deg", "alpha_deg", "tilt_deg", "stage", "wing"):
            continue
        kw[k] = tuple(v) if k == "ribs" else v
    kw["Phi"], kw["alpha"] = np.radians(job["Phi_deg"]), np.radians(job["alpha_deg"])
    if "tilt_deg" in job:
        kw["tilt"] = np.radians(job["tilt_deg"])
    out = dict(job)
    try:
        m, o, s, S = skin.simulate(**kw)
        h = m.pop("hist")
        st = m["steps"]
        last, prev = h[-st:], h[-2 * st:-st]
        for k in ("L", "T", "L_ns", "T_ns", "P", "P_ns", "wmax", "wmin"):
            out[k] = m[k + "_m"]
        sdown = np.sin(2 * np.pi * kw["f"] * o["t"]) > 0
        out.update(P_shaft=m["P_shaft"], tau_peak=m["tau_peak"], unstable=m["unstable"], periodic=float(np.sqrt(np.mean((last - prev) ** 2)) / max(np.sqrt(np.mean(last ** 2)), 1e-9)),
                   w_tip_amp=float(0.5 * (o["w_tip"].max() - o["w_tip"].min())), w_mid_amp=float(0.5 * (o["w_mid"].max() - o["w_mid"].min())), wmax_peak=float(o["wmax"].max()), wmin_peak=float(o["wmin"].min()),
                   L_down=float(o["L"][sdown].mean()), L_up=float(o["L"][~sdown].mean()), mass_mg=skin.masses(dict(skin.PARAMS, **kw))["total"] * 1e6)
    except Exception as e:
        out["error"] = repr(e)
    out["wall"] = time.time() - t0
    return out


def stage_r():
    c = []
    W = [("plain", dict()), ("one rib", dict(ribs=(0.5,))), ("two ribs", dict(ribs=(1 / 3, 2 / 3))), ("four ribs", dict(ribs=(0.2, 0.4, 0.6, 0.8))),
         ("one stiff rib", dict(ribs=(0.5,), rib_EI=4.3e-5, rib_mg_mm=0.155)), ("two stiff ribs", dict(ribs=(1 / 3, 2 / 3), rib_EI=4.3e-5, rib_mg_mm=0.155)), ("four stiff ribs", dict(ribs=(0.2, 0.4, 0.6, 0.8), rib_EI=4.3e-5, rib_mg_mm=0.155)),
         ("chord tension x10", dict(N_c0=0.85)), ("chord tension x30", dict(N_c0=2.55))]
    for (nm, w), k, m in itertools.product(W, KIN, ("off", "down", "up")):
        c.append(dict(wing=nm, mode=m, **w, **k))
    return c


def stage_f():
    c = []
    for k in KIN:
        c.append(dict(mode="off", **k))
        for cap in (1.2, 1.6):
            c.append(dict(mode="off", cn_cap=cap, **k))
        for m, tau in itertools.product(("down", "up"), (0.0, 0.010)):
            c.append(dict(mode=m, tau=tau, **k))
            for cap, d in itertools.product((1.2, 1.6), (-0.4, -0.2, 0.0, 0.2, 0.4)):
                c.append(dict(mode=m, tau=tau, cn_cap=cap, cn_cap_soft=d, **k))
    return c


def stage_b():
    c = []
    for (b, ch), fo, ti, m, k in itertools.product(((0.05, 0.025), (0.07, 0.03)), (0.0, 0.2, 0.4), (0.0, 20.0), ("off", "up", "down"), KIN):
        c.append(dict(b=b, c=ch, fold=fo, tilt_deg=ti, mode=m, **k))
    return c


STIFF = dict(ribs=(0.5,), rib_EI=4.3e-5, rib_mg_mm=0.155)    # carbon strip 0.5 x 0.2 mm


def stage_c():
    """Wider closure: plain or one stiff rib, fold and tilt, both sizes."""
    c = []
    K = [dict(U=U, f=f, Phi_deg=P, alpha_deg=a) for U, f, P, a in itertools.product((3.0, 4.0), (15.0, 20.0, 25.0), (20.0, 30.0), (8.0, 12.0, 16.0))]
    for (b, ch), (nm, w), fo, ti, m, k in itertools.product(((0.05, 0.025), (0.07, 0.03)), (("plain", dict()), ("one stiff rib", STIFF)), (0.0, 0.2, 0.4), (0.0, 20.0), ("off", "up"), K):
        c.append(dict(wing=nm, b=b, c=ch, fold=fo, tilt_deg=ti, mode=m, **w, **k))
    return c


def stage_m():
    """Mesh check on the rib result."""
    c = []
    K = [dict(f=25.0, Phi_deg=30.0, alpha_deg=8.0), dict(f=25.0, Phi_deg=20.0, alpha_deg=12.0)]
    for (nm, w), (ns, nc), m, k in itertools.product((("plain", dict()), ("one rib", dict(ribs=(0.5,))), ("one stiff rib", STIFF), ("four stiff ribs", dict(STIFF, ribs=(0.2, 0.4, 0.6, 0.8)))), ((10, 5), (20, 5), (20, 10)), ("off", "up", "down"), K):
        c.append(dict(wing=nm, ns=ns, nc=nc, mode=m, **w, **k))
    return c


def stage_t():
    """Heavy trailing edge: fibre inertia against rib, voltage and damping."""
    c = []
    K = [dict(f=25.0, Phi_deg=30.0, alpha_deg=8.0), dict(f=25.0, Phi_deg=20.0, alpha_deg=12.0), dict(f=20.0, Phi_deg=30.0, alpha_deg=8.0), dict(f=15.0, Phi_deg=30.0, alpha_deg=12.0)]
    for (nm, w), tm, m, z, k in itertools.product((("plain", dict()), ("one stiff rib", STIFF)), (0.1, 0.25, 0.5, 1.0, 1.5, 2.0), ("off", "up", "down"), (0.015, 0.05), K):
        c.append(dict(wing=nm, te_mass=tm, mode=m, zeta=z, **w, **k))
    return c


if __name__ == "__main__":
    tag = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    OUT = os.path.join(HERE, "data", f"coupled_{tag}.jsonl")
    key = lambda d: json.dumps({k: v for k, v in d.items() if k not in ("wall",) and not (k in ("L", "T", "L_ns", "T_ns", "P", "P_ns", "wmax", "wmin", "P_shaft", "tau_peak", "unstable", "periodic", "w_tip_amp", "w_mid_amp", "wmax_peak", "wmin_peak", "L_down", "L_up", "mass_mg", "error"))}, sort_keys=True, default=list)
    seen = {key(json.loads(l)) for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = [c for c in dict(r=stage_r, f=stage_f, b=stage_b, c=stage_c, m=stage_m, t=stage_t)[tag]() if key(c) not in seen]
    print(len(todo), "cases", flush=True)
    with mp.Pool(workers) as pool, open(OUT, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(run, todo)):
            fh.write(json.dumps(r, default=list) + "\n"); fh.flush()
            if i % 25 == 0:
                print(i, r.get("error"), round(r["wall"], 1), flush=True)
    print("DONE", flush=True)
