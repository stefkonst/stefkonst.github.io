"""W5. Coupled flapping studies with skin.py. One wing; L, T, P are per wing, in the last cycle.
Designs take their tensions from the settled finite-element states. Stages:
  a  timing x design x batten x damping x kinematics
  b  voltage phase sweep and response lag
  c  stroke map for lift and thrust, trim
Run: python timing.py <stage> [workers] -> data/timing_<stage>.jsonl, safe to stop and rerun."""
import itertools, json, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skin

# from data/fe_design.jsonl, theta 5, lam_n 1.06, 1.6 kV
DESIGNS = dict(
    two20=dict(n=2, w_s=0.020, w_e=0.005, N_s0=2.13, N_c0=0.092, F0=0.0215),
    one30=dict(n=1, w_s=0.030, w_e=0.005, N_s0=1.46, N_c0=0.070, F0=0.0215),
)
COMMON = dict(ns_te=0.67, cycles=4)


def run(job):
    t0 = time.time()
    d = job["design"]
    kw = dict(COMMON, **DESIGNS[d])
    for k, v in job.items():
        if k in ("design", "Phi_deg", "alpha_deg"):
            continue
        kw[k] = v
    kw["Phi"], kw["alpha"] = np.radians(job["Phi_deg"]), np.radians(job["alpha_deg"])
    out = dict(job)
    try:
        m, o, s, S = skin.simulate(**kw)
        h = m.pop("hist")
        st = m["steps"]
        last, prev = h[-st:], h[-2 * st:-st]
        for k in ("L", "T", "L_ns", "T_ns", "P", "P_ns", "Nc", "Ns", "F", "wmax", "wmin", "tau_a", "w_tip", "w_mid"):
            out[k] = m[k + "_m"]
        out.update(P_shaft=m["P_shaft"], tau_peak=m["tau_peak"], tau_aero_peak=m["tau_aero_peak"], tau_in_peak=m["tau_in_peak"], unstable=m["unstable"],
                   periodic=float(np.sqrt(np.mean((last - prev) ** 2)) / max(np.sqrt(np.mean(last ** 2)), 1e-9)), steps=st,
                   w_tip_amp=float(0.5 * (o["w_tip"].max() - o["w_tip"].min())), w_mid_amp=float(0.5 * (o["w_mid"].max() - o["w_mid"].min())),
                   wmax_peak=float(o["wmax"].max()), wmin_peak=float(o["wmin"].min()),
                   L_down=float(o["L"][np.sin(2 * np.pi * kw["f"] * o["t"]) > 0].mean()), L_up=float(o["L"][np.sin(2 * np.pi * kw["f"] * o["t"]) <= 0].mean()),
                   T_down=float(o["T"][np.sin(2 * np.pi * kw["f"] * o["t"]) > 0].mean()), T_up=float(o["T"][np.sin(2 * np.pi * kw["f"] * o["t"]) <= 0].mean()),
                   mass_mg=skin.masses(dict(skin.PARAMS, **kw))["total"] * 1e6)
    except Exception as e:
        out["error"] = repr(e)
    out["wall"] = time.time() - t0
    return out


KIN = [dict(f=25.0, Phi_deg=30.0, alpha_deg=8.0), dict(f=20.0, Phi_deg=30.0, alpha_deg=8.0), dict(f=25.0, Phi_deg=30.0, alpha_deg=4.0),
       dict(f=25.0, Phi_deg=30.0, alpha_deg=12.0), dict(f=15.0, Phi_deg=40.0, alpha_deg=8.0)]


def stage_a():
    return [dict(design=d, tip_fixed=t, zeta=z, mode=m, **k) for d, t, z, m, k in itertools.product(DESIGNS, (True, False), (0.015, 0.05), ("off", "on", "down", "up", "sin"), KIN)]


def stage_b():
    jobs = []
    k = dict(f=25.0, Phi_deg=30.0, alpha_deg=8.0)
    for d, t in itertools.product(DESIGNS, (True, False)):
        for psi in np.arange(0, 360, 45):
            jobs.append(dict(design=d, tip_fixed=t, zeta=0.05, mode="sin", psi=float(np.radians(psi)), **k))
        for tau in (0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16):
            jobs.append(dict(design=d, tip_fixed=t, zeta=0.05, mode="down", tau=tau, **k))
            jobs.append(dict(design=d, tip_fixed=t, zeta=0.05, mode="sin", tau=tau, **k))
    return jobs


def stage_c():
    return [dict(design=d, tip_fixed=True, zeta=0.05, mode=m, U=U, f=f, Phi_deg=p, alpha_deg=a)
            for d, m, U, f, p, a in itertools.product(("two20",), ("off", "down"), (3.0, 4.0, 5.0, 6.0), (15.0, 20.0, 25.0, 30.0, 40.0), (20.0, 30.0, 40.0), (4.0, 8.0, 12.0))]


def stage_d():
    """Spring at the batten hinge: feathering without the free swing."""
    jobs = []
    for k_, m, (f, p) in itertools.product((1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3), ("off", "down"), ((20.0, 30.0), (25.0, 30.0), (15.0, 30.0))):
        jobs.append(dict(design="two20", tip_fixed=False, batten_k=k_, zeta=0.05, mode=m, f=f, Phi_deg=p, alpha_deg=8.0))
    return jobs


def stage_e():
    """Convergence of the baseline."""
    b = dict(design="two20", tip_fixed=True, zeta=0.05, f=25.0, Phi_deg=30.0, alpha_deg=8.0)
    jobs = []
    for m in ("off", "down"):
        for x in (dict(ns=14, nc=7), dict(ns=20, nc=8), dict(ns=10, nc=8), dict(free_wake=True), dict(steps_per=80), dict(cycles=7), dict(cn_cap=1.2), dict(cn_cap=1.6), dict(cd0=0.08)):
            jobs.append(dict(b, mode=m, **x))
    return jobs


def stage_f():
    """Small strokes with the section force capped: what survives if the flow will not stay attached."""
    return [dict(design="two20", tip_fixed=True, zeta=0.05, mode=m, cn_cap=cap, U=U, f=f, Phi_deg=p, alpha_deg=a)
            for cap, m, U, f, p, a in itertools.product((1.2, 1.6), ("off", "down"), (3.0, 4.0, 5.0), (15.0, 20.0, 25.0, 30.0), (10.0, 15.0, 20.0, 30.0), (4.0, 8.0, 12.0, 16.0))]


def stage_g():
    """Bigger wing, same film and tensions (no finite-element check at these sizes)."""
    jobs = []
    for b, c, ns_, nc_ in ((0.05, 0.035, 10, 7), (0.07, 0.025, 14, 5), (0.07, 0.035, 14, 7), (0.10, 0.04, 16, 6)):
        for cap, m, U, f, p, a in itertools.product((None, 1.6), ("off", "down"), (3.0, 4.0, 5.0), (10.0, 15.0, 20.0), (15.0, 20.0, 30.0), (8.0, 12.0)):
            jobs.append(dict(design="two20", tip_fixed=True, zeta=0.05, mode=m, cn_cap=cap, U=U, f=f, Phi_deg=p, alpha_deg=a, b=b, c=c, ns=ns_, nc=nc_))
    return jobs


if __name__ == "__main__":
    stage = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    OUT = os.path.join(HERE, "data", f"timing_{stage}.jsonl")
    jobs = dict(a=stage_a, b=stage_b, c=stage_c, d=stage_d, e=stage_e, f=stage_f, g=stage_g)[stage]()
    key = lambda j: json.dumps({k: j[k] for k in sorted(j) if k in ("design", "tip_fixed", "zeta", "mode", "psi", "tau", "U", "f", "Phi_deg", "alpha_deg", "eta", "cn_cap", "free_wake", "ns", "nc", "w_s", "batten_k", "steps_per", "cycles", "cd0", "b", "c")}, sort_keys=True)
    seen = {key(json.loads(l)) for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = [j for j in jobs if key(j) not in seen]
    print(stage, len(todo), "of", len(jobs), flush=True)
    with mp.Pool(workers) as pool, open(OUT, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(run, todo, chunksize=1)):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if i % 20 == 0:
                print(i, {k: r.get(k) for k in ("design", "mode", "L", "T", "wall", "error")}, flush=True)
    print("TIMING DONE", stage, flush=True)
