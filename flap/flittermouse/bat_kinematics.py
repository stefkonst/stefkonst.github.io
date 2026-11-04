"""Bat kinematics in the lattice: span fold on the upstroke, tilted stroke plane, twist. Wing kinematics are prescribed, chord flat, no skin.
phi = Phi cos(wt), downstroke 0 < wt < pi. Span scale = 1 - fold (1 - sin wt) / 2, largest on the upstroke (one sinusoidal channel).
Air power includes the work of folding against the air, not the inertia of the fold. L, T, P per wing, last cycle.
Run: python bat_kinematics.py [workers] -> data/bat_kinematics.jsonl, safe to stop and rerun."""
import itertools, json, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uvlm import UVLM

RHO, CD0 = 1.225, 0.04
OUT = os.path.join(HERE, "data", "bat_kinematics.jsonl")


def simulate(b, c, U, f, Phi_deg, alpha_deg, fold=0.0, tilt_deg=0.0, twist=0.0, ns=10, nc=5, cycles=3, caps=(1.2, 1.6)):
    w, Phi, a, tl = 2 * np.pi * f, np.radians(Phi_deg), np.radians(alpha_deg), np.radians(tilt_deg)
    steps = int(np.ceil(1 / f / (c / nc / U)))
    dt = 1 / f / steps
    S0, C0 = np.linspace(0, 1, ns + 1), np.linspace(0, c, nc + 1)
    ca, sa = np.cos(a), np.sin(a)

    def grid(t):
        phi, dphi = Phi * np.cos(w * t), -Phi * w * np.sin(w * t)
        sc = 1 - fold * (1 - np.sin(w * t)) / 2
        s = S0 * b * sc
        wz = twist * np.arctan2(-s[:, None] * dphi, U) * C0[None, :]   # nose follows the flap, trailing edge lags
        Y, Z1 = s[:, None] * np.cos(phi) - wz * np.sin(phi), s[:, None] * np.sin(phi) + wz * np.cos(phi)
        Xw = np.repeat(C0[None, :], ns + 1, 0)
        X = np.zeros((ns + 1, nc + 1, 3))
        X[..., 0] = Xw * ca + Z1 * sa + s[:, None] * np.sin(phi) * np.sin(tl)   # tilt: downstroke moves forward, upstroke back
        X[..., 1] = Y
        X[..., 2] = -Xw * sa + Z1 * ca
        return X

    air = UVLM(grid, ns, nc, dt, rho=RHO, U=U, free=False, core=1e-4, wake_max=2 * steps + 10)
    rows = []
    for k in range(steps * cycles):
        r = air.step()
        if k < steps * (cycles - 1):
            continue
        t = air.t
        phi, dphi = Phi * np.cos(w * t), -Phi * w * np.sin(w * t)
        vel, Fv, Fn = r["vel"], r["F"], r["Fn"][..., None] * r["n"]
        vr = air.Uinf - vel
        spd = np.linalg.norm(vr, axis=-1, keepdims=True)
        visc = 0.5 * RHO * spd * vr * r["area"][..., None] * CD0
        # strip force coefficient, for the caps: applied to the forces only, as in skin.py
        y = r["cp"][..., 1]
        strip_area = r["area"].sum(1)
        qs = 0.5 * RHO * (spd[..., 0] ** 2 * r["area"]).sum(1) / strip_area
        cn = np.abs(r["Fn"].sum(1)) / (qs * strip_area)
        row = [(Fv + visc)[..., 2].sum(), -(Fv + visc)[..., 0].sum(), (Fn + visc)[..., 2].sum(), -(Fn + visc)[..., 0].sum(),
               -(Fv * vel).sum() + (0.5 * RHO * spd ** 3 * r["area"][..., None] * CD0).sum(), r["area"].sum()]
        for cap in caps:
            ff = np.minimum(1.0, cap / np.maximum(cn, 1e-6))[:, None, None]
            F = Fv * ff
            row += [(F + visc)[..., 2].sum(), -(F + visc)[..., 0].sum(), -(F * vel).sum() + (0.5 * RHO * spd ** 3 * r["area"][..., None] * CD0).sum()]
        rows.append(row)
    R = np.array(rows)
    m = R.mean(0)
    out = dict(L=m[0], T=m[1], L_ns=m[2], T_ns=m[3], P=m[4], area_min=float(R[:, 5].min()), area_max=float(R[:, 5].max()))
    for i, cap in enumerate(caps):
        out.update({f"L_{cap}": m[6 + 3 * i], f"T_{cap}": m[7 + 3 * i], f"P_{cap}": m[8 + 3 * i]})
    return {k: float(v) for k, v in out.items()}


def run(job):
    t0 = time.time()
    out = dict(job)
    try:
        out.update(simulate(**job))
    except Exception as e:
        out["error"] = repr(e)
    out["wall"] = time.time() - t0
    return out


def cases():
    c = []
    for (b, ch), U, f, P, a, fo, ti, tw in itertools.product(((0.05, 0.025), (0.07, 0.03)), (3.0, 4.0), (15.0, 25.0), (20.0, 30.0), (8.0, 12.0, 16.0), (0.0, 0.2, 0.4), (0.0, 20.0), (0.0, 0.3)):
        c.append(dict(b=b, c=ch, U=U, f=f, Phi_deg=P, alpha_deg=a, fold=fo, tilt_deg=ti, twist=tw))
    return c


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    key = lambda d: json.dumps({k: d[k] for k in ("b", "c", "U", "f", "Phi_deg", "alpha_deg", "fold", "tilt_deg", "twist")}, sort_keys=True)
    seen = {key(json.loads(l)) for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = [c for c in cases() if key(c) not in seen]
    print(len(todo), "cases", flush=True)
    with mp.Pool(workers) as pool, open(OUT, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(run, todo)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if i % 40 == 0:
                print(i, r.get("error"), round(r["wall"], 1), flush=True)
    print("DONE", flush=True)
