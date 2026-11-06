"""B-cube wing in time: the coupled sheet + edge + LE of flittermouse/skin.py (unsteady vortex lattice, uvlm.py), set to the
built wing, with two things added: gravity (for the bench, wing flat) and the bench drive (sine squared from 0 V at its own
frequency, air off). Same equations as skin.py:

  sheet   m w_tt + c w_t = N_c w_cc + N_s w_ss + dp - m s phi_tt - m g          (string-membrane, both directions)
  edge    m_f w_tt = F w_ss - N_c w_c                                           (string with mass, pulled by the sheet)
  F   = F_0 - dF_f v + EA_f <w_s^2>/2                 edge tension: loses dF_f at full voltage, gains it back by sagging
  N_c = (8 F / L^2)(b_0 + d)                          the sheet is held in chord by the bowed edge
  N_s = N_s0 - P (...) + G t (3 e_s + 2 e_c)          span tension, Maxwell term P = eta n eps V^2 / H
v = (V / V_max)^2. Parameters: BCUBE below (measured where marked in robot/params.py).
Run: python flap_dyn.py bench | flight | test -> data/dyn_<stage>.json(l)"""
import json, math, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flittermouse"))
import numpy as np
import skin
from uvlm import UVLM

BCUBE = dict(b=0.050, c=0.019, ns=10, nc=5, U=4.0, rho_air=1.2,
             n=1, H=30e-6, V=1800.0, eta=0.15, N_s0=0.83, N_c0=0.51, ns_te=0.67,      # sheet: FE means at 0 V (fe_bench)
             F0=0.043, eta_f=0.4, w_s=0.023, w_e=0.020, active=True,               # edge: 43 mN, 20 mm active, 2 films (n_f below)
             le_kind="cross", le_w=1.5e-3, le_t=0.2e-3, rho_cf=830.0, batten_mg=1.0, tip_fixed=True,   # LE 23 mg; no batten, the curved tip is the LE
             f=25.0, Phi=math.radians(20.0), alpha=math.radians(10.0), zeta=0.03, zeta_f=45.0, cd0=0.04, cycles=4, n_f=2,
             g=0.0, air=True, drive=None, f_v=None, tau=0.0)


def voltage(p, t):
    """Command v = (V/V_max)^2 at time t. drive='sin2' is the bench: V = V_max sin^2(pi f_v t). Otherwise skin.voltage on the stroke phase."""
    if p["drive"] == "sin2":
        return math.sin(math.pi * p["f_v"] * t) ** 4
    if p["drive"] == "sine":                               # the TREK plain sine, 0 to V_max (it responds at f, so it is not bipolar)
        return (0.5 * (1 - math.cos(2 * math.pi * p["f_v"] * t))) ** 2
    return skin.voltage(p, 2 * math.pi * p["f"] * t)


def simulate(**kw):
    p = dict(skin.PARAMS, **BCUBE)
    p.update(kw)
    S = skin.Structure(p)
    S.dFf *= p["n_f"] / (1 + (p["n"] > 1))                 # skin.py counts 1 film in the roll for n = 1; the roll has 2
    ns, nc, b, c, U = p["ns"], p["nc"], p["b"], p["c"], p["U"]
    f = p["f"] if p["air"] else 0.0
    if p["air"]:
        steps = p["steps_per"] or int(np.ceil(1 / f / (c / nc / U)))
        Tc = 1 / f
    else:                                                  # bench: one drive period, dt well below the edge mode
        Tc = 1 / p["f_v"]
        steps = max(int(np.ceil(Tc / 4e-4)), 200)
    dt = Tc / steps
    om = 2 * np.pi * f
    ca, sa = np.cos(p["alpha"]), np.sin(p["alpha"])
    S0, C0 = np.linspace(0, b, ns + 1), np.linspace(0, c, nc + 1)

    def kin(t):
        if f <= 0:
            return 0.0, 0.0, 0.0
        return p["Phi"] * np.cos(om * t), -p["Phi"] * om * np.sin(om * t), -p["Phi"] * om ** 2 * np.cos(om * t)

    def to_global(wf, phi):
        Y = S0[:, None] * np.cos(phi) - wf * np.sin(phi)
        Z = S0[:, None] * np.sin(phi) + wf * np.cos(phi)
        X = np.zeros((ns + 1, nc + 1, 3))
        X[..., 0] = np.repeat(C0[None, :], ns + 1, 0) * ca + Z * sa
        X[..., 1] = Y
        X[..., 2] = -np.repeat(C0[None, :], ns + 1, 0) * sa + Z * ca
        return X

    air = UVLM(None, ns, nc, dt, rho=p["rho_air"], U=U, free=False, core=p["core"], wake_max=int(steps * 2) + 10) if p["air"] else None
    nd = S.nd
    u, v, a = np.zeros(nd), np.zeros(nd), np.zeros(nd)
    beta, gam = 0.3, 0.6
    alpha_d = 2 * p["zeta"] * 2 * np.pi * p["zeta_f"]
    beta_d = 2 * p["zeta"] / (2 * np.pi * p["zeta_f"])
    full = lambda uu: (S.T @ uu).reshape(ns + 1, nc + 1)

    def panels_to_nodes(Fn):
        nf = np.zeros((ns + 1, nc + 1))
        nf[:-1, :-1] += Fn / 4; nf[1:, :-1] += Fn / 4; nf[1:, 1:] += Fn / 4; nf[:-1, 1:] += Fn / 4
        return nf.ravel()

    n_total = steps * p["cycles"]
    rows = []
    Fn_last = np.zeros(S.N)
    vs = 0.0
    snaps = []
    for k in range(n_total):
        tn = (k + 1) * dt
        phi, dphi, ddphi = kin(tn)
        cmd = voltage(p, tn)
        vs = cmd if p["tau"] <= 0 else vs + (1 - np.exp(-dt / p["tau"])) * (cmd - vs)
        f_in = -S.Mfull * S.s * ddphi - S.Mfull * p["g"]
        f_red = S.T.T @ f_in
        un = u + dt * v + dt ** 2 * (0.5 - beta) * a
        vn = v + dt * (1 - gam) * a
        u_new, Fn_load = u.copy(), Fn_last.copy()
        for it in range(p["iters"]):
            wf = full(u_new)
            Nc, Ns, F, bow = S.tensions(wf, vs)
            K = Nc * S.Kc + Ns * S.Ks + dphi ** 2 * S.Ksc + F * S.Kf + dphi ** 2 * S.Kfc + S.Kle
            C = alpha_d * S.M + beta_d * (K - S.Kle)
            A = S.M / (beta * dt ** 2) + C * gam / (beta * dt) + K
            if air is not None:
                r = air.step(to_global(wf, phi), commit=False)
                Fn_nodes = panels_to_nodes(r["Fn"])
                Fn_load = (1 - p["relax"]) * Fn_load + p["relax"] * Fn_nodes if it or k else Fn_nodes
            rhs = f_red + S.T.T @ Fn_load + S.M @ (un / (beta * dt ** 2)) + C @ (un * gam / (beta * dt) - vn)
            u_new = np.linalg.solve(A, rhs)
        wf = full(u_new)
        a_new = (u_new - un) / (beta * dt ** 2)
        v = vn + gam * dt * a_new
        u, a = u_new, a_new
        Fn_last = Fn_load
        twist = math.degrees(math.atan2(wf[ns // 2, nc] - wf[ns // 2, 0], c))            # mid-span chord pitch from the edge
        if air is not None:
            r = air.step(to_global(wf, phi), commit=True)
            vel = r["vel"]
            vr = air.Uinf - vel
            spd = np.linalg.norm(vr, axis=-1, keepdims=True)
            visc = 0.5 * p["rho_air"] * spd * vr * r["area"][..., None] * p["cd0"]
            tot = r["F"] + visc
            Fnorm = r["Fn"][..., None] * r["n"] + visc
            P_air = -float((r["F"] * vel).sum()) + float((0.5 * p["rho_air"] * spd ** 3 * r["area"][..., None] * p["cd0"]).sum())
            rows.append((tn, phi, vs, tot[..., 2].sum(), -tot[..., 0].sum(), Fnorm[..., 2].sum(), -Fnorm[..., 0].sum(), P_air, Nc, Ns, F, wf[ns // 2, nc], wf[ns // 2, nc // 2], twist))
        else:
            rows.append((tn, 0.0, vs, 0, 0, 0, 0, 0, Nc, Ns, F, wf[ns // 2, nc], wf[ns // 2, nc // 2], twist))
        if k >= n_total - steps and (k - (n_total - steps)) % max(steps // 12, 1) == 0:
            snaps.append(wf.round(7).tolist())
    R = np.array(rows)
    names = "t phi v L T L_ns T_ns P Nc Ns F w_te w_mid twist".split()
    last = R[-steps:]
    out = {n: last[:, i] for i, n in enumerate(names)}
    m = {n + "_m": float(last[:, i].mean()) for i, n in enumerate(names)}
    m.update(steps=steps, dt=dt, unstable=bool(not np.isfinite(R).all() or np.abs(R[:, 11]).max() > 0.02),
             mass_mg=skin.masses(p)["total"] * 1e6, w_te_amp=float(0.5 * (out["w_te"].max() - out["w_te"].min())), twist_amp=float(0.5 * (out["twist"].max() - out["twist"].min())),
             w_te_hi=float(out["w_te"].min()), w_te_lo=float(out["w_te"].max()))
    # first harmonic of the edge at the drive (bench) or stroke frequency
    fr = p["f_v"] if not p["air"] else p["f"]
    ph = 2 * np.pi * fr * out["t"]
    for key in ("w_te", "twist"):
        z = out[key] - out[key].mean()
        m[key + "_a1"] = float(2 * np.abs(np.mean(z * np.exp(-1j * ph))))
        m[key + "_ph1"] = float(np.degrees(np.angle(np.mean(z * np.exp(-1j * ph)))))
    m["phase_phi"] = float(np.degrees(np.angle(np.mean(out["phi"] * np.exp(-1j * ph))))) if p["air"] else 0.0
    return m, out, snaps


def run(job):
    t0 = time.time()
    o = dict(job)
    try:
        kw = {k: v for k, v in job.items() if k not in ("design", "stage", "Phi_deg", "alpha_deg")}
        if "Phi_deg" in job:
            kw["Phi"] = math.radians(job["Phi_deg"])
        if "alpha_deg" in job:
            kw["alpha"] = math.radians(job["alpha_deg"])
        m, out, snaps = simulate(**kw)
        o.update(m)
        o["series"] = {k: np.asarray(v)[:: max(len(v) // 80, 1)].round(6).tolist() for k, v in out.items()}
        o["snaps"] = snaps
    except Exception as e:
        import traceback
        o["error"] = traceback.format_exc()[-500:]
    o["wall"] = time.time() - t0
    return o


DES = {"passive": dict(eta=0.0, eta_f=0.0), "skin": dict(eta=0.15, eta_f=0.0), "skin strong": dict(eta=0.4, eta_f=0.0),
       "edge": dict(eta=0.0, eta_f=0.4), "both": dict(eta=0.15, eta_f=0.4), "both strong": dict(eta=0.4, eta_f=0.8)}


def jobs(stage):
    J = []
    if stage == "test":
        return [dict(design="both", mode="off"), dict(design="both", mode="on"), dict(design="both", air=False, g=9.81, drive="sin2", f_v=1.0, cycles=3, **DES["both"])]
    if stage == "bench":                                    # the 5 Nov sweep: 1800 V, 1-20 Hz, and 1 Hz at 1000 / 1400 / 1800 V
        for d in ("both", "both strong", "edge"):
            for fv in (1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0):
                for zt in (0.03, 0.1):
                    J.append(dict(design=d, air=False, g=9.81, drive="sine", f_v=fv, cycles=4, zeta=zt, **DES[d]))
            for V in (1000.0, 1400.0, 1800.0):
                J.append(dict(design=d, air=False, g=9.81, drive="sin2", f_v=1.0, cycles=3, V=V, zeta=0.03, **DES[d]))
        return J
    if stage == "lag":                                     # electromechanical lag of the roll force, fitted to the 1-20 Hz sweep
        for tau in (0.02, 0.05, 0.1, 0.2):
            for fv in (1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0):
                J.append(dict(design="both", air=False, g=9.81, drive="sine", f_v=fv, cycles=6, zeta=0.1, tau=tau, **DES["both"]))
        return J
    if stage == "lagflight":                               # the flight cases again with the fitted lag
        for tau in (0.05, 0.1):
            for d in ("skin strong", "edge", "both", "both strong"):
                for mode in ("on", "down", "up", "sin"):
                    J.append(dict(design=d, mode=mode, alpha_deg=10.0, tau=tau, cycles=8, **DES[d]))
        return J
    if stage == "flight":
        for d, kw in DES.items():
            for mode in ("off", "on", "down", "up"):
                if d == "passive" and mode != "off":
                    continue
                for U, a in ((4.0, 6.0), (4.0, 10.0), (4.0, 14.0), (3.0, 10.0), (6.0, 6.0)):
                    J.append(dict(design=d, mode=mode, U=U, alpha_deg=a, **kw))
        for d in ("both", "both strong"):                     # flapping frequency against the edge mode
            for fq in (15.0, 20.0, 30.0):
                for mode in ("off", "on"):
                    J.append(dict(design=d, mode=mode, f=fq, alpha_deg=10.0, **DES[d]))
            for tau in (0.005, 0.010):
                for mode in ("down", "up"):
                    J.append(dict(design=d, mode=mode, tau=tau, alpha_deg=10.0, **DES[d]))
        return J


if __name__ == "__main__":
    stage = sys.argv[1]
    J = jobs(stage)
    fn = os.path.join(HERE, "data", "dyn_%s.jsonl" % stage)
    with mp.Pool(min(12, len(J))) as pool, open(fn, "w") as fh:
        for r in pool.imap_unordered(run, J):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["design"], r["error"], flush=True); continue
            print("%-12s %-5s air %s fv %s U %s a %s | L %.2f T %.2f P %.2f mW | F %.1f mN Nc %.2f | wTE amp %.3f mm a1 %.3f hi %.3f twist %.1f deg | %s %.0fs" % (
                r["design"], r.get("mode", "-"), r.get("air", True), r.get("f_v", "-"), r.get("U", 4.0), r.get("alpha_deg", "-"), r["L_m"] * 1e3, r["T_m"] * 1e3, r["P_m"] * 1e3,
                r["F_m"] * 1e3, r["Nc_m"], r["w_te_amp"] * 1e3, r["w_te_a1"] * 1e3, r["w_te_hi"] * 1e3, r["twist_amp"], "UNSTABLE" if r["unstable"] else "", r["wall"]), flush=True)
    print("DONE", flush=True)

