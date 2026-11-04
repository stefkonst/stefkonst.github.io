"""Coupled wing: DEA skin + carbon LE + rolled TE fibre + tip batten, in the unsteady lattice (uvlm.py).

Structure, in the flapping frame (s span, c chord, w normal):
  m w_tt + damping = N_c w_cc + N_s w_ss + [centrifugal tension] + dp - m s phi_tt    sheet
  m_f w_tt = F w_ss - N_c w_c + ...                                                    TE fibre (a string with mass)
  LE: Euler-Bernoulli cantilever from the root. Tip: straight batten hinged on the LE. Root: w = 0, on the stroke axis.
A membrane carries no bending, so the LE takes no torque; twist comes from the TE moving, not from the rod.

Tension, in-plane quasi-static, three scalars in time. v = (V/V_max)^2, filtered by the electromechanical lag:
  P   = eta n eps0 eps_r V_max^2 / H * v                  sheet Maxwell term
  F   = F_0 - dF_f v + EA_f <w_s^2>/2                     fibre
  N_s = N_s0 - P (1/2..1) + G t (3 e_s + 2 e_c)           span is held by root and batten; P/2 when the chord is slack
  N_c = (8 F / L^2) (b_0 + d),                            the sheet is held in chord by the BOWED fibre
        d = bow change from the chord balance             so a soft edge reduces the chord tension drop
Wrinkling is a floor on N. Parameters at the top: PARAMS.
"""
import numpy as np
from uvlm import UVLM

EPS0 = 8.854e-12

PARAMS = dict(
    b=0.050, c=0.025, ns=10, nc=5, rho_air=1.225, U=4.0,
    # sheet
    n=1, H=35e-6, eps_r=2.8, V=1600.0, eta=0.5, C1=42842.0, rho_f=1070.0, N_min=0.02,
    N_s0=1.0, N_c0=1.0,
    ns_te=1.0,                      # span tension at the TE relative to the LE; the FE gives about 0.67 (linear across the chord, same mean)
    k_Nc=1.0, k_st=1.0,             # calibration against the FE (skin_calibrate.py): chord tension factor, stretch-stiffening factor
    # fibre
    F0=0.020, eta_f=0.4, w_s=0.010, w_e=0.003, active=True,
    # LE and batten
    E_le=130e9, le_w=1.5e-3, le_t=0.2e-3, le_kind="cross", rho_cf=1550.0, batten_mg=3.0, EI_override=None,
    batten_k=0.0,                   # N m/rad, torsional spring at the batten hinge on the LE, when tip_fixed is False
    tip_fixed=False,                # True: batten bonded to the LE, so the tip cannot twist. False: batten hinged on the LE
    # kinematics: phi = Phi cos(2 pi f t). alpha: stroke axis pitch. Downstroke is 0 < 2 pi f t < pi.
    f=25.0, Phi=np.radians(30.0), alpha=np.radians(8.0),
    # timing: 'off', 'on', 'down', 'up', 'sin'; psi shifts the voltage phase; tau is the electromechanical lag
    mode="off", psi=0.0, tau=0.0, duty=0.5,
    feather=0.0,                    # prescribed twist, rigid chord: pitch = -feather * atan(s phi'/U), cancels that share of the flap angle. A requirement test, not a mechanism
    # numerics and viscous
    zeta=0.05, zeta_f=80.0, cd0=0.04, cn_cap=None, cn_cap_soft=0.0, free_wake=False, cycles=3, iters=3, relax=0.6, core=1e-4, steps_per=None,
)


def le_EI(p):
    if p["EI_override"] is not None:
        return p["EI_override"]
    w, t = p["le_w"], p["le_t"]
    if p["le_kind"] == "cross":
        return p["E_le"] * (w ** 3 * t + w * t ** 3) / 12
    if p["le_kind"] == "flat":
        return p["E_le"] * w * t ** 3 / 12
    if p["le_kind"] == "edge":
        return p["E_le"] * t * w ** 3 / 12
    if p["le_kind"] == "tube":
        return p["E_le"] * np.pi / 64 * (w ** 4 - (w - 2 * t) ** 4)
    raise ValueError(p["le_kind"])


def le_area(p):
    w, t = p["le_w"], p["le_t"]
    return dict(cross=2 * w * t - t ** 2, flat=w * t, edge=w * t, tube=np.pi / 4 * (w ** 2 - (w - 2 * t) ** 2))[p["le_kind"]]


def masses(p):
    """kg, one wing."""
    t = (p["n"] + 1) * p["H"]
    sheet = p["rho_f"] * t * p["b"] * p["c"]
    fibre = p["rho_f"] * t * p["w_s"] * p["b"]
    le = le_area(p) * p["b"] * p["rho_cf"]
    return dict(sheet=sheet, fibre=fibre, le=le, batten=p["batten_mg"] * 1e-6, total=sheet + fibre + le + p["batten_mg"] * 1e-6)


class Structure:
    """Q1 elements on the lattice mesh. Reduced dofs: all nodes except the root row; tip row tied to a straight batten."""

    def __init__(self, p):
        self.p = p
        ns, nc = p["ns"], p["nc"]
        self.ns, self.nc = ns, nc
        ds, dc = p["b"] / ns, p["c"] / nc
        self.ds, self.dc = ds, dc
        N = (ns + 1) * (nc + 1)
        idx = lambda i, j: i * (nc + 1) + j
        self.idx = idx
        self.s = np.repeat(np.arange(ns + 1) * ds, nc + 1)
        Kc, Ks, Ksc = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
        t = (p["n"] + 1) * p["H"]
        ms = p["rho_f"] * t
        gp = np.array([-1, 1]) / np.sqrt(3)
        for i in range(ns):
            for j in range(nc):
                nodes = [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                for a in gp:
                    for b_ in gp:
                        # a along span, b along chord
                        dNa = np.array([-(1 - b_), (1 - b_), (1 + b_), -(1 + b_)]) / 4 * 2 / ds
                        dNb = np.array([-(1 - a), -(1 + a), (1 + a), (1 - a)]) / 4 * 2 / dc
                        J = ds * dc / 4
                        sg = (i + (a + 1) / 2) * ds
                        xi = (j + (b_ + 1) / 2) / nc
                        Ks[np.ix_(nodes, nodes)] += J * np.outer(dNa, dNa) * (1 + (p["ns_te"] - 1) * xi) / (0.5 * (1 + p["ns_te"]))
                        Kc[np.ix_(nodes, nodes)] += J * np.outer(dNb, dNb)
                        Ksc[np.ix_(nodes, nodes)] += J * np.outer(dNa, dNa) * ms * (p["b"] ** 2 - sg ** 2) / 2
        # fibre along the TE
        mf = p["rho_f"] * t * p["w_s"]
        Kf, Kfc = np.zeros((N, N)), np.zeros((N, N))
        for i in range(ns):
            a_, b_ = idx(i, nc), idx(i + 1, nc)
            e = np.array([[1, -1], [-1, 1]]) / ds
            Kf[np.ix_([a_, b_], [a_, b_])] += e
            sg = (i + 0.5) * ds
            Kfc[np.ix_([a_, b_], [a_, b_])] += e * mf * (p["b"] ** 2 - sg ** 2) / 2
        # lumped mass
        M = np.zeros(N)
        wt_s = np.ones(ns + 1); wt_s[[0, -1]] = 0.5
        wt_c = np.ones(nc + 1); wt_c[[0, -1]] = 0.5
        for i in range(ns + 1):
            for j in range(nc + 1):
                M[idx(i, j)] += ms * ds * dc * wt_s[i] * wt_c[j]
        mle = le_area(p) * p["rho_cf"]
        for i in range(ns + 1):
            M[idx(i, 0)] += mle * ds * wt_s[i]
            M[idx(i, nc)] += mf * ds * wt_s[i]
        for j in range(nc + 1):
            M[idx(ns, j)] += p["batten_mg"] * 1e-6 / (nc + 1)
        # LE stiffness from the cantilever flexibility
        ss = (np.arange(ns) + 1) * ds
        EI = le_EI(p)
        Fx = np.zeros((ns, ns))
        for a in range(ns):
            for b_ in range(ns):
                x, y = min(ss[a], ss[b_]), max(ss[a], ss[b_])
                Fx[a, b_] = x ** 2 * (3 * y - x) / (6 * EI)
        Kb = np.linalg.inv(Fx)
        Kle = np.zeros((N, N))
        le_nodes = [idx(i + 1, 0) for i in range(ns)]
        Kle[np.ix_(le_nodes, le_nodes)] = Kb
        # reduction: drop root row; tip row follows the batten
        free = [idx(i, j) for i in range(1, ns) for j in range(nc + 1)] + [idx(ns, 0)] + ([] if p["tip_fixed"] else [idx(ns, nc)])
        T = np.zeros((N, len(free)))
        for k, q in enumerate(free):
            T[q, k] = 1.0
        if p["tip_fixed"]:
            for j in range(1, nc + 1):
                T[idx(ns, j), len(free) - 1] = 1.0
        else:
            ka, kb = len(free) - 2, len(free) - 1
            for j in range(1, nc):
                T[idx(ns, j), ka], T[idx(ns, j), kb] = 1 - j / nc, j / nc
        self.T, self.N, self.nd = T, N, len(free)
        if not p["tip_fixed"] and p["batten_k"] > 0:
            ka_, kb_ = len(free) - 2, len(free) - 1
            kq = p["batten_k"] / p["c"] ** 2
            self._Kbat = (ka_, kb_, kq)
        else:
            self._Kbat = None
        self.Mfull = M
        self.M = T.T @ np.diag(M) @ T
        self.Kc, self.Ks, self.Ksc, self.Kf, self.Kfc, self.Kle = [T.T @ k @ T for k in (Kc, Ks, Ksc, Kf, Kfc, Kle)]
        if self._Kbat:
            ka_, kb_, kq = self._Kbat
            self.Kle[np.ix_([ka_, kb_], [ka_, kb_])] += kq * np.array([[1.0, -1.0], [-1.0, 1.0]])
        self.area_nodes = ds * dc
        # scalars
        self.Gt = 2 * p["C1"] * t
        self.EAf = 6 * p["C1"] * t * p["w_s"]
        self.Pmax = p["eta"] * p["n"] * EPS0 * p["eps_r"] * p["V"] ** 2 / p["H"]
        n_f = 1 + (p["n"] > 1)
        self.dFf = p["eta_f"] * n_f * EPS0 * p["eps_r"] * p["V"] ** 2 * p["w_e"] / p["H"] if p["active"] else 0.0
        self.L2 = p["b"] ** 2
        self.b0 = p["N_c0"] * self.L2 / (8 * max(p["F0"], 1e-4))
        self.kc_edge = 4 * self.Gt * (2 / 3) / p["c"]

    def tensions(self, w_full, v):
        """w_full (ns+1, nc+1), v in [0, 1]. Returns N_c, N_s, F, in-plane bow."""
        p = self.p
        ws = np.diff(w_full, axis=0) / self.ds
        wc = np.diff(w_full, axis=1) / self.dc
        es, ec = 0.5 * np.mean(ws ** 2), 0.5 * np.mean(wc ** 2)
        ef = 0.5 * np.mean(ws[:, -1] ** 2)
        P = self.Pmax * v
        es, ec, ef = p["k_st"] * es, p["k_st"] * ec, p["k_st"] * ef
        F = max(p["F0"] - self.dFf * v + self.EAf * ef, 1e-4)
        kb = 8 * F / self.L2
        a = p["N_c0"] - P + self.Gt * (4 * ec + 2 * es)
        d = (a - kb * self.b0) / (self.kc_edge + kb)
        Nc = max(p["k_Nc"] * kb * (self.b0 + d), p["N_min"])
        # chord nearly slack: uniaxial, span force falls by P/2 (bench law A8); taut in both directions: by P
        Ns = max(p["N_s0"] - P * (0.5 + 0.5 * Nc / max(p["N_s0"], Nc)) + self.Gt * (3 * es + 2 * ec), p["N_min"])
        return Nc, Ns, F, self.b0 + d


def voltage(p, wt):
    """Command v(t) in [0, 1]. Sinusoid: the DC-biased drive shared with the core."""
    a = (wt + p["psi"]) % (2 * np.pi)
    if p["mode"] == "off":
        return 0.0
    if p["mode"] == "on":
        return 1.0
    if p["mode"] == "down":
        return 1.0 if a < 2 * np.pi * p["duty"] else 0.0
    if p["mode"] == "up":
        return 0.0 if a < 2 * np.pi * p["duty"] else 1.0
    if p["mode"] == "sin":
        return 0.5 * (1 + np.sin(a))
    raise ValueError(p["mode"])


def simulate(**kw):
    p = dict(PARAMS, **kw)
    S = Structure(p)
    ns, nc, f, b, c, U = p["ns"], p["nc"], p["f"], p["b"], p["c"], p["U"]
    steps = p["steps_per"] or (int(np.ceil(1 / f / (c / nc / U))) if f > 0 else 1)
    Tc = 1 / f if f > 0 else 2 * c * 8 / U
    if f <= 0:
        steps = max(int(np.ceil(Tc / (c / nc / U))), 1)
    dt = Tc / steps
    om = 2 * np.pi * f
    ca, sa = np.cos(p["alpha"]), np.sin(p["alpha"])
    S0, C0 = np.linspace(0, b, ns + 1), np.linspace(0, c, nc + 1)

    def kin(t):
        if f <= 0:
            return 0.0, 0.0, 0.0
        return p["Phi"] * np.cos(om * t), -p["Phi"] * om * np.sin(om * t), -p["Phi"] * om ** 2 * np.cos(om * t)

    def to_global(wf, phi, dphi=0.0):
        if p["feather"]:
            wf = wf + p["feather"] * np.arctan2(-S0[:, None] * dphi, U) * C0[None, :]
        Y = S0[:, None] * np.cos(phi) - wf * np.sin(phi)
        Z = S0[:, None] * np.sin(phi) + wf * np.cos(phi)
        Xw = np.repeat(C0[None, :], ns + 1, 0)
        X = np.zeros((ns + 1, nc + 1, 3))
        X[..., 0] = Xw * ca + Z * sa
        X[..., 1] = Y
        X[..., 2] = -Xw * sa + Z * ca
        return X

    air = UVLM(None, ns, nc, dt, rho=p["rho_air"], U=U, free=p["free_wake"], core=p["core"],
               wake_max=int(steps * 2) + 10 if f > 0 else 40)
    nd = S.nd
    u, v, a = np.zeros(nd), np.zeros(nd), np.zeros(nd)
    beta, gam = 0.3, 0.6
    Ksign = None
    tau = p["tau"]
    vs = voltage(p, 0.0) if tau <= 0 else 0.5 if p["mode"] in ("down", "up", "sin") else voltage(p, 0.0)
    n_total = steps * p["cycles"]
    rows, snaps = [], []
    Fn_last = np.zeros(S.N)

    def full(uu):
        return (S.T @ uu).reshape(ns + 1, nc + 1)

    def panels_to_nodes(Fn):
        nf = np.zeros((ns + 1, nc + 1))
        nf[:-1, :-1] += Fn / 4; nf[1:, :-1] += Fn / 4; nf[1:, 1:] += Fn / 4; nf[:-1, 1:] += Fn / 4
        return nf.ravel()

    alpha_d = 2 * p["zeta"] * 2 * np.pi * p["zeta_f"]      # mass damping tuned near the skin mode
    beta_d = 2 * p["zeta"] / (2 * np.pi * p["zeta_f"])
    res = None
    for k in range(n_total):
        t = k * dt
        tn = t + dt
        phi, dphi, ddphi = kin(tn)
        cmd = voltage(p, om * tn)
        vs = cmd if tau <= 0 else vs + (1 - np.exp(-dt / tau)) * (cmd - vs)
        # inertial load in the flapping frame
        f_in = -S.Mfull * S.s * ddphi
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
            X = to_global(wf, phi, dphi)
            r = air.step(X, commit=False)
            Fn = r["Fn"].copy()
            if p["cn_cap"] is not None:
                q_loc = 0.5 * p["rho_air"] * (U ** 2 + (S0[:-1, None] + 0.5 * S.ds) ** 2 * dphi ** 2)
                strip = Fn.sum(1) / (q_loc[:, 0] * S.ds * c)
                ff = np.minimum(1.0, (p["cn_cap"] + p["cn_cap_soft"] * vs) / np.maximum(np.abs(strip), 1e-6))
                Fn = Fn * ff[:, None]
            Fn_nodes = panels_to_nodes(Fn)
            Fn_load = (1 - p["relax"]) * Fn_load + p["relax"] * Fn_nodes if it or k else Fn_nodes
            rhs = f_red + S.T.T @ Fn_load + S.M @ (un / (beta * dt ** 2)) + C @ (un * gam / (beta * dt) - vn)
            u_new = np.linalg.solve(A, rhs)
        wf = full(u_new)
        X = to_global(wf, phi, dphi)
        r = air.step(X, commit=True)
        a_new = (u_new - un) / (beta * dt ** 2)
        v = vn + gam * dt * a_new
        u, a = u_new, a_new
        Fn_last = Fn_load
        # outputs: forces on the wing in the wind frame, with and without suction
        Fv, Fnorm = r["F"], r["Fn"][..., None] * r["n"]
        factor = 1.0
        if p["cn_cap"] is not None:
            Fv = Fv * ff[:, None, None]
            Fnorm = Fnorm * ff[:, None, None]
        vel = r["vel"]
        vr = air.Uinf - vel
        spd = np.linalg.norm(vr, axis=-1, keepdims=True)
        visc = 0.5 * p["rho_air"] * spd * vr * r["area"][..., None] * p["cd0"]
        tot, nos = Fv + visc, Fnorm + visc
        # hinge torque about the stroke axis, in the flapping frame, sign: positive lifts the wing
        ss_c = 0.5 * (S0[:-1] + S0[1:])
        tau_aero = float((r["Fn"].sum(1) * ss_c).sum()) if p["cn_cap"] is None else float((r["Fn"] * ff[:, None]).sum(1) @ ss_c)
        wdd = (S.T @ a_new)
        tau_in = float((S.Mfull * S.s * (S.s * ddphi + wdd)).sum())
        P_air = -float((Fv * vel).sum()) + float((visc * spd).sum()) * 0 + float((0.5 * p["rho_air"] * spd ** 3 * r["area"][..., None] * p["cd0"]).sum())
        P_noS = -float((Fnorm * vel).sum())
        rows.append((tn, phi, dphi, vs, tot[..., 2].sum(), -tot[..., 0].sum(), nos[..., 2].sum(), -nos[..., 0].sum(), P_air, P_noS,
                     tau_aero, tau_in, Nc, Ns, F, bow, wf[:, -1][-1], wf[ns // 2, nc], wf[ns, 0], wf.max(), wf.min()))
        if k >= n_total - steps and (k - (n_total - steps)) % max(steps // 16, 1) == 0:
            snaps.append(wf.copy())
    R = np.array(rows)
    names = "t phi dphi v L T L_ns T_ns P P_ns tau_a tau_i Nc Ns F bow w_tip w_mid z_le wmax wmin".split()
    last = R[-steps:]
    out = {n: last[:, i] for i, n in enumerate(names)}
    m = {f"{n}_m": float(last[:, i].mean()) for i, n in enumerate(names)}
    m.update(steps=steps, dt=dt, hist=R[:, 4], unstable=bool(not np.isfinite(R).all() or np.abs(R[:, 19]).max() > 0.02))
    # torque first harmonics and hinge power
    tau_tot = last[:, 10 - 0] * -1 + last[:, 11]   # aero load opposes as a lifting moment: required torque = inertia - aero
    m["P_shaft"] = float(np.mean(tau_tot * last[:, 2]))
    m["tau_peak"] = float(np.abs(tau_tot).max())
    m["tau_aero_peak"] = float(np.abs(last[:, 10]).max())
    m["tau_in_peak"] = float(np.abs(last[:, 11]).max())
    return m, out, snaps, S


if __name__ == "__main__":
    import time, sys
    t0 = time.time()
    m, o, sn, S = simulate(mode="off", f=0.0, alpha=np.radians(6.0), cycles=3)
    print("steady 0 V", {k: round(m[k], 4) for k in ("L_m", "T_m", "Nc_m", "Ns_m", "F_m", "bow_m", "wmax_m")}, "wall", round(time.time() - t0, 1), flush=True)
    q = 0.5 * 1.225 * 16 * 0.05 * 0.025
    print(" CL", m["L_m"] / q)
    m, o, sn, S = simulate(mode="on", f=0.0, alpha=np.radians(6.0), cycles=3)
    print("steady V", {k: round(m[k], 4) for k in ("L_m", "T_m", "Nc_m", "Ns_m", "F_m", "bow_m", "wmax_m")}, "CL", m["L_m"] / q, flush=True)
    t0 = time.time()
    m, o, sn, S = simulate(mode="off")
    print("flap", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k not in ("hist",)}, "wall", round(time.time() - t0, 1), flush=True)
