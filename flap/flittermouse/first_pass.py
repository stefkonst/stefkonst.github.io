"""Flittermouse: first-pass feasibility, 5 cm DEA-skin wings on 2 cm rolled DEAs.

Back-of-envelope only. Parameters at the top, one function per question.
Writes ../data/feasibility.json. Figures: figures.py.

Conventions: phi(t) = Phi cos(wt), so downstroke is 0 < wt < pi.
Voltage LOWERS tension (bench -11..-17 %), so voltage on = SOFT skin.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

EPS0 = 8.854e-12
RHO_AIR = 1.225

P = dict(
    # wing (assumption: 5 cm is one wing's span, chord set by me)
    b=0.05, c=0.025,
    # skin: stack 0.5D+E+D+E+0.5D, t = (n+1) H (full_tension/FINDINGS.md)
    n=1, H=35e-6, rho_f=1070.0, eps_r=2.8, V=1600.0, C1=42842.0,
    roll_w=0.010,                   # strip rolled into the TE fibre
    # LE, carbon strip(s)
    le_w=1.5e-3, le_t=0.2e-3, E_cf=130e9, G_cf=5e9, rho_cf=1550.0,
    # core DEA (2 cm is the user's number, mass is the free choice)
    L_core=0.020, m_core=0.5e-3, strain_amp=0.05,
    # rest of the robot, tethered: frame, hinges, spring (my guess)
    m_frame=0.20e-3,
    # flow and kinematics
    U=4.0, f=20.0, Phi_deg=25.0, alpha_deg=8.0,
    twist_tip=0.5,                  # share of the flap angle removed at the tip by passive twist (guess)
    # aero: slope with AR 4; gains are the FE lift ratio to a flat wing at 4 m/s, 6 deg
    a=2 * np.pi / (1 + 2 / 4.0), G_stiff=1.35, G_soft_05=2.40,
    CLmax=1.0, cd0=0.04, k_ind=0.10, body_CdA=0.5e-4,
)
ETA_ISO, ETA_CYC = 0.40, 0.15       # your bench: isometric, cyclic
E_CYCLE_CHEN = 500.0 / 400.0        # J/kg per cycle: Ren, Kim et al. 2022, >500 W/kg at 400 Hz


def sigma(p, eta):
    """Active stress of the core: eta chi eps E^2, chi taken as 1 at many layers."""
    return eta * EPS0 * p["eps_r"] * (p["V"] / p["H"]) ** 2


def e_cycle(p, eta):
    """J/kg per cycle. Force amplitude sigma/2 in phase with velocity, strain amplitude s:
    W = pi (sigma/2) s per unit volume."""
    return np.pi * sigma(p, eta) / 2 * p["strain_amp"] / p["rho_f"]


def core(p, eta=ETA_ISO):
    area = p["m_core"] / p["rho_f"] / p["L_core"]
    film = p["m_core"] / p["rho_f"] / p["H"]        # total electrode area
    k = 6 * p["C1"] * area / p["L_core"]
    return dict(
        cross_mm2=area * 1e6, solid_dia_mm=2e3 * np.sqrt(area / np.pi),
        film_cm2=film * 1e4, layers_on_6cm_coater=film / (0.055 * p["L_core"]),
        F_block_mN=sigma(p, eta) * area * 1e3, k_N_m=k,
        stroke_pkpk_mm=2 * p["strain_amp"] * p["L_core"] * 1e3,
        C_nF=p["eps_r"] * EPS0 * film / p["H"] * 1e9,
    )


def wing_mass(p):
    areal = p["rho_f"] * (p["n"] + 1) * p["H"]
    skin = areal * p["b"] * p["c"]
    roll = areal * p["roll_w"] * p["b"]
    a_le = 2 * p["le_w"] * p["le_t"] - p["le_t"] ** 2      # cross
    le = a_le * p["b"] * p["rho_cf"]
    return dict(skin=skin, roll=roll, le=le, total=skin + roll + le, areal=areal)


def inertia(p):
    """Flapping inertia about the root: skin, roll, LE and added air."""
    m = wing_mass(p)
    added = RHO_AIR * np.pi * p["c"] ** 2 / 4 * p["b"]
    return (m["skin"] + m["roll"] + m["le"] + added) * p["b"] ** 2 / 3, added


def le_options(p, w_mm2_load=None):
    """Tip bending of the LE as a cantilever, flap load (inertia) and chord load (skin tension)."""
    wm = wing_mass(p)
    w = 2 * np.pi * p["f"]
    acc = np.radians(p["Phi_deg"]) * w ** 2 * p["b"] / 2
    q_flap = wm["total"] * acc / p["b"]
    q_chord = 1.2 * 0.081 / 0.041                       # N/m: 81 mN line over 41 mm, scaled for this wing
    b = p["b"]
    out = {}
    def row(name, I_flap, I_chord, A, J):
        out[name] = dict(
            mass_mg=A * b * p["rho_cf"] * 1e6,
            flap_mm=q_flap * b ** 4 / (8 * p["E_cf"] * I_flap) * 1e3,
            chord_mm=q_chord * b ** 4 / (8 * p["E_cf"] * I_chord) * 1e3,
            GJ_mNm2=p["G_cf"] * J * 1e3,
        )
    for lw, lt in [(1.5e-3, 0.2e-3), (2e-3, 0.2e-3)]:
        I_flat, I_edge = lw * lt ** 3 / 12, lt * lw ** 3 / 12
        row(f"strip {lw*1e3:g} x {lt*1e3:g} mm, flat to flap", I_flat, I_edge, lw * lt, lw * lt ** 3 / 3)
        row(f"strip {lw*1e3:g} x {lt*1e3:g} mm, edge to flap", I_edge, I_flat, lw * lt, lw * lt ** 3 / 3)
        row(f"cross {lw*1e3:g} x {lt*1e3:g} mm", I_edge + I_flat, I_edge + I_flat, 2 * lw * lt - lt ** 2, 2 * lw * lt ** 3 / 3)
    do, di = 1.0e-3, 0.7e-3
    I = np.pi / 64 * (do ** 4 - di ** 4)
    row("tube 1.0/0.7 mm", I, I, np.pi / 4 * (do ** 2 - di ** 2), 2 * I)
    return out, q_flap, q_chord


def theodorsen(k):
    """Jones approximation, magnitude."""
    if k <= 0:
        return 1.0
    return abs(1 - 0.165 / (1 - 0.0455j / k) - 0.335 / (1 - 0.3j / k))


def G_soft(p, eta):
    """Your FE dL: 0.2..0.7 maps close to linearly on eta; the anchor is 14.5 mN at 0.5."""
    return p["G_stiff"] + (p["G_soft_05"] - p["G_stiff"]) * eta / 0.5


def softness(wt, mode, tau, f):
    """0 stiff (0 V), 1 soft (voltage on), first-order lag tau on a square command."""
    if mode == "stiff":
        return np.zeros_like(wt)
    if mode == "soft":
        return np.ones_like(wt)
    cmd = (np.sin(wt) > 0).astype(float)            # V on in the downstroke
    if mode == "up":
        cmd = 1 - cmd
    if tau <= 0:
        return cmd
    dt = (wt[1] - wt[0]) / (2 * np.pi * f)
    s, out = 0.5, np.zeros_like(wt)
    for _ in range(3):                              # settle over three cycles
        for i, c in enumerate(cmd):
            s += dt / tau * (c - s)
            out[i] = s
    return out


def stroke(p, mode="stiff", eta=0.5, tau=0.0, U=None, f=None, Phi=None, alpha=None):
    """Quasi-steady strips with a Theodorsen knock-down on the flap-induced angle.
    Linear membrane gain: camber follows load, on either side."""
    U = p["U"] if U is None else U
    f = p["f"] if f is None else f
    Phi = np.radians(p["Phi_deg"] if Phi is None else Phi)
    alpha = np.radians(p["alpha_deg"] if alpha is None else alpha)
    wt = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    r = (np.arange(30) + 0.5) / 30 * p["b"]
    dr = p["b"] / 30
    C = theodorsen(np.pi * f * p["c"] / U)
    s = softness(wt, mode, tau, f)
    G = p["G_stiff"] + (G_soft(p, eta) - p["G_stiff"]) * s
    phi = Phi * np.cos(wt)
    dphi = -Phi * 2 * np.pi * f * np.sin(wt)
    w = -np.outer(dphi, r)                          # downward speed, positive in the downstroke
    gamma = np.arctan2(w, U)
    a_eff = alpha + (C - p["twist_tip"] * r / p["b"]) * gamma
    cl_raw = p["a"] * G[:, None] * a_eff
    cl = np.clip(cl_raw, -p["CLmax"], p["CLmax"])
    stall = np.abs(cl_raw) > p["CLmax"]
    cd = p["cd0"] + p["k_ind"] * cl ** 2 + stall * np.sin(a_eff) ** 2
    q = 0.5 * RHO_AIR * (U ** 2 + w ** 2)
    l, d = q * p["c"] * cl * dr, q * p["c"] * cd * dr
    Fz = (l * np.cos(gamma) - d * np.sin(gamma)) * np.cos(phi)[:, None]
    Fx = l * np.sin(gamma) - d * np.cos(gamma)
    Fn = l * np.cos(gamma) + d * np.sin(gamma)
    P_aero = (Fn * w).sum(axis=1)
    lift_t, thrust_t = Fz.sum(axis=1), Fx.sum(axis=1)
    body = 0.5 * RHO_AIR * U ** 2 * p["body_CdA"] / 2      # per wing
    return dict(
        wt=wt, lift=lift_t, thrust=thrust_t, power=P_aero, s=s,
        L_mN=lift_t.mean() * 1e3, T_mN=(thrust_t.mean() - body) * 1e3,
        P_mW=P_aero.mean() * 1e3, P_pos_mW=np.clip(P_aero, 0, None).mean() * 1e3,
        stall_frac=float(stall.mean()), C=C,
        a_tip_deg=float(np.degrees(a_eff[:, -1].max())), a_tip_min_deg=float(np.degrees(a_eff[:, -1].min())),
    )


def summarise(p):
    wm, cr, cr15 = wing_mass(p), core(p, ETA_ISO), core(p, ETA_CYC)
    I_w, m_add = inertia(p)
    total = 2 * (p["m_core"] + wm["total"]) + p["m_frame"]
    w = 2 * np.pi * p["f"]
    Phi = np.radians(p["Phi_deg"])
    r = p["strain_amp"] * p["L_core"] / Phi
    k_th = cr["k_N_m"] * r ** 2
    k_need = I_w * w ** 2
    res = dict(
        params={k: (float(v) if not isinstance(v, str) else v) for k, v in p.items()},
        mass_mg=dict(core_each=p["m_core"] * 1e6, skin=wm["skin"] * 1e6, roll=wm["roll"] * 1e6, le=wm["le"] * 1e6,
                     wing_each=wm["total"] * 1e6, frame=p["m_frame"] * 1e6, total=total * 1e6,
                     areal_mg_cm2=wm["areal"] * 1e2),
        weight_mN=total * 9.81 * 1e3,
        core=cr, F_block_cyc_mN=cr15["F_block_mN"],
        e_cycle_J_kg=dict(eta_iso=e_cycle(p, ETA_ISO), eta_cyc=e_cycle(p, ETA_CYC), chen=E_CYCLE_CHEN),
        lever_mm=r * 1e3, St=p["f"] * 2 * p["b"] * np.sin(Phi) / p["U"],
        inertia_kg_m2=I_w, added_mass_mg=m_add * 1e6,
        k_theta=dict(core=k_th, needed=k_need, spring_share=1 - k_th / k_need,
                     f_core_only_Hz=np.sqrt(k_th / I_w) / (2 * np.pi)),
        inertia_torque_mNm=k_need * Phi * 1e3,
        Re=p["U"] * p["c"] / 1.5e-5, k_reduced=np.pi * p["f"] * p["c"] / p["U"],
    )
    # membrane wave speed, skin fundamental across the chord
    tension = 0.081 / 0.041
    for name, t in (("0V", tension), ("V", 0.7 * tension)):
        res[f"f_skin_{name}_Hz"] = 0.5 / p["c"] * np.sqrt(t / (wm["areal"] + RHO_AIR * p["c"] * np.pi / 4))
    # drive
    C_wing = p["eps_r"] * EPS0 * p["n"] * (p["b"] * (p["c"] + p["roll_w"])) / p["H"]
    C_tot = C_wing + cr["C_nF"] * 1e-9
    res["drive"] = dict(C_skin_nF=C_wing * 1e9, C_channel_nF=C_tot * 1e9,
                        I_mA=C_tot * w * p["V"] / 2 * 1e3,
                        P_reactive_mW_per_channel=0.5 * C_tot * p["V"] ** 2 * p["f"] * 1e3)
    return res


def scans(p):
    out = {}
    # lift, thrust, power per wing in each timing, eta 0.5 and 0.15
    for eta in (0.5, 0.15):
        for mode in ("stiff", "soft", "down", "up"):
            r = stroke(p, mode, eta)
            out[f"{mode}_eta{eta}"] = {k: r[k] for k in ("L_mN", "T_mN", "P_mW", "P_pos_mW", "stall_frac", "a_tip_deg", "a_tip_min_deg")}
    base = stroke(p, "stiff", 0.5)
    # lag
    taus = [0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1]
    out["lag"] = [dict(tau=t, **{k: stroke(p, "down", 0.5, t)[k] for k in ("L_mN", "T_mN", "P_mW")}) for t in taus]
    out["base"] = {k: base[k] for k in ("L_mN", "T_mN", "P_mW")}
    # phase-independent: frequency scan at fixed St, and at fixed amplitude
    rows = []
    for f in (10, 15, 20, 30, 40, 60):
        Phi = np.degrees(np.arcsin(min(0.25 * p["U"] / (f * 2 * p["b"]), 0.9)))
        r = stroke(p, "down", 0.5, f=f, Phi=Phi)
        rows.append(dict(f=f, Phi=Phi, L=r["L_mN"], T=r["T_mN"], P=r["P_mW"]))
    out["freq_St025"] = rows
    # map of net thrust and lift against f and amplitude, 4 m/s
    fs, Phis = [10, 15, 20, 25, 30, 40], [10, 15, 20, 25, 30, 35]
    out["map"] = dict(f=fs, Phi=Phis, T=[[stroke(p, "down", 0.5, f=f, Phi=a)["T_mN"] for f in fs] for a in Phis],
                      L=[[stroke(p, "down", 0.5, f=f, Phi=a)["L_mN"] for f in fs] for a in Phis],
                      P=[[stroke(p, "down", 0.5, f=f, Phi=a)["P_mW"] for f in fs] for a in Phis])
    # speed
    out["speed"] = [dict(U=U, L=stroke(p, "down", 0.5, U=U)["L_mN"], T=stroke(p, "down", 0.5, U=U)["T_mN"], P=stroke(p, "down", 0.5, U=U)["P_mW"]) for U in (2, 3, 4, 5, 6)]
    return out


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    res = summarise(P)
    res["le"], q1, q2 = le_options(P)
    res["le_loads_N_m"] = dict(flap=q1, chord=q2)
    res["scans"] = scans(P)
    with open(os.path.join(DATA, "feasibility.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    show = {k: v for k, v in res.items() if k not in ("params", "scans")}
    print(json.dumps(show, indent=1, default=float))
    print(json.dumps(res["scans"], indent=0, default=lambda x: round(float(x), 2))[:6000])
