"""W6 and W7. Leading edge options, transmission parts, skin modes, electrode time constants, drive power, core build.
Closed forms, plus the modes of skin.Structure. Output: data/structure_elec.json."""
import json, os
import numpy as np
import skin

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 8.854e-12 * 2.8
B, C = 0.05, 0.025
E_CF, G_CF, RHO_CF = 130e9, 5e9, 1550.0
Phi, F = np.radians(30.0), 25.0
W = 2 * np.pi * F
out = {}

# ---------- LE: loads from this wing, not from the 7 cm design
m_skin = skin.masses(skin.PARAMS)
acc_mid = Phi * W ** 2 * B / 2
q_flap = (m_skin["sheet"] + m_skin["fibre"]) * Phi * W ** 2 * (2 / 3 * B) * 1.5 / B   # uniform equivalent of a triangular inertia load
q_air = 15e-3 / 2 * 3 / B                                                            # peak lift about 3x mean, per metre
le = []


def row(name, A, I_flap, I_chord, J, N_chord):
    m = A * B * RHO_CF
    q_self = m * Phi * W ** 2 * (2 / 3 * B) * 1.5 / B / B * B
    q = q_flap + q_air + A * RHO_CF * Phi * W ** 2 * B * 0.5
    d_flap = q * B ** 4 / (8 * E_CF * I_flap)
    d_chord = N_chord * B ** 4 / (8 * E_CF * I_chord)
    f1 = (1.875 ** 2 / (2 * np.pi * B ** 2)) * np.sqrt(E_CF * I_flap / (A * RHO_CF + 0.5 * m_skin["sheet"] / B + 1e-12))
    # peak bending strain at the root, flap plane
    h = name_h[name]
    eps = q * B ** 2 / 2 / (E_CF * I_flap) * h / 2
    le.append(dict(name=name, mass_mg=m * 1e6, flap_mm=d_flap * 1e3, chord_mm=d_chord * 1e3, f1_Hz=f1, GJ_mN_m2=G_CF * J * 1e3, root_strain_pct=eps * 100, EI_flap=E_CF * I_flap))


name_h = {}
N_chord = 1.25 * 0.5                                                                  # N/m, upper estimate pulling on the LE at this tension level
for w_, t_ in ((1.0e-3, 0.2e-3), (1.5e-3, 0.2e-3), (2.0e-3, 0.2e-3), (2.0e-3, 0.3e-3)):
    If, Ie = w_ * t_ ** 3 / 12, t_ * w_ ** 3 / 12
    n = f"strip {w_*1e3:g} x {t_*1e3:g}, flat to flap"
    name_h[n] = t_; row(n, w_ * t_, If, Ie, w_ * t_ ** 3 / 3, N_chord)
    n = f"strip {w_*1e3:g} x {t_*1e3:g}, on edge"
    name_h[n] = w_; row(n, w_ * t_, Ie, If, w_ * t_ ** 3 / 3, N_chord)
    n = f"cross {w_*1e3:g} x {t_*1e3:g}"
    name_h[n] = w_; row(n, 2 * w_ * t_ - t_ ** 2, If + Ie, If + Ie, 2 * w_ * t_ ** 3 / 3, N_chord)
for d, i in ((1.0e-3, 0.7e-3), (1.5e-3, 1.0e-3)):
    I = np.pi / 64 * (d ** 4 - i ** 4)
    n = f"tube {d*1e3:g}/{i*1e3:g}"
    name_h[n] = d; row(n, np.pi / 4 * (d ** 2 - i ** 2), I, I, 2 * I, N_chord)
out["le"] = le

# ---------- transmission
kt = 3.3e-3                                                                           # N m/rad, from the drive model at 25 Hz
fl = []
for t_, b_ in ((0.1e-3, 2e-3), (0.15e-3, 2e-3), (0.2e-3, 2e-3)):
    l = E_CF * b_ * t_ ** 3 / (12 * kt)
    fl.append(dict(t_mm=t_ * 1e3, b_mm=b_ * 1e3, l_mm=l * 1e3, strain_pct=t_ * Phi / (2 * l) * 100, mass_mg=b_ * t_ * l * RHO_CF * 1e6))
out["spring_flexure"] = fl
hinge = [dict(t_um=t_ * 1e6, l_mm=l * 1e3, strain_pct=t_ * Phi / (2 * l) * 100, kt=2.5e9 * 3e-3 * t_ ** 3 / (12 * l)) for t_, l in ((12.5e-6, 0.3e-3), (12.5e-6, 0.5e-3), (25e-6, 0.5e-3), (25e-6, 1e-3))]
out["polyimide_hinge"] = hinge
cycles_hr = F * 3600
out["cycles"] = dict(per_hour=cycles_hr, hours_to_2M=2e6 / cycles_hr)
# TE anchor error: an anchor e off the stroke axis changes the fibre length by about e (1 - cos phi) + e_along...
off = []
for e in (0.5e-3, 1e-3, 2e-3, 5e-3):
    # anchor on the body, fibre tip on the wing at span B
    a = np.array([0.0, e])                                                            # anchor offset normal to the span at phi = 0
    Lf = lambda ph: np.hypot(B * np.cos(ph) - a[0], B * np.sin(ph) - a[1])
    d = Lf(Phi) - Lf(-Phi)
    off.append(dict(e_mm=e * 1e3, dL_mm=abs(d) * 1e3, strain_pct=abs(d) / B * 100, dF_mN=abs(d) / B * 6 * 42842.0 * 70e-6 * 0.01 * 1e3))
out["te_anchor_off_axis"] = off

# ---------- skin modes, 0 V and at voltage, no flow
modes = []
for tag, tipf in (("batten bonded", True), ("batten hinged", False)):
    for Ns, Nc, Ff in ((1.25, 0.05, 8e-3), (0.85, 0.03, 7e-3), (3.0, 0.5, 30e-3), (2.5, 0.4, 28e-3)):
        p = dict(skin.PARAMS, tip_fixed=tipf, ns=16, nc=8)
        S = skin.Structure(p)
        K = Nc * S.Kc + Ns * S.Ks + Ff * S.Kf + S.Kle
        ev = np.linalg.eigvals(np.linalg.solve(S.M, K)).real
        fr = np.sort(np.sqrt(np.maximum(ev, 0))) / (2 * np.pi)
        # added air mass per area on the skin, about rho c pi/8 averaged over the chord
        ma = 1.225 * C * np.pi / 8
        ms = 1070 * 70e-6
        modes.append(dict(tip=tag, Ns=Ns, Nc=Nc, F_mN=Ff * 1e3, f_Hz=fr[:4].tolist(), f_with_air_Hz=(fr[:4] * np.sqrt(ms / (ms + ma))).tolist()))
out["modes"] = modes

# ---------- electrics. Ren, Kim et al. Table 1: 6 layers, 35.5 um, 125 mg, 1.83 nF, 22.1 kOhm
A_chen6 = 125e-6 / 1070 / 223e-6
C_check = 5 * EPS * A_chen6 / 35.5e-6
out["chen6"] = dict(sheet_mm2=A_chen6 * 1e6, C_nF_model=C_check * 1e9, C_nF_paper=1.83, RC_us=22.1e3 * 1.83e-9 * 1e6, total_electrode_cm2=A_chen6 * 5e4)
# a core built from sheets 55 mm x 20 mm, H 35.5 um
rows = []
for m_core in (0.25e-3, 0.5e-3, 0.75e-3, 1e-3):
    V_ = m_core / 1070
    layers = V_ / (55e-3 * 20e-3 * 35.5e-6)
    Cc = (layers - 1) * EPS * 55e-3 * 20e-3 / 35.5e-6
    # R scales as 1/layers against the 6-layer device; 22.1 kOhm of the same sheet resistance, same aspect ratio assumed
    R = 22.1e3 * 6 / layers
    t_wall = layers * 35.5e-6
    # spiral: OD from area; core hollow id 1 mm
    od = np.sqrt((1e-3) ** 2 + 4 * V_ / (np.pi * 20e-3))
    rows.append(dict(m_mg=m_core * 1e6, layers=layers, C_nF=Cc * 1e9, R_kOhm=R / 1e3, RC_us=R * Cc * 1e6, od_mm=od * 1e3, electrode_cm2=(layers - 1) * 55e-1 * 20e-1 * 1e0 / 1e0, ratio_to_chen6=((layers - 1) * 11e-4) / (A_chen6 * 5)))
out["core_builds"] = rows
sk = dict(C_nF=EPS * B * C / 35e-6 * 1e9)
Rs = 22.1e3 * 3 * 2                                                                   # kOhm-ish: one pair of the 6-layer device, used as the sheet resistance scale
sk["R_kOhm_feed_along_span"] = Rs * (C / B) / 3 / 1e3
sk["RC_us"] = sk["R_kOhm_feed_along_span"] * 1e3 * sk["C_nF"] * 1e-9 * 1e6
te = dict(C_nF=EPS * B * 3e-3 / 35e-6 * 1e9, R_kOhm=Rs * (B / 3e-3) / 3 / 1e3)
te["RC_us"] = te["C_nF"] * 1e-9 * te["R_kOhm"] * 1e3 * 1e6
out["skin_electrode"], out["te_electrode"] = sk, te
# power per channel at 1.6 kV: charging energy each cycle if nothing is recovered
for name, Cn in (("core 500 mg + skin + fibre", rows[1]["C_nF"] + sk["C_nF"] + te["C_nF"]),):
    pw = []
    for f_ in (15, 25, 40):
        pw.append(dict(f=f_, P_no_recovery_mW=0.5 * Cn * 1e-9 * 1600 ** 2 * f_ * 1e3, I_mA=Cn * 1e-9 * 2 * np.pi * f_ * 800 * 1e3))
    out["drive_power"] = dict(C_nF=Cn, table=pw)
json.dump(out, open(os.path.join(HERE, "data", "structure_elec.json"), "w"), indent=1, default=float)
print("LE")
for r in le:
    print(" %-28s m %5.1f mg  flap %6.2f mm  chord %6.2f mm  f1 %5.0f Hz  GJ %.3f  strain %.2f%%" % (r["name"], r["mass_mg"], r["flap_mm"], r["chord_mm"], r["f1_Hz"], r["GJ_mN_m2"], r["root_strain_pct"]))
print("spring", fl)
print("hinge", hinge, out["cycles"])
print("te off", off)
for r in modes:
    print(r["tip"], r["Ns"], r["Nc"], r["F_mN"], np.round(r["f_Hz"][:3], 0), np.round(r["f_with_air_Hz"][:3], 0))
print(out["chen6"])
for r in rows:
    print(r)
print(sk, te, out["drive_power"])
