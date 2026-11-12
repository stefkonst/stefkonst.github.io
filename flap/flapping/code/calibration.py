"""What the robot's resonance and its quasi-static stroke say about the lever, the moving inertia and the cores.
Stroke about the hinge, cores pulling through a lever r:
    k_theta = 2 k_c r^2                                   (two cores, axial stiffness k_c each, Yeoh fit: 276 N/m)
    f_n = (1/2pi) sqrt(k_theta / I)                       -> I = k_theta / (2 pi f_n)^2
    theta_1 = (F_peak/2) r / k_theta * lag(1 Hz)          (fundamental of the sin^4 force at 1 Hz, quasi-static)
    F_peak = eta_c eps (V/H)^2 A_c                        (blocked force drop of a core; A_c = m_c/(rho L_c))
Measured: f_n = 8.38 Hz, theta_1(1 Hz, 1.2 kV) = 4.22 deg. For each lever r this gives the moving inertia I and the
core efficiency eta_c that the robot must have. The design (r = 2.5 mm, I = wing + air = 1.7e-7 kg m^2) predicted 22.6 Hz.
Run: python calibration.py -> ../data/calibration.json"""
import os, sys, json, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, os.path.join(HERE, "..", "..", "robot"))
import params as P
SW = json.load(open(os.path.join(DATA, "sweepfit.json"))); fn = SW["fits"]["zeta 0.07 (decay)"]["fn"]
R = {r["run"]: r for r in SW["runs"]}; th1 = math.radians(R["full_DEA_core_wing_1hz_1200v_side"]["A"][0])
kc = 275.6; Lc = 20e-3; mc = P.CORE_MASS; Ac = mc / (P.RHO_F * Lc); V = 1200.0
I_wing = 1.6e-7 * 1.05; tau = 0.052; lag1 = 1 / math.sqrt(1 + (2 * math.pi * 1.0 * tau) ** 2)
rows = []
for H in (30e-6, 45e-6):
    sig = P.EPS * (V / H) ** 2
    for r in np.round(np.arange(0.5, 4.01, 0.25) * 1e-3, 6):
        kth = 2 * kc * r * r; I = kth / (2 * math.pi * fn) ** 2
        Fpk = 2 * th1 * kth / (r * lag1); eta = Fpk / (sig * Ac)
        rows.append(dict(H_um=H * 1e6, r_mm=r * 1e3, k_theta=kth, I=I, I_over_wing=I / I_wing, F_peak_mN=Fpk * 1e3, eta_core=eta))
out = dict(f_n=fn, theta1_deg=math.degrees(th1), k_c=kc, A_c_mm2=Ac * 1e6, rows=rows,
           design=dict(r_mm=2.5, I=I_wing, f_pred=math.sqrt(2 * kc * 2.5e-3 ** 2 / I_wing) / 2 / math.pi))
json.dump(out, open(os.path.join(DATA, "calibration.json"), "w"), indent=1)
print("core area %.1f mm2; design predicts %.1f Hz" % (Ac * 1e6, out["design"]["f_pred"]))
for q in rows:
    if q["H_um"] == 30 and q["r_mm"] in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        print("r %.1f mm: inertia %.2e kg m2 (%.1f x wing), core force drop %.0f mN, eta_core %.3f (H 30 um)" % (q["r_mm"], q["I"], q["I_over_wing"], q["F_peak_mN"], q["eta_core"]))
