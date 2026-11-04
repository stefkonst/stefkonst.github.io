"""Transmission for the first build: core size from the sheet, lever, spring. Uses drive.simulate. Layout B: core tension pulls the wing down, spring lifts it.
Sheet: stack of 5 electrodes at H = 30 um (half D, 4 D, half D = 5 H of film), 55 x 20 mm, rolled along the 55 mm side. Run: python transmission.py -> data/transmission.json"""
import json, os
import numpy as np
import drive
HERE = os.path.dirname(os.path.abspath(__file__))
RHO, H = 1070.0, 30e-6
E_CF = 130e9
L0 = 0.020
m_wing, m_rib = 270e-6, 0.0
I_w = (m_wing + 30e-6) * 0.05 ** 2 / 3     # as in budget.py
Om = lambda Phi, f: np.radians(Phi) * 2 * np.pi * f
out = dict(I_w=I_w, sheets=[], runs=[], flexures=[])


def c_a_for(P, Phi, f):
    """Quadratic aero damping from the coupled model's air power."""
    return P / (np.mean(np.abs(np.cos(np.linspace(0, 2 * np.pi, 1001))) ** 3) * Om(Phi, f) ** 3)


for n_el, n_strips in ((5, 1), (5, 2), (5, 3), (5, 4), (7, 1), (7, 2)):
    t = (n_el) * H
    V = 55e-3 * 20e-3 * t * n_strips
    m = V * RHO
    od = np.sqrt(1e-6 + 4 * V / (np.pi * 20e-3))
    out["sheets"].append(dict(n_el=n_el, strips=n_strips, t_um=t * 1e6, m_mg=m * 1e6, od_mm=od * 1e3, area_mm2=V / 20e-3 * 1e6, active=(n_el - 1) * n_strips, turns=np.nan))
    print("electrodes %d strips %d: film %3.0f um, core %4.0f mg, OD %.1f mm" % (n_el, n_strips, t * 1e6, m * 1e6, od * 1e3))
# air power: one stiff rib, 5 x 2.5 cm, coupled_r
AERO = {(25.0, 20.0): 16.7e-3, (25.0, 30.0): 35.8e-3, (15.0, 20.0): 5.0e-3, (15.0, 30.0): 13.0e-3, (20.0, 20.0): 10.0e-3}
for m in (177e-6, 354e-6, 531e-6, 708e-6):
    for f, Phi, sig, nm in ((25.0, 20.0, drive.SIG_ISO, "iso"), (25.0, 20.0, drive.SIG_CYC, "cyc"), (15.0, 20.0, drive.SIG_ISO, "iso"), (15.0, 20.0, drive.SIG_CYC, "cyc")):
        for r_mm in (2.0, 3.0, 4.0):
            ca = c_a_for(AERO[(f, 20.0)], 20.0, f)
            a = drive.simulate(m_core=m, L0=L0, lam0=1.10, r=r_mm * 1e-3, I=I_w, f=f, sig=sig, c_a=ca, layout="B", tau_mean=-2e-3 * 0.0)
            row = dict(m_mg=m * 1e6, f=f, r_mm=r_mm, force=nm, **{k: a[k] for k in ("Phi_deg", "strain_amp", "F_min_mN", "F_max_mN", "slack", "kt_core", "kt_s", "spring_share", "F_block_mN", "P_aero_mW", "P_h_mW", "tau_s0_mNm", "A_mm2")})
            out["runs"].append(row)
            if nm == "iso" or r_mm == 3.0:
                print("%4.0f mg %2.0f Hz r %.0f %s: +-%4.1f deg strain %.3f F %4.0f-%4.0f mN slack %s kt_core %.2e kt_s %.2e share %.2f block %.0f Paero %.1f Ph %.1f tau0 %.2f" % (m * 1e6, f, r_mm, nm, a["Phi_deg"], a["strain_amp"], a["F_min_mN"], a["F_max_mN"], a["slack"], a["kt_core"], a["kt_s"], a["spring_share"], a["F_block_mN"], a["P_aero_mW"], a["P_h_mW"], a["tau_s0_mNm"]))
# flexures: k = E b t^3 / (12 l); strain = t theta / (2 l). Total bend = preload angle + stroke
for kt in (1e-3, 2e-3, 3e-3, 5e-3):
    for t, b in ((0.1e-3, 2e-3), (0.1e-3, 4e-3), (0.2e-3, 1.5e-3)):
        l = E_CF * b * t ** 3 / (12 * kt)
        out["flexures"].append(dict(kt=kt, t_mm=t * 1e3, b_mm=b * 1e3, l_mm=l * 1e3, strain_per_rad=t / (2 * l), mass_mg=l * b * t * 1550 * 1e6))
        print("kt %.0e: t %.2f b %.1f l %5.1f mm, strain at 1 rad %.2f %%" % (kt, t * 1e3, b * 1e3, l * 1e3, t / (2 * l) * 100))
print("I_w", I_w, "I w^2 at 25 Hz", I_w * (2 * np.pi * 25) ** 2, "15 Hz", I_w * (2 * np.pi * 15) ** 2)
json.dump(out, open(os.path.join(HERE, "data", "transmission.json"), "w"), indent=1, default=float)
