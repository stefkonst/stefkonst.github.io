"""Why the lift-change peak near phase 270 deg (full membrane, 4 m/s, 2 Hz)? Sweep the phases up and down through it, at 0 V and 1.6 kV separately,
and record lift and the deflection of panel 2 near the tip. A difference between the two sweep directions means the state depends on its history.
Run: python flap_peak.py -> validation/flap_peak.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
import design as D

U, f = 4.0, 2.0
A, w_ = math.radians(30.0), 2 * math.pi * f
PH = [225, 250, 270, 290, 315]
rate = lambda p: (-A * w_ * math.sin(math.radians(p)), -A * w_ * w_ * math.cos(math.radians(p)))
w = M.Wing(**D.base(U=U, max_iter=10000))
r0 = w.relax(0.0)
w.build_aero(r0["x"])
S, q = w.vlm.area, 0.5 * 1.225 * U ** 2
g2 = w.ids[1]
probe_tip = g2[int(0.7 * g2.shape[0]), g2.shape[1] // 2]      # mid-chord, 70 % along panel 2
probe_in = g2[int(0.25 * g2.shape[0]), g2.shape[1] // 2]
rows = []
for V in (0.0, 1600.0):
    w.set_flap(*rate(180.0))
    x = w.relax(V, x_init=r0["x"], aero=True)["x"]
    for label, seq in (("forward", PH), ("backward", PH[::-1])):
        for ph in seq:
            Om, dOm = rate(ph)
            w.set_flap(Om, dOm)
            a = w.relax(V, x_init=x, aero=True)
            x = a["x"]
            s = w.summary(x, V)
            rows.append(dict(V=V, sweep=label, phase=ph, L_mN=float(a["CL"] * q * S * 1e3), ok=bool(a["ok"]), w_outer_mm=float(x[probe_tip, 2] * 1e3), w_inner_mm=float(x[probe_in, 2] * 1e3), wmax2_mm=float(s[1]["wmax"] * 1e3),
                             alpha_tip=6.0 - math.degrees(math.atan(Om * 0.1042 / U)), r_zero_mm=float(U * math.tan(math.radians(6.0)) / abs(Om) * 1e3) if Om > 0 else None))
            print(V, label, ph, round(rows[-1]["L_mN"], 2), a["ok"], round(rows[-1]["w_outer_mm"], 2), round(rows[-1]["w_inner_mm"], 2), flush=True)
            json.dump(rows, open(os.path.join(HERE, "validation", "flap_peak.json"), "w"), indent=0)
