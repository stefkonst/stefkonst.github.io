"""What B-cube needs to make thrust: stroke reachable at resonance against the stroke that thrust needs.
Reachable stroke amplitude at a new resonance f_r, from the measured one (12.2 deg at 8.4 Hz, 1.2 kV, lag tau0 = 52 ms),
same damping ratio, same cores and voltage:
  route 1, stiffer spring (same inertia):   Phi = Phi0 (f0/f_r)^2 * lag(f_r, tau) / lag(f0, tau0)
  route 2, less moving inertia (same k):    Phi = Phi0           * lag(f_r, tau) / lag(f0, tau0)
  lag(f, tau) = 1 / sqrt(1 + (2 pi f tau)^2)        (first-order lag on the force; tau = 52 ms measured)
Needed: Strouhal f A_tip / U >= 0.15 with A_tip = 2 R sin(Phi), R = 55 mm (the flight model's thrust threshold).
Run: python design_map.py -> ../data/design_map.json"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
f0, Phi0, tau0, R = 8.4, 12.2, 0.052, 0.055
lag = lambda f, t: 1 / np.sqrt(1 + (2 * np.pi * f * t) ** 2)
fr = np.linspace(6, 30, 121)
out = dict(f=fr.tolist(), routes={}, need={})
for tau in (0.052, 0.020, 0.010, 0.005, 0.002):
    out["routes"]["%g" % tau] = dict(stiffer=(Phi0 * (f0 / fr) ** 2 * lag(fr, tau) / lag(f0, tau0)).tolist(), lighter=(Phi0 * lag(fr, tau) / lag(f0, tau0)).tolist())
for U in (2.0, 3.0, 4.0):
    out["need"]["%g" % U] = np.degrees(np.arcsin(np.clip(0.15 * U / (2 * R * fr), 0, 1))).tolist()
# what the inertia calibration says: at the design lever the moving inertia is 7.4 x the wing's; removing the excess
f_wing = f0 * np.sqrt(7.4)
for tau in (0.052, 0.010, 0.005):
    print("tau %2.0f ms: wing-only inertia -> f %.1f Hz, stroke amplitude %.1f deg; need at 3 m/s %.1f deg" % (
        tau * 1e3, f_wing, Phi0 * lag(f_wing, tau) / lag(f0, tau0), np.degrees(np.arcsin(0.15 * 3 / (2 * R * f_wing)))))
out["f_wing_only"] = f_wing
json.dump(out, open(os.path.join(DATA, "design_map.json"), "w"), indent=1)
