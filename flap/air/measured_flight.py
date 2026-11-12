"""The time model (flap_dyn.py) at the robot's measured kinematics: f = 8.4 Hz, stroke amplitude 12.2 deg (half the
24.3 deg peak-to-peak leading-edge angle, side view), against the design point 25 Hz, 20 deg. Passive wing (0 V) and the
edge held at full voltage. Run: python measured_flight.py -> data/dyn_measured.jsonl"""
import json, os, sys, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import flap_dyn as FD
def jobs():
    J = []
    for f, Phi in ((8.4, 12.2), (8.4, 20.0), (12.0, 20.0), (16.0, 20.0), (25.0, 20.0)):
        for U in (1.0, 2.0, 3.0, 4.0):
            for a in (6.0, 10.0):
                J.append(dict(design="passive", mode="off", f=f, Phi_deg=Phi, U=U, alpha_deg=a, **FD.DES["passive"]))
            J.append(dict(design="both", mode="on", f=f, Phi_deg=Phi, U=U, alpha_deg=10.0, tau=0.04, cycles=8, **FD.DES["both"]))
    return J
if __name__ == "__main__":
    J = jobs(); fn = os.path.join(HERE, "data", "dyn_measured.jsonl")
    with mp.Pool(12) as pool, open(fn, "w") as fh:
        for r in pool.imap_unordered(FD.run, J):
            r.pop("series", None); r.pop("snaps", None)
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r: print("ERR", r["f"], r["U"], r["error"][-200:], flush=True); continue
            print("%-8s f %4.1f Phi %4.1f U %.0f a %4.1f | L %.2f T %.2f P %.2f mW %s" % (r["design"], r["f"], r["Phi_deg"], r["U"], r["alpha_deg"], r["L_m"]*1e3, r["T_m"]*1e3, r["P_m"]*1e3, "UNSTABLE" if r["unstable"] else ""), flush=True)
