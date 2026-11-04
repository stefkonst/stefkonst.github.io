"""Prints the heavy trailing edge table from data/coupled_t.jsonl."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = [r for r in map(json.loads, open(os.path.join(HERE, "data", "coupled_t.jsonl"))) if "error" not in r]
for f, P, a in ((25.0, 30.0, 8.0), (25.0, 20.0, 12.0), (20.0, 30.0, 8.0), (15.0, 30.0, 12.0)):
    print("f", f, "Phi", P, "a", a)
    for w in ("plain", "one stiff rib"):
        for z in (0.05, 0.015):
            for tm in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0):
                d = {r["mode"]: r for r in R if r["wing"] == w and r["te_mass"] == tm and r["zeta"] == z and (r["f"], r["Phi_deg"], r["alpha_deg"]) == (f, P, a)}
                o = d["off"]
                print("  %-14s z %.3f te x%.2f m %3.0f: L %5.2f T %5.2f Tns %5.2f P %5.1f tip %.2f mid %.2f peak %.1f per %.3f | up dL %+.2f P %5.1f mid %.2f unst %s | down dL %+.2f P %5.1f mid %.2f unst %s" % (w, z, tm, o["mass_mg"], o["L"] * 1e3, o["T"] * 1e3, o["T_ns"] * 1e3, o["P"] * 1e3, o["w_tip_amp"] * 1e3, o["w_mid_amp"] * 1e3, o["wmax_peak"] * 1e3, o["periodic"], (d["up"]["L"] - o["L"]) * 1e3, d["up"]["P"] * 1e3, d["up"]["w_mid_amp"] * 1e3, d["up"]["unstable"], (d["down"]["L"] - o["L"]) * 1e3, d["down"]["P"] * 1e3, d["down"]["w_mid_amp"] * 1e3, d["down"]["unstable"]))
