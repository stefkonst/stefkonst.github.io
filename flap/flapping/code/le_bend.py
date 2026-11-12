"""Leading-edge bending over the stroke: in each frame, fit a parabola to the tracked leading edge and take its sagitta
(mid-span deviation from the chord line between the ends of the tracked span), in mm. Phase-averaged over driven cycles.
Compares the powered and unpowered wing at 8 Hz, 1200 V (side view; scales 0.152 and 0.135 mm/px)."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); TR = os.path.join(DATA, "tracks")
fs = 200.0
RUNS = {"full_DEA_core_wing_8hz_1200V_side": 0.152, "only_DEA_core_8hz_1200V_side": 0.135, "full_DEA_core_wing_8hz_1400V_side": 0.152,
        "full_DEA_core_wing_7hz_1200V_side": 0.152, "full_DEA_core_wing_9hz_1200V_side": 0.152, "full_DEA_core_wing_5hz_1200v_side": 0.152}
def bend(n, mm):
    d = np.load(os.path.join(TR, n + ".npz")); xs, top, ang = d["xs"].astype(float), d["top"], d["ang"]
    sag = np.full(len(top), np.nan); span = np.nan
    for k in range(len(top)):
        m = np.isfinite(top[k])
        if m.sum() < 15: continue
        x, y = xs[m], top[k][m].astype(float)
        c = np.polyfit(x, y, 2); L = x[-1] - x[0]; span = L * mm
        sag[k] = -c[0] * L * L / 4 * mm          # positive: bowed upward in the image (toward smaller y)
    return sag, ang, span
out = {}
for n, mm in RUNS.items():
    sag, ang, span = bend(n, mm); t = np.arange(len(sag)) / fs
    rest = np.nanmedian(sag[:50]); g = t > 0.6
    if "1400V" in n: g &= t < 2.15
    s = sag[g] - rest; a = ang[g] - np.nanmean(ang[g])
    c = np.polyfit(a[np.isfinite(s)], s[np.isfinite(s)], 1)
    out[n] = dict(span_mm=float(span), rest_sag_mm=float(rest), sag_pp_mm=float(np.nanpercentile(s, 98) - np.nanpercentile(s, 2)),
                  sag_per_deg_um=float(c[0] * 1e3), sag_rms_um=float(np.nanstd(s) * 1e3))
    print("%-38s span %.1f mm  rest sag %+.2f mm  sag pk-pk %.2f mm  slope %+.1f um/deg" % (n, span, rest, out[n]["sag_pp_mm"], c[0] * 1e3))
json.dump(out, open(os.path.join(DATA, "le_bend.json"), "w"), indent=1)
