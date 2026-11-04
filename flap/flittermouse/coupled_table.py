"""Tables from coupled_study.py, and lift over weight on the coupled model's own air power. Run: python coupled_table.py -> data/coupled_summary.json"""
import json, os
import statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
load = lambda t: [r for r in (json.loads(l) for l in open(os.path.join(HERE, "data", f"coupled_{t}.jsonl"))) if "error" not in r and not r.get("unstable")]
ETA_TR, M_FRAME = 0.8, 0.2e-3
KINKEY = lambda r: (r.get("U", 4.0), r["f"], r["Phi_deg"], r["alpha_deg"])
out = {}


def lw(r, e, s=1.0, extra=0.0):
    T = s * r["T"] + (1 - s) * r["T_ns"]
    if r["P"] <= 0:
        return None, T
    m = 2 * (r["P"] / (e * r["f"] * ETA_TR) + r["mass_mg"] * 1e-6 + extra) + M_FRAME
    return 2 * r["L"] / (m * 9.81), T


def best(rows, e, s=1.0, extra=0.0):
    v = [(lw(r, e, s, extra)[0], r) for r in rows if lw(r, e, s, extra)[1] >= 0 and lw(r, e, s, extra)[0]]
    return max(v, key=lambda x: x[0]) if v else (None, None)


# reinforcement
R = load("r")
print("== reinforcement, voltage off: L T T_ns P, median over 6 flight points as % of plain, then best L/W at 0.8 / 1.25 / 2.1 J/kg, full suction, then half suction")
names = []
for r in R:
    if r["wing"] not in names:
        names.append(r["wing"])
plain = {KINKEY(r): r for r in R if r["wing"] == "plain" and r["mode"] == "off"}
rows = []
for n in names:
    S = [r for r in R if r["wing"] == n and r["mode"] == "off"]
    pc = lambda q: st.median(r[q] / plain[KINKEY(r)][q] * 100 for r in S)
    S0 = [r for r in R if r["wing"] == n and r["mode"] in ("off", "down", "up")]
    b = [best(S0, e)[0] for e in (0.8, 1.25, 2.1)]
    bh = [best(S0, e, 0.5)[0] for e in (0.8, 1.25, 2.1)]
    r8 = next(r for r in S if KINKEY(r) == (4.0, 25.0, 30.0, 8.0))
    gain = [(g["L"] - o["L"]) * 1e3 for g in R if g["wing"] == n and g["mode"] == "down" for o in S if KINKEY(o) == KINKEY(g)]
    gainu = [(g["L"] - o["L"]) * 1e3 for g in R if g["wing"] == n and g["mode"] == "up" for o in S if KINKEY(o) == KINKEY(g)]
    rows.append(dict(wing=n, L=pc("L"), T=pc("T"), P=pc("P"), L8=r8["L"] * 1e3, T8=r8["T"] * 1e3, P8=r8["P"] * 1e3, zpk=r8["wmax_peak"] * 1e3, mass=r8["mass_mg"], best=b, best_half=bh, gain_down=st.median(gain), gain_up=st.median(gainu)))
    x = rows[-1]
    print("%-20s L %3.0f%% T %3.0f%% P %3.0f%% | at +-30: L %.2f T %.2f P %.1f bulge %.1f mm m %.0f | L/W %s half %s | dL down %+.2f up %+.2f" % (n, x["L"], x["T"], x["P"], x["L8"], x["T8"], x["P8"], x["zpk"], x["mass"], " ".join("-" if q is None else "%.2f" % q for q in b), " ".join("-" if q is None else "%.2f" % q for q in bh), x["gain_down"], x["gain_up"]))
out["reinforcement"] = rows

# timing flipped
if os.path.exists(os.path.join(HERE, "data", "coupled_f.jsonl")):
    F = load("f")
    print("== timing: gain against 0 V at the same cap (cap unchanged by voltage for the off case), mN, median [min, max] over 6 flight points")
    off = {(KINKEY(r), r.get("cn_cap")): r for r in F if r["mode"] == "off"}
    rows = []
    for tau in (0.0, 0.010):
        for cap in (None, 1.2, 1.6):
            for d in ((0.0,) if cap is None else (-0.4, -0.2, 0.0, 0.2, 0.4)):
                row = dict(tau=tau, cap=cap, d=d)
                for m in ("down", "up"):
                    S = [r for r in F if r["mode"] == m and r.get("tau", 0) == tau and r.get("cn_cap") == cap and r.get("cn_cap_soft", 0.0) == d]
                    g = [(r["L"] - off[(KINKEY(r), cap)]["L"]) * 1e3 for r in S if (KINKEY(r), cap) in off]
                    pw = [r["P"] / off[(KINKEY(r), cap)]["P"] * 100 for r in S if (KINKEY(r), cap) in off]
                    th = [(r["T"] - off[(KINKEY(r), cap)]["T"]) * 1e3 for r in S if (KINKEY(r), cap) in off]
                    row[m] = dict(med=st.median(g), lo=min(g), hi=max(g), P=st.median(pw), T=st.median(th)) if g else None
                rows.append(row)
                if row["down"] and row["up"]:
                    print("tau %2.0f ms cap %-4s d %+.1f: soft down %+.2f [%+.2f %+.2f] P %3.0f%% dT %+.2f | stiff down %+.2f [%+.2f %+.2f] P %3.0f%% dT %+.2f" % (tau * 1e3, cap, d, row["down"]["med"], row["down"]["lo"], row["down"]["hi"], row["down"]["P"], row["down"]["T"], row["up"]["med"], row["up"]["lo"], row["up"]["hi"], row["up"]["P"], row["up"]["T"]))
    out["timing"] = rows

# bat kinematics with the skin
if os.path.exists(os.path.join(HERE, "data", "coupled_b.jsonl")):
    B = load("b")
    print("== bat kinematics, skin compliant, ribs none. L T P as % of no fold, no tilt, voltage off. Best L/W, fold adds 30 mg per wing")
    rows = []
    for b in (0.05, 0.07):
        base = {KINKEY(r): r for r in B if r["b"] == b and r["fold"] == 0 and r["tilt_deg"] == 0 and r["mode"] == "off"}
        for fo in (0.0, 0.2, 0.4):
            for ti in (0.0, 20.0):
                for m in ("off", "up", "down"):
                    S = [r for r in B if r["b"] == b and r["fold"] == fo and r["tilt_deg"] == ti and r["mode"] == m]
                    pc = lambda q: st.median(r[q] / base[KINKEY(r)][q] * 100 for r in S)
                    ex = 30e-6 if fo else 0.0
                    bb = [best(S, e, 1.0, ex)[0] for e in (0.8, 1.25, 2.1)]
                    bh = [best(S, e, 0.5, ex)[0] for e in (0.8, 1.25, 2.1)]
                    rows.append(dict(b=b, fold=fo, tilt=ti, mode=m, L=pc("L"), T=pc("T"), P=pc("P"), best=bb, best_half=bh))
                    print("b %.2f fold %.1f tilt %2.0f %-4s: L %3.0f%% T %3.0f%% P %3.0f%% | L/W %s half %s" % (b, fo, ti, m, pc("L"), pc("T"), pc("P"), " ".join("-" if q is None else "%.2f" % q for q in bb), " ".join("-" if q is None else "%.2f" % q for q in bh)))
    out["bat"] = rows
json.dump(out, open(os.path.join(HERE, "data", "coupled_summary.json"), "w"), indent=1)
