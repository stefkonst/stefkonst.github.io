"""Questions for the first build, steady finite elements at H = 30 um, 1370 V, 4 m/s. Uses wing_shape.run.
  a  triangle or taper, with and without a rib
  b  stretch: chord stretch at the fibre, fibre stretch from the rotation, and pre-stretch along the span before bonding
Run: python build_study.py [workers] -> data/build_study.jsonl, safe to stop and rerun."""
import json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wing_shape as W
OUT = os.path.join(HERE, "data", "build_study.jsonl")
V = dict(V=1370.0)
RIB = W.RIB
T = dict(cr=0.035, ct=0.015, stretch=2.6e-3)
CASES = {
    "rectangle": dict(),
    "rectangle, rib": dict(veins=(RIB(0.5),)),
    "taper 35-15": dict(T),
    "taper 35-15, rib": dict(T, veins=(RIB(0.5),)),
    "taper 40-10": dict(cr=0.040, ct=0.010, stretch=3.0e-3),
    "taper 40-10, rib": dict(cr=0.040, ct=0.010, stretch=3.0e-3, veins=(RIB(0.5),)),
    "taper 45-5": dict(cr=0.045, ct=0.005, stretch=3.0e-3),
    "taper 45-5, rib": dict(cr=0.045, ct=0.005, stretch=3.0e-3, veins=(RIB(0.5),)),
    "triangle 48-2": dict(cr=0.048, ct=0.002, stretch=3.0e-3),
    "triangle 48-2, rib": dict(cr=0.048, ct=0.002, stretch=3.0e-3, veins=(RIB(0.5),)),
    "triangle 48-2, more stretch": dict(cr=0.048, ct=0.002, stretch=4.5e-3),
}
for ln in (1.00, 1.03, 1.06, 1.10):
    for st in (1.3e-3, 2.6e-3, 3.9e-3):
        CASES[f"taper 35-15, chord {ln:.2f}, fibre {st * 1e3:.1f} mm"] = dict(T, lam_n=ln, stretch=st)
for ld in (1.02, 1.04, 1.06):
    CASES[f"taper 35-15, no rotation, span pre-stretch {ld:.2f}"] = dict(cr=0.035, ct=0.015, theta=0.0, stretch=0.0, lam_d=ld)
    CASES[f"rectangle, no rotation, span pre-stretch {ld:.2f}"] = dict(theta=0.0, stretch=0.0, lam_d=ld)


# lighter trailing edge: narrower roll, with or without an elastic or stiff cord in it, taper 35-15
for ws in (0.010, 0.015, 0.020):
    CASES[f"taper 35-15, roll {ws * 1e3:.0f} mm"] = dict(T, ws=ws, we=0.005)
CASES["taper 35-15, roll 10 mm, elastic cord 0.4 N"] = dict(T, ws=0.010, we=0.005, cord_EA=0.4, cord_pre=0.0)
CASES["taper 35-15, roll 10 mm, elastic cord 0.8 N"] = dict(T, ws=0.010, we=0.005, cord_EA=0.8, cord_pre=0.0)
CASES["taper 35-15, roll 10 mm, stiff cord 5 N, 2 % slack"] = dict(T, ws=0.010, we=0.005, cord_EA=5.0, cord_pre=-0.02)
CASES["rectangle, roll 10 mm"] = dict(ws=0.010, we=0.005)
CASES["rectangle, roll 15 mm"] = dict(ws=0.015, we=0.005)


# thickness at a fixed span stretch, same field. Straight cut, rectangle or taper
PS = dict(theta=0.0, stretch=0.0, lam_d=1.04)
TP = dict(PS, cr=0.035, ct=0.015)
for nm, q in (("rectangle", PS), ("taper", TP)):
    for N in (1, 2, 3, 4):
        CASES[f"thick: {nm}, {N} layers, 30 um"] = dict(q, N=N)
    CASES[f"thick: {nm}, 2 layers, 45 um, 2055 V"] = dict(q, H=45e-6, V=2055.0)
    CASES[f"thick: {nm}, 2 layers, 60 um, 2740 V"] = dict(q, H=60e-6, V=2740.0)
    CASES[f"thick: {nm}, 2 layers, 30 um, 30 um dead"] = dict(q, t_extra=30e-6)
    CASES[f"thick: {nm}, 2 layers, 30 um, 60 um dead"] = dict(q, t_extra=60e-6)
    CASES[f"thick: {nm}, 3 layers, 45 um, 2055 V"] = dict(q, N=3, H=45e-6, V=2055.0)
    # small biaxial stretch, as from bonding the frame hot
    for b in (1.01, 1.02, 1.03):
        CASES[f"hot bond: {nm}, biaxial {b:.2f}"] = dict(q, lam_d=b, lam_n=b)
    for ws in (0.010, 0.015):
        CASES[f"pre-stretch: {nm}, roll {ws * 1e3:.0f} mm"] = dict(q, ws=ws)
    CASES[f"pre-stretch: {nm}, rib"] = dict(q, veins=(RIB(0.5),))
    CASES[f"pre-stretch: {nm}, two ribs"] = dict(q, veins=(RIB(1 / 3), RIB(2 / 3)))
    for ld in (1.03, 1.05):
        CASES[f"pre-stretch: {nm}, span {ld:.2f}, chord 1.03"] = dict(q, lam_d=ld, lam_n=1.03)


def cases():
    return [dict(dict(V, name=n, alpha=a), **v) for n, v in CASES.items() for a in (6.0, 10.0)]


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    done = {(d["name"], d["alpha"]) for d in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    todo = [c for c in cases() if (c["name"], c["alpha"]) not in done]
    print(len(todo), "cases", flush=True)
    with mp.Pool(workers) as pool, open(OUT, "a") as fh:
        for r in pool.imap_unordered(W.run, todo):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r["name"], r["alpha"], r.get("ok0"), r.get("ok1"), r.get("error"), "wall %.0f" % r["wall"], flush=True)
    print("DONE", flush=True)
