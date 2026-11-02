"""Lift over a flap cycle, quasi-steady: full membrane (membrane.py) against discrete DEAs in a passive membrane (../plagiopatagiales/model.py).
Same span, chord and area, stroke, pitch, voltage and eta. The older model approximates the planform by a taper of equal area.
Run: python flap_compare.py -> validation/flap_compare.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
import model as OLD
import design as D

OUT = os.path.join(HERE, "validation")
PHI0 = 30.0
PHASES = np.arange(0.0, 360.0, 30.0)
R_AS_TESTED = 0.02                                 # load-path factor of a free trailing edge (Eq. S10)
rep = dict(phases=PHASES.tolist(), Phi0_deg=PHI0, r_as_tested=R_AS_TESTED, cases={})


def rates(f, ph):
    w_, A = 2 * math.pi * f, math.radians(PHI0)
    return -A * w_ * math.sin(math.radians(ph)), -A * w_ * w_ * math.cos(math.radians(ph))


def full_cycle(U, f, V=1600.0, eta=0.4, passes=2):
    """Each state starts from the previous phase; the first pass only settles the path."""
    w = M.Wing(**D.base(U=U, eta_s=eta, max_iter=25000))     # states near the bifurcation may not settle: flagged in ok0, ok1
    r0 = w.relax(0.0)
    w.build_aero(r0["x"])
    S = w.vlm.area
    q = 0.5 * 1.225 * U ** 2
    out = dict(L0=[], L1=[], ok0=[], ok1=[], alpha_tip=[], S_cm2=S * 1e4)
    x0, x1 = r0["x"], None
    for p_ in range(passes):
        for ph in PHASES:
            Om, dOm = rates(f, ph)
            w.set_flap(Om, dOm)
            a0 = w.relax(0.0, x_init=x0, aero=True)
            x0 = a0["x"]
            a1 = w.relax(V, x_init=x0 if x1 is None else x1, aero=True)
            x1 = a1["x"]
            if p_ == passes - 1:
                out["L0"].append(float(a0["CL"] * q * S * 1e3)); out["L1"].append(float(a1["CL"] * q * S * 1e3))
                out["ok0"].append(bool(a0["ok"])); out["ok1"].append(bool(a1["ok"]))
                out["alpha_tip"].append(6.0 - math.degrees(math.atan(Om * (w.pts["E"][0] - w.pts["A"][0]) / U)))
    return out


def old_cycle(U, f, V=1600.0, eta=0.4, pre=1.03):
    c, b = 0.046, 0.1042
    S_target = 33.3e-4
    taper = 2 * S_target / (b * c) - 1
    par = dict(span=b, chord=c, taper=taper, le_mode="rigid", material="aniso", t_mem=180e-6, pre_span=pre, pre_chord=pre, n_dea_chord=8, n_dea_span=0, pre_dea=1.10,
               eta=eta, te_cord=True, U=U, alpha_deg=6.0, flap=True, flap_f=f, flap_amp_deg=PHI0, nx=20, ny=10, max_iter=40000)
    out = dict(L0=[], L1=[], ok0=[], ok1=[], S_cm2=S_target * 1e4, taper=taper)
    x0 = None
    for ph in PHASES:
        w = OLD.Wing(flap_phase_deg=float(ph), **par)
        xa, ia = w.solve(0.0, x_init=x0)
        x0 = xa
        xb, ib = w.solve(V, x_init=xa, ramp_steps=0)
        ra, rb = w.evaluate(xa, 0.0, ia), w.evaluate(xb, V, ib)
        out["L0"].append(ra["L_mN"]); out["L1"].append(rb["L_mN"]); out["ok0"].append(bool(ia["converged"])); out["ok1"].append(bool(ib["converged"]))
    return out


# reference: the flat wing in each aerodynamic model, same incidence
w = M.Wing(**D.base())
r = w.relax(0.0); w.build_aero(r["x"])
rep["CL_flat_vlm"] = float(w.vlm.CL(w.vlm.solve(4.0, math.radians(6.0))))
AR = 2 * 0.1042 ** 2 / 33.3e-4
rep["CL_flat_strip"] = 2 * math.pi * math.radians(6.0) * AR / (AR + 2)
rep["AR"] = AR
print(rep["CL_flat_vlm"], rep["CL_flat_strip"], AR, flush=True)

ONLY = sys.argv[1] if len(sys.argv) > 1 else ""          # one case, saved to its own file (figures.py merges them), so cases can run in parallel
NAME = "flap_compare" + (f"_{ONLY}" if ONLY else "")
for key, fn, kw in (("full_4_1", full_cycle, dict(U=4.0, f=1.0)), ("full_4_2", full_cycle, dict(U=4.0, f=2.0)), ("full_3_2", full_cycle, dict(U=3.0, f=2.0)), ("full_6_2", full_cycle, dict(U=6.0, f=2.0)),
                    ("full_4_2_eta24", full_cycle, dict(U=4.0, f=2.0, eta=0.24)), ("full_4_2_eta15", full_cycle, dict(U=4.0, f=2.0, eta=0.15)),
                    ("old_4_1", old_cycle, dict(U=4.0, f=1.0)), ("old_4_2", old_cycle, dict(U=4.0, f=2.0)), ("old_3_2", old_cycle, dict(U=3.0, f=2.0)), ("old_6_2", old_cycle, dict(U=6.0, f=2.0)),
                    ("old_4_2_pre101", old_cycle, dict(U=4.0, f=2.0, pre=1.01)), ("old_4_2_eta15", old_cycle, dict(U=4.0, f=2.0, eta=0.15))):
    if ONLY and key != ONLY:
        continue
    t = time.time()
    q = fn(**kw)
    q["params"] = kw
    rep["cases"][key] = q
    dl = np.array(q["L1"]) - np.array(q["L0"])
    print(key, "mean L0 %.1f mN, mean dL %.1f mN, range %.1f to %.1f, converged %d/%d" % (np.mean(q["L0"]), dl.mean(), dl.min(), dl.max(), sum(q["ok0"]) + sum(q["ok1"]), 2 * len(q["ok0"])), round(time.time() - t), flush=True)
    json.dump(rep, open(os.path.join(OUT, NAME + ".json"), "w"), indent=1, default=float)
