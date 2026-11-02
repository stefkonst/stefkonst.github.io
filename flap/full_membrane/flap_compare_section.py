"""Discrete DEAs over the flap cycle: verified section model (../plagiopatagiales/section.py, Eqs. S1-S9), strip by strip on the planform of the
full membrane wing, same stroke as flap_compare.py. The older 3D FE does not converge in this flapping case (residual 37 x reference after 40 000 steps).
Corrections so both approaches share a lift reference: 2D loading x CL_flat(vortex lattice) / 2 pi alpha; force change x 1/1.29 (discrete vs smeared DEAs, S_verification).
Framed section, so a best case (r = 1); the free trailing edge has r = 0.02 (Eq. S10). Run: python flap_compare_section.py -> validation/flap_compare_section.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
import section as SEC
import design as D

OUT = os.path.join(HERE, "validation")
PHI0 = math.radians(30.0)
PHASES = np.arange(0.0, 360.0, 30.0)
DISCRETE = 1 / 1.29
R_FREE = 0.02

w = M.Wing(**D.base())
r = w.relax(0.0)
w.build_aero(r["x"])
v = w.vlm
xs = v.xs
chord = np.array([np.sum(v.dyc.reshape(v.nx, v.ny)[i]) for i in range(v.nx)])
xc = 0.5 * (xs[:-1] + xs[1:]) - xs[0]
dx = np.diff(xs)
K3D = v.CL(v.solve(4.0, math.radians(6.0))) / (2 * math.pi * math.radians(6.0))


def cycle(U, f, dF, pre=1.03, spacing=0.012, pre_dea=1.10, V=1600.0):
    L0, L1, nogap = [], [], []
    for ph in PHASES:
        Om = -PHI0 * 2 * math.pi * f * math.sin(math.radians(ph))
        a, b, missing = 0.0, 0.0, 0
        for x_, c_, d_ in zip(xc, chord, dx):
            Ue = math.hypot(U, Om * x_)
            ae = 6.0 - math.degrees(math.atan(Om * x_ / U))
            p = dict(U=Ue, alpha_deg=ae, c=float(c_), pre_chord=pre, pre_span=pre, spacing=spacing, pre_dea=pre_dea, dF1600=dF)
            s0, s1 = SEC.solve(p, 0.0), SEC.solve(p, V)
            if not s0 or not s1:
                missing += 1
                continue
            q = 0.5 * 1.225 * Ue ** 2
            a += q * c_ * d_ * s0["CL"] * K3D
            b += q * c_ * d_ * (s0["CL"] + (s1["CL"] - s0["CL"]) * DISCRETE) * K3D
        L0.append(a * 1e3); L1.append(b * 1e3); nogap.append(missing == 0)
    return dict(L0=L0, L1=L1, ok0=nogap, ok1=nogap, params=dict(U=U, f=f, dF1600_mN=dF * 1e3, pre=pre, spacing=spacing, pre_dea=pre_dea))


ISO, CYC = SEC.DF1600_ISO, SEC.DF1600_CYC
rep = dict(K3D=float(K3D), discrete_factor=DISCRETE, r_free_te=R_FREE, phases=PHASES.tolist(), cases={})
for key, kw in (("old_4_1", dict(U=4.0, f=1.0, dF=ISO)), ("old_4_2", dict(U=4.0, f=2.0, dF=ISO)), ("old_3_2", dict(U=3.0, f=2.0, dF=ISO)), ("old_6_2", dict(U=6.0, f=2.0, dF=ISO)),
                ("old_4_2_cyc", dict(U=4.0, f=2.0, dF=CYC)), ("old_4_2_pre101", dict(U=4.0, f=2.0, dF=ISO, pre=1.01)), ("old_4_2_pre100", dict(U=4.0, f=2.0, dF=ISO, pre=1.00)),
                ("old_3_2_pre101", dict(U=3.0, f=2.0, dF=ISO, pre=1.01))):
    q = cycle(**kw)
    rep["cases"][key] = q
    dl = np.array(q["L1"]) - np.array(q["L0"])
    print(key, "mean L0 %.1f, mean dL %.2f mN, range %.2f to %.2f, phases complete %d/%d" % (np.mean(q["L0"]), np.mean(dl), dl.min(), dl.max(), sum(q["ok0"]), len(q["ok0"])), flush=True)
json.dump(rep, open(os.path.join(OUT, "flap_compare_section.json"), "w"), indent=1, default=float)
