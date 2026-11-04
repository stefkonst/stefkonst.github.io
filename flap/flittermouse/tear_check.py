"""Where is the skin most stretched, and how does that compare with tearing? No root bar, roll end on a fixed thread. 2 and 4 layers, 30 um, 1370 V, 4 m/s.
Tear estimate for a cut of length a at a free edge: G = 2 k W a, k = pi / sqrt(lambda) (Rivlin and Thomas). Run: python tear_check.py -> data/tear_check.json"""
import json, math, multiprocessing as mp, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wing_shape as W
import curved_tip as CT
M = W.M
Q = 0.5 * 1.225 * 16 * 1e3


def peak(w, x, V):
    Pe = w.P_e(V)
    _, d = w.membrane(x, Pe, want=True)
    tr, dd = 0.5 * (d["c00"] + d["c11"]), np.sqrt(0.25 * (d["c00"] - d["c11"]) ** 2 + d["c01"] ** 2)
    l1 = np.sqrt(tr + dd)
    I = l1 ** 2 + 2 / l1                                    # uniaxial, the wrinkled state
    W_ = M.YEOH[0] * (I - 3) + M.YEOH[1] * (I - 3) ** 2
    k = int(np.argmax(l1))
    c = w.x0[w.tri[k]].mean(0) * 1e3
    bl = w.bars(x, V)[2]
    return dict(lam=float(l1[k]), at_mm=[float(c[0] - W.AX * 1e3), float(c[1] - W.AY * 1e3)], W_Jm3=float(W_[k]), lam95=float(np.percentile(l1, 95)), lam_med=float(np.median(l1)), fibre_lam=float(bl.max()),
                sigma_kPa=float(2 * (l1[k] - l1[k] ** -2) * M.W1_of(I[k]) * 1e-3))


def run(job):
    k = dict(W.BASE)
    k.update({q: v for q, v in job.items() if q in k or q in ("root_free", "tensioners")})
    if "curve" in job:
        CT.CUR.update(job["curve"])
        M.coons, M.VLM = CT.coons2, CT.VLM2
    w, x, _ = W.build(k)
    fx = w.fixed.copy()
    r = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)
    xr = r["x"]
    w.build_aero(xr)
    a0 = w.relax(0.0, x_init=xr, fixed=fx, fixed_x=xr, aero=True)
    a1 = w.relax(k["V"], x_init=a0["x"], fixed=fx, fixed_x=xr, aero=True)
    a = float(w.vlm.area * 1e4)
    return dict(name=job["name"], ok=[bool(r["ok"]), bool(a0["ok"]), bool(a1["ok"])], N=job["N"], travel=job["travel"], F0=W.state(w, a0["x"], 0.0)["F_mN"], L0=float(w.aero(a0["x"])[1]) * Q * a * 1e-4, L1=float(w.aero(a1["x"])[1]) * Q * a * 1e-4,
                peak0=peak(w, a0["x"], 0.0), peak1=peak(w, a1["x"], k["V"]), z=[W.state(w, a0["x"], 0.0)["z_max_mm"], W.state(w, a1["x"], k["V"])["z_max_mm"]])


def cases():
    out = []
    for shape, c in (("rectangle", 0.025), ("curved", 0.027)):
        for N in (2, 4):
            for tr, h in ((4.0, 2e-3), (5.0, 2e-3), (6.0, 2e-3), (8.0, 2e-3), (5.0, 1.5e-3)):
                q = dict(name=f"{shape}, {N} layers, pull {tr:.0f} mm, h {h * 1e3:.1f}", N=N, travel=tr, V=1370.0, theta=0.0, stretch=0.0, lam_d=1.0, lam_n=1.0, root_free=True, cr=c, h=h, alpha=6.0,
                         tensioners=(dict(at="D", anchor=(W.AX - 0.02, W.AY + c), kind="line", k=500.0, F0=500.0 * tr * 1e-3 + 0.04 * (N + 1) / 4),))
                if shape == "curved":
                    q.update(curve=dict(xc=W.AX + 0.035, c=c, ct=0.003), ct=0.003)
                else:
                    q.update(ct=c)
                out.append(q)
    return out


def safe(j):
    try:
        return run(j)
    except Exception as e:
        return dict(name=j["name"], error=repr(e)[:60])


if __name__ == "__main__":
    with mp.Pool(13) as pool:
        out = pool.map(safe, cases())
    for r in sorted(out, key=lambda q: q["name"]):
        if "error" in r:
            print(r["name"], r["error"]); continue
        p, q = r["peak0"], r["peak1"]
        print("%-40s %s F %3.0f mN dL %5.2f | 0 V peak %.3f at (%.0f,%.0f) 95%% %.3f med %.3f fibre %.3f sig %.0f kPa G(1 mm) %.1f J/m2 | V peak %.3f" % (r["name"], "".join("Y" if s else "n" for s in r["ok"]), r["F0"], r["L1"] - r["L0"], p["lam"], p["at_mm"][0], p["at_mm"][1], p["lam95"], p["lam_med"], p["fibre_lam"], p["sigma_kPa"], 2 * math.pi / math.sqrt(p["lam"]) * p["W_Jm3"] * 1e-3, q["lam"]))
    json.dump(out, open(os.path.join(HERE, "data", "tear_check.json"), "w"), indent=1)
