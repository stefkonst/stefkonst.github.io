"""Stress and shape across the chord from the finite elements, on the 5 cm panel. Used to set the tension profile in skin.py.
Run: python fe_fields.py -> data/fe_fields.json"""
import json, os, sys
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fe_reference as R
M = R.M


def sections(w, x, V, pe=None):
    """Mean over the middle third of the span and over chord bins: Cauchy resultants in s, c, and shear, height, pressure."""
    Pe = w.P_e(V)
    Sg = w.membrane(x, Pe, want=True)
    tri = w.tri
    xc = x[tri][:, :, :].mean(1)
    S = Sg if isinstance(Sg, dict) else None
    return S, xc


def run(over, tag):
    k = dict(R.BASE, **over)
    w, x = R.build(k)
    fx = w.fixed.copy()
    r = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)
    xr = r["x"]
    w.build_aero(xr)
    a0 = w.relax(0.0, x_init=xr, fixed=fx, fixed_x=xr, aero=True)
    a1 = w.relax(k["V"], x_init=a0["x"], fixed=fx, fixed_x=xr, aero=True)
    out = dict(case=over, tag=tag, CL=[float(a0["CL"]), float(a1["CL"])])
    for name, st, V_, pe in (("ns", xr, 0.0, None), ("V0", a0["x"], 0.0, a0["pe"]), ("V1", a1["x"], k["V"], a1["pe"])):
        f = w.field(st, V_)
        out[name] = {q: (np.asarray(v).tolist() if np.ndim(v) else float(v)) for q, v in f.items()} if isinstance(f, dict) else repr(type(f))
        out[name + "_x"] = st.tolist()
        if pe is not None:
            out[name + "_pe"] = np.asarray(pe).tolist()
    out["tri"] = w.tri.tolist()
    out["fib"] = [int(i) for i in w.fib[0]]
    return out


if __name__ == "__main__":
    res = [run(dict(theta=5.0, lam_n=1.04, alpha=6.0), "base"), run(dict(theta=3.0, lam_n=1.04, alpha=10.0), "low")]
    json.dump(res, open(os.path.join(HERE, "data", "fe_fields.json"), "w"))
    print({k: (type(v).__name__) for k, v in res[0].items()})
    print("keys of field", list(res[0]["V0"].keys()) if isinstance(res[0]["V0"], dict) else res[0]["V0"])
