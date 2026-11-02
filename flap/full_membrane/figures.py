"""Figures for the full membrane page, from the JSON files in validation/ and masks/. Style: ../plagiopatagiales/plotstyle.py.
Run: python figures.py -> validation/F_*.png"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
PS.apply()
V = os.path.join(HERE, "validation")
load = lambda n: json.load(open(os.path.join(V, n + ".json")))
save = lambda fig, n: (fig.tight_layout(), fig.savefig(os.path.join(V, n + ".png")), plt.close(fig))


def fig_tension():
    T, Fd = load("tension"), load("tension_fields")
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.6))
    vm = 2.0
    for a, key, ttl in ((ax[0, 0], "0_tips", "Panel 1, fibre tips only"), (ax[0, 1], "0_edges", "Panel 1, root and batten bars"), (ax[1, 0], "1_tips", "Panel 2, fibre tips only"), (ax[1, 1], "1_edges", "Panel 2, batten and LE bars")):
        f = Fd[key]
        x = np.array(f["x"]) * 1e3
        tp = a.tripcolor(Triangulation(x[:, 0], x[:, 1], np.array(f["tri"])), facecolors=np.array(f["Ne"]), cmap="viridis", vmin=0, vmax=vm)
        a.set_aspect("equal"); a.invert_yaxis(); a.set(xlabel="$x$ in mm", ylabel="$y$ in mm", title=ttl)
        a.title.set_fontsize(9)
    fig.subplots_adjust(left=0.05, right=0.985, wspace=0.34, hspace=0.42, bottom=0.09, top=0.94)
    cax = fig.add_axes([0.665, 0.2, 0.012, 0.6])
    fig.colorbar(tp, cax=cax, label=r"Major resultant $n_1$ in N/m")
    for q in (ax[0, 2], ax[1, 2]):
        p = q.get_position(); q.set_position([p.x0 + 0.08, p.y0, p.width - 0.08, p.height])
    th = T["theta"]
    a = ax[0, 2]
    t = [r["theta"] for r in th]
    a.plot(t, [r["cv0"] * 100 for r in th], "o-", label="Panel 1, bars")
    a.plot(t, [r["cv1"] * 100 for r in th], "s-", label="Panel 2, bars")
    a.axhline(T["strip"]["20"]["N_cv"] * 100, color="k", ls=":", label="External strip grips")
    a.set_yscale("log"); a.set(xlabel=r"Batten tilt $\theta$ in °", ylabel=r"Variation of $n_1$ in %"); a.legend()
    a = ax[1, 2]
    C = load("chord_step")
    for ld, st in ((1.03, "o-"), (1.05, "s-")):
        r = [q for q in C if q["lam_d"] == ld]
        a.plot([q["dn"] for q in r], [q["s"]["1"]["wrinkled"] * 100 for q in r], st, label=rf"$\lambda_d$ = {ld}, panel 2")
    a.set(xlabel=r"LE shift as extra stretch $\Delta\lambda_n$", ylabel="Wrinkled area in %"); a.legend()
    PS.finish(np.array([ax[0, 2], ax[1, 2]]), letters=False)
    for k, q in enumerate((ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2])):
        q.text(-0.12, 1.03, f"{chr(97 + k)})", transform=q.transAxes, fontsize=13, va="bottom")
    fig.savefig(os.path.join(V, "F_tension.png"), bbox_inches="tight"); plt.close(fig)


def fig_sweep():
    R = load("sweep")
    lds = sorted({r["lam_d"] for r in R})
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.4))
    for j, U in enumerate((3.0, 4.0, 6.0)):
        for i, ld in enumerate(lds):
            r = sorted([q for q in R if q["U"] == U and q["lam_d"] == ld], key=lambda q: q["dn"])
            if not r:
                continue
            c = PS.MATLAB[i]
            good = [q for q in r if q.get("ok0") and q.get("ok1") and q.get("dCL") is not None]
            ax[0, j].plot([q["dn"] for q in good], [q["dCL"] for q in good], "o-", color=c, label=rf"$\lambda_d$ = {ld}")
            bad = [q for q in r if q not in good]
            ax[0, j].plot([q["dn"] for q in bad], [0.0 for q in bad], "x", color=c)
            ax[1, j].plot([q["dn"] for q in r if q.get("CL0")], [q["CL0"] for q in r if q.get("CL0")], "o-", color=c)
        ax[0, j].set(xlabel=r"$\lambda_n-\lambda_d^{-1/2}$", ylabel=rf"$\Delta C_L$ at 1.6 kV, $U$ = {U:.0f} m/s", ylim=(-0.02, None))
        ax[1, j].set(xlabel=r"$\lambda_n-\lambda_d^{-1/2}$", ylabel=rf"$C_L$ at 0 V, $U$ = {U:.0f} m/s")
        fl = [q["CL_flat"] for q in R if q["U"] == U]
        if fl:
            ax[1, j].axhline(np.mean(fl), color="k", ls=":", label="Flat wing")
    ax[0, 0].legend(); ax[1, 0].legend()
    PS.finish(ax); save(fig, "F_sweep")


def fig_final():
    F = load("final")
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.4))
    a = ax[0, 0]
    vr = F["V"]
    a.plot([q["V"] / 1e3 for q in vr], [q["CL"] for q in vr], "o-", color=PS.ON)
    a.plot([q["V"] / 1e3 for q in vr], [vr[0]["CL"] + (vr[6]["CL"] - vr[0]["CL"]) * (q["V"] / 1600) ** 2 for q in vr], "k:", label=r"$V^2$ through 1.6 kV")
    a.legend()
    a.set(xlabel="Voltage $V$ in kV", ylabel=r"$C_L$ ($U$ = 4 m/s, $\alpha$ = 6°)")
    a = ax[0, 1]
    for U, st in ((3.0, ":"), (4.0, "-"), (6.0, "--")):
        r = sorted([q for q in F["alpha"] if q["U"] == U and q.get("ok0")], key=lambda q: q["alpha"])
        a.plot([q["alpha"] for q in r], [q["CL0"] for q in r], st, color=PS.OFF, label=f"0 V, {U:.0f} m/s")
        g = [q for q in r if q.get("CL1") is not None and q.get("ok1")]
        a.plot([q["alpha"] for q in g], [q["CL1"] for q in g], st, color=PS.ON, label=f"1.6 kV, {U:.0f} m/s")
    a.axvline(12, color="0.5", lw=0.6); a.set(xlabel=r"$\alpha$ in °", ylabel="$C_L$"); a.legend(fontsize=6)
    a = ax[0, 2]
    for U, st in ((3.0, "o:"), (4.0, "o-"), (6.0, "o--")):
        r = sorted([q for q in F["alpha"] if q["U"] == U and q.get("dCL") is not None and q.get("ok1") and q.get("ok0")], key=lambda q: q["alpha"])
        a.plot([q["alpha"] for q in r], [q["dCL"] for q in r], st, color=PS.MATLAB[1], label=f"{U:.0f} m/s")
    a.set(xlabel=r"$\alpha$ in °", ylabel=r"$\Delta C_L$ at 1.6 kV"); a.legend()
    a = ax[1, 0]
    for k, lbl in enumerate(dict.fromkeys(q["label"] for q in F["stack"])):
        r = [q for q in F["stack"] if q["label"] == lbl and q.get("dCL") is not None and q.get("ok1")]
        a.plot([q["eta"] for q in r], [q["dCL"] for q in r], "o-", label={0: "1 active film, no cover, 35 µm", 1: "½D+E+D+E+½D, 70 µm", 2: "3 active films of 4, 140 µm"}[k])
    a.set(xlabel=r"$\eta$ of the flat sheet", ylabel=r"$\Delta C_L$"); a.legend(fontsize=6)
    a = ax[1, 1]
    for k, ddn in enumerate((-0.005, 0.0, 0.005)):
        r = sorted([q for q in F["tol"] if q["ddn"] == ddn and q.get("dCL") is not None and q.get("ok1")], key=lambda q: q["dld"])
        a.plot([q["dld"] * 100 for q in r], [q["dCL"] for q in r], "o-", label=rf"$\delta\lambda_n$ = {ddn * 100:+.1f} %")
    a.set(xlabel=r"Error in $\lambda_d$ in %", ylabel=r"$\Delta C_L$"); a.legend()
    a = ax[1, 2]
    rods = [q for q in F.get("rods", []) if q.get("dCL") is not None]
    a.barh(range(len(rods)), [q.get("dCL") or 0 for q in rods], color=PS.MATLAB[0])
    a.set_yticks(range(len(rods))); a.set_yticklabels([q["label"] for q in rods], fontsize=7); a.set(xlabel=r"$\Delta C_L$ with an elastic frame")
    PS.finish(ax); save(fig, "F_final")


def fig_shape():
    S = load("shape")
    tri = np.array(S["tri"])
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
    for a, xk, zk, ttl in ((ax[0, 0], "x0", "z0", "0 V"), (ax[0, 1], "x1", "z1", "1.6 kV")):
        x = np.array(S[xk]) * 1e3
        c = a.tripcolor(Triangulation(x[:, 0], x[:, 1], tri), np.array(S[zk]) * 1e3, shading="gouraud", cmap="viridis", vmin=0, vmax=max(np.max(S["z1"]), np.max(S["z0"])) * 1e3)
        fig.colorbar(c, ax=a, label="$w$ in mm", shrink=0.8)
        a.set_aspect("equal"); a.invert_yaxis(); a.set(xlabel="$x$ in mm", ylabel="$y$ in mm", title=ttl)
    x = np.array(S["x1"]) * 1e3
    for a, x0, ttl in ((ax[1, 0], 30.0, "Panel 1, $x$ = 30 mm"), (ax[1, 1], 72.0, "Panel 2, $x$ = 72 mm")):
        for xk, zk, col, lb in (("x0", "z0", PS.OFF, "0 V"), ("x1", "z1", PS.ON, "1.6 kV")):
            xx = np.array(S[xk]) * 1e3
            tr = Triangulation(xx[:, 0], xx[:, 1], tri)
            from matplotlib.tri import LinearTriInterpolator
            ip = LinearTriInterpolator(tr, np.array(S[zk]) * 1e3)
            ys = np.linspace(6.4, 60, 200)
            a.plot(ys, ip(np.full_like(ys, x0), ys), color=col, label=lb)
        a.set(xlabel="$y$ in mm", ylabel="$w$ in mm", title=ttl); a.legend()
    PS.finish(ax); save(fig, "F_shape")


def fig_masks():
    Mk = json.load(open(os.path.join(HERE, "masks", "mask_numbers.json")))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.7))
    labels = dict(root=("Root end", "Batten tip"), tip=("Batten tip", "Tip, by the LE"))
    for a, k, ttl in ((ax[0], "root", "Panel 1 (inboard)"), (ax[1], "tip", "Panel 2 (outboard)")):
        q = Mk[k]
        ac = np.array(q["active"]); a.fill(ac[:, 0], ac[:, 1], color=PS.MATLAB[4], alpha=0.25, label="Active sheet (A and B overlap)")
        s_ = np.array(q["sheet_mm"] + q["sheet_mm"][:1]); a.plot(s_[:, 0], s_[:, 1], "k:", lw=0.8, label="Frame line, as cut")
        st = np.array(q["strip_mm"] + q["strip_mm"][:1]); a.plot(st[:, 0], st[:, 1], color="0.45", lw=0.9, label="Strip rolled into the active fibre")
        tA, tB = np.array(q["tab_A"]), np.array(q["tab_B"])
        st_ = np.array(q["strip_mm"])
        a.fill([tA[:, 0].max(), tB[:, 0].min(), tB[:, 0].min(), tA[:, 0].max()], [tA[:, 1].min(), tA[:, 1].min(), tA[:, 1].max(), tA[:, 1].max()], color=PS.MATLAB[4], alpha=0.25)
        a.fill(tA[:, 0], tA[:, 1], color=PS.OFF, alpha=0.6, label="Contact A")
        a.fill(tB[:, 0], tB[:, 1], color=PS.ON, alpha=0.6, label="Contact B")
        fa, fb = np.array(q["fibre_A"]), np.array(q["fibre_B"])
        a.plot(fa[:, 0], fa[:, 1], color=PS.OFF, lw=0.9, label="Spare rectangle, A")
        a.plot(fb[:, 0], fb[:, 1], color=PS.ON, lw=0.9, ls="--", label="Spare rectangle, B")
        ym = float(np.mean(tA[:, 1]))
        a.text(tA[:, 0].mean(), np.max(st[:, 1]) + 1.2, labels[k][0], ha="center", va="top", fontsize=7)
        a.text(tB[:, 0].mean(), np.max(st[:, 1]) + 1.2, labels[k][1], ha="center", va="top", fontsize=7)
        le_ = np.array(q["sheet_mm"])[:2]
        ang = np.degrees(np.arctan2(le_[1, 1] - le_[0, 1], le_[1, 0] - le_[0, 0]))
        a.text(le_[:, 0].mean(), le_[:, 1].mean() - 1.2, "Leading edge", ha="center", va="bottom", fontsize=7, rotation=-ang, rotation_mode="anchor")
        a.set_aspect("equal"); a.invert_yaxis(); a.set(xlabel="mm", ylabel="mm", title=f"{ttl}: {q['width_mm']:.1f} mm wide")
    PS.finish(ax)
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=7.5)
    fig.savefig(os.path.join(V, "F_masks.png")); plt.close(fig)


def fig_flap():
    import glob
    F = load("flap_compare")
    for fn in glob.glob(os.path.join(V, "flap_compare_*.json")):
        F["cases"].update(json.load(open(fn))["cases"])
    C = F["cases"]
    ph = np.array(F["phases"])
    X = np.r_[ph, 360.0]
    close = lambda y: np.r_[y, y[0]]
    r = load("flap_compare_section")["r_free_te"]

    def dl(k):
        q = np.array(C[k]["L1"], float) - np.array(C[k]["L0"], float)
        q[~(np.array(C[k]["ok0"]) & np.array(C[k]["ok1"]))] = np.nan     # unconverged phases are left as gaps
        return q

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.7))
    a = ax[0]
    a.plot(X, close(np.array(C["full_4_2"]["L0"])), color=PS.OFF, label="Full membrane, 0 V")
    a.plot(X, close(np.array(C["full_4_2"]["L1"])), color=PS.ON, label="Full membrane, 1.6 kV")
    a.plot(X, close(np.array(C["old_4_2"]["L0"])), color=PS.OFF, ls="--", label="Discrete DEAs, 0 V")
    a.plot(X, close(np.array(C["old_4_2"]["L1"])), color=PS.ON, ls="--", label="Discrete DEAs, 1.6 kV")
    a.axvspan(0, 180, color="0.93", zorder=0)
    a.set(xlabel=r"Flap phase $\varphi$ in °", ylabel="Lift $L$ in mN, one wing", xlim=(0, 360)); a.legend(fontsize=6)
    a = ax[1]
    a.plot(X, close(dl("full_4_2")), "o-", ms=2.5, color=PS.MATLAB[0], label=r"Full membrane, $\eta$ = 0.40")
    if "full_4_2_eta24" in C:
        a.plot(X, close(dl("full_4_2_eta24")), "--", color=PS.MATLAB[0], label=r"Full membrane, $\eta$ = 0.24")
    a.plot(X, close(dl("full_4_2_eta15")), ":", color=PS.MATLAB[0], label=r"Full membrane, $\eta$ = 0.15")
    a.plot(X, close(dl("old_4_2_pre100")), "-", color=PS.MATLAB[2], label=r"Discrete, framed, $\lambda_{pre}$ = 1.00")
    a.plot(X, close(dl("old_4_2_pre101")), "-", color=PS.MATLAB[3], label=r"Discrete, framed, $\lambda_{pre}$ = 1.01")
    a.plot(X, close(dl("old_4_2")), "o-", ms=2.5, color=PS.MATLAB[1], label=r"Discrete, framed, $\lambda_{pre}$ = 1.03")
    a.plot(X, close(dl("old_4_2_cyc")), ":", color=PS.MATLAB[1], label=r"Discrete, framed, 1.03, cycle $\Delta F$")
    a.plot(X, close(r * dl("old_4_2")), "-.", color="k", label=f"Discrete, free TE: $r$ = {r} × 1.03")
    a.axvspan(0, 180, color="0.93", zorder=0)
    a.axhline(0, color="k", lw=0.5); a.set(xlabel=r"Flap phase $\varphi$ in °", ylabel=r"$\Delta L$ at 1.6 kV in mN", xlim=(0, 360)); a.legend(fontsize=5.3)
    a = ax[2]
    rows = [("3 m/s\n2 Hz", "3_2"), ("4 m/s\n1 Hz", "4_1"), ("4 m/s\n2 Hz", "4_2"), ("6 m/s\n2 Hz", "6_2")]
    w_ = 0.38
    for i, (lb, k) in enumerate(rows):
        a.bar(i - w_ / 2, np.nanmean(dl("full_" + k)), w_, color=PS.MATLAB[0], label="Full membrane" if i == 0 else None)
        a.bar(i + w_ / 2, np.nanmean(dl("old_" + k)), w_, color=PS.MATLAB[1], label=r"Discrete, framed, $\lambda_{pre}$ = 1.03" if i == 0 else None)
    a.set_xticks(range(len(rows))); a.set_xticklabels([q[0] for q in rows], fontsize=7); a.set(ylabel=r"Cycle-mean $\Delta L$ in mN"); a.legend(fontsize=6.5)
    PS.finish(ax); save(fig, "F_flap")
    out = {k: dict(mean_L0_mN=float(np.mean(C[k]["L0"])), mean_dL_mN=float(np.nanmean(dl(k))), min_dL_mN=float(np.nanmin(dl(k))), max_dL_mN=float(np.nanmax(dl(k))), converged=int(np.sum(C[k]["ok0"]) + np.sum(C[k]["ok1"])), states=2 * len(C[k]["ok0"]), params=C[k].get("params")) for k in C}
    json.dump(out, open(os.path.join(V, "flap_summary.json"), "w"), indent=1)
    for k, v in out.items():
        print(k, {a_: (round(b_, 2) if isinstance(b_, float) else b_) for a_, b_ in v.items() if a_ != "params"})
    d = np.round(dl("full_4_2"), 1); print("full_4_2 dL by phase", d.tolist(), "alpha tip", np.round(C["full_4_2"]["alpha_tip"], 1).tolist())


def fig_main():
    """The comparison alone, for the main page."""
    import glob
    F = load("flap_compare")
    for fn in glob.glob(os.path.join(V, "flap_compare_*.json")):
        F["cases"].update(json.load(open(fn))["cases"])
    C = F["cases"]
    X = np.r_[np.array(F["phases"]), 360.0]
    close = lambda y: np.r_[y, y[0]]
    r = load("flap_compare_section")["r_free_te"]
    d = lambda k: np.array(C[k]["L1"], float) - np.array(C[k]["L0"], float)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9.5})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    a = ax[0]
    a.axvspan(0, 180, color="0.93", zorder=0)
    a.text(90, 15.6, "Downstroke", ha="center", fontsize=9, color="0.35"); a.text(270, 0.9 + 15.0 - 0.3, "Upstroke", ha="center", fontsize=9, color="0.35")
    a.plot(X, close(d("full_4_2")), "o-", color=PS.MATLAB[0], label="Full membrane")
    a.plot(X, close(d("old_4_2")), "o-", color=PS.MATLAB[1], label="Discrete DEAs, ideal rigid trailing edge")
    a.plot(X, close(r * d("old_4_2")), "k-.", label="Discrete DEAs, free trailing edge (estimate)")
    a.set(xlabel=r"Flap phase $\varphi$ in °", ylabel=r"Lift change $\Delta L$ in mN", xlim=(0, 360), ylim=(-0.5, 17)); a.legend(loc="center right", bbox_to_anchor=(1.0, 0.31))
    a = ax[1]
    rows = [("3 m/s", "3_2"), ("4 m/s", "4_2"), ("6 m/s", "6_2")]
    w_ = 0.38
    for i, (lb, k) in enumerate(rows):
        f_ = d("full_" + k).astype(float)
        f_[~(np.array(C["full_" + k]["ok0"]) & np.array(C["full_" + k]["ok1"]))] = np.nan
        a.bar(i - w_ / 2, np.nanmean(f_), w_, color=PS.MATLAB[0], label="Full membrane" if i == 0 else None)
        a.bar(i + w_ / 2, d("old_" + k).mean(), w_, color=PS.MATLAB[1], label="Discrete DEAs, ideal rigid trailing edge" if i == 0 else None)
    a.set_xticks(range(3)); a.set_xticklabels([q[0] for q in rows]); a.set(ylabel=r"Cycle-mean $\Delta L$ in mN"); a.legend(loc="upper left")
    PS.finish(ax); save(fig, "F_main")
    PS.apply()


def fig_design():
    ok = lambda r: r.get("ok0") and r.get("ok1") and r.get("dL_mN") is not None
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
    L = [r for r in load("opt_layers") if ok(r) and r["U"] == 4.0]
    ns = sorted({r["na"][0] for r in L})
    best = [max([r for r in L if r["na"][0] == n], key=lambda r: r["dL_mN"]) for n in ns]
    a = ax[0]
    a.plot(ns, [r["dL_mN"] for r in best], "o-", color=PS.MATLAB[0]); a.set(xlabel="Active layers on both panels", ylabel=r"$\Delta L$ in mN")
    b = a.twinx(); b.plot(ns, [r["mass_mg"] for r in best], "s--", color=PS.MATLAB[1]); b.set_ylabel("Wing mass in mg", color=PS.MATLAB[1])
    a = ax[1]
    P = [r for r in load("opt_panels") if ok(r)]
    for n2, c in zip((1, 2, 3, 4, 5), PS.MATLAB):
        q = sorted([r for r in P if r["na"][1] == n2], key=lambda r: r["mass_mg"])
        a.plot([r["mass_mg"] for r in q], [r["dL_mN"] for r in q], "o-", color=c, label=f"Panel 2: {n2}")
    a.set(xlabel="Wing mass in mg", ylabel=r"$\Delta L$ in mN"); a.legend(fontsize=7, title="Panel 1: 1, 2 along each line", title_fontsize=6.5)
    a = ax[2]
    Rl = [r for r in load("opt_roll") if ok(r) and r["U"] == 4.0]
    Rl = sorted(Rl, key=lambda r: r["ws"][0])
    a.plot([r["ws"][0] * 1e3 for r in Rl], [r["dL_mN"] for r in Rl], "o-", color=PS.MATLAB[0])
    a.axvline(17.5, color="k", ls=":", lw=0.8); a.text(18, min(r["dL_mN"] for r in Rl), "coater limit,\nchord 46 mm", fontsize=7)
    a.set(xlabel="Roll width as cut in mm", ylabel=r"$\Delta L$ in mN")
    a = ax[3]
    Sh = [r for r in load("opt_shape") + load("opt_shape2") if ok(r) and r["U"] == 4.0 and r["na"][0] == 2 and abs(r["theta"] - 14) < 0.1]
    for yc, c in ((42.5, PS.MATLAB[0]), (47.5, PS.MATLAB[1]), (52.5, PS.MATLAB[2])):
        q = sorted({round(r["xc"] * 1e3, 1): r for r in Sh if abs(r["yc"] * 1e3 - yc) < 0.1}.items())
        a.plot([k for k, _ in q], [v["dL_mN"] for _, v in q], "o-", color=c, label=f"Chord line at {yc:g} mm")
    a.set(xlabel="Batten tip position $x$ in mm", ylabel=r"$\Delta L$ in mN"); a.legend(fontsize=7)
    PS.finish(ax); fig.tight_layout(); fig.savefig(os.path.join(V, "F_design.png")); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["tension", "sweep", "final", "shape", "masks", "flap", "main", "design"]
    for w in which:
        globals()["fig_" + w]()
