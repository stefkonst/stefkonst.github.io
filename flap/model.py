"""Active membrane wing model: hyperelastic membrane + rolled DEAs + reinforcement under aerodynamic load.

Physics (all SI units):
  Membrane   : thin incompressible hyperelastic sheet, plane stress, large deformation, constant-strain triangles
               (total Lagrangian). Strain energy per reference volume
                   W = C10 (I1-3) + C20 (I1-3)^2 + C30 (I1-3)^3 + sum_d k1_d (I4_d-1)^2 + k2_d (I4_d-1)^3  (I4_d > 1)
               with I1 = tr C + 1/det C (incompressibility, lambda3^2 = 1/det C) and I4_d = a_d . C a_d.
               2nd Piola-Kirchhoff stress S = 2 W1 (I - C^-1/det C) + 2 W4_d a_d (x) a_d.
               Wrinkling: compressive principal stresses are set to zero (no-compression membrane).
               Constants calibrated on measured uniaxial curves (calibrate.py), valid up to stretch 1.22.
  DEA        : each rolled DEA is a 1D hyperelastic bar (Yeoh, uniaxial) bonded along a mesh line.
               Maxwell stress eps0 epsr E^2 with true field E = V sqrt(lam)/t_act acting on the current active
               cross-section A0 v_act / lam reduces the axial force by
                   dF = eta v_act eps0 epsr (V / t_act)^2 A0       (independent of stretch),
               eta = measured / ideal blocked force (0.37 from the bench, fig2). Bars carry tension only.
  Reinforcement: linear elastic bars (E A), optionally tension-only (thin strips buckle).
  Aerodynamics: quasi-steady thin-airfoil theory applied strip-wise on the DEFORMED membrane: for each spanwise
               station the camber line relative to the LE-TE chord gives Glauert coefficients A0..AN,
                   dCp(theta) = 4 [A0 (1+cos theta)/sin theta + sum An sin n theta],  CL = pi (2 A0 + A1),
               averaged exactly over each mesh cell, applied as a follower pressure normal to the surface.
               3D: lift-slope factor AR/(AR+2) and optional elliptic spanwise loading. Attached flow only.
  Flapping   : quasi-steady stroke Phi(t) = Phi0 cos(phase); section velocity Omega x changes the local angle of
               attack and dynamic pressure; membrane inertia (-m dOmega/dt x) and centrifugal load are included.
  Solver     : dynamic relaxation with kinetic damping (robust for slack/tension-only cable-membrane systems).
"""
import math
import numpy as np

EPS0 = 8.854e-12
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2 (Pyodide) compatibility

# Constitutive constants (Pa) from calibrate.py (measured uniaxial unloading curves, fig1)
# W_s = v_m sum_i C_i (I1-3)^i + sum_d (v_f E_f)_d (sqrt(I4_d) - 1)^2 / 2   (calibrate.py, fit_p7670.py)
MATERIALS = {
    "aniso": dict(label="Anisotropic composite membrane (measured)", C=(5636.1, 15840.8, 0.0), v_m=0.577,
                  Ef_chord=143990.0, Ef_span=68770.0),
    "aniso90": dict(label="Anisotropic composite membrane, rotated 90 deg", C=(5636.1, 15840.8, 0.0), v_m=0.577,
                    Ef_chord=68770.0, Ef_span=143990.0),
    "P6760": dict(label="Isotropic Wacker P6760 (measured)", C=(5636.1, 15840.8, 0.0), v_m=1.0, Ef_chord=0.0, Ef_span=0.0),
    "DS20": dict(label="Isotropic DragonSkin 20 (measured)", C=(63051.2, 0.0, 0.0), v_m=1.0, Ef_chord=0.0, Ef_span=0.0),
    "P7670": dict(label="Isotropic Elastosil P7670 (DEA silicone, measured)", C=(42842.1, 447.0, 0.0), v_m=1.0, Ef_chord=0.0, Ef_span=0.0),
}
LAM_CAL = 1.22
DEA_YEOH = (42842.1, 447.0, 0.0)          # Elastosil P7670, fitted on the rolled-DEA bundle at 0 V (fit_p7670.py)

DEFAULTS = dict(
    span=0.070, chord=0.035, taper=1.0, nx=16, ny=8,     # chord = root chord; taper = tip chord / root chord
    material="aniso", t_mem=180e-6, rho_mem=1100.0, pre_span=1.03, pre_chord=1.03,
    clamp_LE=True, clamp_TE=False, clamp_root=True, clamp_tip=False,
    # leading edge: "rod" = carbon rod cantilevered at the root (linearly tapered), "rigid" = clamped edge
    le_mode="rod", rod_d_root=1.0e-3, rod_d_tip=0.5e-3, rod_E=120e9, rod_rho=1550.0,
    # chordwise carbon digits from the LE to the TE (span fractions), joined stiffly to the LE rod
    digits="0.5, 1.0", digit_d=0.8e-3, rod_EA_cap=100.0,
    # rolled DEAs
    n_dea_chord=4, n_dea_span=0, dea_width=10e-3, t_act=30e-6, t_pass=30e-6, eps_r=2.8, eta=0.40,
    pre_dea=1.10, V=1600.0, act_chord=True, act_span=True, E_bd=100e6,
    # reinforcement (linear elastic strips)
    n_rf_chord=0, n_rf_span=0, rf_E=4e9, rf_width=1e-3, rf_thick=50e-6, rf_tension_only=True, te_cord=True,
    # flow
    U=3.0, alpha_deg=6.0, rho_air=1.225, three_d=True, elliptic=True,
    # flapping (quasi-steady)
    flap=False, flap_f=3.0, flap_amp_deg=30.0, flap_phase_deg=90.0,
    dea_schedule="always",                # "always", "downstroke" or "upstroke" (DEA on only in that half-stroke)
    # solver
    wrinkle_residual=0.0,                 # fraction of compressive membrane stress kept (0 = pure tension field)
    p_uniform=None,                       # validation only: uniform pressure (Pa) instead of aerodynamics
    max_iter=30000, tol=1e-4, aero_every=10, ramp=400, n_glauert=8,
)


def _fibre_pos(n):
    """Fractions of n fibres evenly spread (centred in n equal bands)."""
    return [(k + 0.5) / n for k in range(n)] if n > 0 else []


def _grid(L, nseg, fracs):
    """Uniform grid with mesh lines placed exactly at the fibre positions; returns coords, {frac: index}."""
    pts = list(np.linspace(0, L, nseg + 1))
    h = L / nseg
    for f in fracs:
        pts = [q for k, q in enumerate(pts) if k in (0, len(pts) - 1) or abs(q - f * L) > 0.4 * h or q in [g * L for g in fracs]]
        pts.append(f * L)
    xs = np.array(sorted(set(np.round(pts, 12))))
    return xs, {f: int(np.argmin(np.abs(xs - f * L))) for f in fracs}


class Wing:
    def __init__(self, **kw):
        p = dict(DEFAULTS)
        p.update(kw)
        self.p = p
        b, c = p["span"], p["chord"]
        fd_c, fr_c = _fibre_pos(p["n_dea_chord"]), _fibre_pos(p["n_rf_chord"])     # chordwise lines (span positions)
        fd_s, fr_s = _fibre_pos(p["n_dea_span"]), _fibre_pos(p["n_rf_span"])       # spanwise lines (chord positions)
        dg = p["digits"]
        dg = [float(v) for v in str(dg).replace(";", ",").split(",") if str(v).strip()] if isinstance(dg, str) else list(dg or [])
        self.dig_frac = sorted({min(max(f, 0.0), 1.0) for f in dg}) if p["le_mode"] == "rod" else []
        xs, ix = _grid(b, p["nx"], fd_c + fr_c + [f for f in self.dig_frac if 0 < f < 1])
        ys, iy = _grid(c, p["ny"], fd_s + fr_s)
        nx, ny = len(xs) - 1, len(ys) - 1
        self.nx, self.ny, self.xs, self.ys = nx, ny, xs, ys
        I, J = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
        self.I, self.J = I.ravel(), J.ravel()
        nid = lambda i, j: i * (ny + 1) + j
        self.nid = nid
        self.cx = c * (1 - (1 - p["taper"]) * xs / b)                  # local chord (straight LE at y = 0)
        xf = np.c_[xs[self.I], ys[self.J] / c * self.cx[self.I], np.zeros(self.I.size)]   # frame (mounted) positions
        self.area = b * c * (1 + p["taper"]) / 2
        self.x0 = xf.copy()
        self.X = np.c_[xf[:, 0] / p["pre_span"], xf[:, 1] / p["pre_chord"]]   # unstretched membrane
        # triangles (alternating diagonals), with their cell indices
        tri, cell = [], []
        for i in range(nx):
            for j in range(ny):
                a, b_, c_, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
                if (i + j) % 2 == 0:
                    tri += [(a, b_, c_), (a, c_, d)]
                else:
                    tri += [(a, b_, d), (b_, c_, d)]
                cell += [(i, j), (i, j)]
        self.tri, self.cell = np.array(tri), np.array(cell)
        self.tri_flat = self.tri.ravel()
        P = self.X[self.tri]                                         # (ne,3,2)
        x1, y1, x2, y2, x3, y3 = P[:, 0, 0], P[:, 0, 1], P[:, 1, 0], P[:, 1, 1], P[:, 2, 0], P[:, 2, 1]
        A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        self.A0 = 0.5 * A2
        G = np.zeros((len(tri), 3, 2))
        G[:, 0, 0], G[:, 1, 0], G[:, 2, 0] = (y2 - y3) / A2, (y3 - y1) / A2, (y1 - y2) / A2
        G[:, 0, 1], G[:, 1, 1], G[:, 2, 1] = (x3 - x2) / A2, (x1 - x3) / A2, (x2 - x1) / A2
        self.G = G
        m = MATERIALS[p["material"]]
        self.C = tuple(m["v_m"] * ci for ci in m["C"])                # v_m W_iso folded into the constants
        self.Efc, self.Efs = m["Ef_chord"], m["Ef_span"]
        # boundary conditions
        fixed = np.zeros(self.I.size, bool)
        if p["clamp_root"]: fixed |= self.I == 0
        if p["clamp_tip"]: fixed |= self.I == nx
        self.rod = p["le_mode"] == "rod"
        if self.rod:
            fixed |= (self.J == 0) & (self.I == 0)
        elif p["clamp_LE"]:
            fixed |= self.J == 0
        if p["clamp_TE"]: fixed |= self.J == ny
        self.fixed = fixed
        # bars: (n1, n2, L0, kind, group)
        bars = []
        self.dea_lines = {"chord": sorted({ix[f] for f in fd_c}), "span": sorted({iy[f] for f in fd_s})}
        self.rf_lines = {"chord": sorted({ix[f] for f in fr_c}), "span": sorted({iy[f] for f in fr_s})}
        if p["te_cord"] and not p["clamp_TE"]:
            self.rf_lines["span"] = sorted(set(self.rf_lines["span"] + [ny]))

        def add_line(direction, k, kind, pre):
            if direction == "chord":
                pairs = [(nid(k, j), nid(k, j + 1)) for j in range(ny)]
            else:
                pairs = [(nid(i, k), nid(i + 1, k)) for i in range(nx)]
            for n1, n2 in pairs:
                L = np.linalg.norm(xf[n2] - xf[n1]) / pre
                bars.append((n1, n2, L, kind, direction))

        for d in ("chord", "span"):
            for k in self.dea_lines[d]:
                add_line(d, k, "dea", p["pre_dea"])
            for k in self.rf_lines[d]:
                add_line(d, k, "rf", 1.0)
        dfun = lambda xx: p["rod_d_root"] + (p["rod_d_tip"] - p["rod_d_root"]) * xx / b
        # rod bending: discrete curvature kappa_i = a x_(i-1) + b x_i + c x_(i+1), energy EI_i L_i |kappa_i|^2 / 2.
        # Each rod line starts at a stiff joint: the mirrored ghost x_g = x_1 - 2 h t enforces the initial tangent t
        # (fixed at the clamped root for the LE; following the joint translation for the digits).
        self.dig_lines = []
        if self.rod:
            IM, I0, IP, A_, B_, C_, W_, OFF = [], [], [], [], [], [], [], []

            def rod_line(nodes, sc, d_of_s, t):
                h = np.diff(sc)
                hm, hp = np.r_[h[0], h[:-1]], h
                a_, c_ = 2 / (hm * (hm + hp)), 2 / (hp * (hm + hp))
                EI = p["rod_E"] * math.pi * d_of_s(sc[:-1]) ** 4 / 64
                for k in range(len(nodes) - 1):
                    ghost = k == 0                                       # clamp: mirrored ghost x_g = x_1 - 2 h t
                    IM.append(nodes[k + 1] if ghost else nodes[k - 1]); I0.append(nodes[k]); IP.append(nodes[k + 1])
                    A_.append(a_[k]); B_.append(-(a_[k] + c_[k])); C_.append(c_[k])
                    W_.append(EI[k] * (0.5 * hp[k] if ghost else 0.5 * (hm[k] + hp[k])))
                    OFF.append(-a_[k] * 2 * hm[k] * t if ghost else np.zeros(3))

            rod_line([nid(i, 0) for i in range(nx + 1)], xs, dfun, np.array([1.0, 0, 0]))
            add_line("span", 0, "rod", 1.0)                             # axial stiffness of the LE rod
            self.dig_lines = sorted({int(np.argmin(np.abs(xs - f * b))) for f in self.dig_frac})
            for i in self.dig_lines:
                rod_line([nid(i, j) for j in range(ny + 1)], ys / c * self.cx[i], lambda q: np.full_like(q, p["digit_d"]),
                         np.array([0, 1.0, 0]))
                add_line("chord", i, "rod", 1.0)
            self.rb = dict(im=np.array(IM), i0=np.array(I0), ip=np.array(IP), a=np.array(A_), b=np.array(B_),
                           c=np.array(C_), W=np.array(W_), off=np.array(OFF))
        self.bars = bars
        nb = len(bars)
        self.b1 = np.array([b_[0] for b_ in bars], int)
        self.b2 = np.array([b_[1] for b_ in bars], int)
        self.bL0 = np.array([b_[2] for b_ in bars], float)
        self.bdea = np.array([b_[3] == "dea" for b_ in bars], bool)
        self.brod = np.array([b_[3] == "rod" for b_ in bars], bool)
        self.brf = ~self.bdea & ~self.brod
        xmid = 0.5 * (xf[self.b1, 0] + xf[self.b2, 0]) if nb else np.zeros(0)
        dch = np.array([b_[4] == "chord" for b_ in bars], bool)
        self.EA_rod = p["rod_E"] * math.pi * np.where(dch, p["digit_d"], dfun(xmid)) ** 2 / 4
        # numerical: rods are effectively inextensible (strain < 1e-4 under the membrane loads); their axial EA is capped
        # so the explicit solver converges. Results are insensitive to the cap (validate.py, V8).
        self.EA_rod = np.minimum(self.EA_rod, p["rod_EA_cap"])
        self.bchord = np.array([b_[4] == "chord" for b_ in bars], bool)
        self.A_dea = p["dea_width"] * (p["t_act"] + p["t_pass"])
        self.v_act = p["t_act"] / (p["t_act"] + p["t_pass"])
        self.EA_rf = p["rf_E"] * p["rf_width"] * p["rf_thick"]
        # lumped masses (physical, for inertia) [kg]
        mphys = np.zeros(self.I.size)
        np.add.at(mphys, self.tri.ravel(), np.repeat(p["rho_mem"] * p["t_mem"] * self.A0 / 3, 3))
        rho_rf = 1380.0                                              # PET
        bm = np.where(self.bdea, p["rho_mem"] * self.A_dea, rho_rf * p["rf_width"] * p["rf_thick"]) * self.bL0
        bm = np.where(self.brod, p["rod_rho"] * self.EA_rod / p["rod_E"] * self.bL0, bm)
        np.add.at(mphys, self.b1, bm / 2)
        np.add.at(mphys, self.b2, bm / 2)
        self.mphys = mphys
        self.m_dea = float(np.sum(bm[self.bdea]))
        self.m_total = float(mphys.sum())
        self._dr_masses()
        # Glauert cell bookkeeping
        xi = ys / c
        self.th = np.arccos(1 - 2 * xi)
        self.Bdcp = self._dcp_basis(p["n_glauert"])

    # ---------------- constitutive ----------------
    def _W1(self, I1, C):
        e = I1 - 3
        return C[0] + 2 * C[1] * e + 3 * C[2] * e ** 2

    @staticmethod
    def _W4(I4, Ef):
        """dW_aniso/dI4 for W_aniso = (v_f E_f)(sqrt(I4) - 1)^2 / 2, tension only."""
        r = np.sqrt(np.maximum(I4, 1e-12))
        return np.where(r > 1, Ef * (r - 1) / (2 * r), 0.0)

    def membrane(self, x, want_stress=False):
        xe = x[self.tri]                                               # (ne,3 nodes,3 comps)
        G0, G1 = self.G[:, :, 0:1], self.G[:, :, 1:2]
        F0 = np.sum(xe * G0, 1)                                        # dx/dX (ne,3)
        F1 = np.sum(xe * G1, 1)                                        # dx/dY
        c00 = np.sum(F0 * F0, 1); c11 = np.sum(F1 * F1, 1); c01 = np.sum(F0 * F1, 1)
        det = c00 * c11 - c01 ** 2
        I1 = c00 + c11 + 1 / det
        W1 = self._W1(I1, self.C)
        f = 2 * W1 / det ** 2                                          # inv(C)/det = adj(C)/det^2
        S00 = 2 * W1 - f * c11 + 2 * self._W4(c00, self.Efs)
        S11 = 2 * W1 - f * c00 + 2 * self._W4(c11, self.Efc)
        S01 = f * c01
        # no-compression (wrinkling): clamp negative principal stresses
        tr, dd = 0.5 * (S00 + S11), np.sqrt(0.25 * (S00 - S11) ** 2 + S01 ** 2)
        l2 = tr - dd
        if np.any(l2 < 0):
            l1 = tr + dd
            th = 0.5 * np.arctan2(2 * S01, S00 - S11)
            co, si = np.cos(th), np.sin(th)
            r = self.p["wrinkle_residual"]                              # small residual compressive stiffness (regularisation)
            l1c, l2c = np.where(l1 > 0, l1, r * l1), np.where(l2 > 0, l2, r * l2)
            S00 = l1c * co ** 2 + l2c * si ** 2
            S11 = l1c * si ** 2 + l2c * co ** 2
            S01 = (l1c - l2c) * co * si
        P0 = F0 * S00[:, None] + F1 * S01[:, None]                    # first Piola-Kirchhoff, column X
        P1 = F0 * S01[:, None] + F1 * S11[:, None]                    # column Y
        tA = (self.p["t_mem"] * self.A0)[:, None, None]
        fe = tA * (P0[:, None, :] * G0 + P1[:, None, :] * G1)         # (ne,3 nodes,3 comps)
        fint = np.empty_like(x)
        tri = self.tri_flat
        n = x.shape[0]
        for k in range(3):
            fint[:, k] = np.bincount(tri, fe[:, :, k].ravel(), minlength=n)
        if not want_stress:
            return fint
        Js = np.sqrt(det)
        S = np.stack([np.stack([S00, S01], -1), np.stack([S01, S11], -1)], -2)
        F = np.stack([F0, F1], -1)
        sig = np.einsum("eiJ,eJK,elK->eil", F, S, F) * (self.p["t_mem"] / Js)[:, None, None]   # N/m, current
        lam_max = np.sqrt(np.maximum(0.5 * (c00 + c11) + np.sqrt(0.25 * (c00 - c11) ** 2 + c01 ** 2), 0))
        return fint, sig, lam_max

    def bar_forces(self, x, V_chord, V_span):
        p = self.p
        d = x[self.b2] - x[self.b1]
        l = np.linalg.norm(d, axis=1)
        lam = l / self.bL0
        N = np.zeros_like(lam)
        if self.bdea.any():
            ld = lam[self.bdea]
            I1 = ld ** 2 + 2 / ld
            Np = self.A_dea * 2 * (ld - ld ** -2) * self._W1(I1, DEA_YEOH)
            Vb = np.where(self.bchord[self.bdea], V_chord, V_span)
            dF = p["eta"] * self.v_act * EPS0 * p["eps_r"] * (Vb / p["t_act"]) ** 2 * self.A_dea
            N[self.bdea] = np.maximum(Np - dF, 0.0)
        rf = self.brf
        if rf.any():
            Nr = self.EA_rf * (lam[rf] - 1)
            N[rf] = np.maximum(Nr, 0.0) if p["rf_tension_only"] else Nr
        if self.brod.any():
            N[self.brod] = self.EA_rod[self.brod] * (lam[self.brod] - 1)
        e = d / l[:, None]
        fint = np.zeros_like(x)
        for k in range(3):
            fint[:, k] = np.bincount(self.b2, N * e[:, k], minlength=x.shape[0]) - \
                         np.bincount(self.b1, N * e[:, k], minlength=x.shape[0])
        return fint, N, lam

    def rod_bending(self, x):
        """Internal force dU/dx of the rods (LE and digits), U = sum_i EI_i L_i |kappa_i|^2 / 2."""
        f = np.zeros_like(x)
        if not self.rod:
            return f
        rb = self.rb
        kap = rb["a"][:, None] * x[rb["im"]] + rb["b"][:, None] * x[rb["i0"]] + rb["c"][:, None] * x[rb["ip"]] + rb["off"]
        Wk = rb["W"][:, None] * kap
        np.add.at(f, rb["im"], rb["a"][:, None] * Wk)
        np.add.at(f, rb["i0"], rb["b"][:, None] * Wk)
        np.add.at(f, rb["ip"], rb["c"][:, None] * Wk)
        return f

    # ---------------- loads ----------------
    def flow_state(self):
        """Per-column local angle of attack, dynamic pressure and load factor; flapping kinematics."""
        p = self.p
        xcol = self.xs
        alpha = np.full(self.nx + 1, math.radians(p["alpha_deg"]))
        U = p["U"]
        Om = dOm = 0.0
        if p["flap"]:
            w = 2 * math.pi * p["flap_f"]
            A, ph = math.radians(p["flap_amp_deg"]), math.radians(p["flap_phase_deg"])
            Om, dOm = -A * w * math.sin(ph), -A * w * w * math.cos(ph)
        v = Om * xcol
        alpha = alpha - np.arctan2(v, max(U, 1e-9))
        q = 0.5 * p["rho_air"] * (U ** 2 + v ** 2)
        k3 = 1.0
        if p["three_d"]:
            AR = 2 * p["span"] ** 2 / self.area
            k3 = AR / (AR + 2)
        ell = (4 / math.pi) * np.sqrt(np.clip(1 - (xcol / p["span"]) ** 2, 0, 1)) if p["elliptic"] else np.ones_like(xcol)
        return dict(alpha=alpha, q=q, load=k3 * ell, Om=Om, dOm=dOm, x=xcol)

    def glauert(self, x, fs):
        """Glauert coefficients per spanwise column from the deformed camber line."""
        ny, N = self.ny, self.p["n_glauert"]
        idx = self.nid(np.arange(self.nx + 1)[:, None], np.arange(ny + 1)[None, :])
        y, z = x[idx, 1], x[idx, 2]                                     # (ncol, ny+1)
        dy = y[:, -1:] - y[:, :1]
        cl = np.sqrt(dy[:, 0] ** 2 + (z[:, -1] - z[:, 0]) ** 2)
        xi = np.clip((y - y[:, :1]) / dy, 0, 1)
        zr = z - (z[:, :1] + (z[:, -1:] - z[:, :1]) * xi)
        a_loc = fs["alpha"] + np.arctan2(z[:, 0] - z[:, -1], dy[:, 0])
        th = np.arccos(1 - 2 * xi)
        s = np.diff(zr, axis=1) / (np.diff(xi, axis=1) * cl[:, None] + 1e-30)
        dth = np.diff(th, axis=1)
        A = np.zeros((self.nx + 1, N + 1))
        A[:, 0] = a_loc - np.sum(s * dth, 1) / math.pi
        for n in range(1, N + 1):
            A[:, n] = (2 / math.pi) * np.sum(s * np.diff(np.sin(n * th), axis=1) / n, 1)
        camber = np.max(np.abs(zr), 1) * np.sign(zr[np.arange(zr.shape[0]), np.argmax(np.abs(zr), 1)]) / cl
        return A, camber, cl, zr

    def _dcp_basis(self, N):
        """B[n, j] = exact mean over chordwise cell j of the n-th dCp mode (dCp = A @ B)."""
        t0, t1 = self.th[:-1], self.th[1:]
        dxi = 0.5 * (np.cos(t0) - np.cos(t1))
        B = np.zeros((N + 1, len(t0)))
        B[0] = 2 * ((t1 + np.sin(t1)) - (t0 + np.sin(t0)))
        for n in range(1, N + 1):
            def I(t):
                lo = t if n == 1 else np.sin((n - 1) * t) / (n - 1)
                return 0.25 * (lo - np.sin((n + 1) * t) / (n + 1))
            B[n] = 4 * (I(t1) - I(t0))
        return B / dxi[None, :]

    def cell_dcp(self, A):
        return A @ self.Bdcp

    def pressure_forces(self, x, fs, ramp):
        if self.p["p_uniform"] is not None:
            pe = np.full(len(self.tri), ramp * self.p["p_uniform"])
        else:
            A, *_ = self.glauert(x, fs)
            dcp = self.cell_dcp(A)                                      # (ncol, ny)
            colp = (fs["q"] * fs["load"])[:, None] * dcp
            cellp = 0.5 * (colp[:-1] + colp[1:])                        # (nx, ny)
            pe = ramp * cellp[self.cell[:, 0], self.cell[:, 1]]
        xe = x[self.tri]
        nA = 0.5 * np.cross(xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0])
        fe = (pe / 3)[:, None] * nA
        f = np.zeros_like(x)
        for k in range(3):
            f[:, k] = np.bincount(self.tri.ravel(), np.repeat(fe[:, k], 3), minlength=x.shape[0])
        return f

    def body_forces(self, fs):
        f = np.zeros((self.I.size, 3))
        if self.p["flap"]:
            xn = self.x0[:, 0]
            f[:, 2] = -self.mphys * fs["dOm"] * xn
            f[:, 0] = self.mphys * fs["Om"] ** 2 * xn
        return f

    # ---------------- solver ----------------
    def _dr_masses(self):
        p = self.p
        lam = 1.35
        I1 = lam ** 2 + 2 / lam
        Et = (6 * self._W1(I1, self.C) + max(self.Efc, self.Efs)) * 2.0
        S = np.zeros(self.I.size)
        g = np.linalg.norm(self.G, axis=2)                              # (ne,3)
        ke = p["t_mem"] * self.A0[:, None] * Et * g * g.sum(1, keepdims=True)
        np.add.at(S, self.tri.ravel(), ke.ravel())
        if len(self.bars):
            Ed = 6 * self._W1(I1, DEA_YEOH) * 2.0
            kb = np.where(self.bdea, Ed * self.A_dea, np.where(self.brod, self.EA_rod, self.EA_rf)) / self.bL0 * 2
            np.add.at(S, self.b1, kb)
            np.add.at(S, self.b2, kb)
        if self.rod:
            rb = self.rb
            row = rb["W"] * (np.abs(rb["a"]) + np.abs(rb["b"]) + np.abs(rb["c"]))
            np.add.at(S, rb["im"], row * np.abs(rb["a"]))
            np.add.at(S, rb["i0"], row * np.abs(rb["b"]))
            np.add.at(S, rb["ip"], row * np.abs(rb["c"]))
        self.mdr = 0.5 * S * 1.2 + 1e-12

    def residual(self, x, fext, V):
        Vc = V if self.p["act_chord"] else 0.0
        Vs = V if self.p["act_span"] else 0.0
        fint = self.membrane(x) + self.rod_bending(x)
        fb, N, lam = self.bar_forces(x, Vc, Vs)
        R = fext - fint - fb
        R[self.fixed] = 0
        return R

    def solve(self, V, x_init=None, ramp_steps=None):
        p = self.p
        fs = self.flow_state()
        x = (self.x0 if x_init is None else x_init).copy()
        rs = p["ramp"] if ramp_steps is None else ramp_steps
        v = np.zeros_like(x)
        m = self.mdr[:, None]
        KE_prev = 0.0
        fref = p["t_mem"] * 1e4 * p["chord"] / self.ny * 0.03 + np.max(fs["q"]) * p["span"] * p["chord"] / (self.nx * self.ny) + 1e-9
        it, conv, Rmax = 0, False, np.inf
        R = None
        fbody = self.body_forces(fs)
        for it in range(1, p["max_iter"] + 1):
            ramp = min(1.0, it / rs) if rs > 0 else 1.0
            if R is None or it % p["aero_every"] == 0 or ramp < 1:
                fext = self.pressure_forces(x, fs, ramp) + ramp * fbody
            R = self.residual(x, fext, V)
            v += R / m
            KE = float(np.sum(m * v * v))
            if KE < KE_prev:
                x -= 0.5 * v                                            # step back to the kinetic-energy peak (approx.)
                v[:] = 0
                KE_prev = 0.0
                continue
            KE_prev = KE
            x += v
            if ramp >= 1 and it % 25 == 0:
                Rmax = float(np.max(np.abs(R)))
                if Rmax < p["tol"] * fref:
                    conv = True
                    break
            if not np.all(np.isfinite(x)) or np.max(np.abs(x[:, 2])) > 2 * p["chord"]:
                break
        return x, dict(iterations=it, converged=conv, residual=Rmax / fref, fs=fs)

    # ---------------- post-processing ----------------
    def evaluate(self, x, V, info):
        p, fs = self.p, info["fs"]
        Vc = V if p["act_chord"] else 0.0
        Vs = V if p["act_span"] else 0.0
        _, sig, lam_mem = self.membrane(x, want_stress=True)
        _, N, lam_b = self.bar_forces(x, Vc, Vs)
        A, camber, cl, zr = self.glauert(x, fs)
        CLcol = math.pi * (2 * A[:, 0] + A[:, 1]) * fs["load"]
        Lcol = fs["q"] * cl * CLcol
        L = float(_trapz(Lcol, fs["x"]))
        gam = np.arctan2(fs["Om"] * fs["x"], max(p["U"], 1e-9))         # inflow angle from the plunge velocity
        Phi = math.radians(p["flap_amp_deg"]) * math.cos(math.radians(p["flap_phase_deg"])) if p["flap"] else 0.0
        F_vert = float(_trapz(Lcol * np.cos(gam), fs["x"])) * math.cos(Phi)
        F_thrust = float(_trapz(-Lcol * np.sin(gam), fs["x"]))
        tip = self.nid(self.nx, 0)
        q_inf = 0.5 * p["rho_air"] * p["U"] ** 2
        CL = L / (q_inf * self.area) if q_inf > 0 else 0.0
        mem_area = self.A0 * p["pre_span"] * p["pre_chord"]
        T_chord = float(np.sum(sig[:, 1, 1] * mem_area) / mem_area.sum())
        T_span = float(np.sum(sig[:, 0, 0] * mem_area) / mem_area.sum())
        Ndea = N[self.bdea]
        E_field = Vc * np.sqrt(lam_b[self.bdea]) / p["t_act"] if self.bdea.any() else np.array([0.0])
        # measurement station: spanwise position of peak camber (mid-span can sit on a digit, where camber ~ 0)
        inner = np.arange(1, self.nx) if self.nx > 1 else np.arange(self.nx + 1)
        mid = int(inner[np.argmax(np.abs(camber[inner]))])
        cam_mean = float(_trapz(camber, self.xs) / p['span'])
        nl_c, nl_s = len(self.dea_lines['chord']), len(self.dea_lines['span'])
        Fc = float(np.mean(N[self.bdea & self.bchord])) if nl_c else 0.0
        Fs = float(np.mean(N[self.bdea & ~self.bchord])) if nl_s else 0.0
        idx = self.nid(np.arange(self.nx + 1)[:, None], np.arange(self.ny + 1)[None, :])
        return dict(
            z_mm=(x[idx, 2] * 1e3).round(5).tolist(),
            xy_mm=(x[idx, :2] * 1e3).round(4).tolist(),
            camber_pct=(camber * 100).round(4).tolist(),
            CL_col=CLcol.round(5).tolist(),
            profile_mid_mm=(zr[mid] * 1e3).round(5).tolist(),
            L_mN=L * 1e3, CL=CL, F_vert_mN=F_vert * 1e3, F_thrust_mN=F_thrust * 1e3,
            rod_tip_dy_mm=float((x[tip, 1] - self.x0[tip, 1]) * 1e3), rod_tip_dz_mm=float((x[tip, 2] - self.x0[tip, 2]) * 1e3),
            camber_mid_pct=float(camber[mid] * 100), camber_mean_pct=cam_mean * 100,
            T_chord=T_chord, T_span=T_span,
            T_chord_total=T_chord + nl_c * Fc / p['span'], T_span_total=T_span + nl_s * Fs / p['chord'],
            x_mm=(self.xs * 1e3).round(4).tolist(), y_mm=(self.ys * 1e3).round(4).tolist(),
            dea_force_mean_mN=float(np.mean(Ndea) * 1e3) if Ndea.size else 0.0,
            dea_slack=int(np.sum(Ndea <= 0)) if Ndea.size else 0,
            lam_mem_max=float(lam_mem.max()),
            dea_strain_pct=float((np.mean(lam_b[self.bdea]) / p["pre_dea"] - 1) * 100) if self.bdea.any() else 0.0,
            te_z_mid_mm=float(x[self.nid(mid, self.ny), 2] * 1e3), x_peak_mm=float(self.xs[mid] * 1e3),
            alpha_eff_75_deg=float(np.degrees(np.interp(0.75 * p["span"], self.xs, fs["alpha"]))),
            E_field_MV_m=float(np.max(E_field) / 1e6),
            iterations=info["iterations"], converged=info["converged"], residual=info["residual"],
        )


def dea_active(p):
    """DEA switched on at the current flap phase? Downstroke: Omega < 0 (phase 0-180 deg with Phi = Phi0 cos phase)."""
    if not p["flap"] or p["dea_schedule"] == "always":
        return True
    down = math.sin(math.radians(p["flap_phase_deg"])) > 0
    return down if p["dea_schedule"] == "downstroke" else not down


def run(params=None):
    """Solve DEA off (V = 0) and on (V) and return everything the web page needs (JSON-friendly)."""
    w = Wing(**(params or {}))
    p = w.p
    Von = p["V"] if dea_active(p) else 0.0
    x_off, i_off = w.solve(0.0)
    r_off = w.evaluate(x_off, 0.0, i_off)
    x_on, i_on = w.solve(Von, x_init=x_off, ramp_steps=0)
    r_on = w.evaluate(x_on, Von, i_on)
    fs = i_off["fs"]
    a_eff = np.degrees(fs["alpha"])
    warnings = []
    if np.max(np.abs(a_eff)) > 12:
        warnings.append(f"local angle of attack up to {np.max(np.abs(a_eff)):.0f} deg: thin-airfoil (attached flow) theory not valid beyond ~10-12 deg (stall)")
    if max(r_off["lam_mem_max"], r_on["lam_mem_max"]) > LAM_CAL + 0.03:
        warnings.append(f"membrane stretch {max(r_off['lam_mem_max'], r_on['lam_mem_max']):.2f} exceeds the calibrated range (1.22)")
    if r_on["dea_slack"]:
        warnings.append(f"{r_on['dea_slack']} DEA segments slack at {p['V']:.0f} V: actuation saturates (increase DEA pre-stretch)")
    if r_on["E_field_MV_m"] * 1e6 > p["E_bd"]:
        warnings.append(f"DEA field {r_on['E_field_MV_m']:.0f} V/um exceeds breakdown ({p['E_bd'] / 1e6:.0f} V/um)")
    if not (i_off["converged"] and i_on["converged"]):
        warnings.append("solver did not fully converge (possible aeroelastic divergence or very slack membrane): treat results with caution")
    if p["flap"] and p["U"] > 0:
        k = math.pi * p["flap_f"] * p["chord"] / p["U"]
        if k > 0.2:
            warnings.append(f"reduced frequency k = {k:.2f} > 0.2: quasi-steady aerodynamics underestimates unsteady effects")
    if p["U"] < 0.5:
        warnings.append("flow speed below 0.5 m/s: aerodynamic load negligible")
    if Von == 0 and p["V"] > 0:
        warnings.append(f"DEA schedule '{p['dea_schedule']}': DEAs are off at this flap phase")
    if not p["clamp_TE"] and not p["te_cord"] and p["U"] > 0.5:
        warnings.append("free trailing edge without cord: no static equilibrium may exist (the edge carries no tension; a real membrane would luff)")
    return dict(
        off=r_off, on=r_on, params={k: v for k, v in p.items()},
        alpha_eff_deg=a_eff.round(3).tolist(), q_col=fs["q"].round(4).tolist(),
        flap=dict(Om=fs["Om"], dOm=fs["dOm"]),
        dea_lines={k: v for k, v in w.dea_lines.items()}, rf_lines={k: v for k, v in w.rf_lines.items()},
        dig_lines=list(map(int, w.dig_lines)),
        m_dea_g=w.m_dea * 1e3, m_total_g=w.m_total * 1e3,
        d=dict(camber_mid_pct=r_on["camber_mid_pct"] - r_off["camber_mid_pct"],
               camber_mean_pct=r_on["camber_mean_pct"] - r_off["camber_mean_pct"],
               T_chord_total_pct=100 * (r_on["T_chord_total"] / r_off["T_chord_total"] - 1) if r_off["T_chord_total"] else 0.0,
               T_span_total_pct=100 * (r_on["T_span_total"] / r_off["T_span_total"] - 1) if r_off["T_span_total"] else 0.0,
               L_mN=r_on["L_mN"] - r_off["L_mN"], CL=r_on["CL"] - r_off["CL"],
               T_chord_pct=100 * (r_on["T_chord"] / r_off["T_chord"] - 1) if r_off["T_chord"] else 0.0,
               T_span_pct=100 * (r_on["T_span"] / r_off["T_span"] - 1) if r_off["T_span"] else 0.0,
               authority_mN_per_g=(r_on["L_mN"] - r_off["L_mN"]) / (w.m_dea * 1e3) if w.m_dea > 0 else 0.0),
        warnings=warnings,
    )


SCALARS = ("L_mN", "F_vert_mN", "F_thrust_mN", "rod_tip_dy_mm", "rod_tip_dz_mm", "CL", "camber_mid_pct", "camber_mean_pct", "T_chord", "T_span", "T_chord_total", "T_span_total",
           "dea_force_mean_mN", "dea_strain_pct", "te_z_mid_mm", "alpha_eff_75_deg", "lam_mem_max", "converged")


def sweep(params, kind, values, voltages=(0.0,), aoas=None, progress=None):
    """Experiment-style sweeps with warm starts. Returns {kind, values, curves: {label: {quantity: [...]}}}.
    kind = 'voltage'  : values = voltages, one curve per pitch angle in aoas
           'aoa'      : values = pitch angles, one curve per voltage
           'speed'    : values = flow speeds, one curve per voltage
           'phase'    : values = flap phases, curves per pitch angle x voltage (DEA schedule from params)
           'schedule' : values = flap phases, curves = DEA off / always / downstroke only / upstroke only at max(voltages)
    For phase sweeps the cycle mean of lift, vertical force and thrust is added (phases sampled uniformly)."""
    base = dict(DEFAULTS); base.update(params or {})
    aoas = [base["alpha_deg"]] if aoas is None else list(aoas)
    if kind == "voltage":
        curves = [(a, None, None) for a in aoas]
    elif kind == "schedule":
        Vm = max(voltages) if len(voltages) else base["V"]
        curves = [(aoas[0], 0.0, "always"), (aoas[0], Vm, "always"), (aoas[0], Vm, "downstroke"), (aoas[0], Vm, "upstroke")]
    else:
        curves = [(a, V, base["dea_schedule"]) for a in aoas for V in voltages]
    total, done, out = len(curves) * len(values), 0, {}
    for a, V, sched in curves:
        pp = dict(base); pp["alpha_deg"] = a
        if kind in ("phase", "schedule"):
            pp["flap"] = True
            pp["dea_schedule"] = sched
        w = Wing(**pp)
        x = None
        rec = {k: [] for k in SCALARS}
        for v in values:
            if kind == "voltage":
                Vv = float(v)
            else:
                w.p[{"aoa": "alpha_deg", "phase": "flap_phase_deg", "schedule": "flap_phase_deg", "speed": "U"}[kind]] = float(v)
                Vv = float(V) if dea_active(w.p) else 0.0
            if x is None:
                x, _ = w.solve(0.0)
            x, info = w.solve(Vv, x_init=x, ramp_steps=0)
            r = w.evaluate(x, Vv, info)
            for k in SCALARS:
                rec[k].append(r[k])
            done += 1
            if progress:
                progress(done, total)
        if kind == "voltage":
            label = f"pitch {a:g} deg"
        elif kind == "schedule":
            label = "DEA off" if V == 0 else f"{V:g} V, " + {"always": "always on", "downstroke": "downstroke only", "upstroke": "upstroke only"}[sched]
        else:
            label = f"pitch {a:g} deg, {V:g} V"
        if kind in ("phase", "schedule"):
            vals = np.asarray(values, float)
            keep = ~np.isclose(np.mod(vals, 360), np.mod(vals[0], 360)) | (np.arange(len(vals)) == 0)
            rec["cycle_mean"] = {q: float(np.mean(np.asarray(rec[q])[keep])) for q in ("L_mN", "F_vert_mN", "F_thrust_mN")}
        out[label] = rec
    return dict(kind=kind, values=list(map(float, values)), curves=out)


if __name__ == "__main__":
    import time, json
    t = time.time()
    r = run()
    print(f"{time.time() - t:.2f} s")
    for s in ("off", "on"):
        o = r[s]
        print(s, {k: round(o[k], 4) if isinstance(o[k], float) else o[k] for k in
                  ("L_mN", "CL", "camber_mid_pct", "T_chord", "T_span", "dea_force_mean_mN", "dea_slack", "lam_mem_max", "iterations", "converged", "residual")})
    print(r["d"], r["m_dea_g"], r["warnings"])
