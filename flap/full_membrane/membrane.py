"""Full active membrane wing: two tensioned DEA panels, rolled edge fibres, batten, LE rod, 3D vortex-lattice loading.

Physics (all SI units; x spanwise root to tip, y chordwise LE to TE, z up):
  Sheet      : incompressible Yeoh membrane (Elastosil P7670, fit_p7670.py), constant-strain triangles, total Lagrangian.
               With l1 >= l2 the eigenvalues of C and P = eta n_a eps0 eps_r V^2 / H (N/m), the resultant 2nd Piola-Kirchhoff
               stress is  t S = 2 t W1 (I - C^-1 / det C) - P adj(C),  the gradient of  U = t W(I1) - P det(C) / 2.
               The -P adj(C) term is the Maxwell stress of an active layer at fixed voltage, thinning included.
  Wrinkling  : tension-field relaxation. If the smaller principal stress is compressive, the transverse stretch relaxes to
               the l2w that makes it zero (uniaxial state along the larger stretch, with the Maxwell term retained);
               if no such state carries tension the element is slack. Forces stay the gradient of the relaxed energy.
  Fibres     : rolled edge strip, a 1D Yeoh bar, tension only: N = 2 A (lam - lam^-2) W1 - eta_f n_f eps0 eps_r (V/H)^2 H w_a.
  Frame      : LE rod, batten (3D Euler-Bernoulli frame, frame_K) coupled by fixed-point iteration on the reactions, or rigid.
  Aero       : steady vortex-ring lattice in the plane z = 0 with a mirror plane at the root; surface slope enters the
               boundary condition; attached flow, no unsteady terms. Pressure is a follower load on the deformed sheet.
  Solver     : dynamic relaxation with kinetic damping (robust through slack and wrinkled states).
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from matplotlib.tri import Triangulation

EPS0 = 8.854e-12
YEOH = (42842.1, 447.0, 0.0)   # Elastosil P7670, fitted on the rolled-DEA bundle at 0 V (../plagiopatagiales/fit_p7670.py)

DEFAULTS = dict(
    # planform as tensioned (m). A root LE, D root TE anchor, C batten tip, T fibre 2 tip anchor, E LE tip
    A=(3.8e-3, 6.4e-3), D=(15.0e-3, 52.5e-3), C=(55.0e-3, 52.5e-3), E=(108.0e-3, 6.4e-3), T=(108.0e-3, 9.5e-3),
    theta_deg=14.0,                 # batten tilt from the chordwise axis; the LE end moves inboard
    h=2e-3,                         # target mesh size
    # sheet
    t=70e-6, H=35e-6, eps_r=2.8, n_a=1, eta_s=0.4,
    lam_d=(1.05, 1.05),             # stretch along each panel's fibre
    lam_n=(None, None),             # stretch across the fibre; None = natural necking lam_d^-1/2 (uniaxial, N_n = 0)
    # fibres: strip width as cut (reference), passive fraction of the width excluded, rolled active films
    w_s=(16.5e-3, 16.5e-3), w_a_frac=0.8, n_f=2, eta_f=0.3, active_fibre=False,
    cord_EA=(0.0, 0.0), cord_pre=(0.05, 0.05),   # optional passive cord in the roll: axial stiffness (N) and pre-strain
    # flow
    U=4.0, alpha_deg=6.0, rho_air=1.225, na_x=36, na_y=16, na_cos=True,
    # frame
    fix_te=False, rigid_frame=True, rod_od=3e-3, rod_id=2e-3, rod_od_tip=None, rod_id_tip=None, rod_E=120e9, rod_G=5e9,
    batten_d=0.8e-3, batten_E=120e9, batten_G=5e9, frame_relax=0.7, frame_iter=12,
    # solver
    max_iter=60000, tol=2e-7, aero_every=20,
)


def coons(a, b, c, d, ni, nj):
    """Bilinear patch a(0,0) b(1,0) c(1,1) d(0,1); returns (ni+1, nj+1, 2)."""
    s = np.linspace(0, 1, ni + 1)[:, None, None]
    u = np.linspace(0, 1, nj + 1)[None, :, None]
    a, b, c, d = [np.asarray(q, float)[None, None, :] for q in (a, b, c, d)]
    return (1 - s) * (1 - u) * a + s * (1 - u) * b + s * u * c + (1 - s) * u * d


def homogeneous_F(d_vec, lam_d, lam_n):
    d = np.asarray(d_vec, float) / np.hypot(*d_vec)
    n = np.array([-d[1], d[0]])
    return lam_d * np.outer(d, d) + lam_n * np.outer(n, n)


def W1_of(I1):
    e = I1 - 3
    return YEOH[0] + 2 * YEOH[1] * e + 3 * YEOH[2] * e ** 2


def W_of(I1):
    e = I1 - 3
    return YEOH[0] * e + YEOH[1] * e ** 2 + YEOH[2] * e ** 3


def sheet_stress(c00, c11, c01, t, P):
    """Relaxed resultant stress t S and state: 0 taut, 1 wrinkled, 2 slack. Also returns principal Cauchy resultants."""
    tr, dd = 0.5 * (c00 + c11), np.sqrt(0.25 * (c00 - c11) ** 2 + c01 ** 2)
    l1, l2 = tr + dd, np.maximum(tr - dd, 1e-9)
    det = l1 * l2
    W1 = W1_of(l1 + l2 + 1 / det)
    S2 = 2 * t * W1 * (1 - 1 / (l2 * det)) - P * l1
    S00 = 2 * t * W1 * (1 - c11 / det ** 2) - P * c11
    S11 = 2 * t * W1 * (1 - c00 / det ** 2) - P * c00
    S01 = 2 * t * W1 * c01 / det ** 2 + P * c01
    S1p = 2 * t * W1 * (1 - 1 / (l1 * det)) - P * l2
    state = np.zeros_like(c00, dtype=np.int8)
    N1, N2 = S1p * l1 / np.sqrt(det), S2 * l2 / np.sqrt(det)
    wr = S2 < -1e-9                                           # exact necking (S2 = 0 to round-off) stays in the taut branch
    if np.any(wr):
        l1w, l2o = l1[wr], l2[wr]
        Pw = P[wr] if np.ndim(P) else np.full(l1w.shape, float(P))
        lw = l2o.copy()
        ok = np.ones(l1w.shape, bool)
        for _ in range(3):                                    # W1 varies slowly with stretch: fixed point on l2w
            Wk = W1_of(l1w + lw + 1 / (l1w * lw))
            g = 1 - Pw * l1w / (2 * t * Wk)
            ok = g > 1e-6
            lw = np.where(ok, np.sqrt(1 / (l1w * np.where(ok, g, 1.0))), lw)
        Wk = W1_of(l1w + lw + 1 / (l1w * lw))
        S1 = 2 * t * Wk * (1 - 1 / (l1w ** 2 * lw)) - Pw * lw
        ok &= S1 > 0
        S1 = np.where(ok, S1, 0.0)
        th = 0.5 * np.arctan2(2 * c01[wr], c00[wr] - c11[wr])
        co, si = np.cos(th), np.sin(th)
        S00[wr], S11[wr], S01[wr] = S1 * co * co, S1 * si * si, S1 * co * si
        state[wr] = np.where(ok, 1, 2)
        N1[wr] = S1 * l1w / np.sqrt(l1w * lw)
        N2[wr] = 0.0
    return S00, S11, S01, state, N1, N2


def sheet_energy(c00, c11, c01, t, P):
    """Relaxed energy density per reference area, consistent with sheet_stress (used only to verify the forces)."""
    tr, dd = 0.5 * (c00 + c11), np.sqrt(0.25 * (c00 - c11) ** 2 + c01 ** 2)
    l1, l2 = tr + dd, np.maximum(tr - dd, 1e-9)
    S2 = 2 * t * W1_of(l1 + l2 + 1 / (l1 * l2)) * (1 - 1 / (l2 * l1 * l2)) - P * l1
    lw = l2.copy()
    wr = S2 < 0
    if np.any(wr):
        lk = l2[wr]
        Pw = P[wr] if np.ndim(P) else P
        for _ in range(30):
            g = 1 - Pw * l1[wr] / (2 * t * W1_of(l1[wr] + lk + 1 / (l1[wr] * lk)))
            lk = np.sqrt(1 / (l1[wr] * g))
        lw[wr] = lk
    U = t * W_of(l1 + lw + 1 / (l1 * lw)) - 0.5 * P * l1 * lw
    # slack branch: the relaxed energy is flat at its value in the natural state under P (equibiaxial, no stress)
    Pa = np.full(l1.shape, float(P)) if not np.ndim(P) else P
    S1 = 2 * t * W1_of(l1 + lw + 1 / (l1 * lw)) * (1 - 1 / (l1 ** 2 * lw)) - Pa * lw
    sl = wr & (S1 <= 0)
    if np.any(sl):
        ln = np.ones(sl.sum())
        for _ in range(60):                                   # 2 t W1 (1 - l^-3) = P l, by fixed point
            ln = (1 - Pa[sl] * ln / (2 * t * W1_of(2 * ln + ln ** -2))) ** (-1 / 3)
        U[sl] = t * W_of(2 * ln + ln ** -2) - 0.5 * Pa[sl] * ln ** 2
    return U


# ---------------- vortex lattice ----------------
def _seg_w(px, py, ax, ay, bx, by, core=1e-9):
    """z-velocity at in-plane points p from a unit-strength segment a->b in the plane z = 0."""
    r1x, r1y, r2x, r2y = px - ax, py - ay, px - bx, py - by
    cr = r1x * r2y - r1y * r2x
    n1, n2 = np.hypot(r1x, r1y), np.hypot(r2x, r2y)
    r0x, r0y = bx - ax, by - ay
    dot = r0x * (r1x / np.maximum(n1, core) - r2x / np.maximum(n2, core)) + r0y * (r1y / np.maximum(n1, core) - r2y / np.maximum(n2, core))
    return dot * cr / (cr ** 2 + core ** 2) / (4 * math.pi)


class VLM:
    """Vortex rings on a planform given by LE and TE curves; mirror plane at x = x[0]. Flow along +y."""

    def __init__(self, xs, y_le, y_te, ny, cosine=False):
        self.xs, self.ny = np.asarray(xs), ny
        nx = len(xs) - 1
        self.nx = nx
        f = 0.5 * (1 - np.cos(np.linspace(0, math.pi, ny + 1))) if cosine else np.linspace(0, 1, ny + 1)
        self.f = f
        Xg = np.repeat(self.xs[:, None], ny + 1, 1)
        Yg = y_le[:, None] + (y_te - y_le)[:, None] * f[None, :]
        self.Xg, self.Yg = Xg, Yg
        dy = np.diff(Yg, axis=1)                                  # rings shifted back by a quarter panel
        Xr, Yr = Xg.copy(), np.c_[Yg[:, :-1] + 0.25 * dy, Yg[:, -1] + 0.25 * dy[:, -1]]
        xc = 0.5 * (Xg[:-1, :-1] + Xg[1:, :-1])
        yc = 0.25 * (Yg[:-1, :-1] + Yg[1:, :-1]) + 0.25 * (Yg[:-1, 1:] + Yg[1:, 1:]) + 0.25 * (0.5 * (dy[:-1] + dy[1:]))
        self.xc, self.yc = xc.ravel(), yc.ravel()
        self.dyc = (0.5 * (dy[:-1] + dy[1:])).ravel()
        self.dx = np.repeat(np.diff(self.xs)[:, None], ny, 1).ravel()
        self.area = np.sum(self.dx * self.dyc)
        n = nx * ny
        A = np.zeros((n, n))
        W = 100 * (y_te - y_le).max()
        px, py = self.xc[:, None], self.yc[:, None]
        for sgn, mir in ((1.0, 1.0), (-1.0, -1.0)):
            X = self.xs[0] + mir * (Xr - self.xs[0])
            a = np.zeros((n, n))
            k = 0
            for i in range(nx):
                for j in range(ny):
                    x1, x2 = X[i, j], X[i + 1, j]
                    y1a, y2a, y1b, y2b = Yr[i, j], Yr[i + 1, j], Yr[i, j + 1], Yr[i + 1, j + 1]
                    v = _seg_w(px, py, x1, y1a, x2, y2a) + _seg_w(px, py, x2, y2a, x2, y2b) + \
                        _seg_w(px, py, x2, y2b, x1, y1b) + _seg_w(px, py, x1, y1b, x1, y1a)
                    if j == ny - 1:                           # trailing wake: closed ring far downstream
                        v += _seg_w(px, py, x1, y1b, x2, y2b) + _seg_w(px, py, x2, y2b, x2, y2b + W) + \
                            _seg_w(px, py, x2, y2b + W, x1, y1b + W) + _seg_w(px, py, x1, y1b + W, x1, y1b)
                    a[:, k] = v[:, 0]
                    k += 1
            A += sgn * a
        self.Ainv = np.linalg.inv(A)

    def solve(self, U, alpha, slope=None):
        """Panel pressure jump per unit rho U^2, from slope dz/dy at the control points."""
        s = np.zeros(self.xc.size) if slope is None else slope
        G = (self.Ainv @ (s - alpha)).reshape(self.nx, self.ny)
        d = np.diff(np.c_[np.zeros(self.nx), G], axis=1)
        return -d.ravel() / self.dyc                          # Dp = rho U^2 * this; rings bound along +x, flow +y: lift = -Gamma

    def CL(self, dp_hat):
        return 2 * np.sum(dp_hat * self.dx * self.dyc) / self.area


# ---------------- wing ----------------
class Wing:
    def __init__(self, panels=(0, 1), **kw):
        p = dict(DEFAULTS)
        p.update(kw)
        self.p = p
        self.panels = panels
        A, D, C, E, T = [np.array(p[k], float) for k in "ADCET"]
        B = np.array([C[0] - (C[1] - A[1]) * math.tan(math.radians(p["theta_deg"])), A[1]])
        self.pts = dict(A=A, B=B, C=C, D=D, E=E, T=T)
        h = p["h"]
        nj = max(int(round(np.hypot(*(C - B)) / h)), 4)
        ni1 = max(int(round(0.5 * (np.hypot(*(B - A)) + np.hypot(*(C - D))) / h)), 4)
        ni2 = max(int(round(0.5 * (np.hypot(*(E - B)) + np.hypot(*(T - C))) / h)), 6)
        nodes, tri, tp = [], [], []
        idx = {}

        def add_patch(k, grid):
            ni_, nj_ = grid.shape[0] - 1, grid.shape[1] - 1
            ids = np.zeros((ni_ + 1, nj_ + 1), int)
            for i in range(ni_ + 1):
                for j in range(nj_ + 1):
                    key = tuple(np.round(grid[i, j] * 1e7).astype(int))
                    if key not in idx:
                        idx[key] = len(nodes)
                        nodes.append(grid[i, j])
                    ids[i, j] = idx[key]
            for i in range(ni_):
                for j in range(nj_):
                    a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
                    tri_q = [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
                    tri.extend(tri_q)
                    tp.extend([k, k])
            return ids

        self.ids = {}
        if 0 in panels:
            self.ids[0] = add_patch(0, coons(A, B, C, D, ni1, nj))
        if 1 in panels:
            self.ids[1] = add_patch(1, coons(B, E, T, C, ni2, nj))
        self.x0 = np.c_[np.array(nodes), np.zeros(len(nodes))]
        self.tri = np.array(tri)
        self.tp = np.array(tp)
        self.tri_flat = self.tri.ravel()
        # per-panel homogeneous pre-stretch: reference triangle = F^-1 (target triangle). This inverse map IS the mask.
        self.dirs = {0: C - D, 1: T - C}
        self.Finv = {}
        G = np.zeros((len(tri), 3, 2))
        A0 = np.zeros(len(tri))
        for k in panels:
            ld = p["lam_d"][k]
            ln = p["lam_n"][k] if p["lam_n"][k] is not None else ld ** -0.5
            Fi = np.linalg.inv(homogeneous_F(self.dirs[k], ld, ln))
            self.Finv[k] = Fi
            e = self.tp == k
            Pp = self.x0[self.tri[e]][:, :, :2] @ Fi.T
            x1, y1, x2, y2, x3, y3 = Pp[:, 0, 0], Pp[:, 0, 1], Pp[:, 1, 0], Pp[:, 1, 1], Pp[:, 2, 0], Pp[:, 2, 1]
            A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            A0[e] = 0.5 * np.abs(A2)
            G[e, 0, 0], G[e, 1, 0], G[e, 2, 0] = (y2 - y3) / A2, (y3 - y1) / A2, (y1 - y2) / A2
            G[e, 0, 1], G[e, 1, 1], G[e, 2, 1] = (x3 - x2) / A2, (x1 - x3) / A2, (x2 - x1) / A2
        self.G, self.A0 = G, A0
        # supports
        fixed = np.zeros(len(nodes), bool)
        self.fib = {}
        self.le_nodes, self.seam, self.root = [], [], []
        if 0 in panels:
            g = self.ids[0]
            fixed[g[:, 0]] = True; fixed[g[0, :]] = True; fixed[g[-1, :]] = True
            self.fib[0] = g[:, -1]
            self.le_nodes += list(g[:, 0])
            self.seam = list(g[-1, :])
            self.root = list(g[0, :])
        if 1 in panels:
            g = self.ids[1]
            fixed[g[:, 0]] = True; fixed[g[0, :]] = True; fixed[g[-1, :]] = True
            self.fib[1] = g[:, -1]
            self.le_nodes += list(g[1 if 0 in panels else 0:, 0])
            self.seam = list(g[0, :])
            self.tip = list(g[-1, :])
        if p["fix_te"]:
            for ids in self.fib.values():
                fixed[ids] = True
        self.fixed = fixed
        # fibre bars, stretched with their panel's lam_d (rolled with the sheet)
        b1, b2, L0, Ab, dF, EAc, cpre, bp = [], [], [], [], [], [], [], []
        for k, ids in self.fib.items():
            w = p["w_s"][k]
            for n1, n2 in zip(ids[:-1], ids[1:]):
                b1.append(n1); b2.append(n2)
                L0.append(np.linalg.norm(self.x0[n2] - self.x0[n1]) / p["lam_d"][k])
                Ab.append(w * p["t"])
                EAc.append(p["cord_EA"][k]); cpre.append(p["cord_pre"][k]); bp.append(k)
        self.b1, self.b2 = np.array(b1, int), np.array(b2, int)
        self.bL0, self.bA = np.array(L0), np.array(Ab)
        self.bEAc, self.bcpre, self.bp = np.array(EAc), np.array(cpre), np.array(bp, int)
        self._masses()
        self.vlm = None
        self.pe = None
        self.f_ext = None                                     # known nodal loads: verification, and flapping inertia
        self.flap = None

    def set_flap(self, Om, dOm, rho_s=1100.0):
        """Stroke rate and acceleration (rad/s, rad/s^2): plunge load -m dOm r and centrifugal load m Om^2 r on the sheet and fibres."""
        self.flap = dict(Om=Om, dOm=dOm)
        m = np.zeros(self.x0.shape[0])
        np.add.at(m, self.tri_flat, np.repeat(rho_s * self.p["t"] * self.A0 / 3, 3))
        if self.b1.size:
            np.add.at(m, self.b1, 0.5 * rho_s * self.bA * self.bL0)
            np.add.at(m, self.b2, 0.5 * rho_s * self.bA * self.bL0)
        r = self.x0[:, 0] - self.pts["A"][0]
        self.f_ext = np.c_[m * Om ** 2 * r, np.zeros_like(m), -m * dOm * r]

    # ---------------- forces ----------------
    def P_e(self, V):
        """Maxwell resultant per element (N/m). V is a scalar or (V_panel1, V_panel2)."""
        p = self.p
        V = (V, V) if np.isscalar(V) else V
        Pv = np.array([p["eta_s"] * p["n_a"] * EPS0 * p["eps_r"] * v ** 2 / p["H"] for v in V])
        return Pv[self.tp]

    def membrane(self, x, Pe, want=False):
        xe = x[self.tri]
        G0, G1 = self.G[:, :, 0:1], self.G[:, :, 1:2]
        F0 = np.sum(xe * G0, 1)
        F1 = np.sum(xe * G1, 1)
        c00, c11, c01 = np.sum(F0 * F0, 1), np.sum(F1 * F1, 1), np.sum(F0 * F1, 1)
        S00, S11, S01, st, N1, N2 = sheet_stress(c00, c11, c01, self.p["t"], Pe)
        P0 = F0 * S00[:, None] + F1 * S01[:, None]
        P1 = F0 * S01[:, None] + F1 * S11[:, None]
        tA = self.A0[:, None, None]
        fe = tA * (P0[:, None, :] * G0 + P1[:, None, :] * G1)
        f = np.empty_like(x)
        for k in range(3):
            f[:, k] = np.bincount(self.tri_flat, fe[:, :, k].ravel(), minlength=x.shape[0])
        if not want:
            return f
        return f, dict(state=st, N1=N1, N2=N2, c00=c00, c11=c11, c01=c01, F0=F0, F1=F1)

    def bars(self, x, V):
        p = self.p
        d = x[self.b2] - x[self.b1]
        l = np.linalg.norm(d, axis=1)
        lam = l / self.bL0
        Vb = np.array((V, V) if np.isscalar(V) else V, float)[self.bp]
        dF = p["eta_f"] * p["n_f"] * EPS0 * p["eps_r"] * Vb ** 2 / p["H"] * p["w_a_frac"] * np.array(p["w_s"])[self.bp] if p["active_fibre"] else 0.0
        Ns = np.maximum(self.bA * 2 * (lam - lam ** -2) * W1_of(lam ** 2 + 2 / lam) - dF, 0.0)
        Nc = np.maximum(self.bEAc * (lam / np.array(p["lam_d"])[self.bp] - 1 + self.bcpre), 0.0)
        N = Ns + Nc
        e = d / l[:, None]
        f = np.zeros_like(x)
        for k in range(3):
            f[:, k] = np.bincount(self.b2, N * e[:, k], minlength=x.shape[0]) - np.bincount(self.b1, N * e[:, k], minlength=x.shape[0])
        return f, N, lam

    def pressure(self, x, pe):
        xe = x[self.tri]
        nA = 0.5 * np.cross(xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0])
        nA = nA * np.sign(nA[:, 2:3] + 1e-30)                   # upward normal for positive pressure
        fe = (pe / 3)[:, None] * nA
        f = np.zeros_like(x)
        for k in range(3):
            f[:, k] = np.bincount(self.tri_flat, np.repeat(fe[:, k], 3), minlength=x.shape[0])
        return f

    def _masses(self):
        g = np.linalg.norm(self.G, axis=2)
        Et = 2 * 12 * YEOH[0] * 1.5
        ke = self.p["t"] * self.A0[:, None] * Et * g * g.sum(1, keepdims=True)
        S = np.zeros(self.x0.shape[0])
        np.add.at(S, self.tri.ravel(), ke.ravel())
        if self.b1.size:
            kb = (6 * YEOH[0] * self.bA * 2 * 1.5 + self.bEAc) / self.bL0 * 2
            np.add.at(S, self.b1, kb)
            np.add.at(S, self.b2, kb)
        self.m = 0.6 * S + 1e-12

    def _rhs(self, x, Pe, V, pe):
        f = -self.membrane(x, Pe)
        if self.b1.size:
            f -= self.bars(x, V)[0]
        if pe is not None:
            f += self.pressure(x, pe)
        if self.f_ext is not None:
            f += self.f_ext
        return f

    # ---------------- aero coupling ----------------
    def build_aero(self, x):
        """Lattice on the current planform (TE follows the bowed fibres) and its sparse map to the mesh."""
        p = self.p
        order = [self.fib[k] for k in self.panels]
        te = np.concatenate([x[o, :2] for o in order])
        xr = self.pts["A"][0]
        # root edge from A to D closes the planform inboard of the TE
        A, D = self.pts["A"], self.pts["D"]
        xte, yte = np.r_[A[0], te[:, 0]], np.r_[A[1], te[:, 1]]
        s = np.argsort(xte)
        xte, yte = xte[s], yte[s]
        xs = xr + (self.pts["E"][0] - xr) * (1 - np.cos(np.linspace(0, math.pi / 2, p["na_x"] + 1)))
        y_te = np.interp(xs, xte, yte)
        y_le = np.full_like(xs, A[1])
        y_te = np.maximum(y_te, y_le + 1.5e-3)
        self.vlm = VLM(xs, y_le, y_te, p["na_y"], p["na_cos"])
        self.tri2d = Triangulation(x[:, 0], x[:, 1], self.tri)
        self.tf = self.tri2d.get_trifinder()
        v = self.vlm
        eps = 0.5 * v.dyc                                     # slope averaged over one panel length around the control point
        self.Gs = self._interp(v.xc, v.yc + eps, x) - self._interp(v.xc, v.yc - eps, x)
        self.Gs = sp.diags(1 / (2 * eps)) @ self.Gs
        # lattice panel -> triangle: each triangle averages the panels under 28 interior sample points, so load is not aliased
        nb = 6
        bary = np.array([(i + 1 / 3, j + 1 / 3) for i in range(nb) for j in range(nb - i)] + [(i + 2 / 3, j + 2 / 3) for i in range(nb - 1) for j in range(nb - 1 - i)]) / nb
        xe = x[self.tri][:, :, :2]
        rows, cols, vals = [], [], []
        for a, b in bary:
            q = xe[:, 0] * (1 - a - b) + xe[:, 1] * a + xe[:, 2] * b
            ix = np.clip(np.searchsorted(xs, q[:, 0]) - 1, 0, v.nx - 1)
            yl, yt = np.interp(q[:, 0], xs, y_le), np.interp(q[:, 0], xs, y_te)
            jx = np.clip(np.searchsorted(v.f, (q[:, 1] - yl) / (yt - yl)) - 1, 0, v.ny - 1)
            rows.append(np.arange(len(self.tri))); cols.append(ix * v.ny + jx); vals.append(np.full(len(self.tri), 1 / len(bary)))
        self.W_tp = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(self.tri), v.nx * v.ny))

    def _interp(self, px, py, x):
        tt = self.tf(px, py)
        I, J, V_ = [], [], []
        for k, (a, b, t_) in enumerate(zip(px, py, tt)):
            if t_ < 0:
                n = int(np.argmin((x[:, 0] - a) ** 2 + (x[:, 1] - b) ** 2))
                I.append(k); J.append(n); V_.append(1.0)
                continue
            q = self.tri[t_]
            T_ = np.c_[x[q[1], :2] - x[q[0], :2], x[q[2], :2] - x[q[0], :2]]
            l = np.linalg.solve(T_, np.array([a, b]) - x[q[0], :2])
            for n, w in zip(q, (1 - l.sum(), l[0], l[1])):
                I.append(k); J.append(n); V_.append(w)
        return sp.csr_matrix((V_, (I, J)), shape=(len(px), x.shape[0]))

    def aero(self, x):
        """Element pressures and CL from the current shape."""
        p = self.p
        slope = self.Gs @ x[:, 2]
        alpha = math.radians(p["alpha_deg"])
        scale = 1.0
        if self.flap is not None:                             # quasi-steady stroke, as Eq. S12: local incidence and dynamic pressure
            r = self.vlm.xc - self.pts["A"][0]
            alpha = alpha - np.arctan(self.flap["Om"] * r / p["U"])
            scale = 1 + (self.flap["Om"] * r / p["U"]) ** 2
        dp_hat = self.vlm.solve(p["U"], alpha, slope) * scale
        dp = p["rho_air"] * p["U"] ** 2 * dp_hat
        return self.W_tp @ dp, self.vlm.CL(dp_hat)

    # ---------------- solver ----------------
    def relax(self, V=0.0, x_init=None, aero=False, uniform_p=None, fixed=None, tol=None, max_iter=None, fixed_x=None):
        p = self.p
        tol = p["tol"] if tol is None else tol
        max_iter = p["max_iter"] if max_iter is None else max_iter
        x = (self.x0 if x_init is None else x_init).copy()
        fx = self.fixed if fixed is None else fixed
        if fixed_x is not None:
            x[fx] = fixed_x[fx]
        fr = ~fx
        Pe = self.P_e(V)
        v = np.zeros_like(x)
        m = self.m[:, None]
        pe = None if uniform_p is None else np.full(len(self.tri), float(uniform_p))
        cl = None
        if aero:
            pe, cl = self.aero(x)
        KEo = 0.0
        rmax = np.inf
        ok = False
        for it in range(max_iter):
            if aero and it % p["aero_every"] == 0:
                pe, cl = self.aero(x)
            R = self._rhs(x, Pe, V, pe)
            R[fx] = 0
            v += R / m
            KE = np.sum(m * v * v)
            if KE < KEo:
                x -= v                                       # step back to about the kinetic energy peak
                v[:] = 0
                KEo = 0.0
                continue
            KEo = KE
            x += v
            if it % 50 == 0:
                rmax = np.abs(R[fr]).max()
                zmax = np.abs(x[:, 2]).max()
                if zmax > 0.05:
                    return dict(x=x, ok=False, it=it, rmax=rmax, CL=cl, diverged=True, pe=pe)
                if rmax < tol and it > 100:
                    ok = True
                    break
        if aero:
            pe, cl = self.aero(x)
        return dict(x=x, ok=ok, it=it, rmax=rmax, CL=cl, diverged=False, pe=pe)

    # ---------------- diagnostics ----------------
    def field(self, x, V=0.0):
        f, d = self.membrane(x, self.P_e(V), want=True)
        return d

    def fibre_report(self, x, V=0.0):
        out = {}
        _, N, lam = self.bars(x, V)
        for k, ids in self.fib.items():
            a, b = x[ids[0], :2], x[ids[-1], :2]
            u = (b - a) / np.linalg.norm(b - a)
            nrm = np.array([-u[1], u[0]])
            sag = np.abs((x[ids, :2] - a) @ nrm).max()
            out[k] = dict(F=float(N[self.bp == k].mean()), lam=float(lam[self.bp == k].mean()), sag=float(sag),
                          span=float(np.linalg.norm(b - a)), sag_frac=float(sag / np.linalg.norm(b - a)),
                          w_edge=float(np.abs(x[ids, 2]).max()))
        return out

    def summary(self, x, V=0.0):
        d = self.field(x, V)
        s = {}
        for k in self.panels:
            e = self.tp == k
            A = self.A0[e]
            s[k] = dict(taut=float(A[d["state"][e] == 0].sum() / A.sum()), wrinkled=float(A[d["state"][e] == 1].sum() / A.sum()),
                        slack=float(A[d["state"][e] == 2].sum() / A.sum()), N1=float(np.sum(A * d["N1"][e]) / A.sum()),
                        N2=float(np.sum(A * d["N2"][e]) / A.sum()), N1_cv=float(np.std(d["N1"][e]) / (np.abs(np.mean(d["N1"][e])) + 1e-12)),
                        wmax=float(np.abs(x[np.unique(self.tri[e]), 2]).max()))
        return s

    def reactions(self, x, V=0.0, pe=None):
        """Force exerted by the sheet and fibres on each supported node (N)."""
        return self._rhs(x, self.P_e(V), V, pe)

    # ---------------- masks ----------------
    def mask_panel(self, k):
        """Reference (as-cut, flat) outline of the sheet and of the strip to roll: corners from F^-1, in mm, in the panel's frame."""
        p = self.p
        Fi = self.Finv[k]
        P = self.pts
        corners = dict(root=(P["A"], P["B"], P["C"], P["D"]), tip=(P["B"], P["E"], P["T"], P["C"]))["root" if k == 0 else "tip"]
        ref = np.array([Fi @ c for c in corners])
        a, b = ref[3], ref[2]                                  # fibre line in reference
        u = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-u[1], u[0]])
        if nrm @ (a - ref[0]) < 0:
            nrm = -nrm
        w = p["w_s"][k]
        return dict(sheet=ref * 1e3, strip=np.array([a, b, b + nrm * w, a + nrm * w]) * 1e3, u=u, n=nrm, w=w * 1e3)


# ---------------- 3D frame: LE rod + batten ----------------
def _beam12(p1, p2, EA, EIy, EIz, GJ, up=(0, 0, 1.0)):
    L = np.linalg.norm(p2 - p1)
    ex = (p2 - p1) / L
    ez = np.asarray(up, float)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    R3 = np.array([ex, ey, ez])
    T = np.zeros((12, 12))
    for i in range(4):
        T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R3
    k = np.zeros((12, 12))
    k[0, 0] = k[6, 6] = EA / L; k[0, 6] = k[6, 0] = -EA / L
    k[3, 3] = k[9, 9] = GJ / L; k[3, 9] = k[9, 3] = -GJ / L
    a, b, c, d = 12 * EIz / L ** 3, 6 * EIz / L ** 2, 4 * EIz / L, 2 * EIz / L
    for (i, j, v) in [(1, 1, a), (1, 5, b), (1, 7, -a), (1, 11, b), (5, 5, c), (5, 7, -b), (5, 11, d), (7, 7, a), (7, 11, -b), (11, 11, c)]:
        k[i, j] = k[j, i] = v
    a, b, c, d = 12 * EIy / L ** 3, 6 * EIy / L ** 2, 4 * EIy / L, 2 * EIy / L
    for (i, j, v) in [(2, 2, a), (2, 4, -b), (2, 8, -a), (2, 10, -b), (4, 4, c), (4, 8, b), (4, 10, d), (8, 8, a), (8, 10, b), (10, 10, c)]:
        k[i, j] = k[j, i] = v
    return T.T @ k @ T


def frame_K(wing):
    """Frame on the sheet's LE and batten nodes, clamped at the LE root. Returns a solver f -> u at the mesh nodes."""
    p = wing.p
    x = wing.x0
    le = sorted(set(int(n) for n in wing.le_nodes), key=lambda n: x[n, 0])
    seam = list(wing.seam)
    all_n = le + [n for n in seam if n not in le]
    pos = {n: i for i, n in enumerate(all_n)}
    N = len(all_n)
    I, J, Vv = [], [], []
    L_tot = x[le[-1], 0] - x[le[0], 0]

    def add(n1, n2, EA, EIy, EIz, GJ):
        k = _beam12(x[n1], x[n2], EA, EIy, EIz, GJ)
        d = np.r_[6 * pos[n1] + np.arange(6), 6 * pos[n2] + np.arange(6)]
        for a in range(12):
            for b in range(12):
                I.append(d[a]); J.append(d[b]); Vv.append(k[a, b])

    od_t = p["rod_od_tip"] or p["rod_od"]
    id_t = p["rod_id_tip"] if p["rod_id_tip"] is not None else p["rod_id"]
    for n1, n2 in zip(le[:-1], le[1:]):
        f = 0.5 * (x[n1, 0] + x[n2, 0] - 2 * x[le[0], 0]) / L_tot
        od, id_ = p["rod_od"] + (od_t - p["rod_od"]) * f, p["rod_id"] + (id_t - p["rod_id"]) * f
        Ix = math.pi * (od ** 4 - id_ ** 4) / 64
        add(n1, n2, p["rod_E"] * math.pi * (od ** 2 - id_ ** 2) / 4, p["rod_E"] * Ix, p["rod_E"] * Ix, p["rod_G"] * 2 * Ix)
    if seam:
        Ib = math.pi * p["batten_d"] ** 4 / 64
        s = sorted(seam, key=lambda n: x[n, 1])
        for n1, n2 in zip(s[:-1], s[1:]):
            add(n1, n2, p["batten_E"] * math.pi * p["batten_d"] ** 2 / 4, p["batten_E"] * Ib, p["batten_E"] * Ib, p["batten_G"] * 2 * Ib)
    K = sp.csc_matrix((Vv, (I, J)), shape=(6 * N, 6 * N))
    clamp = 6 * pos[le[0]] + np.arange(6)
    free = np.setdiff1d(np.arange(6 * N), clamp)
    lu = spl.splu(K[free][:, free].tocsc())

    def solve_f(f_nodes):
        b = np.zeros(6 * N)
        for n, i in pos.items():
            b[6 * i:6 * i + 3] = f_nodes[n]
        u = np.zeros(6 * N)
        u[free] = lu.solve(b[free])
        out = np.zeros_like(f_nodes)
        for n, i in pos.items():
            out[n] = u[6 * i:6 * i + 3]
        return out

    return solve_f, np.array(all_n), le


def solve_elastic(w, V=0.0, x_init=None, aero=False):
    """Sheet with fixed frame nodes <-> linear frame under the reactions (fixed point). Root rib stays rigid."""
    p = w.p
    fk, nodes, le = frame_K(w)
    x = (w.x0 if x_init is None else x_init).copy()
    r = None
    for it in range(p["frame_iter"]):
        r = w.relax(V, x_init=x, aero=aero, fixed_x=x, tol=p["tol"] * 3, max_iter=p["max_iter"])
        if r.get("diverged"):
            return r
        x = r["x"]
        f = w.reactions(x, V, r["pe"])
        u = fk(f)
        new = w.x0[nodes] + u[nodes]
        dx = new - x[nodes]
        err = float(np.abs(dx).max())
        x[nodes] += p["frame_relax"] * dx
        if err < 3e-6:
            break
    r = w.relax(V, x_init=x, aero=aero, fixed_x=x, tol=p["tol"] * 3, max_iter=p["max_iter"])
    r["frame_err"] = err
    r["frame_u"] = u
    return r
