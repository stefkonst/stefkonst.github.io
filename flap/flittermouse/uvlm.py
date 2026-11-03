"""Unsteady 3D vortex lattice with a shed wake (Katz and Plotkin, ch. 13), for a wing that moves and deforms.

Axes: x downstream, y span (root to tip, mirror plane y = 0), z up. Vortex rings on the surface, quarter-panel
offset, control points at three-quarter panel. Wake rows are shed from the trailing edge (Kelvin) and convected
with the free stream, plus their induced velocity if free=True. Attached flow only.

Two forces per panel, as bounds on leading-edge suction:
  F  = rho Gamma (V x l) on every segment + rho dGamma/dt A n   (full suction)
  Fn = the component of F along n                               (no suction)
grid(t) returns nodes of shape (ns+1, nc+1, 3). SI units.
"""
import numpy as np


MIRROR = np.array([1.0, -1.0, 1.0])

try:
    from numba import njit, prange
except ImportError:                      # same results, about 50x slower
    prange = range

    def njit(*args, **kwargs):
        return lambda f: f


@njit(parallel=False, fastmath=True, cache=True)
def _seg(px, py, pz, ax, ay, az, bx, by, bz, core2):
    """Velocity at p per unit strength of segment a->b, divided by 4 pi later."""
    r1x, r1y, r1z = px - ax, py - ay, pz - az
    r2x, r2y, r2z = px - bx, py - by, pz - bz
    cx = r1y * r2z - r1z * r2y
    cy = r1z * r2x - r1x * r2z
    cz = r1x * r2y - r1y * r2x
    c2 = cx * cx + cy * cy + cz * cz
    r0x, r0y, r0z = bx - ax, by - ay, bz - az
    n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z) + 1e-14
    n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z) + 1e-14
    d = (r0x * r1x + r0y * r1y + r0z * r1z) / n1 - (r0x * r2x + r0y * r2y + r0z * r2z) / n2
    l2 = r0x * r0x + r0y * r0y + r0z * r0z
    f = d / (c2 + core2 * l2 + 1e-30)
    return cx * f, cy * f, cz * f


@njit(parallel=False, fastmath=True, cache=True)
def biot(P, A, B, G, core2):
    """Velocity at P from segments A->B, strengths G, with the image in y = 0 (mirrored, strength reversed)."""
    v = np.zeros_like(P)
    k = 1.0 / (4 * np.pi)
    for i in range(P.shape[0]):
        px, py, pz = P[i, 0], P[i, 1], P[i, 2]
        sx, sy, sz = 0.0, 0.0, 0.0
        for j in range(A.shape[0]):
            g = G[j]
            if g == 0.0:
                continue
            ux, uy, uz = _seg(px, py, pz, A[j, 0], A[j, 1], A[j, 2], B[j, 0], B[j, 1], B[j, 2], core2)
            mx, my, mz = _seg(px, py, pz, A[j, 0], -A[j, 1], A[j, 2], B[j, 0], -B[j, 1], B[j, 2], core2)
            sx += g * (ux - mx)
            sy += g * (uy - my)
            sz += g * (uz - mz)
        v[i, 0], v[i, 1], v[i, 2] = sx * k, sy * k, sz * k
    return v


@njit(parallel=False, fastmath=True, cache=True)
def influence(P, n, A, B, idx, nr, core2):
    """Normal velocity at P per unit ring strength. idx maps each segment to its ring."""
    M = np.zeros((P.shape[0], nr))
    k = 1.0 / (4 * np.pi)
    for i in range(P.shape[0]):
        px, py, pz = P[i, 0], P[i, 1], P[i, 2]
        for j in range(A.shape[0]):
            ux, uy, uz = _seg(px, py, pz, A[j, 0], A[j, 1], A[j, 2], B[j, 0], B[j, 1], B[j, 2], core2)
            mx, my, mz = _seg(px, py, pz, A[j, 0], -A[j, 1], A[j, 2], B[j, 0], -B[j, 1], B[j, 2], core2)
            M[i, idx[j]] += k * ((ux - mx) * n[i, 0] + (uy - my) * n[i, 1] + (uz - mz) * n[i, 2])
    return M


class UVLM:
    def __init__(self, grid, ns, nc, dt, rho=1.225, U=4.0, free=True, core=1e-4, wake_max=None):
        self.grid, self.ns, self.nc, self.dt, self.rho, self.U, self.free = grid, ns, nc, dt, rho, U, free
        self.Uinf = np.array([U, 0.0, 0.0])
        self.t = 0.0
        self.G = np.zeros((ns, nc))
        self.wn = np.zeros((0, ns + 1, 3))          # wake nodes, row 0 newest
        self.wG = np.zeros((0, ns))
        self.core = core
        self.wake_max = wake_max
        self.X_old = None
        self._wc = None

    @staticmethod
    def geom(X):
        """Rings, control points, unit normals (up for x downstream, y out), areas from nodes X."""
        d = X[:, 1:] - X[:, :-1]
        R = np.concatenate((X[:, :-1] + 0.25 * d, X[:, -1:] + 0.25 * d[:, -1:]), axis=1)
        cp = 0.5 * ((X[:-1, :-1] + 0.75 * d[:-1]) + (X[1:, :-1] + 0.75 * d[1:]))
        a, b = R[1:, 1:] - R[:-1, :-1], R[:-1, 1:] - R[1:, :-1]
        n = np.cross(b, a)
        area = 0.5 * np.sqrt((n * n).sum(-1))
        n = n / np.maximum(2 * area, 1e-20)[..., None]
        return R, cp, n, area

    def _wing_segments(self, R):
        p00, p10, p01, p11 = R[:-1, :-1], R[1:, :-1], R[:-1, 1:], R[1:, 1:]
        A = np.stack((p00, p10, p11, p01), axis=2).reshape(-1, 3)
        B = np.stack((p10, p11, p01, p00), axis=2).reshape(-1, 3)
        return A, B, np.repeat(np.arange(self.ns * self.nc), 4)

    def _wake_segments(self):
        if len(self.wG) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        Rw = self.wn
        p00, p10, p01, p11 = Rw[:-1, :-1], Rw[:-1, 1:], Rw[1:, :-1], Rw[1:, 1:]
        A = np.stack((p00, p10, p11, p01), axis=2).reshape(-1, 3)
        B = np.stack((p10, p11, p01, p00), axis=2).reshape(-1, 3)
        return A, B, np.repeat(self.wG.ravel(), 4)

    def step(self, X=None, commit=True):
        """Advance one dt. X overrides grid(t), for the coupled solver. commit=False leaves the state untouched."""
        dt, ns, nc = self.dt, self.ns, self.nc
        X = self.grid(self.t) if X is None else X
        vs = np.zeros_like(X) if self.X_old is None else (X - self.X_old) / dt
        R, cp, n, area = self.geom(X)
        vel_cp = 0.5 * ((0.25 * vs[:-1, :-1] + 0.75 * vs[:-1, 1:]) + (0.25 * vs[1:, :-1] + 0.75 * vs[1:, 1:]))
        A, B, idx = self._wing_segments(R)
        if len(self.wG):
            self.wn = self.wn.copy()
            self.wn[0] = R[:, -1]
        wA, wB, wGs = self._wake_segments()
        core2 = self.core ** 2
        Cp, nf = cp.reshape(-1, 3), n.reshape(-1, 3)
        sA, sB = R[:-1, :-1], R[1:, :-1]
        cA, cB = R[:, :-1], R[:, 1:]
        smid, cmid = (0.5 * (sA + sB)).reshape(-1, 3), (0.5 * (cA + cB)).reshape(-1, 3)
        if self._wc is None:
            self._wc = (biot(Cp, wA, wB, wGs, core2), biot(smid, wA, wB, wGs, core2), biot(cmid, wA, wB, wGs, core2))
        rhs = -(((self.Uinf - vel_cp.reshape(-1, 3) + self._wc[0]) * nf).sum(1))
        G = np.linalg.solve(influence(Cp, nf, A, B, idx, ns * nc, core2), rhs).reshape(ns, nc)
        dGdt = (G - self.G) / dt
        ws = np.repeat(G.ravel(), 4)
        induced = lambda p: biot(p, A, B, ws, core2) + biot(p, wA, wB, wGs, core2)
        # spanwise bound segments, shared by chord neighbours
        dG = G.copy()
        dG[:, 1:] -= G[:, :-1]
        V = self.Uinf + biot(smid, A, B, ws, core2) + self._wc[1] - (0.5 * (vs[:-1, :-1] + vs[1:, :-1])).reshape(-1, 3)
        F = (self.rho * dG.ravel()[:, None] * np.cross(V, (sB - sA).reshape(-1, 3))).reshape(ns, nc, 3)
        # chordwise bound segments, net strength along +x; the root one cancels with its image
        Gc = np.zeros((ns + 1, nc))
        Gc[-1] = G[-1]
        Gc[1:-1] = G[:-1] - G[1:]
        Vc = self.Uinf + biot(cmid, A, B, ws, core2) + self._wc[2] - (0.5 * (vs[:, :-1] + vs[:, 1:])).reshape(-1, 3)
        Fc = (self.rho * Gc.ravel()[:, None] * np.cross(Vc, (cB - cA).reshape(-1, 3))).reshape(ns + 1, nc, 3)
        F = F + 0.5 * (Fc[:-1] + Fc[1:]) + self.rho * (dGdt * area)[..., None] * n
        Fn = (F * n).sum(-1)
        res = dict(F=F, Fn=Fn, n=n, area=area, cp=cp, vel=vel_cp, X=X, G=G)
        if not commit:
            return res
        self.G, self.X_old, self._wc = G, X.copy(), None
        # convect the wake, then shed a new row with the trailing-edge strength
        if len(self.wG):
            Vn = self.Uinf
            if self.free:
                Vn = self.Uinf + induced(self.wn.reshape(-1, 3)).reshape(self.wn.shape)
            self.wn = self.wn + dt * Vn
            self.wn = np.concatenate((R[:, -1][None], self.wn), axis=0)
            self.wG = np.concatenate((G[:, -1][None], self.wG), axis=0)
        else:
            self.wn = np.concatenate((R[:, -1][None], R[:, -1][None] + dt * self.Uinf), axis=0)
            self.wG = G[:, -1][None].copy()
        if self.wake_max and len(self.wG) > self.wake_max:
            self.wn, self.wG = self.wn[:self.wake_max + 1], self.wG[:self.wake_max]
        self.t += dt
        return res
