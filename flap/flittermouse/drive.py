"""W4. Drive: rolled core DEA + lever + bias spring + wing, one degree of freedom.

  I phi'' + tau_aero(phi') + k_s (phi - phi_s) r_s... : written as a torque balance about the stroke axis
  I phi'' = F(lam, v) r - k_t (phi - phi_0) - c_h phi' - tau_a(phi')

Layout A: core tension lifts the wing, so voltage (less tension) lets the bias spring drive the downstroke.
          One channel per wing: skin and core share v(t), and at resonance peak v falls in mid downstroke.
Layout B: core tension pulls the wing down and reacts lift directly, so voltage drives the upstroke.
          Skin needs v high on the downstroke, so it needs its own channel.

Core: F = A [2 C1 (lam - lam^-2) + 4 C2 (I1 - 3)(lam - lam^-2)] - sigma_a A v - c_eq lam' L0 A
sigma_a from the bench bundle (isometric or slow-cycle force). Tension only: a slack core carries nothing.
tau_a from data/aero_table.json if present, otherwise a quadratic fit passed by the caller.
"""
import json, os
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
C1, C2, RHO = 42842.0, 447.0, 1070.0
SIG_ISO, SIG_CYC = 44.7e-3 / 3e-6, 17.2e-3 / 3e-6        # Pa at 1.6 kV, bench bundle a
LOSS_5 = 0.38                                             # J/kg per cycle at +-5 %, upper bound from the bench branches


def core_force(lam, A):
    I1 = lam ** 2 + 2 / lam
    return 2 * A * (lam - lam ** -2) * (C1 + 2 * C2 * (I1 - 3))


def stiffness(lam, A, L0):
    d = 1e-4
    return (core_force(lam + d, A) - core_force(lam - d, A)) / (2 * d) / L0


def simulate(m_core=0.5e-3, L0=0.020, lam0=1.10, r=1.9e-3, I=1.7e-7, f=25.0, sig=SIG_ISO, loss5=LOSS_5, k_ratio=None, c_a=None, tau_mean=0.0,
             n=30, layout="A", V_scale=1.0, f_n=None):
    """Periodic response to v = (1 + sin wt) / 2. f_n: natural frequency the spring is chosen for (default f).
    c_a: quadratic aero damping, torque = c_a |phi'| phi'. tau_mean: mean aerodynamic lifting torque."""
    A = m_core / RHO / L0
    w = 2 * np.pi * f
    k_core = stiffness(lam0, A, L0)
    wn = 2 * np.pi * (f_n or f)
    kt_core = k_core * r ** 2
    kt_s = max(I * wn ** 2 - kt_core, 0.0)
    s_core = 1 if layout == "A" else -1                 # A: tension lifts
    sigA = sig * A * V_scale ** 2
    # the spring zero sets the mean so that the core sits at lam0 with mean voltage and mean lift
    F_mean = core_force(lam0, A) - 0.5 * sigA
    tau_s0 = s_core * F_mean * r + tau_mean             # torque the spring must balance at phi = 0

    def rhs(t, y):
        phi, om = y
        lam = lam0 - s_core * r * phi / L0
        lamd = -s_core * r * om / L0
        v = 0.5 * (1 + np.sin(w * t))
        F = core_force(lam, A) - sigA * v
        slack = F < 0
        F = max(F, 0.0)
        # hysteretic loss as an equivalent viscous term at this frequency
        ampl = max(abs(y[0]) * r / L0, 0.002)
        return [om, (s_core * F * r - kt_s * phi - tau_s0 + tau_mean - c_a * abs(om) * om - h * om) / I]

    # hysteretic loss: W = LOSS (s/0.05)^2 per kg per cycle = pi kt_eq... equivalent c from the amplitude, iterated
    h = 0.0
    y0 = [0.0, 0.0]
    out = None
    Tc = 1 / f
    for it in range(5):
        sol = solve_ivp(rhs, (0, n * Tc), y0, max_step=Tc / 40, rtol=1e-6, atol=1e-9, dense_output=False, t_eval=np.linspace((n - 2) * Tc, n * Tc, 401))
        phi = sol.y[0]
        Phi = 0.5 * (phi.max() - phi.min())
        s = Phi * r / L0
        W_h = loss5 * (s / 0.05) ** 2 * m_core
        h = W_h / (np.pi * w * max(Phi, 1e-4) ** 2)
        y0 = [sol.y[0][-1], sol.y[1][-1]]
    t, om = sol.t, sol.y[1]
    lam = lam0 - s_core * r * phi / L0
    v = 0.5 * (1 + np.sin(w * t))
    F = core_force(lam, A) - sigA * v
    tau_el = s_core * np.maximum(F, 0) * r
    last = t >= (n - 1) * Tc
    P_active = np.mean((sigA * (-v) * s_core * r * om)[last])
    P_aero = np.mean((c_a * np.abs(om) ** 3)[last])
    # phase of voltage peak relative to mid downstroke: + means voltage peak comes after it
    ph = np.angle(np.mean(np.exp(1j * w * t[last]) * (-om[last])))
    return dict(Phi_deg=float(np.degrees(Phi)), strain_amp=float(s), F_min_mN=float(F.min() * 1e3), F_max_mN=float(F.max() * 1e3), slack=bool(F.min() <= 0),
                k_core=float(k_core), kt_core=float(kt_core), kt_s=float(kt_s), spring_share=float(kt_s / max(kt_s + kt_core, 1e-12)), F_block_mN=float(sigA * 1e3),
                P_aero_mW=float(P_aero * 1e3), W_h_mJ=float(W_h * 1e3), P_h_mW=float(W_h * f * 1e3), tau_s0_mNm=float(tau_s0 * 1e3), phase_v_vs_midstroke_deg=float(np.degrees(ph)),
                mean_phi_deg=float(np.degrees(phi[last].mean())), A_mm2=float(A * 1e6))


if __name__ == "__main__":
    # the aero damping is fixed to reach 9.4 mW at 25 Hz, +-30 deg: replaced by the coupled model in W5
    Om = np.radians(30) * 2 * np.pi * 25
    ca = 9.4e-3 / (np.mean(np.abs(np.cos(np.linspace(0, 2 * np.pi, 1001))) ** 3) * Om ** 3)
    print("c_a", ca)
    print("spring-tuned, sigma iso", json.dumps(simulate(c_a=ca), indent=0)[:600])
    for f in (10, 15, 20, 25, 30, 40):
        a = simulate(c_a=ca, f=f)
        print(f, "Phi %.1f s %.3f Fmin %.0f P_aero %.1f P_h %.1f spring %.2f" % (a["Phi_deg"], a["strain_amp"], a["F_min_mN"], a["P_aero_mW"], a["P_h_mW"], a["spring_share"]))
