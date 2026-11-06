"""One place for every number the robot study uses, with where it comes from.
M = measured on the bench (4-5 Nov 2026), F = fitted to those measurements, P = published, A = assumption (my judgement)."""
import math

EPS0, EPS_R = 8.854e-12, 2.8
EPS = EPS0 * EPS_R
RHO_F = 1100.0          # A  silicone film density, kg/m3
C1 = 42842.0            # F  Yeoh C1 of P7670, Pa (rolled-DEA bundle)
RHO_AIR, NU_AIR, G = 1.2, 1.5e-5, 9.81

# wing, as built (first wing, 2 electrodes)
WING_MASS = 273e-6      # M  kg, incl. cables, glue, leading edge
LE_MASS = 23e-6         # M
SPAN = 0.050            # M  one wing, about 50 mm along the leading edge
CHORD_ROOT = 0.024      # M  from the outline (geometry.json)
AREA = 1.25e-3          # M  about 12.5 cm2 per wing (outline)
EDGE_L = 0.0588         # M  trailing edge, root to tip
ROLL_CUT = 52.94e-3     # M  mask
ROLL_LEN = (61.6e-3, 64.8e-3)   # M  in the front view
T_EDGE = (0.050, 0.096) # F  N, from length and the P7670 fit
N_ACROSS = (0.15, 0.29) # F  N/m, skin pull across the edge, from T * curvature
ROLL_W, ROLL_WA = 23e-3, 20e-3   # M  strip and active width
ETA_F = (0.4, 0.8)      # F  edge efficiency the weight model needs (5 Nov FE)
ETA_S = (0.05, 0.4)     # F  skin efficiency, open
H = 30e-6               # A  one dielectric layer (unmeasured; 30 um is the working value)
FILM_T = 60e-6          # A  skin total, 2 electrodes: 0.5D + E + D + E + 0.5D
ROLL_MASS = 1100.0 * ROLL_W * FILM_T * ROLL_CUT   # A  about 80 mg
CORE_MASS = 1131e-6 / 2 # M  each core
E_CYCLE = {"bench, isometric": 0.8, "published (Ren, Kim 2022)": 1.25, "60 V/um estimate": 2.1}   # J/kg per cycle

# 5 Nov videos
Z_HIGH = {1000: 0.28, 1400: 0.82, 1800: 1.84}   # M  mm, side view, out of plane
A1_FREQ = {1: 0.58, 5: 0.17, 10: 0.10, 15: 0.23, 20: 0.10}   # M  mm, diagonal view, 1800 V
BUMP_HZ = 15.0          # M

# materials for an edge cord
CORDS = {
    "rolled DEA (as built)": dict(E=6 * C1 * 1.5, A=ROLL_W * FILM_T, rho=RHO_F, note="active, 2 films"),
    "nylon 0.10 mm": dict(E=2.5e9, A=math.pi / 4 * 0.10e-3 ** 2, rho=1140.0, note="passive, creeps little"),
    "nylon 0.20 mm": dict(E=2.5e9, A=math.pi / 4 * 0.20e-3 ** 2, rho=1140.0, note="passive"),
    "TPU 1 x 0.1 mm": dict(E=25e6, A=1e-3 * 0.1e-3, rho=1200.0, note="passive, creeps"),
    "TPU 2 x 0.2 mm": dict(E=25e6, A=2e-3 * 0.2e-3, rho=1200.0, note="passive, creeps"),
}


def maxwell(V, H=H):
    """Maxwell stress eps (V/H)^2, Pa."""
    return EPS * (V / H) ** 2


def roll_dF(V, eta_f, layers=2, wa=ROLL_WA, H=H):
    """Force the rolled edge loses at voltage, N: eta_f n eps (V/H)^2 H w_a (blocked)."""
    return eta_f * layers * maxwell(V, H) * H * wa
