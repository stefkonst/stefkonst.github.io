"""Small schematic next to each parameter plot: orange is what the plot varies, the text says what stays fixed. Inline SVG, so it follows the page's theme.
Run: python schematics.py -> rewrites the blocks between <!--s:NAME--> and <!--/s--> in the pages. Safe to rerun."""
import math, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUT, SHEET, HI, ACC = "var(--ink)", "var(--muted)", "var(--accent-dim)", "var(--ps-hi)", "var(--accent)"

CSS = """.plotrow{display:grid;grid-template-columns:minmax(0,1fr) 248px;gap:14px;align-items:start}
.plotrow>img{min-width:0}
@media (max-width:820px){.plotrow{grid-template-columns:minmax(0,1fr)}.ps{max-width:380px}}
.pcard{--ps-hi:#c8440a}
.pcard svg text.h{fill:var(--ps-hi);font-weight:600}
.ps{--ps-hi:#c8440a;border:1px solid var(--rule);border-radius:8px;padding:8px 10px;font-size:12.5px;line-height:1.4;background:var(--card)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .ps,:root:not([data-theme="light"]) .pcard{--ps-hi:#f08a55}}
:root[data-theme="dark"] .ps,:root[data-theme="dark"] .pcard{--ps-hi:#f08a55}
.ps svg{width:100%;height:auto;display:block;margin-bottom:6px}
.ps svg text{fill:var(--ink);font-family:'Source Sans 3',system-ui,sans-serif;font-size:10.5px}
.ps svg text.h{fill:var(--ps-hi);font-weight:600}
.ps svg text.m{fill:var(--muted)}
.ps p{margin:.25em 0;color:var(--ink);font-size:12.5px}
.ps b{color:var(--ps-hi)}
.ps p+p b{color:var(--muted)}
"""


# ---------------- drawing ----------------
def txt(x, y, s, k="", a="start"):
    """Text; a_b is a subscript. k: h (orange), m (muted)."""
    s = re.sub(r"_(\{[^}]*\}|\w+)", lambda m: f'<tspan dy="2.5" font-size="8">{m[1].strip("{}")}</tspan><tspan dy="-2.5">​</tspan>', s)
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{a}"' + (f' class="{k}"' if k else "") + f">{s}</text>"


def line(x1, y1, x2, y2, c=INK, w=1.2, d=""):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"' + (f' stroke-dasharray="{d}"' if d else "") + "/>"


def poly(p, fill="none", c="none", w=1.0, d=""):
    return '<path d="M' + " L".join(f"{x:.1f} {y:.1f}" for x, y in p) + f' Z" fill="{fill}" stroke="{c}" stroke-width="{w}"' + (f' stroke-dasharray="{d}"' if d else "") + "/>"


def pl(p, c=INK, w=1.2, d=""):
    return '<path d="M' + " L".join(f"{x:.1f} {y:.1f}" for x, y in p) + f'" fill="none" stroke="{c}" stroke-width="{w}"' + (f' stroke-dasharray="{d}"' if d else "") + "/>"


def arrow(x1, y1, x2, y2, c=HI, w=1.4, both=False):
    def head(xa, ya, xb, yb):
        u = math.atan2(yb - ya, xb - xa)
        return poly([(xb, yb), (xb - 6 * math.cos(u - .42), yb - 6 * math.sin(u - .42)), (xb - 6 * math.cos(u + .42), yb - 6 * math.sin(u + .42))], c, c, .4)
    return line(x1, y1, x2, y2, c, w) + head(x1, y1, x2, y2) + (head(x2, y2, x1, y1) if both else "")


def arc(cx, cy, r, a0, a1, c=HI, w=1.4, end=True, both=False):
    """Arc in degrees, counter-clockwise on the page from a0 to a1."""
    P = [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / 24)), cy - r * math.sin(math.radians(a0 + (a1 - a0) * i / 24))) for i in range(25)]
    s = pl(P, c, w)
    if end:
        s += arrow(*P[-3], *P[-1], c, w)
    if both:
        s += arrow(*P[2], *P[0], c, w)
    return s


def svg(h, body, label):
    return f'<svg viewBox="0 0 240 {h}" role="img" aria-label="{label}">{body}</svg>'


class Wing:
    """Top view to scale, LE at the top, chord downwards. 2 px per mm by default."""
    def __init__(self, ox=10, oy=22, s=2.0, xb=60.0, chord=41.1, xt=108.0, drop=3.1, xr=3.8):
        self.ox, self.oy, self.s, self.xb, self.c, self.xt, self.dr, self.xr = ox, oy, s, xb, chord, xt, drop, xr

    def p(self, x, y):
        return self.ox + self.s * x, self.oy + self.s * y

    def te(self, x):
        return self.c if x <= self.xb else self.c + (self.dr - self.c) * (x - self.xb) / (self.xt - self.xb)

    def outline(self):
        return [self.p(self.xr, 0), self.p(self.xt, 0), self.p(self.xt, self.dr), self.p(self.xb, self.c), self.p(self.xr, self.c)]

    def full(self, fibre=ACC, rods=True, sheet=SHEET):
        s = poly(self.outline(), sheet, MUT, .7)
        s += line(*self.p(self.xr - 2, 0), *self.p(self.xt + 1, 0), INK, 2.4)
        if rods:
            s += line(*self.p(self.xr, 0), *self.p(self.xr, self.c), INK, 1.4) + line(*self.p(self.xb, 0), *self.p(self.xb, self.c), INK, 1.4)
        return s + pl([self.p(self.xr, self.c), self.p(self.xb, self.c), self.p(self.xt, self.dr)], fibre, 2.6)

    def discrete(self, dea=INK, n=8, te=MUT, te_w=.8):
        s = poly(self.outline(), SHEET, MUT, .7)
        for i in range(n):
            x = self.xr + (self.xt - self.xr) * (i + .6) / (n + .2)
            s += line(*self.p(x, 0), *self.p(x, self.te(x)), dea, 2)
        s += pl([self.p(self.xr, self.c), self.p(self.xb, self.c), self.p(self.xt, self.dr)], te, te_w)
        return s + line(*self.p(self.xr - 2, 0), *self.p(self.xt + 1, 0), INK, 2.4)

    def flow(self, c=MUT, n=4, y=-16):
        s = ""
        for i in range(n):
            x, yy = self.p(self.xr + (self.xt - self.xr) * (i + .5) / n, 0)
            s += arrow(x, yy + y, x, yy - 4, c, 1.1)
        return s


def tilted(W, tb, tr):
    """The sheet as cut: root edge and batten leaning outboard by tr and tb, dashed."""
    tn = lambda d: math.tan(math.radians(d))
    return pl([W.p(W.xr, 0), W.p(W.xr + W.c * tn(tr), W.c), W.p(W.xb + W.c * tn(tb), W.c), W.p(W.xt, W.dr)], HI, 1.1, "4 3") + line(*W.p(W.xb, 0), *W.p(W.xb + W.c * tn(tb), W.c), HI, 1.1, "4 3")


def section(ox, oy, L, al=8.0, cam=.1, c=ACC, w=2.2):
    """Chord in side view, LE on the left, pitched nose up, with a bulge on the suction side."""
    a = math.radians(al)
    P = [(ox + L * t * math.cos(a) - L * cam * 4 * t * (1 - t) * math.sin(a), oy + L * t * math.sin(a) - L * cam * 4 * t * (1 - t) * math.cos(a)) for t in [i / 24 for i in range(25)]]
    return pl(P, c, w), P


def stack(ox, oy, w, n, c=HI, t=7):
    """Half dielectric, then electrode, dielectric, ..., electrode, half dielectric."""
    s, y = "", oy
    s += poly([(ox, y), (ox + w, y), (ox + w, y + t / 2), (ox, y + t / 2)], SHEET, MUT, .5); y += t / 2
    for i in range(n + 1):
        s += line(ox, y, ox + w, y, c, 1.8)
        if i < n:
            s += poly([(ox, y), (ox + w, y), (ox + w, y + t), (ox, y + t)], SHEET, MUT, .5); y += t
    s += poly([(ox, y), (ox + w, y), (ox + w, y + t / 2), (ox, y + t / 2)], SHEET, MUT, .5)
    return s, y + t / 2


def stroke(ox=24, oy=78, L=150, hi=True):
    """Front view: stroke axis at the body, wing at the top and bottom of the stroke."""
    a = math.radians(30)
    s = poly([(ox - 10, oy - 16), (ox, oy - 8), (ox, oy + 8), (ox - 10, oy + 16)], MUT, MUT, .5)
    s += line(ox, oy, ox + L * math.cos(a), oy - L * math.sin(a), INK, 1.8) + line(ox, oy, ox + L * math.cos(a), oy + L * math.sin(a), MUT, 1.2, "4 3") + line(ox, oy, ox + L, oy, MUT, .6, "2 3")
    if hi:
        s += arc(ox, oy, L * .78, 27, -27, HI, 1.6, both=True)
    return s


def bundle(ox, oy, w, n=5, gap=6):
    """Rolled DEAs side by side between a fixed clamp and a load cell."""
    s = poly([(ox - 10, oy - 8), (ox, oy - 8), (ox, oy + gap * (n - 1) + 8), (ox - 10, oy + gap * (n - 1) + 8)], MUT, MUT, .5)
    s += poly([(ox + w, oy - 8), (ox + w + 22, oy - 8), (ox + w + 22, oy + gap * (n - 1) + 8), (ox + w, oy + gap * (n - 1) + 8)], "none", INK, 1)
    for i in range(n):
        s += line(ox, oy + i * gap, ox + w, oy + i * gap, ACC, 2.2)
    return s + txt(ox + w + 11, oy + gap * (n - 1) / 2 + 3.5, "F", "", "middle")


# ---------------- one per plot ----------------
def sk_stroke():
    return svg(156, stroke() + txt(24, 12, "Front view, one wing", "m") + txt(150, 74, "Stroke phase φ", "h", "middle") + txt(170, 26, "φ = 0", "m") + txt(170, 146, "φ = 180°", "m") + arrow(214, 100, 214, 124, MUT, 1.1) + txt(226, 104, "90°", "m", "middle") + txt(226, 134, "", "m"), "Front view of a wing flapping 30 degrees either side, phase highlighted")


def sk_speed():
    W = Wing(oy=40)
    return svg(136, W.flow(HI, 4, -22) + W.full() + txt(10, 12, "Flow speed U", "h") + txt(120, 134, "Full membrane; discrete wing the same way", "m", "middle"), "Wing from above with the oncoming flow highlighted")


def sk_eta():
    s, y = stack(20, 30, 130, 2, MUT)
    return svg(108, txt(10, 14, "Fraction of Maxwell stress delivered, η", "h") + s + arrow(90, 42, 90, 34, HI) + arrow(90, 46, 90, 54, HI) + arrow(150, 45, 178, 45, HI) + arrow(20, 45, -8 + 20, 45, HI) + txt(186, 49, "η·P", "h") + txt(20, y + 16, "Film thins and grows in area", "m"), "Side view of the film stack, with the actuation stress scaled by eta highlighted")


def sk_voltage():
    W = Wing(oy=30)
    s = W.full()
    a, b = W.p(32, 20), W.p(32, 41.1)
    return svg(138, s + txt(10, 14, "Voltage V across the sheet", "h") + arrow(*W.p(10, 20), *W.p(56, 20), HI, 1.4, True) + arrow(*W.p(32, 4), *W.p(32, 36), HI, 1.4, True) + txt(*W.p(46, 18), "n_x", "h") + txt(*W.p(34, 32), "n_y", "h") + txt(10, 132, "b) resultants plotted on panel 1", "m"), "Panel of the wing with the span and chord tension resultants that fall with voltage")


def sk_stretch():
    D = Wing(oy=20, s=1.05, ox=4)
    F = Wing(oy=20, s=1.05, ox=122)
    s = D.discrete(n=6)
    s += arrow(*D.p(18, 20), *D.p(-2, 20), HI) + arrow(*D.p(50, 42), *D.p(50, 56), HI)
    s += F.full(rods=True) + tilted(F, 14, 25)
    return svg(112, txt(4, 12, "a) Pre-stretch λ_pre", "h") + txt(122, 12, "b) Tilt θ_b, θ_r = 1.77 θ_b", "h") + s + txt(4, 106, "Discrete", "m") + txt(122, 106, "Full, as cut (dashed)", "m"), "Discrete wing with its membrane pre-stretch, and full membrane with its tilted cut")


def sk_tendon():
    W = Wing(oy=24)
    s = W.discrete(te=HI, te_w=2.6)
    return svg(136, txt(10, 14, "Trailing-edge tension T_c", "h") + s + arrow(*W.p(3.8, 41.1), *W.p(-4, 49), HI) + arrow(*W.p(108, 3.1), *W.p(114, 9), HI) + txt(*W.p(30, 48), "T_c along the edge cord", "h") + txt(10, 132, "", "m"), "Discrete wing with the trailing edge cord highlighted")


def sk_pitch():
    c, P = section(46, 62, 150, 8, .05, ACC)
    return svg(136, txt(10, 14, "a) Mean pitch α", "h") + c + line(46, 62, 214, 62, MUT, .7, "3 3") + arc(46, 62, 92, 0, -8, HI, 1.5, end=False) + txt(144, 88, "α", "h") + arrow(6, 62, 36, 62, MUT, 1.1) + txt(6, 54, "U", "m") + stroke_mini(70, 116) + txt(10, 112, "b) f, U", "h"), "Side view of a chord at a pitch angle, and the flap frequency")


def stroke_mini(ox, oy):
    return arc(ox, oy + 2, 26, 35, -15, HI, 1.3, both=True) + line(ox, oy + 2, ox + 30, oy - 12, INK, 1.4) + txt(ox + 40, oy + 2, "α_eff > 0, outer panel", "m")


def sk_rotation():
    W = Wing(oy=26)
    s = W.full() + tilted(W, 14, 25)
    bx, by = W.p(60, 0)
    return svg(148, txt(10, 14, "Tilt as cut: θ_b (batten), θ_r (root)", "h") + s + txt(*W.p(66, 28), "θ_b", "h") + txt(*W.p(11, 28), "θ_r", "h") + arc(bx, by, 62, -90, -76, HI, 1.2, end=False) + arc(W.p(3.8, 0)[0], by, 62, -90, -66, HI, 1.2, end=False) + txt(10, 132, "a) moment on the batten about its LE hinge", "m") + txt(10, 144, "b) stretch of fibre 1, in blue", "m"), "Wing with dashed tilted outline and the two tilt angles highlighted")


def sk_layers():
    a, y = stack(14, 26, 92, 1, HI, 6)
    b, y2 = stack(132, 26, 92, 3, HI, 6)
    return svg(92, txt(10, 14, "Active layers n", "h") + a + b + txt(14, 76, "n = 1", "h") + txt(132, 76, "n = 3", "h") + txt(10, 89, "Mid-dielectrics active; half-layers cover", "m"), "Film stacks with one and three active layers")


def sk_tension():
    W = Wing(oy=28, s=1.05, ox=4, xb=60)
    V = Wing(oy=28, s=1.05, ox=122, xb=60)
    a = W.full(rods=False, fibre=ACC) + arrow(*W.p(3.8, 41.1), *W.p(-2, 56), HI) + arrow(*W.p(60, 41.1), *W.p(66, 56), HI) + arrow(*W.p(108, 3.1), *W.p(112, 16), HI)
    b = V.full(rods=False) + line(*V.p(3.8, 0), *V.p(3.8, 41.1), HI, 2.4) + line(*V.p(60, 0), *V.p(60, 41.1), HI, 2.4)
    return svg(112, txt(4, 14, "a, d) Pull at fibre tips", "h") + txt(122, 14, "b, e) Rigid bars", "h") + a + b + txt(4, 106, "c) batten tilt θ", "h") + txt(122, 106, "f) LE shift Δλ_n", "h"), "Two ways of pulling the sheet: at the fibre tips or with rigid bars")


def sk_design():
    W = Wing(oy=24, ox=8)
    tn = W.p(60, 41.1)
    s = W.full() + arrow(*W.p(30, 46), *W.p(60, 46), HI) + txt(*W.p(34, 54), "d) batten x", "h") + arrow(*W.p(72, 0), *W.p(72, 28), HI, 1.4, True) + txt(*W.p(75, 22), "d) y", "h")
    b, y = stack(8, 136, 66, 2, HI, 5)
    r = poly([(120, 138), (170, 138), (170, 148), (120, 148)], "none", HI, 1) + arrow(120, 156, 170, 156, HI, 1.2, True)
    return svg(172, txt(8, 14, "Earlier stretched design", "m") + s + b + txt(8, 170, "a, b) layers", "h") + r + txt(120, 170, "c) roll width", "h"), "Wing with batten position, chord line position, layers and roll width highlighted")


def sk_final():
    W = Wing(oy=20, s=1.8)
    s = W.full() + arrow(*W.p(16, 10), *W.p(50, 10), HI, 1.3, True) + txt(*W.p(24, 8), "e) λ_d", "h") + arrow(*W.p(34, 14), *W.p(34, 38), HI, 1.3, True) + txt(*W.p(36, 30), "e) λ_n", "h") + line(*W.p(1.8, 0), *W.p(109, 0), HI, 3) + txt(*W.p(70, -3), "f) LE rod", "h")
    c, P = section(30, 118, 100, 8, .06, ACC)
    return svg(150, s + c + line(30, 118, 150, 118, MUT, .7, "3 3") + arc(30, 118, 74, 0, -8, HI, 1.4, end=False) + txt(108, 138, "b, c) α, U", "h") + txt(150, 106, "a) V", "h") + txt(150, 118, "d) η, film stack", "h"), "Wing with stretch errors and leading edge rod highlighted, and a chord with pitch angle")


def sk_shape():
    W = Wing(oy=20)
    s = W.full() + line(*W.p(30, 0), *W.p(30, 41.1), HI, 1.8, "4 2") + line(*W.p(72, 0), *W.p(72, W.te(72)), HI, 1.8, "4 2") + txt(*W.p(23, -4), "x = 30 mm", "h") + txt(*W.p(66, -4), "x = 72 mm", "h")
    c, _ = section(20, 136, 80, 0, .09, MUT, 1.2)
    d, _ = section(20, 136, 80, 0, .2, HI, 1.8)
    return svg(148, s + c + d + txt(108, 124, "Bulge w along each cut:", "m") + txt(108, 137, "0 V and ", "m") + txt(146, 137, "1.6 kV", "h"), "Wing with two chordwise cuts highlighted, and the bulge at 0 volts and 1.6 kilovolts")


def sk_sweep():
    W = Wing(oy=42)
    s = W.full(rods=True) + arrow(*W.p(10, 24), *W.p(52, 24), HI, 1.4, True) + txt(*W.p(14, 20), "along fibre, λ_d", "h") + arrow(*W.p(32, 2), *W.p(32, 40), HI, 1.4, True) + txt(*W.p(34, 34), "λ_n", "h") + W.flow(HI, 4, -14)
    return svg(136, txt(10, 11, "Stretches, at U = 3, 4, 6 m/s", "h") + s, "Wing with the stretch along and across the fibre highlighted")


def sk_vocal():
    s = f'<circle cx="120" cy="70" r="50" fill="{SHEET}" stroke="{MUT}" stroke-width="1"/>'
    a, _ = section(80, 82, 80, 0, .14, ACC, 2.4)
    b, _ = section(80, 58, 80, 0, -.14, ACC, 2.4)
    return svg(142, txt(10, 12, "Voltage V; fibre motion at mid-length, h", "h") + s + a + b + arrow(120, 70, 120, 58, HI, 1.3) + arrow(120, 70, 120, 82, HI, 1.3) + txt(125, 75, "h", "h") + arrow(80, 94, 160, 94, MUT, 1, True) + txt(120, 106, "32 mm", "m", "middle") + txt(10, 138, "From above, no flow, ring about 40 mm", "m"), "Ring with two rolled fibres that move apart with voltage")


def sk_flapcompare():
    return svg(156, stroke() + txt(24, 12, "a, b) Stroke phase φ", "h") + txt(150, 74, "φ", "h", "middle") + txt(176, 134, "c) U, f", "h") + txt(12, 150, "b) η, λ_pre", "h"), "Front view of a flapping wing, with the phase highlighted")


def sk_roll():
    s = poly([(40, 22), (190, 22), (190, 76), (40, 76)], "none", MUT, .9) + poly([(40, 22), (190, 22), (190, 36), (40, 36)], HI, HI, .3)
    s += arrow(206, 22, 206, 76, HI, 1.4, True) + txt(212, 53, "w_s", "h") + arrow(26, 22, 26, 36, HI, 1.2, True) + txt(6, 33, "w_e", "h") + txt(115, 60, "Passive core", "m", "middle") + txt(115, 14, "Sheet side", "m", "middle")
    return svg(106, s + txt(10, 94, "F_0 grows with w_s, the voltage drop with w_e", "m"), "Rolled strip as cut, with its active band next to the sheet and passive core")


def sk_bench():
    return svg(112, txt(10, 12, "a) Bundle of 5 DEAs, length held", "m") + bundle(30, 32, 150) + arrow(40, 74, 170, 74, HI, 1.3, True) + txt(105, 88, "λ = 1.00, 1.12; voltage V", "h", "middle") + txt(10, 106, "b–e) wings on video, no flow or flapping", "m"), "Five rolled actuators held between a clamp and a load cell")


def sk_section():
    s = poly([(20, 28), (220, 28), (220, 92), (20, 92)], SHEET, MUT, .7) + line(20, 28, 220, 28, INK, 2.2) + line(20, 92, 220, 92, INK, 2.2)
    for x in (50, 90, 130, 170, 210):
        s += line(x, 28, x, 92, ACC, 2)
    s += arrow(50, 104, 90, 104, HI, 1.3, True) + txt(70, 117, "s", "h", "middle") + arrow(100, 36, 100, 86, HI, 1.3, True) + txt(104, 64, "λ_c0", "h") + arrow(178, 38, 178, 82, HI, 1.3, True) + txt(182, 64, "λ_d0", "h")
    return svg(140, txt(10, 14, "Framed section, chord c = 5 cm", "m") + s + txt(120, 117, "α, U = 3–6 m/s", "h") + txt(10, 134, "Rigid LE and TE, r = 1, 1.6 kV", "m"), "Framed section with DEA spacing, membrane and DEA pre-stretch highlighted")


def sk_load():
    s = ""
    for i, y in enumerate((30, 44, 58)):
        s += line(20, y, 104, y, MUT, 2) + (line(20, y + 3.5, 104, y + 3.5, HI, .8) if i < 2 else "")
    s += arrow(112, 44, 112, 47.5, HI, 1, True) + txt(116, 50, "g", "h")
    W = Wing(ox=132, oy=30, s=.9)
    w = W.discrete(n=5, te=HI, te_w=2.2)
    return svg(112, txt(10, 14, "a, b) Air gap g, film H", "h") + s + txt(132, 14, "c) What holds the TE", "h") + w + txt(10, 84, "Roll cross-section", "m") + txt(132, 84, "Free, tendon or rod", "h") + txt(132, 98, "6 and 12 cm wings", "m"), "Air gaps between the films of a roll, and the anchor at the trailing edge")


def sk_flapsec():
    return svg(156, stroke() + txt(24, 12, "a, b) U and f", "h") + txt(12, 152, "Tip R = 0.15 m, and 0.75R", "m") + txt(150, 74, "c) φ", "h", "middle"), "Front view of a flapping wing with stroke highlighted")


def sk_static():
    c, _ = section(40, 62, 150, 8, .09, ACC)
    return svg(118, txt(10, 14, "a–c) Voltage V, per pitch α", "h") + c + line(40, 62, 206, 62, MUT, .7, "3 3") + arc(40, 62, 112, 0, -8, HI, 1.4, end=False) + txt(160, 92, "d) α", "h") + arrow(4, 62, 32, 62, MUT, 1) + txt(96, 34, "camber h", "h") + txt(10, 110, "e) LE rod tip; f) bench, U = 0", "m"), "Chord at a pitch angle with camber highlighted")


def sk_schedule():
    a = stroke(ox=20, oy=72, L=100, hi=True)
    rows = [("Off", (0, 0)), ("Always on", (1, 1)), ("Downstroke", (1, 0)), ("Upstroke", (0, 1))]
    s = ""
    for i, (n, on) in enumerate(rows):
        y = 30 + 25 * i
        lv = lambda q: y + (5 if q else 11)
        s += txt(142, y - 1, n, "m") + pl([(142, lv(on[0])), (172, lv(on[0])), (172, lv(on[1])), (202, lv(on[1]))], HI, 1.7)
    return svg(136, txt(10, 12, "DEA schedule over the stroke", "h") + a + s + txt(10, 132, "3 Hz, Φ_0 = 30°, α = 6°", "m"), "Front view of a flapping wing and four schedules for switching the actuators")


def sk_p3():
    W = Wing(oy=22, ox=12)
    s = W.discrete(n=5) + arrow(*W.p(18, 45), *W.p(36, 45), HI, 1.3, True) + txt(*W.p(38, 48), "a) number of DEAs", "h") + arrow(*W.p(-1, 4), *W.p(-1, 36), HI, 1.3, True) + txt(*W.p(58, 54), "b) DEA λ, c) membrane λ", "h")
    return svg(150, txt(10, 12, "Discrete wing at 1.6 kV", "m") + s + txt(10, 144, "d) DEA and strip directions; e, f) planform", "h"), "Discrete wing with number of DEAs, stretches, directions and planform highlighted")


def sk_comparison():
    s = bundle(22, 30, 58, 3, 7) + txt(10, 74, "a) Bench, λ = 1.12", "h")
    W = Wing(ox=134, oy=24, s=.8)
    s += W.discrete(n=5) + txt(134, 74, "b) Static wings", "h")
    s += arc(26, 96, 32, 28, -28, HI, 1.3, both=True) + line(26, 96, 56, 84, INK, 1.5)
    return svg(126, txt(10, 12, "Same model, three measurements", "m") + s + txt(72, 100, "c) Flapping wing,", "h") + txt(72, 113, "ΔC_L measured and predicted", "h"), "Bench bundle, static wing and flapping wing")


def sk_fit():
    return svg(112, txt(10, 12, "Bundle of 5 DEAs on the test stand", "m") + bundle(30, 32, 150) + arrow(40, 74, 170, 74, HI, 1.3, True) + txt(105, 88, "a) stretch λ, 0 V", "h", "middle") + txt(105, 102, "b) η from two bench methods", "h", "middle"), "Five rolled actuators held between a clamp and a load cell, with stretch highlighted")


def sk_fibre():
    """The edge seen from behind: a string between the root rod and batten, lifted by the air load."""
    P = [(30 + 180 * t, 66 - 34 * 4 * t * (1 - t)) for t in [i / 24 for i in range(25)]]
    s = line(30, 66, 210, 66, MUT, .7, "3 3") + pl(P, ACC, 2.6) + line(30, 40, 30, 92, INK, 1.6) + line(210, 40, 210, 92, INK, 1.6)
    s += arrow(62, 40, 32, 28, HI, 1.5) + arrow(178, 40, 208, 28, HI, 1.5) + txt(44, 28, "F", "h") + txt(190, 28, "F", "h")
    for x in (70, 100, 150, 180):
        s += arrow(x, 88, x, 72, MUT, 1)
    s += arrow(120, 66, 120, P[12][1], HI, 1.3, True) + txt(124, 54, "z_TE", "h") + txt(120, 104, "air load near the edge, q C_L ℓ", "m", "middle") + txt(120, 14, "Lower F: the edge rises, the wing loses angle", "m", "middle")
    return svg(110, s, "Trailing edge seen from behind, as a string pulled by fibre force F and lifted by the air load")


def sk_choice():
    W = Wing(oy=24)
    s = W.full() + tilted(W, 12, 20) + arrow(*W.p(84, 2), *W.p(84, 14), HI, 1.3, True) + txt(*W.p(87, 11), "λ_n", "h")
    return svg(122, txt(10, 14, "a, b) Tilts θ_b, θ_r; c) chord stretch λ_n", "h") + s + txt(10, 118, "Tilts drawn larger than real", "m"), "Wing with tilts and chord stretch highlighted")


def sk_tol():
    W = Wing(oy=24)
    t = math.tan(.18)
    s = W.full() + line(*W.p(60, 0), *W.p(60 + 41.1 * t, 41.1), HI, 1.2, "4 3") + line(*W.p(60, 0), *W.p(60 - 41.1 * t, 41.1), HI, 1.2, "4 3") + line(*W.p(3.8, 0), *W.p(3.8 + 41.1 * t, 41.1), HI, 1.2, "4 3")
    return svg(132, txt(10, 14, "a, b) Rods short of or past vertical", "h") + s + txt(10, 128, "c) film thickness; d) η_f of the roll", "h"), "Wing with rods short of and past vertical highlighted")


def sk_expect():
    c, _ = section(34, 64, 150, 8, .09, ACC)
    return svg(156, c + line(34, 64, 200, 64, MUT, .7, "3 3") + arc(34, 64, 110, 0, -8, HI, 1.3, end=False) + arrow(4, 64, 26, 64, HI, 1.3) + txt(4, 56, "b) U", "h") + txt(150, 94, "α", "h") + txt(84, 30, "a) V; c) η", "h") + arc(52, 132, 24, 35, -25, HI, 1.3, both=True) + line(52, 132, 76, 118, INK, 1.4) + txt(90, 136, "d) stroke, mean pitch", "h"), "Chord at pitch with voltage, speed and stroke highlighted")


def sk_mapcut():
    W = Wing(oy=22)
    s = W.full() + line(*W.p(32, 0), *W.p(32, 41.1), HI, 1.8, "4 2") + line(*W.p(72, 0), *W.p(72, W.te(72)), HI, 1.8, "4 2") + txt(*W.p(25, -4), "x = 32 mm", "h") + txt(*W.p(66, -4), "x = 72 mm", "h")
    return svg(120, s + txt(10, 116, "Height w of the sheet, 0 V and 1.6 kV", "h"), "Wing with two chordwise cuts")


def sk_acc():
    s = ""
    for i in range(9):
        s += line(14 + i * 12, 26, 14 + i * 12, 86, MUT, .5)
    for j in range(6):
        s += line(14, 26 + j * 12, 110, 26 + j * 12, MUT, .5)
    for i in range(8):
        for j in range(5):
            s += line(14 + i * 12, 26 + j * 12, 26 + i * 12, 38 + j * 12, MUT, .5)
    s += txt(14, 14, "a) mesh h", "h") + line(132, 40, 226, 40, HI, 3) + line(180, 40, 180, 90, HI, 1.6) + txt(132, 14, "c) real rods", "h") + txt(132, 32, "LE tube", "m") + txt(184, 72, "batten", "m")
    return svg(120, s + txt(14, 106, "b) vortex lattice; d) solver settling", "h"), "Mesh and flexible rods")


def card():
    """To scale: the sheet as cut, dashed, with its strip and active band; solid, both edges vertical."""
    W = Wing(ox=78, oy=46, s=4.6, xb=60.0)
    tn = lambda d: math.tan(math.radians(d))
    c = W.c
    R0, B0 = (W.xr + c * tn(7.5), c), (60 + c * tn(4.5), c)
    T0 = (108.0, 3.1)

    def strip(a, b, wd):
        ux, uy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(ux, uy)
        nx, ny = -uy / L, ux / L
        if ny < 0:
            nx, ny = -nx, -ny
        return [a, b, (b[0] + nx * wd, b[1] + ny * wd), (a[0] + nx * wd, a[1] + ny * wd)]
    P = lambda q: [W.p(*a) for a in q]
    s = ""
    for a, b in ((R0, B0), (B0, T0)):
        s += poly(P(strip(a, b, 20)), "none", MUT, .8) + poly(P(strip(a, b, 5)), HI, HI, .4)
    s += poly(W.outline(), SHEET, MUT, .8) + line(*W.p(1, 0), *W.p(109, 0), INK, 3.2)
    s += line(*W.p(W.xr, 0), *W.p(W.xr, c), INK, 1.8) + line(*W.p(60, 0), *W.p(60, c), INK, 1.8)
    s += pl([W.p(W.xr, c), W.p(60, c), W.p(108, 3.1)], ACC, 3)
    s += pl([W.p(W.xr, 0), W.p(*R0), W.p(*B0), W.p(108, 3.1)], INK, 1, "5 4") + line(*W.p(60, 0), *W.p(*B0), INK, 1, "5 4")
    s += txt(*W.p(1, -3), "LE rod, to the crank") + txt(*W.p(8, 20), "θ_r = 7.5°") + txt(*W.p(63, 16), "θ_b = 4.5°")
    s += arrow(*W.p(W.xr, c), *W.p(-5, c), INK, 1.5) + txt(4, W.p(0, c)[1] - 12, "Line to the body,") + txt(4, W.p(0, c)[1] + 16, "about 81 mN")
    s += txt(*W.p(14, 50), "Active band 5 mm", "h") + txt(*W.p(14, 66), "Strip 20 mm: the passive core rolls first")
    s += txt(*W.p(72, 58), "Solid: both edges vertical") + txt(*W.p(72, 62), "Dashed: as cut, chord 39.1 mm")
    return f'<svg viewBox="0 0 660 370" role="img" aria-label="Wing to scale as cut and with both edges vertical, with the rolled strip and its active band">{s}</svg>'


STROKE_FIX = r"$U$ = 4 m/s, $f$ = 2 Hz, $\Phi_0$ = 30°, $\alpha$ = 6°. 0 V and 1.6 kV."
CYCLE_FIX = r"$f$ = 2 Hz, $\Phi_0$ = 30°, $\alpha$ = 6°, 1.6 kV."
S = {
    "M1": (sk_stroke, "Phase over a cycle, for each wing.", STROKE_FIX),
    "M2": (sk_speed, "$U$ from 2 to 7 m/s.", CYCLE_FIX),
    "M3": (sk_eta, r"$\eta$ from 0.05 to 0.8.", r"$U$ = 4 m/s, " + CYCLE_FIX),
    "M4": (sk_voltage, "$V$ from 0 to 1.8 kV.", r"$U$ = 4 m/s, $\alpha$ = 6°, no stroke."),
    "M5": (sk_stretch, r"a) $\lambda_{pre}$ 1.00–1.05. b) $\theta_b$ 3–14°.", r"$U$ = 4 m/s, " + CYCLE_FIX),
    "M6": (sk_tendon, r"$T_c$ from 0.01 to 10 N. b) also $\lambda_{pre}$ 1.03 and 1.01.", r"Discrete wing, $U$ = 4 m/s, " + CYCLE_FIX),
    "M8": (sk_pitch, r"a) $\alpha$ = 6, 8, 10°, over the stroke. b) $f$ and $U$.", r"$\Phi_0$ = 30°, 1.6 kV. Full membrane."),
    "M9": (sk_rotation, r"a, b) $\theta_r$, at $\theta_b$ = 7, 10, 14°. c) 18 cases of tilt, layers, $\eta$, $\lambda_n$.", r"a, b) No flow, both edges vertical. c) $U$ = 4 m/s, $\alpha$ = 6°, 1.6 kV."),
    "M7": (sk_layers, "a) $n$ from 1 to 6. b) 40 cases of stretch and layers.", r"Earlier stretched design, $\eta$ = 0.4, passive fibre, 4 m/s, 6°, 1.6 kV."),
    "F_tension": (sk_tension, "How the sheet is pulled, and c) $\\theta$, f) $\\Delta\\lambda_n$.", "Earlier stretched design. In-plane only: no flow, 0 V."),
    "F_design": (sk_design, "a) Layers 1–6. b) Layers per panel. c) Roll 9–25 mm. d) Batten tip $x$ and chord line $y$.", r"$U$ = 4 m/s, $\alpha$ = 6°, 1.6 kV, $\eta$ = 0.4, rigid frame."),
    "F_final": (sk_final, r"a) $V$. b, c) $\alpha$, $U$. d) $\eta$, stack. e) Errors in $\lambda_d$, $\lambda_n$. f) Rod.", r"Earlier design: 1 layer, $\lambda_d$ = 1.05, $\eta$ = 0.4, rigid frame unless varied."),
    "F_shape": (sk_shape, "0 V or 1.6 kV. Position along the chord.", r"Earlier design, $U$ = 4 m/s, $\alpha$ = 6°."),
    "F_sweep": (sk_sweep, r"$\lambda_n$ above necking, $\lambda_d$, and $U$.", r"Earlier design, $\alpha$ = 6°, $\eta$ = 0.4."),
    "F_vocal": (sk_vocal, "a, b) Voltage. c) Time, at 1.8 kV.", "No flow, 1 Hz drive. One device, measured."),
    "F_flap": (sk_flapcompare, r"Phase; $\eta$ and $\lambda_{pre}$; $U$ and $f$.", r"Earlier design, finite elements. $\Phi_0$ = 30°, $\alpha$ = 6°, 1.6 kV."),
    "O": (sk_bench, r"a) Stretch and voltage. b–e) Voltage, on tested wings.", "Measured, not modelled."),
    "S_trends": (sk_section, r"a, b) $\lambda_{c,0}$. c) $s$. d) $\lambda_{d,0}$. e, f) $\alpha$, $U$.", r"$\lambda_{s,0}$ = 1.03, $s$ = 12 mm, $\lambda_{d,0}$ = 1.10 unless varied."),
    "S_loadpath": (sk_load, r"a, b) Air gap and film thickness. c) Anchor at the TE.", "1.6 kV. a) measured against the ideal roll."),
    "S_flapping": (sk_flapsec, "a, b) $U$, at each $f$. c) Phase, at 1 Hz.", r"$\Phi_0$ = 30°, mean pitch 6° in c), $c$ = 7 cm."),
    "P1": (sk_static, r"$V$ and $\alpha$.", "Default discrete wing, no stroke."),
    "P2": (sk_schedule, "Phase, and when the DEAs are on.", "Default discrete wing. Assumes switching within a half-stroke."),
    "P3": (sk_p3, "a) DEA number. b) DEA stretch. c) Membrane stretch. d) Directions. e, f) Planform.", "Default discrete wing, 1.6 kV."),
    "C": (sk_comparison, "Setup: bench, static wing, flapping wing.", "Each model input as measured on that setup."),
    "F_p7670": (sk_fit, r"a) Stretch. b) Measurement method.", "P7670, the 5-DEA bundle."),
}

S.update({
    "D_layers": (sk_layers, "Active layers $n$, active and passive fibre.", r"$\theta_b$ 4.5°, $\theta_r$ 7.5°, $\lambda_n$ 1.05, $w_s$ 20 mm, $w_e$ 5 mm. 4 m/s, 6°, 1.6 kV, $\eta$ 0.5."),
    "D_fibre": (sk_fibre, "a) $V$. b, c) All cases of the sweeps: layers, roll, band, tilt.", r"4 m/s, 6°, $\eta$ 0.5 unless varied."),
    "D_shape_choice": (sk_choice, r"a) $\theta_b$. b) $\theta_r$. c) $\lambda_n$.", r"$n$ = 2, $w_s$ 20 mm, $w_e$ 5 mm. a, c) 4 m/s, 6°, 1.6 kV; c) $\theta_b$ 5°, the run before the final pick. b) No flow."),
    "D_roll": (sk_roll, "a) $w_s$. b, c) $w_e$.", "The design otherwise."),
    "D_tolerance": (sk_tol, r"Rod angles, film thickness, $\eta_f$.", "The design otherwise. 4 m/s, 6°, 1.6 kV."),
    "D_expect": (sk_expect, r"a) $V$. b) $U$. c) $\eta$, active and passive fibre. d) Phase, at 6° and 8° mean pitch.", r"The design otherwise. d) 4 m/s, 2 Hz, $\Phi_0$ = 30°."),
    "D_map": (sk_mapcut, "Voltage: 0 and 1.6 kV.", "The design, 4 m/s, 6°."),
    "D_accuracy": (sk_acc, "a) Mesh. b) Lattice. c) Rods. d) Solver.", "The design otherwise."),
})

PAGES = {
    "design/index.html": {"../full_membrane/validation/D_layers.png": "D_layers", "../full_membrane/validation/D_fibre.png": "D_fibre", "../full_membrane/validation/D_shape_choice.png": "D_shape_choice", "../full_membrane/validation/D_roll.png": "D_roll", "../full_membrane/validation/D_tolerance.png": "D_tolerance", "../full_membrane/validation/D_expect.png": "D_expect", "../full_membrane/validation/D_map.png": "D_map", "../full_membrane/validation/D_accuracy.png": "D_accuracy"},
    "index.html": {"model/M1_stroke.png": "M1", "model/M2_speed.png": "M2", "model/M3_eta.png": "M3"},
    "manufacturing/index.html": {},
    "model/index.html": {"M4_voltage.png": "M4", "M5_stretch.png": "M5", "M6_tendon.png": "M6", "M8_pitch.png": "M8", "M9_rotation.png": "M9", "M7_check.png": "M7"},
    "full_membrane/index.html": {"../model/M9_rotation.png": "M9", "validation/F_tension.png": "F_tension", "validation/F_design.png": "F_design", "validation/F_final.png": "F_final", "validation/F_shape.png": "F_shape", "validation/F_sweep.png": "F_sweep", "validation/F_vocal.png": "F_vocal", "validation/F_flap.png": "F_flap"},
    "plagiopatagiales/index.html": {"validation/O_observations.png": "O", "validation/S_trends.png": "S_trends", "validation/S_loadpath.png": "S_loadpath", "validation/S_flapping.png": "S_flapping", "validation/P1_quasistatic.png": "P1", "validation/P2_flapping.png": "P2", "validation/P3_design.png": "P3", "validation/C_comparison.png": "C", "validation/F_p7670.png": "F_p7670"},
}


def block(key):
    f, var, fix = S[key]
    return f'<!--s:{key}--><aside class="ps">{f()}<p><b>Varied:</b> {var}</p><p><b>Fixed:</b> {fix}</p></aside><!--/s-->'


def main():
    for page, items in PAGES.items():
        p = os.path.join(HERE, page)
        if not os.path.exists(p):
            continue
        s = open(p, encoding="utf8").read()
        if ".plotrow{" not in s:
            s = s.replace("</style>", CSS + "</style>", 1)
        else:
            s = re.sub(r"\.plotrow\{.*?\.ps p\+p b\{color:var\(--muted\)\}\n", lambda m: CSS, s, count=1, flags=re.S)
        s = re.sub(r"<!--card-->.*?<!--/card-->", lambda _: "<!--card--><div class=\"pcard\">" + card() + "</div><!--/card-->", s, flags=re.S)
        for src, key in items.items():
            m = re.search(r'<img src="' + re.escape(src) + r'"[^>]*>', s)
            assert m, src
            tag = m[0]
            mark = re.compile(r"<!--s:" + re.escape(key) + r"-->.*?<!--/s-->", re.S)
            if s[m.end():].startswith("<!--s:"):
                s = s[:m.end()] + mark.sub(lambda _: block(key), s[m.end():], count=1)
            else:
                s = s[:m.start()] + '<div class="plotrow">' + tag + block(key) + "</div>" + s[m.end():]
        open(p, "w", encoding="utf8").write(s)
        print(page, len(items))


HEX = {"var(--ink)": "#14171a", "var(--muted)": "#5d646b", "var(--accent-dim)": "#dce7f5", "var(--ps-hi)": "#c8440a", "var(--accent)": "#1f64b5"}


def export(folder):
    """Each schematic as a file of its own, with the colours and text styles written in."""
    os.makedirs(folder, exist_ok=True)
    st = '<style>text{font-family:Arial,sans-serif;font-size:10.5px;fill:#14171a}text.h{fill:#c8440a;font-weight:600}text.m{fill:#5d646b}</style>'
    for name, sv in [(k, S[k][0]()) for k in S] + [("design_card", card())]:
        for a, b in HEX.items():
            sv = sv.replace(a, b)
        sv = sv.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1).replace(">", ">" + st, 1)
        open(os.path.join(folder, name + ".svg"), "w", encoding="utf8").write(sv)


if __name__ == "__main__":
    import sys
    main()
    if len(sys.argv) > 1:
        export(sys.argv[1])
