"""Schematic 'block' figures for each result, as hand-built SVG (gradients, shadows, crisp text).
One visual language: blue = DEA skin, maroon tube = rolled DEA edge, carbon = leading edge, amber = voltage,
teal = air. Writes results/blocks/B1_robot.svg ... B8_flight.svg (+ B1_system.svg).
Run: python blocks.py  (then render.py to check them as PNG)"""
import os, math
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "blocks"); os.makedirs(OUT, exist_ok=True)

INK, MUTED, LINE = "#0f172a", "#64748b", "#cbd5e1"
BLUE, BLUE_D, BLUE_L = "#3b82f6", "#1d4ed8", "#dbeafe"
MAROON, ROSE = "#750014", "#fb7185"
AMBER, AMBER_L = "#f59e0b", "#fef3c7"
TEAL, TEAL_L = "#0d9488", "#ccfbf1"
FONT = "Helvetica,Arial,sans-serif"

DEFS = f"""<defs>
<linearGradient id="skin" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#eff6ff"/><stop offset="0.55" stop-color="#bfdbfe"/><stop offset="1" stop-color="#60a5fa"/></linearGradient>
<linearGradient id="skinV" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fffbeb"/><stop offset="0.5" stop-color="#fde68a"/><stop offset="1" stop-color="#f59e0b"/></linearGradient>
<linearGradient id="carbon" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#475569"/><stop offset="0.5" stop-color="#0f172a"/><stop offset="1" stop-color="#334155"/></linearGradient>
<linearGradient id="core" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#bfdbfe"/><stop offset="0.35" stop-color="#eff6ff"/><stop offset="0.7" stop-color="#60a5fa"/><stop offset="1" stop-color="#1e3a8a"/></linearGradient>
<linearGradient id="frame" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#e2e8f0"/></linearGradient>
<linearGradient id="card" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#f8fafc"/></linearGradient>
<linearGradient id="air" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#f0fdfa"/><stop offset="1" stop-color="#ccfbf1"/></linearGradient>
<radialGradient id="glow"><stop offset="0" stop-color="#fbbf24" stop-opacity="0.55"/><stop offset="1" stop-color="#fbbf24" stop-opacity="0"/></radialGradient>
<radialGradient id="ball" cx="0.35" cy="0.3"><stop offset="0" stop-color="#ffffff"/><stop offset="0.4" stop-color="#93c5fd"/><stop offset="1" stop-color="#1d4ed8"/></radialGradient>
<radialGradient id="ballR" cx="0.35" cy="0.3"><stop offset="0" stop-color="#ffffff"/><stop offset="0.4" stop-color="#fda4af"/><stop offset="1" stop-color="#750014"/></radialGradient>
<filter id="sh" x="-20%" y="-20%" width="140%" height="160%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.16"/></filter>
<filter id="soft" x="-10%" y="-10%" width="120%" height="140%"><feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.18"/></filter>
<marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>
<marker id="ahA" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{AMBER}"/></marker>
<marker id="ahM" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MAROON}"/></marker>
<marker id="ahT" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{TEAL}"/></marker>
<marker id="ahB" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{BLUE_D}"/></marker>
</defs>"""


import re as _re


def journal(body):
    """Strip the infographic look: no card frames, no shadows, no glow, smaller and lighter type."""
    body = body.replace('filter="url(#sh)"', "").replace('filter="url(#soft)"', "")
    body = _re.sub(r'<rect x="[^"]+" y="[^"]+" width="[^"]+" height="[^"]+" rx="1[468]" fill="(#fff|#ffffff|url\(#card\))" stroke="#e2e8f0"\s*/>', "", body)
    body = body.replace('stroke="#e2e8f0"', 'stroke="none"')
    body = _re.sub(r'<circle [^>]*fill="url\(#glow\)"/>', "", body)
    def fs(m):
        v = float(m.group(1))
        n = 11.5 if v <= 14 else 12.5 if v <= 16 else 13.5 if v <= 18 else 16 if v <= 24 else 20
        return 'font-size="%g"' % n
    body = _re.sub(r'font-size="([0-9.]+)"', fs, body)
    body = body.replace('font-weight="800"', 'font-weight="600"').replace('font-weight="700"', 'font-weight="600"')
    return body


def letter(x, y, ch):
    return f'<text x="{x}" y="{y}" font-size="17" font-weight="700" fill="{INK}">{ch}</text>'


def svg(w, h, body, title=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 72 {w} {h - 72}" width="{w}" height="{h - 72}" font-family="{FONT}" role="img" aria-label="{title}">'
            f'{DEFS}<rect x="0" y="72" width="{w}" height="{h - 72}" fill="#ffffff"/>{journal(body)}</svg>')


def _svg_old(w, h, body, title=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}" role="img" aria-label="{title}">'
            f'{DEFS}<rect x="0" y="0" width="{w}" height="{h}" rx="18" fill="url(#card)" stroke="#e2e8f0"/>{body}</svg>')


def t(x, y, s, size=15, color=INK, weight=400, anchor="start", italic=False, op=1):
    st = ' font-style="italic"' if italic else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" opacity="{op}"{st}>{s}</text>'


def tt(x, y, lines, size=14, color=INK, weight=400, anchor="start", lh=1.25):
    return "".join(t(x, y + i * size * lh, l, size, color, weight, anchor) for i, l in enumerate(lines))


def path(d, stroke=INK, w=2, fill="none", extra=""):
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{w}" fill="{fill}" stroke-linecap="round" stroke-linejoin="round" {extra}/>'


def line(x1, y1, x2, y2, stroke=INK, w=2, extra=""):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}" stroke-linecap="round" {extra}/>'


def arrow(x1, y1, x2, y2, color=INK, w=2.2, mk="ah", extra=""):
    return line(x1, y1, x2, y2, color, w, f'marker-end="url(#{mk})" {extra}')


def poly(pts, fill, stroke="none", w=1, extra=""):
    return f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" stroke-linejoin="round" {extra}/>'


def qbez(p0, c, p1):
    return f"M{p0[0]:.1f},{p0[1]:.1f} Q{c[0]:.1f},{c[1]:.1f} {p1[0]:.1f},{p1[1]:.1f}"


def tube(d, w=11):
    """Rolled DEA edge: a tube with a highlight."""
    return (path(d, MAROON, w, extra='filter="url(#soft)"') + path(d, "#be123c", w * 0.55) + path(d, "#fecdd3", w * 0.18, extra='opacity="0.9"'))


def rod(x1, y1, x2, y2, w=7):
    return line(x1, y1, x2, y2, "#0f172a", w) + line(x1, y1 - w * 0.22, x2, y2 - w * 0.22, "#94a3b8", w * 0.25, 'opacity="0.8"')


def cylinder(x, y, L, r, horizontal=True):
    """Rolled DEA core: a cylinder with rings."""
    s = f'<rect x="{x}" y="{y - r}" width="{L}" height="{2 * r}" rx="{r}" fill="url(#core)" stroke="{BLUE_D}" stroke-width="1.4"/>'
    for k in range(1, 8):
        xx = x + L * k / 8
        s += f'<path d="M{xx:.1f},{y - r + 1.5} q-3,{r - 1.5} 0,{2 * r - 3}" stroke="{BLUE_D}" stroke-width="0.9" fill="none" opacity="0.55"/>'
    return s


def bolt(x, y, s=1.0, label=None):
    pts = [(0, 0), (-9, 20), (-1, 20), (-7, 40), (12, 13), (3, 13), (9, 0)]
    out = f'<circle cx="{x + 1 * s:.1f}" cy="{y + 20 * s:.1f}" r="{30 * s:.1f}" fill="url(#glow)"/>'
    out += poly([(x + px * s, y + py * s) for px, py in pts], AMBER, "#b45309", 1.2)
    if label: out += t(x + 18 * s, y + 26 * s, label, 14, "#b45309", 700)
    return out


def spring(x1, y1, x2, y2, n=8, amp=8, color=INK, w=2):
    dx, dy = x2 - x1, y2 - y1; L = math.hypot(dx, dy); ux, uy = dx / L, dy / L; px, py = -uy, ux
    pts = [(x1, y1), (x1 + ux * L * 0.12, y1 + uy * L * 0.12)]
    for k in range(n):
        f = 0.12 + 0.76 * (k + 0.5) / n; sgn = 1 if k % 2 == 0 else -1
        pts.append((x1 + ux * L * f + px * amp * sgn, y1 + uy * L * f + py * amp * sgn))
    pts += [(x1 + ux * L * 0.88, y1 + uy * L * 0.88), (x2, y2)]
    return f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"/>'


def chip(x, y, s, fill=BLUE_L, color=BLUE_D, size=14, w=None, weight=600):
    return t(x, y + size * 1.3, s, size, color if color != INK else "#334155", 400)


def _chip_old(x, y, s, fill=BLUE_L, color=BLUE_D, size=14, w=None, weight=600):
    w = w or (len(s) * size * 0.52 + 22)
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{size * 1.9:.1f}" rx="{size * 0.95:.1f}" fill="{fill}"/>' +
            t(x + w / 2, y + size * 1.32, s, size, color, weight, "middle"))


def tile(x, y, w, h, big, small, color=BLUE_D):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#ffffff" stroke="#e2e8f0" filter="url(#sh)"/>' +
            t(x + 16, y + h * 0.52, big, 30, color, 700) + t(x + 16, y + h * 0.82, small, 13, MUTED))


def num(x, y, k, color=INK):
    return f'<circle cx="{x}" cy="{y}" r="11" fill="{color}"/>' + t(x, y + 4.8, str(k), 13, "#ffffff", 700, "middle")


def wave(x, y, w, h, n=1.5, asym=0.0, color=BLUE_D, sw=2.5, phase=0.0):
    pts = []
    for i in range(121):
        u = i / 120; a = 2 * math.pi * n * u + phase
        v = math.sin(a) + asym * math.cos(2 * a)
        pts.append((x + w * u, y - h * v / (1 + abs(asym))))
    return f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'


def head(x, y, title, sub=None):
    return ""


def _head_old(x, y, title, sub=None):
    s = t(x, y, title, 21, INK, 700)
    if sub: s += t(x, y + 22, sub, 14, MUTED)
    return s


LETTERS = {"B1_robot": [(28, 104, "a"), (745, 104, "b")], "B2_bench": [(28, 104, "a"), (622, 104, "b")],
           "B3_mechanism": [(28, 104, "a"), (428, 104, "b"), (792, 104, "c")], "B4_flapping": [(52, 104, "a"), (572, 104, "b")],
           "B5_wing_state": [(28, 104, "a"), (558, 104, "b")], "B6_model": [(28, 104, "a"), (602, 104, "b")],
           "B7_air": [(28, 104, "a"), (682, 104, "b")], "B8_flight": [(28, 104, "a"), (558, 104, "b"), (28, 372, "c")]}


def write(name, w, h, body, title):
    return _write(name, w, h, body, title)


def _write(name, w, h, body, title):
    open(os.path.join(OUT, name + ".svg"), "w", encoding="utf-8").write(svg(w, h, body, title))


# ============================== B1: the robot ==============================
def robot_side(ox, oy, sc=1.0, stroke_deg=0.0, ghosts=True, edge_state="tense", show_labels=True):
    """Perspective-ish view of B-cube from the side/above. Returns svg string."""
    s = ""
    X = lambda x: ox + x * sc; Y = lambda y: oy + y * sc
    # body: frame box in light 3/4 view
    fx, fy, fw, fh, dz = -175, -45, 150, 95, 26
    top = [(X(fx), Y(fy)), (X(fx + fw), Y(fy)), (X(fx + fw + dz), Y(fy - dz * 0.6)), (X(fx + dz), Y(fy - dz * 0.6))]
    side = [(X(fx + fw), Y(fy)), (X(fx + fw + dz), Y(fy - dz * 0.6)), (X(fx + fw + dz), Y(fy + fh - dz * 0.6)), (X(fx + fw), Y(fy + fh))]
    s += poly(top, "#f1f5f9", "#334155", 1.6) + poly(side, "#e2e8f0", "#334155", 1.6)
    s += f'<rect x="{X(fx):.1f}" y="{Y(fy):.1f}" width="{fw * sc:.1f}" height="{fh * sc:.1f}" rx="{8 * sc:.1f}" fill="url(#frame)" stroke="#334155" stroke-width="1.8" filter="url(#sh)"/>'
    s += cylinder(X(fx + 14), Y(fy + 28), 118 * sc, 13 * sc) + cylinder(X(fx + 14), Y(fy + 66), 118 * sc, 13 * sc)
    # hinge and lever
    hx, hy = X(0), Y(-10)
    s += line(X(fx + 132), Y(fy + 28), X(-12), Y(-14), "#334155", 3.2 * sc) + line(X(-12), Y(-14), hx, hy, "#334155", 3.2 * sc)
    anc = (X(-14), Y(78))
    R = 330 * sc

    def wing(deg, alpha, ghost=False):
        a = math.radians(deg)
        tip = (hx + R * math.cos(a), hy - R * math.sin(a) * 0.55 - R * 0.03)
        mid_te = ((tip[0] + anc[0]) / 2 + 40 * sc, (tip[1] + anc[1]) / 2 + (40 if edge_state == "slack" else 26) * sc)
        out = ""
        mem = f"M{hx:.1f},{hy:.1f} L{tip[0]:.1f},{tip[1]:.1f} Q{mid_te[0]:.1f},{mid_te[1]:.1f} {anc[0]:.1f},{anc[1]:.1f} Z"
        if ghost:
            out += f'<path d="{mem}" fill="#93c5fd" opacity="{alpha}"/>' + line(hx, hy, tip[0], tip[1], "#0f172a", 3 * sc, f'opacity="{alpha * 1.6}"')
            return out
        out += f'<path d="{mem}" fill="url(#skin)" stroke="#93c5fd" stroke-width="1" opacity="0.96" filter="url(#sh)"/>'
        # skin texture: faint chordwise lines (electrode pattern)
        for k in range(1, 9):
            f = k / 9; p = (hx + (tip[0] - hx) * f, hy + (tip[1] - hy) * f)
            q = (anc[0] + (tip[0] - anc[0]) * f, anc[1] + (tip[1] - anc[1]) * f + (mid_te[1] - (anc[1] + tip[1]) / 2) * 4 * f * (1 - f))
            out += line(p[0], p[1], q[0], q[1], "#ffffff", 1.1, 'opacity="0.55"')
        te = qbez(anc, mid_te, tip)
        if edge_state == "slack":
            out += path(te, MAROON, 10 * sc, extra='stroke-dasharray="2 0" opacity="0.95"') + path(te, "#be123c", 5 * sc) + path(te, "#fecdd3", 1.6 * sc)
        else:
            out += tube(te, 10 * sc)
        out += rod(hx, hy, tip[0], tip[1], 7 * sc)
        out += f'<circle cx="{anc[0]:.1f}" cy="{anc[1]:.1f}" r="{6 * sc:.1f}" fill="{MAROON}" stroke="#fff" stroke-width="1.5"/>'
        return out, tip
    if ghosts:
        s += wing(stroke_deg + 13, 0.16, True) + wing(stroke_deg - 13, 0.16, True)
    w_, tip = wing(stroke_deg, 1.0)
    s += w_
    s += f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="{9 * sc:.1f}" fill="#fff" stroke="#0f172a" stroke-width="2.4"/><circle cx="{hx:.1f}" cy="{hy:.1f}" r="{3 * sc:.1f}" fill="#0f172a"/>'
    return s, dict(hinge=(hx, hy), tip=tip, anchor=anc, core1=(X(fx + 14), Y(fy + 28)), core2=(X(fx + 14), Y(fy + 66)), frame=(X(fx), Y(fy)))


def B1():
    W, H = 1100, 520
    b = head(36, 52, "B-cube: a flapping robot whose wing is an actuator", "the skin and its rolled trailing edge are both dielectric elastomer actuators (DEAs)")
    r, P = robot_side(330, 300, 1.0)
    b += r
    # stroke arc
    hx, hy = P["hinge"]
    b += path(f"M{hx + 250:.1f},{hy - 105:.1f} A270,270 0 0 1 {hx + 262:.1f},{hy + 40:.1f}", INK, 2, extra='marker-start="url(#ah)" marker-end="url(#ah)" opacity="0.65"')
    b += t(hx + 272, hy - 30, "stroke", 15, MUTED, 600)
    b += bolt(95, 395, 1.0, "0–1.4 kV")
    b += path("M118,385 C150,370 160,360 175,345", AMBER, 2.2, extra='marker-end="url(#ahA)" stroke-dasharray="5 4"')
    # callouts
    c1 = P["core1"]; c2 = P["core2"]; tip = P["tip"]; anc = P["anchor"]
    calls = [((c1[0] + 30, c1[1]), (c1[0] - 10, 160), 1, "DEA core (drives)"), ((c2[0] + 30, c2[1]), (c2[0] - 10, 448), 2, "DEA core (spring)"),
             (((hx + tip[0]) / 2 + 40, (hy + tip[1]) / 2 - 3), (470, 135), 3, "carbon leading edge, 50 mm"),
             (((hx + tip[0]) / 2 + 30, (hy + tip[1]) / 2 + 30), (640, 345), 4, "DEA skin"),
             ((anc[0] + 150, anc[1] + 2), (420, 470), 5, "rolled DEA edge: actuator + contacts")]
    for (px, py), (lx, ly), k, txt in calls:
        b += line(px, py, lx, ly, MUTED, 1.2, 'stroke-dasharray="3 3"') + f'<circle cx="{lx}" cy="{ly}" r="2.5" fill="{MUTED}"/>' + t(lx + 8, ly + 5, txt, 15, INK, 400)
    # stack inset
    ix, iy = 760, 92
    b += f'<rect x="{ix}" y="{iy}" width="310" height="182" rx="16" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
    b += t(ix + 18, iy + 30, "skin, about 60 µm", 15, INK, 700)
    lay = [(BLUE_L, 9), (MAROON, 3), ("#bfdbfe", 18), (MAROON, 3), (BLUE_L, 9)]
    yy = iy + 46
    for k, (col, hh) in enumerate(lay):
        sk = 14
        b += poly([(ix + 22, yy), (ix + 150, yy), (ix + 150 + sk, yy - sk * 0.55), (ix + 22 + sk, yy - sk * 0.55)], col if col != MAROON else "#9f1239", "#ffffff", 0.6)
        b += f'<rect x="{ix + 22}" y="{yy:.1f}" width="128" height="{hh * 2.2:.1f}" fill="{col}" stroke="#fff" stroke-width="0.6"/>'
        yy += hh * 2.2
    b += t(ix + 22, yy + 22, "silicone · CNT electrodes", 12.5, MUTED)
    # roll spiral
    cx, cy = ix + 240, iy + 100
    sp = []
    for i in range(400):
        th = i / 400 * 4.3 * math.pi; rr = 6 + 26 * th / (4.3 * math.pi); sp.append((cx + rr * math.cos(th), cy + rr * math.sin(th)))
    pts = " ".join(f"{a:.1f},{c:.1f}" for a, c in sp)
    b += f'<polyline points="{pts}" fill="none" stroke="#bfdbfe" stroke-width="7" stroke-linecap="round"/><polyline points="{pts}" fill="none" stroke="{MAROON}" stroke-width="1.6"/>'
    b += t(cx, iy + 52, "rolled edge", 13, MUTED, 600, "middle")
    # stat tiles
    tiles = []
    for k, (bg, sm) in enumerate(tiles):
        b += tile(760 + (k % 2) * 160, 296 + (k // 2) * 102, 150, 88, bg, sm, [BLUE_D, BLUE_D, INK, "#b45309"][k])
    write("B1_robot", W, H, b, "B-cube robot schematic")


# ============================== B1 system diagram ==============================
def card(x, y, w, h, title, lines, accent=BLUE, icon=""):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
    s += f'<rect x="{x}" y="{y}" width="7" height="{h}" rx="3.5" fill="{accent}"/>'
    s += icon + t(x + 22, y + 30, title, 17, INK, 700)
    s += tt(x + 22, y + 54, lines, 13.5, "#334155")
    return s


def B1_system():
    W, H = 1100, 420
    b = head(36, 52, "One drive, two actuators, two modes", "how the parts act on each other, and what the model keeps")
    b += card(36, 100, 200, 150, "Drive", ["sine squared, 0–1.4 kV", "1–15 Hz", "one signal for both"], AMBER)
    b += wave(60, 228, 150, 10, 2.0, 0, AMBER, 2.2)
    b += card(286, 90, 230, 110, "DEA core", ["force ∝ V²", "lag τ ≈ 40 ms"], BLUE)
    b += card(286, 250, 230, 130, "Wing DEA", ["skin + rolled edge", "edge tension T₀ − ΔF(V)", "robot: T₀ ≈ 4 mN"], BLUE)
    b += card(586, 70, 250, 140, "Stroke mode", ["8.4 Hz, damping 0.07", "θ″ + 2ζωθ′ + ω²θ + βθ²", "     = G·u(t) − μ·ω_w²·q"], INK)
    b += card(586, 250, 250, 140, "Wing mode", ["17 Hz at 0 V (edge ≈ 4 mN)", "≈ 10 Hz when the edge is slack", "q″ + 2ζ_wω_w q′ + ω_w² q = −θ″"], MAROON)
    b += card(906, 160, 160, 120, "Video", ["leading-edge angle", "θ + q, 200 fps"], TEAL)
    b += arrow(227, 160, 282, 145, AMBER, 2.6, "ahA") + arrow(227, 190, 282, 300, AMBER, 2.6, "ahA")
    b += arrow(517, 145, 582, 140, INK, 2.4) + t(530, 132, "u(t)", 14, MUTED, 600, italic=True)
    b += arrow(517, 315, 582, 320, INK, 2.4) + t(533, 306, "T", 14, MUTED, 600, italic=True)
    b += arrow(680, 212, 680, 246, MAROON, 2.8, "ahM") + arrow(742, 246, 742, 212, MAROON, 2.8, "ahM")
    b += t(626, 235, "−θ″", 15, MAROON, 700) + t(752, 235, "−μω_w² q", 15, MAROON, 700)
    b += arrow(837, 140, 902, 200, TEAL, 2.4, "ahT") + arrow(837, 320, 902, 245, TEAL, 2.4, "ahT")
    b += chip(286, 408, "voltage slackens the edge  →  the wing mode leaves the band  →  the stroke is freed", AMBER_L, "#b45309", 14, 550)
    write("B1_system", W, H + 40, b, "System block diagram")


# ============================== B2: bench ==============================
def B2():
    W, H = 1100, 430
    b = head(36, 52, "On the bench, the edge drops out of plane", "wing clamped at the leading edge; sine-squared voltage")
    # clamped wing in perspective
    L0 = (90, 150); L1 = (560, 120)      # leading edge
    A = (110, 300)                       # root of the edge
    T0 = [(110, 300), (330, 268), (560, 120)]
    b += f'<rect x="60" y="130" width="40" height="40" rx="6" fill="#334155"/>' + t(80, 192, "clamp", 12, MUTED, 600, "middle")
    mem0 = f"M{L0[0]},{L0[1]} L{L1[0]},{L1[1]} Q{T0[1][0]},{T0[1][1]} {A[0]},{A[1]} Z"
    b += f'<path d="{mem0}" fill="url(#skin)" filter="url(#sh)"/>'
    # voltage state: edge sagging out of plane (ghost, amber)
    memV = f"M{L0[0]},{L0[1]} L{L1[0]},{L1[1]} Q330,350 {A[0]},{A[1]} Z"
    b += f'<path d="M{A[0]},{A[1]} Q330,268 {L1[0]},{L1[1]} Q330,350 {A[0]},{A[1]} Z" fill="{AMBER}" opacity="0.28"/>'
    b += tube(qbez(A, (330, 268), L1), 10)
    b += path(qbez(A, (330, 350), L1), AMBER, 9, extra='opacity="0.85"') + path(qbez(A, (330, 350), L1), "#fde68a", 2.5)
    b += rod(L0[0], L0[1], L1[0], L1[1], 7)
    b += arrow(330, 278, 330, 318, AMBER, 2.6, "ahA") + t(342, 306, "1.8 mm out of plane", 15, "#b45309", 700)
    b += arrow(250, 230, 218, 230, INK, 2, "ah", 'opacity="0.6"') + t(140, 222, "0.3 mm in plane", 13, INK, 600)
    b += bolt(150, 330, 0.9, "1.8 kV, 1 Hz")
    # response sketch: flat, steep, saturating
    gx, gy, gw, gh = 640, 110, 400, 250
    b += f'<rect x="{gx - 20}" y="{gy - 30}" width="{gw + 40}" height="{gh + 70}" rx="16" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
    b += arrow(gx + 10, gy + gh, gx + gw, gy + gh, INK, 1.8) + arrow(gx + 10, gy + gh, gx + 10, gy, INK, 1.8)
    b += t(gx + gw - 4, gy + gh + 26, "voltage", 14, MUTED, 600, "end") + t(gx + 18, gy + 6, "edge drop", 14, MUTED, 600)
    pts = []
    for i in range(101):
        v = i / 100; y = 1 / (1 + math.exp(-(v - 0.62) * 14))
        pts.append((gx + 20 + v * (gw - 40), gy + gh - 12 - y * (gh - 50)))
    b += f'<polyline points="{" ".join(f"{a:.1f},{c:.1f}" for a, c in pts)}" fill="none" stroke="{AMBER}" stroke-width="4" stroke-linecap="round"/>'
    for (vx, lab) in ((0.25, "flat"), (0.62, "steep"), (0.92, "saturates")):
        px = gx + 20 + vx * (gw - 40); py = gy + gh - 12 - (1 / (1 + math.exp(-(vx - 0.62) * 14))) * (gh - 50)
        b += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="#fff" stroke="{AMBER}" stroke-width="2.5"/>' + t(px, py - 14, lab, 14, INK, 700, "middle")
    b += chip(gx + 30, gy + 20, "cycle high ∝ V^3.2", AMBER_L, "#b45309", 14)
    b += chip(gx + 30, gy + 60, "13× less with more tension", "#f1f5f9", INK, 14)
    write("B2_bench", W, H, b, "Bench result schematic")


# ============================== B3: mechanism ==============================
def B3():
    W, H = 1100, 440
    b = head(36, 52, "The edge is a tension switch", "two springs hold the edge; voltage takes one away")
    def panel(x0, title, slack):
        s = f'<rect x="{x0}" y="92" width="330" height="290" rx="18" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
        s += t(x0 + 22, 124, title, 17, INK, 700)
        nx, ny = x0 + 165, 240 + (55 if slack else 12)
        # string (edge tension) left-right anchors
        s += f'<rect x="{x0 + 18}" y="200" width="14" height="80" rx="4" fill="#334155"/><rect x="{x0 + 298}" y="200" width="14" height="80" rx="4" fill="#334155"/>'
        if slack:
            s += path(f"M{x0 + 32},240 C{x0 + 80},300 {x0 + 110},320 {nx},{ny} C{x0 + 220},320 {x0 + 250},300 {x0 + 298},240", MAROON, 6, extra='opacity="0.35" stroke-dasharray="2 7"')
            s += t(x0 + 40, 196, "edge slack: T ≈ 0", 14, MAROON, 700)
        else:
            s += tube(f"M{x0 + 32},240 L{nx},{ny} L{x0 + 298},240", 9)
            s += arrow(nx - 30, ny - 4, x0 + 60, 238, MAROON, 2, "ahM") + arrow(nx + 30, ny - 4, x0 + 270, 238, MAROON, 2, "ahM")
            s += t(x0 + 40, 196, "edge tension T", 14, MAROON, 700)
        # skin spring from below
        s += f'<rect x="{nx - 40}" y="368" width="80" height="10" rx="3" fill="#94a3b8"/>'
        s += spring(nx, 368, nx, ny + 8, 9, 9, BLUE_D, 2.4) + t(nx + 18, 352, "skin N/c", 14, BLUE_D, 700)
        # load
        for dx in (-26, 0, 26):
            s += arrow(nx + dx, ny - 70, nx + dx, ny - 16, TEAL, 2.2, "ahT")
        s += t(nx + 36, ny - 52, "load q", 14, TEAL, 700)
        s += f'<circle cx="{nx}" cy="{ny}" r="9" fill="url(#ballR)" stroke="#fff" stroke-width="1.5"/>'
        return s
    b += panel(36, "0 V", False)
    b += bolt(400, 205, 1.0)
    b += arrow(386, 290, 420, 290, AMBER, 3, "ahA")
    b += panel(436, "at voltage", True)
    # equation and edge mode
    x0 = 800
    b += f'<rect x="{x0}" y="92" width="268" height="290" rx="18" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
    b += chip(x0 + 18, 142, "Δz = q / (8T/L² + N/c)", BLUE_L, BLUE_D, 14.5, 232)
    b += t(x0 + 20, 300, "edge mode", 14, INK, 400)
    b += wave(x0 + 20, 330, 100, 14, 3.0, 0, MAROON, 2.4) + t(x0 + 128, 335, "45 Hz", 14, MAROON, 700)
    b += wave(x0 + 20, 362, 100, 22, 1.2, 0, AMBER, 2.4) + t(x0 + 128, 367, "11–27 Hz", 14, "#b45309", 700)
    write("B3_mechanism", W, H, b, "Tension switch schematic")


# ============================== B4: flapping / resonance ==============================
def B4():
    W, H = 1100, 430
    b = head(36, 52, "The robot is a resonator with a lagging push", "one stroke mode, driven by the DEA core through a first-order lag")
    # mechanical analogue
    x0, y0 = 60, 110
    b += f'<rect x="{x0}" y="{y0}" width="470" height="270" rx="18" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
    b += f'<rect x="{x0 + 24}" y="{y0 + 60}" width="16" height="160" rx="4" fill="#334155"/>'
    b += spring(x0 + 40, y0 + 110, x0 + 220, y0 + 110, 10, 12, BLUE_D, 2.6) + t(x0 + 50, y0 + 84, "stiffness k (cores)", 13.5, BLUE_D, 700)
    # damper
    b += line(x0 + 40, y0 + 175, x0 + 120, y0 + 175, INK, 2.4) + f'<rect x="{x0 + 120}" y="{y0 + 160}" width="50" height="30" fill="none" stroke="{INK}" stroke-width="2.2"/>' + line(x0 + 145, y0 + 175, x0 + 220, y0 + 175, INK, 2.4)
    b += line(x0 + 150, y0 + 163, x0 + 150, y0 + 187, INK, 2.4) + t(x0 + 48, y0 + 222, "damping ζ = 0.07", 13.5, MUTED, 700)
    b += f'<rect x="{x0 + 220}" y="{y0 + 70}" width="110" height="140" rx="14" fill="url(#ball)" stroke="{BLUE_D}" stroke-width="1.5" filter="url(#sh)"/>'
    b += t(x0 + 275, y0 + 145, "wing +", 15, "#fff", 700, "middle") + t(x0 + 275, y0 + 165, "leading edge", 13, "#fff", 600, "middle")
    # actuator with clock (lag)
    b += arrow(x0 + 440, y0 + 140, x0 + 336, y0 + 140, AMBER, 3.2, "ahA")
    b += f'<circle cx="{x0 + 400}" cy="{y0 + 86}" r="24" fill="#fff" stroke="{AMBER}" stroke-width="2.5"/>' + line(x0 + 400, y0 + 86, x0 + 400, y0 + 70, "#b45309", 2.5) + line(x0 + 400, y0 + 86, x0 + 413, y0 + 92, "#b45309", 2.5)
    b += t(x0 + 400, y0 + 128, "lag ≈ 40 ms", 14, "#b45309", 700, "middle") + t(x0 + 390, y0 + 172, "F ∝ V²", 14, "#b45309", 700, "middle")
    b += bolt(x0 + 430, y0 + 190, 0.8)
    # resonance curve
    gx, gy, gw, gh = 580, 110, 470, 270
    b += f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="18" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
    ox, oy, pw, ph = gx + 40, gy + gh - 46, gw - 70, gh - 90
    b += arrow(ox, oy, ox + pw, oy, INK, 1.8) + arrow(ox, oy, ox, oy - ph, INK, 1.8)
    b += t(ox + pw, oy + 28, "drive frequency", 14, MUTED, 600, "end") + t(ox + 8, oy - ph + 4, "stroke", 14, MUTED, 600)
    fx = lambda f: ox + f / 28 * pw
    def resp(f, lag):
        r = f / 8.4; v = 1 / math.sqrt((1 - r * r) ** 2 + (0.14 * r) ** 2)
        return v / math.sqrt(1 + (2 * math.pi * f * lag) ** 2)
    for lag, col, dash in ((0.0, "#94a3b8", "6 6"), (0.04, BLUE_D, None)):
        pts = [(fx(f), oy - min(resp(f, lag), 9) / 9 * (ph - 10)) for f in [0.3 + k * 0.1 for k in range(270)]]
        b += f'<polyline points="{" ".join(f"{a:.1f},{c:.1f}" for a, c in pts)}" fill="none" stroke="{col}" stroke-width="{3.5 if not dash else 2}" {"stroke-dasharray=" + chr(34) + dash + chr(34) if dash else ""}/>'
    b += f'<rect x="{fx(20):.1f}" y="{oy - ph:.1f}" width="{fx(25) - fx(20):.1f}" height="{ph:.1f}" fill="{AMBER_L}" opacity="0.8"/>' + t((fx(20) + fx(25)) / 2, oy - ph + 20, "design", 13, "#b45309", 700, "middle") + t((fx(20) + fx(25)) / 2, oy - ph + 36, "20–25 Hz", 13, "#b45309", 700, "middle")
    b += line(fx(8.4), oy, fx(8.4), oy - ph + 10, BLUE_D, 1.2, 'stroke-dasharray="3 3"') + t(fx(8.4) + 6, oy + 18, "8.4 Hz", 14, BLUE_D, 700)
    b += t(fx(11), oy - ph * 0.86, "no lag", 13, MUTED, 600) + t(fx(10.5), oy - ph * 0.45, "40 ms lag", 13, BLUE_D, 700)
    write("B4_flapping", W, H, b, "Resonance schematic")


# ============================== B5: powered against unpowered ==============================
def B5():
    W, H = 1100, 450
    b = head(36, 52, "Powering the wing frees the stroke", "8 Hz, 1.2 kV on the core in both runs; only the wing's state differs")
    for k, (x0, title, slack, ang, big, harm, col) in enumerate(((36, "wing powered", True, 15, "24.3°", "2nd harmonic 0.7°", BLUE_D), (566, "wing unpowered", False, 11, "20.2°", "2nd harmonic 1.7°", MAROON))):
        b += f'<rect x="{x0}" y="88" width="500" height="330" rx="18" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
        b += t(x0 + 22, 120, title, 18, col, 700)
        if slack: b += bolt(x0 + 200, 98, 0.62)
        hx, hy = x0 + 70, 225; R = 300
        for sgn in (1, -1):
            a = math.radians(ang * sgn)
            tip = (hx + R * math.cos(a), hy - R * math.sin(a))
            anc = (hx - 6, hy + 48)
            mid = ((tip[0] + anc[0]) / 2 + 8, (tip[1] + anc[1]) / 2 + (24 if slack else 8))
            b += f'<path d="M{hx},{hy} L{tip[0]:.1f},{tip[1]:.1f} Q{mid[0]:.1f},{mid[1]:.1f} {anc[0]},{anc[1]} Z" fill="{"url(#skinV)" if slack else "url(#skin)"}" opacity="0.42"/>'
            if slack:
                b += path(qbez(anc, mid, tip), AMBER, 4, extra='opacity="0.8" stroke-dasharray="1 6"')
            else:
                b += path(qbez(anc, mid, tip), MAROON, 4.5, extra='opacity="0.85"')
            b += rod(hx, hy, tip[0], tip[1], 5)
        b += path(f"M{hx + R + 14:.1f},{hy - R * math.sin(math.radians(ang)) :.1f} A{R},{R} 0 0 1 {hx + R + 14:.1f},{hy + R * math.sin(math.radians(ang)):.1f}", col, 2.4, extra='marker-start="url(#ah)" marker-end="url(#ah)"')
        b += t(x0 + 480, 128, big, 34, col, 800, "end")
        b += f'<circle cx="{hx}" cy="{hy}" r="8" fill="#fff" stroke="#0f172a" stroke-width="2.2"/>'
        b += t(x0 + 22, 352, "edge slack" if slack else "edge tense, ≈ 4 mN", 14, "#b45309" if slack else MAROON, 400)
        b += wave(x0 + 22, 392, 300, 15 if slack else 11, 2.0, 0.0 if slack else 0.42, col, 2.6)
        b += t(x0 + 340, 396, harm, 14, col, 700)
    write("B5_wing_state", W, H, b, "Powered against unpowered wing")


# ============================== B6: model ==============================
def B6():
    W, H = 1100, 450
    b = head(36, 52, "A wing mode explains it", "the edge's tension sets a second mode that steals from the stroke")
    x0 = 36
    b += f'<rect x="{x0}" y="88" width="540" height="330" rx="18" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
    # wall, stroke mass, wing mass
    b += f'<rect x="{x0 + 22}" y="150" width="16" height="160" rx="4" fill="#334155"/>'
    b += spring(x0 + 38, 230, x0 + 150, 230, 8, 11, BLUE_D, 2.6)
    b += f'<rect x="{x0 + 150}" y="175" width="120" height="110" rx="14" fill="url(#ball)" stroke="{BLUE_D}" filter="url(#sh)"/>' + t(x0 + 210, 225, "stroke", 16, "#fff", 700, "middle") + t(x0 + 210, 246, "8.4 Hz", 14, "#fff", 600, "middle")
    b += spring(x0 + 270, 230, x0 + 380, 230, 7, 11, MAROON, 2.6) + t(x0 + 290, 205, "coupling μ", 13.5, MAROON, 700)
    b += f'<rect x="{x0 + 380}" y="190" width="90" height="80" rx="12" fill="url(#ballR)" stroke="{MAROON}" filter="url(#sh)"/>' + t(x0 + 425, 228, "wing", 15, "#fff", 700, "middle") + t(x0 + 425, 247, "mode", 13, "#fff", 600, "middle")
    b += arrow(x0 + 120, 150, x0 + 150, 190, AMBER, 2.8, "ahA") + t(x0 + 52, 140, "DEA core push", 13.5, "#b45309", 700)
    # tension knob controlling wing mode
    kx, ky = x0 + 425, 340
    b += f'<circle cx="{kx}" cy="{ky}" r="30" fill="#fff" stroke="{MAROON}" stroke-width="3" filter="url(#soft)"/>' + line(kx, ky, kx - 14, ky - 22, MAROON, 4)
    b += path(f"M{kx - 44},{ky + 6} A44,44 0 0 1 {kx + 40},{ky - 18}", MAROON, 2, extra='marker-end="url(#ahM)" opacity="0.6"')
    b += t(kx - 60, ky + 50, "edge tension T: 4 mN → 0 at voltage", 13.5, MAROON, 700, "middle")
    b += arrow(kx, ky - 32, kx, 274, MAROON, 2, "ahM", 'opacity="0.6"')
    # frequency ruler
    gx, gy, gw = 610, 120, 455
    b += f'<rect x="{gx}" y="88" width="{gw}" height="330" rx="18" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
    fx = lambda f: gx + 40 + (f - 5) / 20 * (gw - 80)
    yl = 250
    b += line(fx(5), yl, fx(25), yl, INK, 2) + "".join(line(fx(f), yl - 6, fx(f), yl + 6, INK, 1.5) + t(fx(f), yl + 26, f"{f}", 13, MUTED, 600, "middle") for f in (5, 10, 15, 20, 25))
    b += t(fx(25), yl + 46, "Hz", 13, MUTED, 600, "end")
    def mark(f, lab, col, up=True, dy=0):
        yy = yl - 40 - dy if up else yl + 60 + dy
        return line(fx(f), yl, fx(f), yy + (8 if up else -8), col, 2.2) + f'<circle cx="{fx(f):.1f}" cy="{yl}" r="7" fill="{col}"/>' + t(fx(f), yy, lab, 14, col, 700, "middle")
    b += mark(8, "stroke 8", BLUE_D, True, 0) + mark(16, "2nd harmonic 16", INK, True, 40) + mark(17, "wing mode 17 (0 V)", MAROON, False, 0)
    b += mark(10, "slack edge ≈ 10", "#b45309", False, 70)
    b += path(f"M{fx(17):.1f},{yl + 14} C{fx(15):.1f},{yl + 60} {fx(12):.1f},{yl + 60} {fx(10.3):.1f},{yl + 22}", AMBER, 2.4, extra='marker-end="url(#ahA)"')
    b += f'<rect x="{fx(15.6):.1f}" y="{yl - 90:.1f}" width="{fx(17.6) - fx(15.6):.1f}" height="80" fill="{MAROON}" opacity="0.1"/>'
    write("B6_model", W, H, b, "Two-mode model schematic")


# ============================== B7: air ==============================
def B7():
    W, H = 1100, 430
    b = head(36, 52, "In air, the active skin cambers the wing", "only ~15 % of the electrostatic stress becomes a tension drop, but the wing sits near divergence")
    x0 = 36
    b += f'<rect x="{x0}" y="88" width="620" height="320" rx="18" fill="url(#air)" stroke="#e2e8f0" filter="url(#sh)"/>'
    for k in range(7):
        y = 130 + k * 40
        b += path(f"M{x0 + 20},{y} C{x0 + 200},{y - 8 - (12 if 2 <= k <= 4 else 0)} {x0 + 420},{y - 14 - (16 if 2 <= k <= 4 else 0)} {x0 + 600},{y - 6}", TEAL, 1.6, extra='opacity="0.55" marker-end="url(#ahT)"')
    le, te = (x0 + 140, 250), (x0 + 520, 262)
    b += path(f"M{le[0]},{le[1]} Q{(le[0] + te[0]) / 2},{le[1] - 34} {te[0]},{te[1]}", BLUE_D, 5)
    b += path(f"M{le[0]},{le[1]} Q{(le[0] + te[0]) / 2},{le[1] - 78} {te[0]},{te[1]}", AMBER, 5, extra='stroke-dasharray="10 6"')
    b += f'<circle cx="{le[0]}" cy="{le[1]}" r="8" fill="#0f172a"/><circle cx="{te[0]}" cy="{te[1]}" r="7" fill="{MAROON}"/>'
    b += t((le[0] + te[0]) / 2, le[1] - 92, "at voltage: more camber", 15, "#b45309", 700, "middle") + t((le[0] + te[0]) / 2, le[1] + 22, "0 V", 15, BLUE_D, 700, "middle")
    b += bolt(x0 + 60, 120, 0.8)
    b += chip(x0 + 24, 352, "active skin: ΔC_L +0.09 to +0.37", AMBER_L, "#b45309", 14.5)
    b += chip(x0 + 330, 352, "C_T ≈ 3.3, divergence at 1.73", "#ffffff", INK, 14.5)
    # discrete fibres vs full skin
    gx = 690
    b += f'<rect x="{gx}" y="88" width="374" height="320" rx="18" fill="#fff" stroke="#e2e8f0" filter="url(#sh)"/>'
    b += t(gx + 22, 124, "discrete fibres, same wing", 14, INK, 400)
    b += poly([(gx + 30, 150), (gx + 340, 150), (gx + 300, 250), (gx + 50, 250)], "url(#skin)", "#93c5fd", 1)
    for k in range(4):
        x = gx + 80 + k * 65
        b += path(f"M{x},150 Q{x + 14},200 {x - 4},{250}", MAROON, 4.5)
    b += path(f"M{gx + 50},250 Q{gx + 175},280 {gx + 300},250", INK, 2, extra='stroke-dasharray="4 4"')
    b += t(gx + 22, 296, "fibres keep 29 % of their pre-tension", 13.5, "#334155")
    b += chip(gx + 22, 330, "half the skin's ΔC_L, +51 mg", "#fee2e2", MAROON, 14)
    write("B7_air", W, H, b, "Wing in air schematic")


# ============================== B8: flight ==============================
def B8():
    W, H = 1100, 430
    b = head(36, 52, "Thrust needs a faster stroke", "flapping makes thrust only above a Strouhal number of about 0.15")
    def wake(x0, title, amp, n, thrust, col, label):
        s = f'<rect x="{x0}" y="88" width="500" height="250" rx="18" fill="url(#air)" stroke="#e2e8f0" filter="url(#sh)"/>'
        s += t(x0 + 22, 120, title, 17, col, 700)
        pts = [(x0 + 40 + u * 420, 215 - amp * math.sin(2 * math.pi * n * u)) for u in [k / 200 for k in range(201)]]
        s += f'<polyline points="{" ".join(f"{a:.1f},{c:.1f}" for a, c in pts)}" fill="none" stroke="{col}" stroke-width="3" stroke-dasharray="2 5" stroke-linecap="round"/>'
        for k in range(int(n * 2)):
            u = (k + 0.25) / (2 * n); xx = x0 + 40 + u * 420; yy = 215 - amp * math.sin(2 * math.pi * n * u)
            r = 6 + amp * 0.18; cw = (k % 2 == 0)
            s += f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="{r:.1f}" fill="none" stroke="{TEAL}" stroke-width="2"/>'
            s += arrow(xx + (r if cw else -r), yy, xx + (r if cw else -r), yy + (-6 if cw else 6), TEAL, 2, "ahT")
        s += arrow(x0 + (470 if thrust else 400), 305, x0 + (410 if thrust else 460), 305, col, 3.4, "ahB" if thrust else "ahM")
        s += t(x0 + 390, 310, label, 15, col, 700, "end")
        return s
    b += wake(36, "as built: 8.4 Hz, ±12°", 14, 2.0, False, MAROON, "net drag")
    b += wake(566, "16–25 Hz, ±20°", 46, 2.0, True, BLUE_D, "net thrust")
    # St scale
    gx, gy, gw = 60, 380, 980
    fx = lambda s: gx + s / 0.4 * gw
    b += line(fx(0), gy, fx(0.4), gy, INK, 2) + "".join(line(fx(v), gy - 5, fx(v), gy + 5, INK, 1.5) + t(fx(v), gy + 22, f"{v:g}", 12.5, MUTED, 600, "middle") for v in (0, 0.1, 0.2, 0.3, 0.4))
    b += f'<rect x="{fx(0.15):.1f}" y="{gy - 10}" width="{fx(0.4) - fx(0.15):.1f}" height="20" fill="{BLUE_L}" opacity="0.9"/>' + t(fx(0.27), gy - 16, "thrust (animals cruise at 0.2–0.4)", 13, BLUE_D, 700, "middle")
    b += f'<circle cx="{fx(0.05):.1f}" cy="{gy}" r="8" fill="{MAROON}"/>' + t(fx(0.05), gy - 16, "robot now, 0.05", 13, MAROON, 700, "middle")
    b += t(fx(0.0), gy - 16, "St = f A / U", 13, MUTED, 700)
    write("B8_flight", W, H + 10, b, "Flight schematic")


if __name__ == "__main__":
    for f in (B1, B1_system, B2, B3, B4, B5, B6, B7, B8):
        f(); print("wrote", f.__name__)
