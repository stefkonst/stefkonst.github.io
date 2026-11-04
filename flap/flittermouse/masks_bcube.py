"""Electrode masks for the first B-cube build, on the 114 mm plates of full_membrane/masks. Straight cut: span as cut = 50 mm / lam_d. Chord unstretched.
Electrodes alternate A, B. Sheet active area inset from the bond edges; passive roll, each electrode reaches its own end tab. Core: two 55 x 23 mm strips per plate.
Run: python masks_bcube.py -> masks/mask_bcube_*.svg, masks/mask_bcube_numbers.json"""
import json, math, os, sys
import xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "full_membrane", "masks", "mask_2partwing_passive_fibre.svg")
OUT = os.path.join(HERE, "masks"); os.makedirs(OUT, exist_ok=True)
LAYER = np.array([63.002573, 44.99419])
CENTRES = {("tip", "A"): (58.15, 91.25), ("tip", "B"): (174.2, 89.75), ("root", "A"): (57.15, 209.5), ("root", "B"): (172.6, 209.25)}
M_LE, M_END, TAB, STRIP, M_STRIP, LAM = 2.0, 1.5, 4.8, 20.0, 1.0, 1.04
COATER, FILTER = 60.0, 90.0
ns = "{http://www.w3.org/2000/svg}"


def wing(cr, ct):
    """Returns electrode outlines A and B and the cut outline, fibre along x, in mm, centred."""
    S = 50.0 / LAM
    p = np.array([(0, 0), (S, 0), (S, ct), (0, cr)], float)         # LE root, LE tip, TE tip, TE root
    ang = -math.atan2(p[2, 1] - p[3, 1], p[2, 0] - p[3, 0])
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    q = p @ R.T
    y_f = q[3, 1]                                                    # the fibre line is horizontal after rotation
    u = np.array([1.0, 0.0])
    strip = np.array([q[3], q[2], q[2] + [0, STRIP], q[3] + [0, STRIP]])
    ln = np.array([[math.cos(ang), math.sin(ang)]])
    # active sheet: inset from LE, root and tip
    sheet = Polygon(q)
    le_n = R @ np.array([0.0, 1.0])
    root_n, tip_n = R @ np.array([1.0, 0.0]), R @ np.array([-1.0, 0.0])
    def cut(poly, a, n, m):
        L = 500.0
        t = np.array([-n[1], n[0]])
        return poly.intersection(Polygon([a + m * n - L * t, a + m * n + L * t, a + m * n + L * t + L * n, a + m * n - L * t + L * n]))
    act = cut(cut(cut(sheet, q[0], le_n, M_LE), q[0], root_n, M_END), q[1], tip_n, M_END)
    x0, x1 = q[3, 0], q[2, 0]
    ta = box(x0, y_f, x0 + TAB, y_f + STRIP - M_STRIP)
    tb = box(x1 - TAB, y_f, x1, y_f + STRIP - M_STRIP)
    A, B = unary_union([act.buffer(1e-3), ta.buffer(1e-3)]), unary_union([act.buffer(1e-3), tb.buffer(1e-3)])
    c = 0.5 * (np.r_[q, strip].min(0) + np.r_[q, strip].max(0))
    sh = lambda g: Polygon(np.array(g.exterior.coords) - c)
    corners = np.r_[q, strip] - c
    return dict(A=[sh(A)], B=[sh(B)], cut=[Polygon(corners[:4]), Polygon(corners[4:])], width=float(np.ptp(corners[:, 1])), length=float(np.ptp(corners[:, 0])), radius=float(np.hypot(*corners.T).max()),
                active_mm2=float(act.area), rotation_deg=math.degrees(ang), span_cut=S, fibre_cut=float(np.linalg.norm(q[2] - q[3])), corners_mm=(corners.round(2)).tolist())


def curved(xc=35.0, c=27.0, ct=3.0):
    """Straight LE to xc (tensioned mm), then a quarter ellipse aft to a short tip edge. Fibre straight along the chord c."""
    S = 50.0 / LAM
    xc = xc / LAM
    x = np.r_[0.0, xc, xc + (S - xc) * np.sin(np.linspace(0, np.pi / 2, 40))[1:]]
    t = np.clip((x - xc) / (S - xc), 0, 1)
    yl = (c - ct) * (1 - np.sqrt(1 - t ** 2))
    sheet = Polygon(np.c_[np.r_[x, S, 0.0], np.r_[yl, c, c]])
    # inset from LE, curve and tip, not from the fibre line
    ext = Polygon(np.c_[np.r_[x, S, S, 0.0, 0.0], np.r_[yl, c, c + 6.0, c + 6.0, 0.0]])
    act = ext.buffer(-M_LE, join_style=1).intersection(sheet).intersection(box(M_END, -1, S + 1, c))
    ta = box(0.0, c, TAB, c + STRIP - M_STRIP)
    tb = box(S - TAB, c, S, c + STRIP - M_STRIP)
    A, B = unary_union([act.buffer(1e-3), ta.buffer(1e-3)]), unary_union([act.buffer(1e-3), tb.buffer(1e-3)])
    assert A.geom_type == "Polygon" and B.geom_type == "Polygon"
    pts = np.r_[np.array(sheet.exterior.coords), [(0, c + STRIP), (S, c + STRIP)]]
    cen = 0.5 * (pts.min(0) + pts.max(0))
    sh = lambda g: Polygon(np.array(g.exterior.coords) - cen)
    return dict(A=[sh(A)], B=[sh(B)], cut=[sh(sheet)], width=float(np.ptp(pts[:, 1])), length=float(np.ptp(pts[:, 0])), radius=float(np.hypot(*(pts - cen).T).max()),
                active_mm2=float(act.area), rotation_deg=0.0, span_cut=S, fibre_cut=S, corners_mm=None)


def core(L=55.0, h=23.0, gap=3.0, off=3.0, side=1.0):
    """Two strips. A runs to one long edge, B to the other, overlap h - off = 20 mm along the core axis; side margins so the roll ends stay insulated."""
    A, B, cuts = [], [], []
    for yc in (-(h + gap) / 2, (h + gap) / 2):
        y0, y1 = yc - h / 2, yc + h / 2
        A.append(box(-L / 2 + side, y0, L / 2 - side, y1 - off))
        B.append(box(-L / 2 + side, y0 + off, L / 2 - side, y1))
        cuts.append(box(-L / 2, y0, L / 2, y1))
    return dict(A=A, B=B, cut=cuts, width=2 * h + gap, length=L, radius=float(np.hypot(L / 2, h + gap / 2)), active_mm2=2 * (L - 2 * side) * (h - off), rotation_deg=0.0, span_cut=L, fibre_cut=None, corners_mm=None)


def path_d(poly):
    pts = np.array(poly.simplify(0.01).exterior.coords)[:-1]
    return "M " + " L ".join(f"{a:.3f},{b:.3f}" for a, b in pts) + " Z"


def write(fn, root_part, tip_part):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
    ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
    tree = ET.parse(SRC)
    layer = tree.getroot().find(f"{ns}g")
    for parent in list(layer.iter()):
        for e in list(parent):
            if (e.get("id") or "").startswith("model_"):
                parent.remove(e)
    for (name, elec), cen0 in CENTRES.items():
        part = root_part if name == "root" else tip_part
        cen = np.array(cen0) + LAYER
        for i, poly in enumerate(part[elec]):
            e = ET.SubElement(layer, f"{ns}path")
            e.set("id", f"bcube_{name}_{elec}_{i}")
            e.set("style", "fill:none;stroke:#000000;stroke-width:0.0746456;stroke-opacity:1")
            e.set("d", path_d(Polygon(np.array(poly.exterior.coords) + cen)))
    tree.write(os.path.join(OUT, fn), xml_declaration=True, encoding="UTF-8")


D = dict(rectangle=wing(25.0, 25.0), taper=wing(35.0, 15.0), steep_taper=wing(40.0, 10.0), curved_tip=curved(), core=core())
write("mask_bcube_taper_rectangle.svg", D["taper"], D["rectangle"])
write("mask_bcube_steep_taper_core.svg", D["steep_taper"], D["core"])
write("mask_bcube_curved_core.svg", D["curved_tip"], D["core"])
rep = {k: {q: v[q] for q in ("width", "length", "radius", "active_mm2", "rotation_deg", "span_cut", "fibre_cut", "corners_mm")} for k, v in D.items()}
rep["notes"] = dict(lam_span=LAM, margins_mm=dict(LE=M_LE, root_tip=M_END, strip=M_STRIP), tab_mm=TAB, strip_mm=STRIP, coater=COATER, filter=FILTER)
json.dump(rep, open(os.path.join(OUT, "mask_bcube_numbers.json"), "w"), indent=1)
for k, v in D.items():
    C = (8.854e-12 * 2.8 * v["active_mm2"] * 1e-6 / 30e-6) * 1e9
    print("%-12s width %.1f (coater %.0f) length %.1f radius %.1f (filter %.0f) active %.0f mm2, %.2f nF per dielectric, fibre cut %s" % (k, v["width"], COATER, v["length"], v["radius"], FILTER / 2, v["active_mm2"], C, v["fibre_cut"]))
