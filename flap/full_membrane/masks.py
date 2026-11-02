"""Electrode masks from the model: reference outline = F^-1 (tensioned planform), on the 114 mm plates of mask_2partwing.svg.
Contacts as in the original mask: electrodes A and B run into opposite ends of the strip and overlap between the end tabs, so the rolled fibre is active.
The spare rectangles of the original mask (same size, A and B offset along their length) sit beside each panel.
Run: python masks.py -> masks/mask_2partwing_passive_fibre.svg, masks/mask_2partwing_active_fibre.svg, and numbers"""
import json, math, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane as M

SRC = r"C:\Users\stefk\Documents\YEAR2\full_tension\mask_2partwing.svg"
OUT = os.path.join(HERE, "masks"); os.makedirs(OUT, exist_ok=True)
LAYER = np.array([63.002573, 44.99419])                       # layer translate of the source file
CENTRES = {("tip", "A"): (58.15, 91.25), ("tip", "B"): (174.2, 89.75), ("root", "A"): (57.15, 209.5), ("root", "B"): (172.6, 209.25)}   # plate centres, from the holes
OLD = {("root", "A"): "path3", ("root", "B"): "path3-4", ("tip", "A"): "path4", ("tip", "B"): "path4-1"}
M_LE, M_SEAM, M_ROOT, M_STRIP = 2.0, 1.5, 1.5, 1.0   # passive margins (mm): LE bond, batten, root rib, strip edge
TAB = 4.8                                             # contact tab along the fibre (mm), as in the original mask
COATER = 60.0
EXTRA = {"root": dict(L=36.9, w=20.6, off=3.0), "tip": dict(L=40.0, w=16.3, off=3.0)}   # spare rectangles, e.g. a DEA tensioner: length, width, A-B offset (mm)
FIB_GAP = 2.0                                         # clearance between a panel's outline and its additional fibre


def inset(poly, a, b, m, inside):
    """Cut poly along the line a-b moved inward by m."""
    a, b = np.asarray(a), np.asarray(b)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    if n @ (np.asarray(inside) - a) < 0:
        n = -n
    L = 1000.0
    half = Polygon([a - L * u + m * n, b + L * u + m * n, b + L * u + L * n, a - L * u + L * n])
    return poly.intersection(half)


def design_panel(w, k, lam_note=None):
    q = w.mask_panel(k)
    s, st, u, n = q["sheet"], q["strip"], q["u"], q["n"]
    ang = math.atan2(u[1], u[0])                               # rotate so the fibre runs along the plate x axis
    R = np.array([[math.cos(-ang), -math.sin(-ang)], [math.sin(-ang), math.cos(-ang)]])
    s2, st2 = (s @ R.T), (st @ R.T)
    c = 0.5 * (np.r_[s2, st2].min(0) + np.r_[s2, st2].max(0))
    s2, st2 = s2 - c, st2 - c
    sheet = Polygon(s2)
    cen = s2.mean(0)
    # corners: 0 and 1 are on the LE, 2 and 3 on the fibre; root panel: 0-3 root, 1-2 seam; tip panel: 0-3 seam, 1-2 tip edge
    act = inset(sheet, s2[0], s2[1], M_LE, cen)
    act = inset(act, s2[0], s2[3], M_ROOT if k == 0 else M_SEAM, cen)
    act = inset(act, s2[1], s2[2], M_SEAM, cen)
    a, b = st2[0], st2[1]
    ue = (b - a) / np.linalg.norm(b - a)
    ne = (st2[3] - st2[0]) / np.linalg.norm(st2[3] - st2[0])
    L = np.linalg.norm(b - a)
    wd = q["w"]

    def rect(s0, s1):
        p = lambda ss, tt: a + ue * ss + ne * tt
        return Polygon([p(s0, 0), p(s1, 0), p(s1, wd - M_STRIP), p(s0, wd - M_STRIP)])

    half_a = rect(0.0, TAB)
    half_b = rect(L - TAB, L)
    strip_a, strip_b = rect(0.0, L - TAB), rect(TAB, L)
    # electrode A = sheet + strip without its far tab, electrode B = sheet + strip without its near tab: they overlap in the sheet and in the fibre
    if not ACTIVE:                                             # passive fibre: each electrode reaches only its own end tab, with no overlap in the roll
        strip_a, strip_b = half_a, half_b
    ea, eb = unary_union([act.buffer(1e-3), strip_a.buffer(1e-3)]), unary_union([act.buffer(1e-3), strip_b.buffer(1e-3)])
    corners = np.r_[s2, st2]
    # additional fibre: long axis across the coater width, placed past the panel along the fibre direction; the group is recentred on the plate
    name = "root" if k == 0 else "tip"
    ex = EXTRA[name]
    xmax = float(np.r_[s2, st2][:, 0].max()) + FIB_GAP
    ycen = float(0.5 * (np.r_[s2, st2][:, 1].min() + np.r_[s2, st2][:, 1].max()))
    fib = lambda d: Polygon([(xmax, ycen - ex["L"] / 2 + d), (xmax + ex["w"], ycen - ex["L"] / 2 + d), (xmax + ex["w"], ycen + ex["L"] / 2 + d), (xmax, ycen + ex["L"] / 2 + d)])
    fa, fb = fib(-ex["off"] / 2), fib(ex["off"] / 2)
    shift = -0.5 * (float(np.r_[s2, st2][:, 0].min()) + xmax + ex["w"])
    from shapely.affinity import translate
    tr = lambda g: translate(g, shift, 0.0)
    ea, eb, act, half_a, half_b, fa, fb = tr(ea), tr(eb), tr(act), tr(half_a), tr(half_b), tr(fa), tr(fb)
    s2, st2 = s2 + [shift, 0.0], st2 + [shift, 0.0]
    corners = np.r_[s2, st2]
    return dict(sheet=s2, strip=st2, active=act, A=ea, B=eb, half_a=half_a, half_b=half_b, fibre_A=fa, fibre_B=fb,
                width=float(np.ptp(corners[:, 1])), length=float(np.ptp(corners[:, 0])), angle_deg=float(math.degrees(ang)),
                active_area=float(act.area), fibre_ref=float(L), fibre_target=float(np.linalg.norm(w.x0[w.fib[k][-1]] - w.x0[w.fib[k][0]]) * 1e3))


def path_d(poly):
    pts = np.array(poly.simplify(0.01).exterior.coords)[:-1]
    return "M " + " L ".join(f"{a:.3f},{b:.3f}" for a, b in pts) + " Z"


def main(theta=5.0, theta_r=8.85, lam_n=1.04, w_s=13.0, xb=60.0, yc=47.5, active_fibre=False):
    """The sheet is cut unstretched along the fibre, tilted, with a small chord stretch lam_n set at bonding. The tension comes from rotating the edges to vertical."""
    lam_d, ln = 1.0, lam_n
    C_ = ((xb * 1e-3 + (yc * 1e-3 - 6.4e-3) * math.tan(math.radians(theta))), yc * 1e-3)
    D_ = (3.8e-3 + (yc * 1e-3 - 6.4e-3) * math.tan(math.radians(theta_r)), yc * 1e-3)
    global ACTIVE
    ACTIVE = active_fibre
    w = M.Wing(lam_d=(lam_d, lam_d), lam_n=(ln, ln), w_s=(w_s * 1e-3, w_s * 1e-3), theta_deg=theta, C=C_, D=D_, h=3e-3)
    P = {"root": design_panel(w, 0), "tip": design_panel(w, 1)}
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
    ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
    ns = "{http://www.w3.org/2000/svg}"
    tree = ET.parse(SRC)
    root = tree.getroot()
    layer = root.find(f"{ns}g")
    kept = []
    old_ids = set(OLD.values()) | {"rect4", "rect4-5", "rect18-6", "rect18-6-9"}
    for parent in layer.iter():
        for e in list(parent):
            if e.get("id") in old_ids:
                parent.remove(e)
                kept.append(e.get("id"))
    for (name, elec), oldid in OLD.items():
        cen = np.array(CENTRES[(name, elec)]) + LAYER
        for tag, poly in ((f"model_{name}_{elec}", P[name][elec]), (f"model_{name}_{elec}_additional_fibre", P[name]["fibre_" + elec])):
            moved = Polygon(np.array(poly.exterior.coords) + cen)
            e = ET.SubElement(layer, f"{ns}path")
            e.set("id", tag)
            e.set("style", "fill:none;stroke:#000000;stroke-width:0.0746456;stroke-opacity:1")
            e.set("d", path_d(moved))
    tag = "active_fibre" if active_fibre else "passive_fibre"
    fn = os.path.join(OUT, f"mask_2partwing_{tag}.svg")
    tree.write(fn, xml_declaration=True, encoding="UTF-8")
    rep = dict(lam_n=ln, w_s_mm=w_s, theta_deg=theta, theta_r_deg=theta_r, xb_mm=xb, yc_mm=yc, active_fibre=active_fibre, margins_mm=dict(LE=M_LE, batten=M_SEAM, root=M_ROOT, strip=M_STRIP), tab_mm=TAB, replaced_in_source=kept)
    for k, v in P.items():
        rep[k] = dict(width_mm=v["width"], length_mm=v["length"], rotation_deg=v["angle_deg"], active_area_mm2=v["active_area"], fibre_as_cut_mm=v["fibre_ref"], fibre_tensioned_mm=v["fibre_target"],
                      fits_coater=bool(v["width"] <= COATER), total_length_with_fibre_mm=float(unary_union([v["A"], v["B"], v["fibre_A"], v["fibre_B"]]).bounds[2] - unary_union([v["A"], v["B"], v["fibre_A"], v["fibre_B"]]).bounds[0]), hole_clearance_mm=float(min(unary_union([v["A"], v["B"], v["fibre_A"], v["fibre_B"]]).distance(Polygon([(hx - 0.8, hy - 0.8), (hx + 0.8, hy - 0.8), (hx + 0.8, hy + 0.8), (hx - 0.8, hy + 0.8)])) for hx, hy in ((0, -49.4), (49.4, 0), (0, 49.4), (-49.4, 0)))), tab_A=np.array(v["half_a"].exterior.coords).tolist(), tab_B=np.array(v["half_b"].exterior.coords).tolist(), fibre_A=np.array(v["fibre_A"].exterior.coords).tolist(), fibre_B=np.array(v["fibre_B"].exterior.coords).tolist(), sheet_mm=v["sheet"].tolist(), strip_mm=v["strip"].tolist(), A=np.array(v["A"].exterior.coords).tolist(), B=np.array(v["B"].exterior.coords).tolist(),
                      active=np.array(v["active"].exterior.coords).tolist())
    rep["root"]["fibre_tensioned_mm"] = xb - 3.8                # with the root edge and batten vertical
    rep["tip"]["fibre_tensioned_mm"] = math.hypot(108.0 - xb, yc - 6.4 - 3.1)
    json.dump(rep, open(os.path.join(OUT, f"mask_numbers_{tag}.json"), "w"), indent=1)
    for k in ("root", "tip"):
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in rep[k].items() if a not in ("sheet_mm", "strip_mm", "A", "B", "active", "tab_A", "tab_B", "fibre_A", "fibre_B")})
    print("removed:", kept)
    return rep


if __name__ == "__main__":
    main()                                                     # best lift: 5 and 8.85 degrees, passive fibre
    main(theta=10.0, theta_r=18.0, active_fibre=True)          # active fibre needs more tilt to keep its tension
