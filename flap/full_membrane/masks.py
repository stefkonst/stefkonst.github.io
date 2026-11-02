"""Electrode masks from the model: reference outline = F^-1 (tensioned planform), on the 114 mm plates of mask_2partwing.svg.
Active area stops at the fibre line, so the rolled strip stays passive: in the strip each electrode covers one half of the length.
Run: python masks.py [lam_d] [dn] [w_s_mm] -> masks/mask_2partwing_model.svg, masks/mask_numbers.json"""
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
M_LE, M_SEAM, M_ROOT, M_STRIP, GAP = 2.0, 1.5, 1.5, 1.0, 4.0  # passive margins (mm): LE bond, batten, root rib, strip edge and ends; gap between strip electrodes
COATER = 60.0


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

    half_a = rect(M_STRIP, 0.5 * (L - GAP))
    half_b = rect(0.5 * (L + GAP), L - M_STRIP)
    # electrode A = active + strip half A, electrode B = active + strip half B
    ea, eb = unary_union([act.buffer(1e-3), half_a.buffer(1e-3)]), unary_union([act.buffer(1e-3), half_b.buffer(1e-3)])
    corners = np.r_[s2, st2]
    return dict(sheet=s2, strip=st2, active=act, A=ea, B=eb, half_a=half_a, half_b=half_b,
                width=float(np.ptp(corners[:, 1])), length=float(np.ptp(corners[:, 0])), angle_deg=float(math.degrees(ang)),
                active_area=float(act.area), fibre_ref=float(L), fibre_target=float(np.linalg.norm(w.x0[w.fib[k][-1]] - w.x0[w.fib[k][0]]) * 1e3))


def path_d(poly):
    pts = np.array(poly.simplify(0.01).exterior.coords)[:-1]
    return "M " + " L ".join(f"{a:.3f},{b:.3f}" for a, b in pts) + " Z"


def main(lam_d=1.03, dn=0.01, w_s=16.5, theta=14.0):
    ln = lam_d ** -0.5 + dn
    w = M.Wing(lam_d=(lam_d, lam_d), lam_n=(ln, ln), w_s=(w_s * 1e-3, w_s * 1e-3), theta_deg=theta, h=3e-3)
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
        poly = P[name][elec]
        cen = np.array(CENTRES[(name, elec)]) + LAYER
        moved = Polygon(np.array(poly.exterior.coords) + cen)
        e = ET.SubElement(layer, f"{ns}path")
        e.set("id", f"model_{name}_{elec}")
        e.set("style", "fill:none;stroke:#000000;stroke-width:0.0746456;stroke-opacity:1")
        e.set("d", path_d(moved))
    fn = os.path.join(OUT, "mask_2partwing_model.svg")
    tree.write(fn, xml_declaration=True, encoding="UTF-8")
    rep = dict(lam_d=lam_d, lam_n=ln, dn=dn, w_s_mm=w_s, theta_deg=theta, margins_mm=dict(LE=M_LE, batten=M_SEAM, root=M_ROOT, strip=M_STRIP, gap=GAP), removed_from_source=kept)
    for k, v in P.items():
        rep[k] = dict(width_mm=v["width"], length_mm=v["length"], rotation_deg=v["angle_deg"], active_area_mm2=v["active_area"], fibre_as_cut_mm=v["fibre_ref"], fibre_tensioned_mm=v["fibre_target"],
                      fits_coater=bool(v["width"] <= COATER), sheet_mm=v["sheet"].tolist(), strip_mm=v["strip"].tolist(), A=np.array(v["A"].exterior.coords).tolist(), B=np.array(v["B"].exterior.coords).tolist(),
                      active=np.array(v["active"].exterior.coords).tolist())
    json.dump(rep, open(os.path.join(OUT, "mask_numbers.json"), "w"), indent=1)
    for k in ("root", "tip"):
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in rep[k].items() if a not in ("sheet_mm", "strip_mm", "A", "B", "active")})
    print("removed:", kept)
    return rep


if __name__ == "__main__":
    a = [float(q) for q in sys.argv[1:]]
    main(*a)
