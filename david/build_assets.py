"""Cut the page's images out of David's portfolio figures.
Usage: python build_assets.py <Portfolio_David/Figures> . Writes img/*.webp next to this file."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SRC = Path(sys.argv[1])
OUT = Path(__file__).parent / "img"
OUT.mkdir(exist_ok=True)

# name: (file, (left, top, right, bottom)) in original pixels, clear of the (i)/(ii) panel labels
CROPS = {
    "sem_array":   ("micro_sla.png", (113, 27, 832, 598)),
    "needles":     ("micro_sla.png", (952, 27, 1672, 598)),
    "cathedral":   ("csem.png", (60, 25, 659, 466)),
    "chip":        ("csem.png", (115, 494, 722, 835)),
    "channel":     ("csem.png", (723, 494, 1304, 835)),
    "wafer":       ("3d_liga.png", (1404, 45, 2053, 617)),
    "turbine":     ("lmm.png", (1182, 13, 1998, 729)),
    "alumina_macro": ("ceramic_sla.png", (16, 70, 1043, 691)),
    "alumina_pencil": ("ceramic_sla.png", (1075, 70, 1976, 691)),
    "cad_vs_lmm":  ("metal_sla.png", (75, 445, 1062, 872)),
    "watch":       ("metal_sla.png", (1135, 173, 1982, 507)),
    "stud":        ("metal_sla.png", (1136, 517, 1443, 833)),
    "blobs":       ("metal_sla.png", (1454, 517, 1984, 833)),
    "wired":       ("pdm.png", (73, 28, 438, 372)),
    "coils_kapton": ("pdm.png", (68, 390, 436, 781)),
    "fold_a":      ("pdm.png", (468, 80, 874, 412)),
    "fold_b":      ("pdm.png", (468, 415, 874, 776)),
    "coils_pomac": ("pdm.png", (946, 28, 1450, 404)),
    "fold_pomac":  ("pdm.png", (1019, 435, 1395, 790)),
    "mold":        ("sicn.png", (87, 17, 519, 393)),
    "pdc1":        ("sicn.png", (619, 17, 1189, 223)),
    "pdc2":        ("sicn.png", (1218, 17, 1788, 223)),
    "pdc3":        ("sicn.png", (619, 240, 1189, 393)),
    "pdc4":        ("sicn.png", (1218, 240, 1788, 393)),
    "foot_3d":     ("happy_feet.png", (51, 790, 1225, 1030)),
    "foot_blocks": ("happy_feet.png", (1435, 280, 2490, 780)),
    "mag5":        ("mems.png", (52, 140, 514, 462)),
    "mag20":       ("mems.png", (620, 140, 1039, 462)),
    "mag50":       ("mems.png", (34, 508, 532, 831)),
    "mag100":      ("mems.png", (578, 508, 1080, 831)),
    "defect_a":    ("mems.png", (1140, 61, 1511, 338)),
    "defect_b":    ("mems.png", (1520, 61, 1890, 338)),
    "defect_c":    ("mems.png", (1140, 386, 1511, 662)),
    "defect_d":    ("mems.png", (1520, 386, 1890, 662)),
    "defect_e":    ("mems.png", (1330, 710, 1700, 987)),
    "breadboard":  ("telecommande.png", (65, 14, 993, 660)),
    "sofia":       ("sofia.png", (40, 60, 1915, 755)),
}
for name, (fn, box) in CROPS.items():
    Image.open(SRC / fn).convert("RGB").crop(box).save(OUT / f"{name}.webp", quality=88, method=6)

# schematics: blank out the panel labels only
def blanked(fn, box, rects, name):
    im = Image.open(SRC / fn).convert("RGB").crop(box)
    d = ImageDraw.Draw(im)
    for r in rects:
        d.rectangle(r, fill=(255, 255, 255))
    im.save(OUT / f"{name}.webp", quality=90, method=6)
blanked("telecommande.png", (1028, 12, 1986, 385), [(0, 0, 40, 40)], "emitter")
blanked("telecommande.png", (1010, 408, 1986, 660), [(0, 20, 52, 70)], "receiver")

# foot sketch: the figure has "Pressur Sensors" typed in it, relabel that one word
im = Image.open(SRC / "happy_feet.png").convert("RGB").crop((51, 32, 1160, 712))
d = ImageDraw.Draw(im)
d.rectangle((0, 0, 30, 110), fill=(255, 255, 255))
d.rectangle((48, 560, 280, 600), fill=(255, 255, 255))
try:
    f = ImageFont.truetype("timesbd.ttf", 31)
except OSError:
    f = ImageFont.load_default()
d.text((48, 561), "Pressure Sensors", fill=(20, 20, 20), font=f)
im.save(OUT / "foot_sketch.webp", quality=90, method=6)

# hero: one light base for the overview, native-resolution patches for the three close-ups
master = Image.open(SRC / "BU-T_Digital_Journey_Trophy-1.jpg").convert("RGB")
W, H = master.size
base = master.resize((2800, round(2800 * H / W)), Image.LANCZOS)
base.save(OUT / "hero_base.webp", quality=84, method=6)
PATCHES = {"hero_polymer": (.215, .190, .545, .800), "hero_metal": (.525, .495, .705, .810), "hero_ceramic": (.712, .630, .830, .825)}
for name, (a, b, c, e) in PATCHES.items():
    master.crop((round(a * W), round(b * H), round(c * W), round(e * H))).save(OUT / f"{name}.webp", quality=82, method=6)
print({k: v for k, v in PATCHES.items()}, master.size)
