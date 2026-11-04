"""Render the zoom-in/zoom-out loops from the page's stills with ffmpeg. Usage: python build_videos.py"""
import subprocess
from pathlib import Path
from PIL import Image

HERE = Path(__file__).parent
IMG, OUT = HERE / "img", HERE / "video"
OUT.mkdir(exist_ok=True)
FPS, SECS = 30, 12

def ease(t):
    return t * t * (3 - 2 * t)

def window(im, view, size):
    """view = (cx, cy, w) as fractions of the image width/height; the window keeps the video aspect."""
    W, H = im.size
    cx, cy, w = view
    ww = w * W
    wh = ww * size[1] / size[0]
    if wh > H:
        wh, ww = H, H * size[0] / size[1]
    l = min(max(cx * W - ww / 2, 0), W - ww)
    t = min(max(cy * H - wh / 2, 0), H - wh)
    return im.resize(size, Image.BICUBIC, box=(l, t, l + ww, t + wh))

def lerp_view(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] * (b[2] / a[2]) ** t)

def render(name, size, shots):
    """shots = [(image, [view, view, ...])]; adjacent shots dissolve. Plays forward then back so it loops."""
    ims = [Image.open(IMG / f"{n}.webp").convert("RGB") for n, _ in shots]
    half = FPS * SECS // 2
    segs = sum(len(v) - 1 for _, v in shots)
    frames = []
    seg_len = half / segs
    for (n, views), im in zip(shots, ims):
        for a, b in zip(views, views[1:]):
            for i in range(round(seg_len)):
                frames.append((im, lerp_view(a, b, ease(i / seg_len))))
    fade = 18
    cuts = []
    k = 0
    for n, views in shots[:-1]:
        k += round(seg_len) * (len(views) - 1)
        cuts.append(k)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                          "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "24", "-preset", "slow", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", str(OUT / f"{name}.mp4")], stdin=subprocess.PIPE)
    seq = []
    for i, (im, v) in enumerate(frames):
        f = window(im, v, size)
        for c in cuts:
            if c <= i < c + fade:  # dissolve from the last view of the previous shot
                prev_im, prev_v = frames[c - 1]
                f = Image.blend(window(prev_im, prev_v, size), f, ease((i - c + 1) / fade))
        seq.append(f)
        if i == 0:
            f.save(OUT / f"{name}.webp", quality=80)
    for f in seq + seq[::-1]:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print(name, (OUT / f"{name}.mp4").stat().st_size // 1024, "kB")

render("pencil", (1280, 720), [("alumina_pencil", [(.5, .5, 1), (.735, .36, .3)]), ("alumina_macro", [(.5, .5, .85), (.49, .58, .36)])])
render("chip", (1024, 576), [("chip", [(.5, .5, 1), (.55, .5, .3)]), ("channel", [(.5, .5, 1), (.32, .42, .55)])])
render("needles", (960, 720), [("needles", [(.5, .47, 1), (.44, .22, .42)])])
render("turbine", (960, 720), [("turbine", [(.5, .47, 1), (.58, .62, .42)])])
render("wafer", (960, 720), [("wafer", [(.5, .5, .96), (.45, .47, .42)])])
render("watch", (1024, 576), [("watch", [(.25, .5, .5), (.13, .42, .3), (.76, .35, .3)])])
render("cathedral", (960, 720), [("cathedral", [(.34, .72, .62), (.34, .4, .5)])])
