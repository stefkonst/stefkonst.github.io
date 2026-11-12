"""Render the block SVGs to PNG with headless Chrome (for checking, and for the paper later).
Run: python render_blocks.py -> blocks/png/*.png"""
import os, re, subprocess, glob
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, "blocks"); P = os.path.join(B, "png"); os.makedirs(P, exist_ok=True)
CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"
for f in sorted(glob.glob(os.path.join(B, "*.svg"))):
    s = open(f, encoding="utf-8").read(); m = re.search(r'width="(\d+)" height="(\d+)"', s); w, h = int(m.group(1)), int(m.group(2))
    html = os.path.join(P, "_tmp.html")
    open(html, "w", encoding="utf-8").write(f'<html><body style="margin:0;background:#fff"><img src="file:///{f.replace(os.sep, "/")}" width="{w}" height="{h}"></body></html>')
    out = os.path.join(P, os.path.basename(f)[:-4] + ".png")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={w},{h}", f"--screenshot={out}", "file:///" + html.replace(os.sep, "/")], capture_output=True, timeout=60)
    print(os.path.basename(out), os.path.exists(out))
os.remove(os.path.join(P, "_tmp.html"))
