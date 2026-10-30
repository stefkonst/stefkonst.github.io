import cv2, numpy as np, sys
path, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(path); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
fr = np.array(fr); n = len(fr); fps = n / 5.0
# region signal: tip panel vs inboard panel, relative to the time-mean image
mean = fr.mean(0)
tip = (slice(330, 520), slice(760, 1080)); inb = (slice(500, 620), slice(380, 780))
st = np.abs(fr[:, tip[0], tip[1]] - mean[tip]).mean((1, 2)); si = np.abs(fr[:, inb[0], inb[1]] - mean[inb]).mean((1, 2))
F = np.abs(np.fft.rfft(st - st.mean())); fq = np.fft.rfftfreq(n, 1 / fps)
print("fps %.0f, dominant freq tip region %.2f Hz" % (fps, fq[np.argmax(F[1:]) + 1]))
# phase-lock at 1 Hz: on/off images from the projection onto the tip signal
ph = (np.arange(n) / fps) % 1.0
bins = np.array([fr[(ph >= a) & (ph < a + 0.05)].mean(0) for a in np.arange(0, 1, 0.05)])
d = [np.abs(b - bins[0])[tip].mean() for b in bins]
k = int(np.argmax(d)); print("phase of max change: %.2f cycle" % (k * 0.05), "tip region diff profile", np.round(d, 2))
lo, hi = bins[0].astype(np.uint8), bins[k].astype(np.uint8)
flow = cv2.calcOpticalFlowFarneback(lo, hi, None, 0.5, 4, 25, 5, 7, 1.5, 0)
mag = np.hypot(flow[..., 0], flow[..., 1]); mmpx = 10 / 52.6
for lab, r in (("tip panel (outboard of digit)", tip), ("inboard panel", inb)):
    m = mag[r]; print(f"{lab}: median {np.median(m)*mmpx:.2f} mm, p95 {np.percentile(m,95)*mmpx:.2f} mm, max {m.max()*mmpx:.2f} mm")
vis = cv2.cvtColor(hi, cv2.COLOR_GRAY2BGR)
mm = np.clip(mag / max(np.percentile(mag[tip], 99), 1e-3), 0, 1)
heat = cv2.applyColorMap((mm * 255).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite(out, cv2.addWeighted(vis, 0.55, heat, 0.45, 0))
np.savetxt(out.replace(".png", "_sig.txt"), np.c_[np.arange(n) / fps, st, si], fmt="%.4f")
