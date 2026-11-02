import cv2, numpy as np, sys, os
out = sys.argv[1]
for path in sys.argv[2:]:
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    fr = np.array(fr); n = len(fr)
    ref = fr[:5].mean(0)
    sig = np.abs(fr - ref).mean(axis=(1, 2))
    name = os.path.splitext(os.path.basename(path))[0]
    np.savetxt(os.path.join(out, name + "_sig.txt"), sig, fmt="%.4f")
    # on/off frames: sort by signal
    hi = fr[sig > np.percentile(sig, 85)].mean(0); lo = fr[sig < np.percentile(sig, 15)].mean(0)
    d = np.abs(hi - lo); d = np.clip(d / max(np.percentile(d, 99.9), 1e-3) * 255, 0, 255)
    cv2.imwrite(os.path.join(out, name + "_frame.png"), fr[0].astype(np.uint8))
    cv2.imwrite(os.path.join(out, name + "_diff.png"), d.astype(np.uint8))
    print(name, n, "sig min/max %.3f %.3f" % (sig.min(), sig.max()), "diff p99.9 %.2f" % np.percentile(np.abs(hi - lo), 99.9))
