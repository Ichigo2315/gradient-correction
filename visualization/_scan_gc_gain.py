import os
import numpy as np
from PIL import Image

GT = "aot-benchmark/datasets/DAVIS/Annotations/480p"
RES = "aot-benchmark/results/davis2017"
NOGC = os.path.join(RES, "davis2017_val_noGCfull_AOTT_PRE_ckpt_unknown", "Annotations", "480p")
GC = os.path.join(RES, "davis2017_val_legacy20k1full_AOTT_PRE_ckpt_unknown", "Annotations", "480p")


def load(p):
    return np.array(Image.open(p))


def frame_J(gt, pred, objs):
    js = []
    for o in objs:
        g, p = (gt == o), (pred == o)
        u = np.logical_or(g, p).sum()
        if u == 0:
            continue
        js.append(np.logical_and(g, p).sum() / u)
    return float(np.mean(js)) if js else np.nan


rows = []
for seq in sorted(os.listdir(GT)):
    gtdir = os.path.join(GT, seq)
    nd, gd = os.path.join(NOGC, seq), os.path.join(GC, seq)
    if not (os.path.isdir(nd) and os.path.isdir(gd)):
        continue
    files = sorted(f for f in os.listdir(gtdir) if f.endswith(".png"))
    objs = [o for o in np.unique(load(os.path.join(gtdir, files[0]))) if o != 0]
    for f in files[1:]:
        g = load(os.path.join(gtdir, f))
        try:
            n = load(os.path.join(nd, f))
            c = load(os.path.join(gd, f))
        except FileNotFoundError:
            continue
        jn, jc = frame_J(g, n, objs), frame_J(g, c, objs)
        if np.isnan(jn) or np.isnan(jc):
            continue
        rows.append((seq, f, jn, jc, jc - jn))

rows.sort(key=lambda r: r[4], reverse=True)
print("=== Top 15 frames by J gain (legacy GC - noGC) ===")
print("%-20s %-9s %8s %8s %8s" % ("seq", "frame", "noGC_J", "GC_J", "delta"))
for seq, f, jn, jc, d in rows[:15]:
    print("%-20s %-9s %8.4f %8.4f %+8.4f" % (seq, f, jn, jc, d))
