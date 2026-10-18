"""Isolated per-stage cost of one displayed frame (median of N, ms), 1080p and 4K, Qt display path; run under the shared GPU lock."""
import sys, time, json, statistics, os
WT = os.getcwd(); sys.path.insert(0, WT)
import tests.isolation
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap
app = QApplication([])
from nodebased import imaging, color, gpudisplay, playback, media
from nodebased.core import Dispatcher
from nodebased.tileexec import TileExecutor
from nodebased.tiles import TileRegion
def med(f, n=9):
    f(); ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 2)
out = {}
for name, (W, H) in (("1080p", (1920, 1080)), ("4K", (3840, 2160))):
    r = {}
    d = Dispatcher(); doc = d.document
    d.execute({"op": "create", "type": "Checker", "id": "c", "params": {"width": W, "height": H, "size": 64}})
    ex = TileExecutor()
    reg = TileRegion(0, 0, W, H, full_x=0, full_y=0, full_width=W, full_height=H)
    fr = [0]
    def comp():
        ex.cache.clear() if hasattr(ex, "cache") and hasattr(ex.cache, "clear") else None
        fr[0] += 1; return ex.compose_region(d.document, "c", reg, frame=fr[0], tier=1)
    r["compose_region Checker (cold, per frame)"] = med(comp)
    frame = comp().pixels
    r["frame_float32_MB"] = round(frame.nbytes / 1048576, 1)
    cache = playback.DisplayCache()
    r["DisplayCache.put (half_safe + f16)"] = med(lambda: cache.put("k", frame, None))
    r["  half_safe alone"] = med(lambda: media.half_safe(frame))
    r["DisplayCache.get (f16->f32 copy)"] = med(lambda: cache.get("k", None))
    # to_qimage pieces
    alpha = frame[..., 3:4]
    rgb = frame[..., :3] * np.float32(1)
    def pre():
        weight = np.clip(alpha, 0, 1); straight = np.divide(rgb, weight, out=np.zeros_like(rgb), where=weight > 1e-8); return weight, straight
    r["to_qimage: unpremult (clip+divide+zeros_like)"] = med(pre)
    weight, straight = pre()
    r["to_qimage: display_rgb ACES 2.0 (Qt GL upload+shader+glReadPixels float)"] = med(lambda: color.display_rgb(straight, "ACES 2.0"))
    r["to_qimage: display_rgb sRGB (OCIO CPU, threaded)"] = med(lambda: color.display_rgb(straight, "sRGB"))
    disp = color.display_rgb(straight, "ACES 2.0")
    def post():
        x = disp * weight; return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    r["to_qimage: *weight, clip, *255, astype uint8"] = med(post)
    rgb8 = post()
    r["to_qimage: QImage(...).copy()"] = med(lambda: QImage(rgb8.data, W, H, rgb8.strides[0], QImage.Format.Format_RGB888).copy())
    r["to_qimage TOTAL (as shipped)"] = med(lambda: imaging.to_qimage(frame, 0.0, "RGB", "black", "ACES 2.0", None))
    img = imaging.to_qimage(frame, 0.0, "RGB", "black", "ACES 2.0", None)
    r["main: QPixmap.fromImage"] = med(lambda: QPixmap.fromImage(img))
    r["main: QImage.scaled x2 FastTransformation (proxy tier 2)"] = med(lambda: img.scaled(W * 2, H * 2) if False else QImage(W // 2, H // 2, QImage.Format.Format_RGB888).scaled(W, H, aspectMode=__import__('PySide6.QtCore', fromlist=['Qt']).Qt.AspectRatioMode.IgnoreAspectRatio, mode=__import__('PySide6.QtCore', fromlist=['Qt']).Qt.TransformationMode.FastTransformation))
    out[name] = r
    print(name, json.dumps(r, indent=1), flush=True)
json.dump(out, open(os.environ.get("NB_BASELINE_OUT", "micro.json"), "w"), indent=1)
os._exit(0)
