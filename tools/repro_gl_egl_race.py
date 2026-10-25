"""Reproduce the OpenGL abort that closed the 0.43.0 and 0.43.1 package builds (see nodebased/glguard.py).

This script ABORTS the Python process on purpose in the `enum` and `twodev` modes (exit code 134, Rust panic
"BadAccess" at wgpu-hal-29.0.3/src/gles/egl.rs:182 inside an extern "C" function). `single` is the control
that survives. Run it on a machine with wgpu and an OpenGL adapter, forcing Mesa's software EGL if you want
llvmpipe:

    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1 \\
        python tools/repro_gl_egl_race.py enum

  single   one OpenGL device used from two threads: fine (one adapter, one lock)
  enum     a worker thread uses the device while the main thread enumerates adapters: aborts
  twodev   two devices, made from two enumerations, used from two threads: aborts
"""
import sys
import threading
import time

import numpy as np
import wgpu


def gl_adapter():
    adapters = [a for a in wgpu.gpu.enumerate_adapters_sync() if a.info["backend_type"] == "OpenGL"]
    return next((a for a in adapters if a.info["adapter_type"] == "CPU"), adapters[0] if adapters else None)


def churn(device, stop, counts):
    while not stop.is_set():
        buffer = device.create_buffer(size=4096, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        device.queue.write_buffer(buffer, 0, np.zeros(1024, np.float32))
        buffer.destroy()
        counts.append(1)


def main(mode):
    adapter = gl_adapter()
    if adapter is None:
        print("no OpenGL adapter here")
        return 2
    print("adapter:", adapter.info["device"], adapter.info["backend_type"], flush=True)
    first = adapter.request_device_sync()
    second = gl_adapter().request_device_sync() if mode == "twodev" else first
    stop, counts = threading.Event(), []
    worker = threading.Thread(target=churn, args=(first, stop, counts))
    worker.start()
    start = time.time()
    while time.time() - start < 8:
        if mode == "enum":
            wgpu.gpu.enumerate_adapters_sync()
        else:
            buffer = second.create_buffer(size=4096, usage=wgpu.BufferUsage.COPY_DST)
            buffer.destroy()
    stop.set()
    worker.join()
    print(f"survived {mode}: {len(counts)} worker iterations", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "enum"))
