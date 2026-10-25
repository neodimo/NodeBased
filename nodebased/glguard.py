"""Why the Direct picture never presents through OpenGL on Linux, and the one gate for adapter enumeration.

wgpu's OpenGL backend on Linux (wgpu-hal 29.0.3, src/gles/egl.rs) gives every adapter it hands out the same
EGL context but a lock of its own, and `Instance::enumerate_adapters` / `create_surface` make that context
current without taking any adapter's lock. EGL lets one thread hold a context at a time, so when two threads
reach it together (the filler thread of a presenter, the main thread asking for adapters; or two devices made
from two enumerations) `eglMakeCurrent` answers BadAccess, the `unwrap()` on it panics inside an `extern "C"`
function, and Rust aborts the whole process. A Python `try/except` cannot catch that, and the artist loses
unsaved work. Reproduced on llvmpipe: tools/repro_gl_egl_race.py. Vulkan, D3D12 and Metal have no such shared
context. A single device used from several threads is fine on OpenGL (one adapter, one lock); two are not.

So on Linux the Direct picture asks for a Vulkan adapter only and refuses to start, with a plain reason and
before any device or buffer exists, when the machine offers nothing else. Adapter enumeration goes through one
process-wide gate and is done once (`gpu3d`), so a second thread never enumerates while the first still holds
the context.
"""
import sys
import threading

GATE = threading.RLock()     # held around every call that makes wgpu enumerate or request adapters

NO_VULKAN = ("this computer's graphics driver only offers OpenGL, and the Direct picture cannot use OpenGL on "
             "Linux without risking the application closing. Install a Vulkan driver (for example Mesa's "
             "Vulkan package) to use it; the standard viewer is unaffected")
OPENGL_ADAPTER = ("the graphics adapter {name} is an OpenGL adapter, which the Direct picture does not use on "
                  "Linux (it can close the application). Use a Vulkan adapter, or the standard viewer")


class Refused(RuntimeError):
    """A graphics path that is not used here because it could take the whole process down."""


def affected(platform=None):
    """True where wgpu's OpenGL backend is the shared-EGL-context one: Linux."""
    return (platform or sys.platform).startswith("linux")


def is_opengl(info):
    return str(info.get("backend_type", "")).strip().lower() == "opengl"


def refuse_adapter(info, platform=None):
    """Raise `Refused` for an OpenGL adapter on Linux; nothing else is touched."""
    if affected(platform) and is_opengl(info):
        raise Refused(OPENGL_ADAPTER.format(name=info.get("device") or info.get("description") or "this adapter"))


def refuse_gl_only(adapters, platform=None):
    """Raise `Refused` when `adapters` (dicts with a `backend` key, as `gpu3d.enumerate_adapters` returns) are all
    OpenGL on Linux. An empty list is no claim: wgpu may simply be absent, and the normal start-up error says so."""
    if not affected(platform) or not adapters:
        return
    if all(str(a.get("backend", "")).strip().lower() == "opengl" for a in adapters):
        raise Refused(NO_VULKAN)
