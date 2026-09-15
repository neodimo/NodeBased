"""GPU-accelerated OCIO display transform via Qt's OpenGL bindings.

Builds each view's OCIO GPU shader once (`GpuShaderDesc` -> GLSL text plus its LUT
textures) and reuses it every call: bind the input image as a texture, draw one
full-screen triangle into a float FBO, read the pixels back. This is a readback design,
not a viewer-texture integration -- the Viewer widget still receives a plain QImage
through the existing `to_qimage`/`DisplayCache` path in `app.py`, so capture,
`reference_context`, and the display cache all keep working unchanged.

Threading: a `QOpenGLContext` is thread-affine -- it may only be current on the thread
that uses it. NodeBased already funnels every preview render through one dedicated
worker thread (`MainWindow.executor`, `max_workers=1` in `app.py`), so a `GpuDisplay`
instance is built lazily on whichever thread first calls `render()` and is used
exclusively from that thread afterward; nothing here is safe to call concurrently from
two threads. `QOffscreenSurface` itself must be *created* on the GUI thread, so
`app.py` creates one at startup and hands it to this module via `configure_surface()`
before the worker thread's first display request. When no surface has been configured
(benchmarks, tests, and any other single-threaded caller), a surface is created lazily
on the calling thread instead, which is fine as long as that thread is the only caller.

Fallback: `NODEBASED_DISPLAY_GPU=0` forces the CPU path unconditionally. Any failure to
create a context, compile a shader, or render -- no GPU, no display attached, an
offscreen QPA platform without GL support, a mid-session driver hiccup -- marks the GPU
path unavailable for the rest of the process and every caller falls back to the
threaded CPU path in `color.py`. A GPU failure never raises past `display_rgb` and never
produces a partially-transformed image.
"""
from __future__ import annotations

import os
import threading

import numpy as np

_GL_RGBA32F = 0x8814
_GL_RGB = 0x1907
_GL_FLOAT = 0x1406
_GL_TRIANGLES = 0x0004
_GL_PACK_ALIGNMENT = 0x0D05

VERTEX_SHADER = """#version 400 core
out vec2 vUV;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""

FRAGMENT_TEMPLATE = """#version 400 core
{ocio_src}
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D inputImage;
void main() {{
    // The input texture is RGB32F, not RGBA: padding a per-pixel alpha of 1.0 in Python
    // before every upload cost as much as the whole rest of the GPU path combined
    // (docs/BENCHMARKS-v0.16-display.md), so the constant alpha is supplied here instead.
    vec3 c = texture(inputImage, vUV).rgb;
    fragColor = {function_name}(vec4(c, 1.0));
}}
"""


class GpuUnavailable(Exception):
    """Raised internally when the GPU path cannot serve this call; always caught."""


_lock = threading.Lock()
_surface = None
_instance = None
_failed_reason = None


def configure_surface(surface):
    """Register a `QOffscreenSurface` created on the GUI thread. Call once at startup."""
    global _surface
    _surface = surface


def gpu_enabled():
    return os.environ.get('NODEBASED_DISPLAY_GPU', '1') != '0'


def status():
    """Human-readable backend name for the viewer status text."""
    if not gpu_enabled():
        return 'CPU (forced)'
    if _failed_reason is not None:
        return f'CPU (GPU unavailable: {_failed_reason})'
    if _instance is not None:
        return 'GPU'
    return 'CPU'


def get_display(force=False):
    """Return the process-wide `GpuDisplay`, building it lazily on the calling thread.

    Returns None when the GPU path is disabled or has already failed once; `force`
    bypasses the "already failed" memo so a benchmark or test can retry after fixing
    the environment (e.g. switching QT_QPA_PLATFORM) without restarting the process.
    """
    global _instance, _failed_reason
    if not gpu_enabled():
        return None
    if _failed_reason is not None and not force:
        return None
    with _lock:
        if _instance is not None:
            return _instance
        try:
            _instance = GpuDisplay(_surface)
            _failed_reason = None
        except Exception as error:
            _failed_reason = str(error)
            return None
    return _instance


def _release_instance():
    global _instance
    with _lock:
        instance, _instance = _instance, None
    if instance is not None:
        try:
            instance._destroy()
        except GpuUnavailable:
            pass


def shutdown():
    """Release GL resources with the context current. Call from the thread that built them.

    Optional: process exit reclaims GPU resources regardless. This only avoids the
    "destroy called without a current context" warning Qt prints when a `QOpenGLTexture`
    or FBO is garbage-collected after its context has gone away.
    """
    _release_instance()


def reset_for_testing():
    """Drop the memoized instance/failure so tests can exercise fresh init paths.

    Releases GL resources first (same as `shutdown()`) so repeated construction across
    tests in one process doesn't print "destroy called without a current context" for
    every texture the garbage collector happens to finalize later, on no particular
    thread and with no context current.
    """
    global _failed_reason
    _release_instance()
    with _lock:
        _failed_reason = None


class GpuDisplay:
    """Owns one GL context and renders OCIO display views through it.

    Every method must be called from the same thread that constructed this instance.
    """

    def __init__(self, surface=None):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat

        # QOffscreenSurface (and the QOpenGLContext bound to it) needs a live
        # Q[Gui]Application; creating one without it does not raise a catchable Qt error,
        # it segfaults. Any caller of `display_rgb` that has not made a QApplication --
        # a bare script, an early-import test module -- must fall back to CPU instead.
        if QCoreApplication.instance() is None:
            raise GpuUnavailable('no QApplication/QGuiApplication instance exists yet')

        self._owns_surface = surface is None
        fmt = QSurfaceFormat()
        # Matches GPU_LANGUAGE_GLSL_4_0 in _build_program: the OCIO-generated shader text
        # and this module's own vertex/fragment wrapper must agree on GLSL dialect, since
        # the two are concatenated into one compilation unit.
        fmt.setVersion(4, 0)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        if surface is None:
            surface = QOffscreenSurface()
            surface.setFormat(fmt)
            surface.create()
        if not surface.isValid():
            raise GpuUnavailable('QOffscreenSurface is not valid on this platform')
        self._surface = surface

        context = QOpenGLContext()
        context.setFormat(fmt)
        if not context.create():
            raise GpuUnavailable('QOpenGLContext.create() failed')
        if not context.makeCurrent(surface):
            raise GpuUnavailable('QOpenGLContext.makeCurrent() failed (no GL on this platform)')
        try:
            self._thread = threading.get_ident()
            self._context = context
            self._functions = context.extraFunctions()
            self._functions.initializeOpenGLFunctions()
            self._programs = {}      # view -> (program, program_id, input_loc, lut bindings)
            self._input_texture = None
            self._input_size = None
            self._fbo = None
            self._fbo_size = None
            self._readback = None
            self._vao = None
            self._init_vao()
        finally:
            context.doneCurrent()

    def _check_thread(self):
        if threading.get_ident() != self._thread:
            raise GpuUnavailable('GpuDisplay used from a different thread than it was created on')

    def _init_vao(self):
        vao = np.zeros(1, dtype=np.uint32)
        self._functions.glGenVertexArrays(1, vao)
        self._vao = int(vao[0])

    def _build_program(self, view):
        from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
        from . import color

        gpu_processor = color.display_gpu_processor(view)
        import PyOpenColorIO as ocio
        desc = ocio.GpuShaderDesc.CreateShaderDesc()
        desc.setLanguage(ocio.GPU_LANGUAGE_GLSL_4_0)
        desc.setFunctionName('OCIODisplayTransform')
        gpu_processor.extractGpuShaderInfo(desc)

        fragment_src = FRAGMENT_TEMPLATE.format(ocio_src=desc.getShaderText(),
                                                 function_name='OCIODisplayTransform')
        program = QOpenGLShaderProgram()
        if not program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER):
            raise GpuUnavailable(f'vertex shader: {program.log()}')
        if not program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragment_src):
            raise GpuUnavailable(f'fragment shader ({view}): {program.log()}')
        if not program.link():
            raise GpuUnavailable(f'link ({view}): {program.log()}')
        program_id = program.programId()

        luts = []
        unit = 1
        for texture in desc.getTextures():
            luts.append(self._make_lut_texture(texture, is_3d=False))
            unit += 1
        for texture in desc.get3DTextures():
            luts.append(self._make_lut_texture(texture, is_3d=True))
            unit += 1
        # Assign texture units after collection so ordering is deterministic.
        bound = []
        fn = self._functions
        input_loc = fn.glGetUniformLocation(program_id, 'inputImage')
        for index, (sampler_name, qt_texture, channel) in enumerate(luts):
            loc = fn.glGetUniformLocation(program_id, sampler_name)
            bound.append((loc, qt_texture, index + 1))
        # Keep the QOpenGLShaderProgram alive (owns the compiled shader objects).
        self._programs[view] = (program, program_id, input_loc, bound)

    def _make_lut_texture(self, texture, is_3d):
        from PySide6.QtOpenGL import QOpenGLTexture
        values = np.asarray(texture.getValues(), dtype=np.float32)
        channel_name = str(texture.channel)
        is_rgb = 'RGB' in channel_name
        pixel_format = QOpenGLTexture.PixelFormat.RGB if is_rgb else QOpenGLTexture.PixelFormat.Red
        tex_format = QOpenGLTexture.TextureFormat.RGB32F if is_rgb else QOpenGLTexture.TextureFormat.R32F
        interpolation = str(texture.interpolation)
        gl_filter = (QOpenGLTexture.Filter.Nearest if 'NEAREST' in interpolation
                     else QOpenGLTexture.Filter.Linear)

        qt_texture = QOpenGLTexture(QOpenGLTexture.Target.Target3D if is_3d else
                                     (QOpenGLTexture.Target.Target1D if texture.height == 1
                                      else QOpenGLTexture.Target.Target2D))
        if is_3d:
            edge = round(round(values.size / (3 if is_rgb else 1)) ** (1 / 3))
            qt_texture.setSize(edge, edge, edge)
        else:
            qt_texture.setSize(texture.width, max(1, texture.height))
        qt_texture.setFormat(tex_format)
        qt_texture.allocateStorage()
        qt_texture.setData(pixel_format, QOpenGLTexture.PixelType.Float32, values.tobytes())
        qt_texture.setMinificationFilter(gl_filter)
        qt_texture.setMagnificationFilter(gl_filter)
        qt_texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        return texture.samplerName, qt_texture, texture.channel

    def _ensure_input_texture(self, width, height):
        from PySide6.QtOpenGL import QOpenGLTexture
        if self._input_texture is not None and self._input_size == (width, height):
            return
        if self._input_texture is not None:
            self._input_texture.destroy()
        texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        texture.setFormat(QOpenGLTexture.TextureFormat.RGB32F)
        texture.setSize(width, height)
        texture.allocateStorage()
        texture.setMinificationFilter(QOpenGLTexture.Filter.Nearest)
        texture.setMagnificationFilter(QOpenGLTexture.Filter.Nearest)
        texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        self._input_texture = texture
        self._input_size = (width, height)

    def _ensure_fbo(self, width, height):
        from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
        if self._fbo is not None and self._fbo_size == (width, height):
            return
        if self._fbo is not None:
            self._fbo.release()
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setInternalTextureFormat(_GL_RGBA32F)
        fbo = QOpenGLFramebufferObject(width, height, fbo_format)
        if not fbo.isValid():
            raise GpuUnavailable('QOpenGLFramebufferObject is not valid')
        self._fbo = fbo
        self._fbo_size = (width, height)
        # RGB float32 rows are already 4-byte aligned (12 bytes/pixel), so a tightly packed
        # GL_RGB readback needs GL_PACK_ALIGNMENT=1 and skips both the alpha channel's
        # bandwidth and the extra "drop the 4th channel" numpy copy a GL_RGBA readback
        # would need. Persistent per-size buffer: a fresh multi-megabyte bytearray every
        # call was itself a measurable fraction of the total (see the module docstring).
        self._functions.glPixelStorei(_GL_PACK_ALIGNMENT, 1)
        self._readback = bytearray(width * height * 3 * 4)

    def _destroy(self):
        self._check_thread()
        if not self._context.makeCurrent(self._surface):
            return
        try:
            if self._input_texture is not None:
                self._input_texture.destroy()
            for _view, (_program, _pid, _loc, luts) in self._programs.items():
                for _loc2, qt_texture, _unit in luts:
                    qt_texture.destroy()
            if self._fbo is not None:
                self._fbo.release()
        finally:
            self._context.doneCurrent()
        if self._owns_surface:
            self._surface.destroy()

    def render(self, rgb, view):
        """Apply `view` to straight (unassociated) HxWx3 float32 `rgb`. Returns HxWx3 float32."""
        self._check_thread()
        rgb = np.ascontiguousarray(rgb, dtype=np.float32)
        height, width = rgb.shape[:2]
        if height == 0 or width == 0:
            return rgb.copy()
        if not self._context.makeCurrent(self._surface):
            raise GpuUnavailable('makeCurrent failed mid-session')
        try:
            if view not in self._programs:
                self._build_program(view)
            program, program_id, input_loc, luts = self._programs[view]

            self._ensure_input_texture(width, height)
            self._ensure_fbo(width, height)

            from PySide6.QtOpenGL import QOpenGLTexture
            fn = self._functions
            self._input_texture.setData(QOpenGLTexture.PixelFormat.RGB,
                                         QOpenGLTexture.PixelType.Float32, rgb.tobytes())

            self._fbo.bind()
            fn.glViewport(0, 0, width, height)
            fn.glBindVertexArray(self._vao)
            fn.glUseProgram(program_id)
            self._input_texture.bind(0)
            fn.glUniform1i(input_loc, 0)
            for loc, qt_texture, unit in luts:
                qt_texture.bind(unit)
                fn.glUniform1i(loc, unit)
            fn.glDrawArrays(_GL_TRIANGLES, 0, 3)

            fn.glReadPixels(0, 0, width, height, _GL_RGB, _GL_FLOAT, self._readback)
            self._fbo.release()

            out = np.frombuffer(self._readback, dtype=np.float32).reshape(height, width, 3)
            return out.copy()
        except GpuUnavailable:
            raise
        except Exception as error:
            raise GpuUnavailable(str(error)) from error
        finally:
            self._context.doneCurrent()
