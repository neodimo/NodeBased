"""The idle state of a discrete graphics card, and the short ramp the first played frames after a rest use to get past it.

After about 3.5 s without work a discrete card lowers its clocks and its link to the processor (the RTX 3080 Ti: 210 MHz, PCIe
generation 1 of 4). Until the link climbs back, reading a picture back is four to seven times as slow: 33 MB (a 1080p float picture) takes
84 ms against 12, and the first two or three played frames of a play pressed after a rest arrive 60 to 150 ms apart. The climb is
staged and follows how busy the link is (`benchmarks/3d-playback/laptop-1023/clock-ramp-probes/link_probe*.py`, 10/24): copies to the
processor that keep it about half busy for 0.65 s raise it one stage (8 MB every 42 ms: 22 ms a copy, then 11.6); copies that keep it
all busy raise it to the top in the same time (16 MB every 42 ms: 42 ms a copy, then 6); a burst of compute raises the shader clock but not
the link; 2 and 4 MB every 42 ms raise nothing. A full-size frame (33 MB) after a half-busy stage takes 51 ms for three frames until the
driver sees the link full, and the top stage steps down again about 1.5 s after the copies that raised it stop.

`Ramp` is the start of one play after a rest. The window draws the first frames at half size (8 MB, 22 ms on the idle link: on time), and
each of those frames is followed by a copy to the processor of as many more bytes as the rest of the frame's time can carry at the speed
the frame's own readback showed (`after_frame`), so the link is full for that time. When a frame's readback has become fast (`Ramp.up`)
the last few frames at the lower size are held a little longer than the one before (the first frame at the chosen size arrives later
after its start than they do), then the window goes back to the chosen size (`Ramp.switch`), and the extra copies taper off over two
seconds. The extra traffic lasts for the first 1 to 3.5 s of a play pressed after a rest and never longer; during a rest, and in a play
pressed on a warm card, nothing is held or copied.
"""
from __future__ import annotations

import statistics
import time

# A card that has had no work for this long may have lowered its clocks (the RTX does so after about 3.5 s).
REST_SECONDS = 3.0
# Frames at the lower tier are drawn at least this long (a frame or two of the idle link are not yet evidence), and at most this long.
MIN_SECONDS = 0.35
MAX_SECONDS = 1.5
# The link is up when a frame took under this fraction of the median of the first three.
UP_FACTOR = 0.35

_last_busy = [None]        # time.monotonic() when the app last waited on the card for a render


def note_busy():
    """The app has just waited on the card for a render."""
    _last_busy[0] = time.monotonic()


def rest_seconds():
    """Seconds since the app last used the card for a render, or None before the first."""
    return None if _last_busy[0] is None else max(0.0, time.monotonic() - _last_busy[0])


def clock_idles(info):
    """True for an adapter whose clocks and link fall when it has no work: a discrete card (an integrated one shares the processor's
    memory and power states, which a frame wakes in a few milliseconds)."""
    return str((info or {}).get('adapter_type', '')).lower().replace('_', '').replace(' ', '') == 'discretegpu'


# The extra copy of a frame: at most BALLAST_MAX_BYTES, at least BALLAST_MIN_BYTES (under that it is not sent), and it takes at most
# BALLAST_FILL of the frame's spare time at the speed the frame's own readback showed.
BALLAST_MAX_BYTES = 16 * 2 ** 20
BALLAST_MIN_BYTES = 1 * 2 ** 20
BALLAST_FILL = 0.8
# After the link is up the extra copies taper to nothing over this long (stopping at once let the link step down again about 1.5 s later:
# three frames of 51 ms at 2.3 to 2.9 s in half of the cube runs, 10/24).
TAPER_SECONDS = 2.0
# The first frame at the chosen size takes longer after its start than a frame at the lower size: about three times what the lower size
# took (HOLD_FACTOR) plus the time it spends packing its scene again. The last few frames at the lower size are held to make up the
# difference, each at most HOLD_STEP_SECONDS longer than the one before so no interval grows by more than that, up to HOLD_MAX_SECONDS.
HOLD_FACTOR = 3.0
HOLD_MAX_SECONDS = 0.060
HOLD_STEP_SECONDS = 0.010

_active = [None]           # the Ramp of the play that is starting, while its frames are drawn at the lower size


class Ramp:
    """The start of one play after a rest (see the module doc). `budget` is the seconds a frame has (1 / the frame rate)."""

    def __init__(self, budget, clock=time.monotonic):
        self.budget = float(budget)
        self.clock = clock
        self.began = clock()
        self.times = []
        self.up = False            # the link is fast again
        self.up_at = None          # when the first frame after that was read back; the extra copies taper off from then
        self.after_up = 0          # frames read back since
        self.hold_step = 0.0       # seconds each of those frames is held longer than the one before (see `after_frame`)
        self.hold_frames = 0
        self.switch = False        # the window draws the chosen size from the next frame on
        self.build_seconds = 0.0   # the longest a frame spent packing and uploading its scene since the ramp began
        self.extra_bytes = 0
        _active[0] = self

    def observe(self, seconds):
        """A frame's submit-to-readback time. Sets `up` when the link is fast again: the last two frames took under UP_FACTOR of the
        median of the first three, after MIN_SECONDS; or MAX_SECONDS have passed."""
        self.times.append(float(seconds))
        elapsed = self.clock() - self.began
        if elapsed >= MAX_SECONDS:
            self.up = True
        elif elapsed >= MIN_SECONDS and len(self.times) >= 5:
            baseline = sorted(self.times[:3])[1]
            self.up = max(self.times[-2:]) <= UP_FACTOR * baseline
        return self.up

    def finish(self):
        """The play has stopped: the extra copies stop."""
        if _active[0] is self:
            _active[0] = None

    def fill(self):
        """The share of a frame's spare time the extra copy may take: all of BALLAST_FILL until the link is up, then tapering to nothing
        over TAPER_SECONDS (None when it is over). The link climbed by the copies steps down again about 1.5 s after they stop if they stop
        at once, which cost three frames of 51 ms at 2.3 to 2.9 s in half of the runs (10/24); full-size frames alone (28 percent of the
        link) do not hold it that high, and a taper lets it settle where they do."""
        if self.up_at is None:
            return BALLAST_FILL
        elapsed = self.clock() - self.up_at
        return None if elapsed >= TAPER_SECONDS else BALLAST_FILL * (1.0 - elapsed / TAPER_SECONDS)


def note_build(seconds):
    """A frame has just spent `seconds` packing and uploading its scene for the card. The first frame at the chosen size does it again (the
    scene it draws is not the one the lower size drew), so the ramp counts it into the time that frame will take longer."""
    ramp = _active[0]
    if ramp is not None:
        ramp.build_seconds = max(ramp.build_seconds, float(seconds))


def after_frame(state, readback_bytes, seconds):
    """Called by a render right after its picture has been read back, `seconds` after it was submitted. While a `Ramp` is running, notes the
    time and queues a copy to the processor of the bytes the rest of the frame's time can carry at the speed this readback showed
    (never waited for here: it runs while the frame is on its way to the screen and ends before the next one is submitted)."""
    ramp = _active[0]
    if ramp is None:
        return
    if not ramp.up:
        ramp.observe(seconds)
    if ramp.up and ramp.up_at is None:
        ramp.up_at = ramp.clock()
        # A frame at the chosen size arrives longer after its frame starts than one at the lower size (about three times the readback
        # time, and it packs and uploads its scene again: 10 ms for the cube, 25 to 40 ms for the heavier scenes), which would lengthen the interval at the switch by that much
        # (53 to 56 ms for the cube, 66 to 85 for the furnished room and the mixed scene; the gate is 54). So the last few frames at the
        # lower size are each held a little longer than the one before, up to that difference, and the switch comes after them.
        delta = min(HOLD_MAX_SECONDS, HOLD_FACTOR * float(statistics.median(ramp.times[-2:])) + ramp.build_seconds)
        ramp.hold_frames = max(1, int(delta / HOLD_STEP_SECONDS + 0.999))
        ramp.hold_step = delta / ramp.hold_frames
    elif ramp.up_at is not None and not ramp.switch:
        ramp.after_up += 1
    if ramp.up_at is not None and not ramp.switch:
        if ramp.after_up:
            time.sleep(ramp.hold_step * ramp.after_up)
        ramp.switch = ramp.after_up >= ramp.hold_frames
    fill = ramp.fill()
    if fill is None:
        _active[0] = None
        return
    spare = ramp.budget * fill - seconds
    nbytes = min(BALLAST_MAX_BYTES, int(spare * readback_bytes / max(seconds, 1e-3))) // 4 * 4
    if nbytes < BALLAST_MIN_BYTES:
        return
    try:
        wgpu, device = state['wgpu'], state['device']
        buffers = state.get('clock_ballast')
        if buffers is None:         # kept for the next ramp: destroying a buffer waits for the card and stalls every other thread using the device (87 to 135 ms)
            buffers = state['clock_ballast'] = (
                device.create_buffer(size=BALLAST_MAX_BYTES, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST),
                device.create_buffer(size=BALLAST_MAX_BYTES, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST))
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(buffers[0], 0, buffers[1], 0, nbytes)
        device.queue.submit([encoder.finish()])
        ramp.extra_bytes += nbytes
    except Exception:           # an optimisation: the frames just wait for the link
        _active[0] = None
