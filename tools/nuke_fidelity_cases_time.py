"""Behaviour cases for the Time group (2D behaviour lane 2, step I, November 2026).

TimeOffset, FrameHold, Retime, FrameRange, TimeClip, AppendClip, TimeEcho, TimeBlur and TimeWarp act across frames, so the one-frame
fixture cannot see them. Each case runs the node on a frame-stamped plate (`_stamp` in the case's params, built by
`_run_stamped_document` in `tools/nuke_fidelity_cases.py`): a Constant whose red channel is the frame number, so the red value of the
node's picture is, written as a number, the source frame the node picked (a fractional frame reads back as itself, the stamp being a
straight line). Green is 0.25 and blue 0.5 on every frame; AppendClip's clips carry blue 0.2, 0.3, ... so the clip shows too.

`source` is "guide" when the Nuke 17.0 Reference Guide states the frame (page quoted in the statement) and "convention" when only the
form is long-established; the statement says which sentence. A rule the guide leaves open and no source fixes is listed `unscored`
with the reason, so nothing here guesses an off-by-one.

Nothing here has been compared with a real Nuke render. The reference pack writes one-frame scripts and cannot render a sequence, so
these cases have no Nuke script that means anything; they stand on the guide's text. Only names defined above `CASES = _cases()` in
`nuke_fidelity_cases` are imported (this module is imported by that line's hook).
"""
from __future__ import annotations

import numpy as np

H, W = 12, 16
GREEN = 0.25
BLUE = 0.5
GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/time_nodes/"
CLIP20 = [("FrameRange", {"first_frame": 1, "last_frame": 20})]            # a 20-frame clip: the guide's examples use one
NO_FADE = {"fade_in": 0.0, "fade_out": 0.0}


def stamped(value, blue=BLUE, alpha=1.0):
    """The picture of a frame-stamped plate whose stamp is `value`."""
    out = np.empty((H, W, 4), np.float64)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = value, GREEN, blue, alpha
    return out


def shows(value, blue=BLUE):
    """A reference that is the stamped plate at `value`, everywhere."""
    return lambda fx: (stamped(value, blue), np.ones((H, W, 4), bool))


def black(fx):
    """Black: the colour is zero; the guide does not say what a black frame's alpha is, so alpha is not compared."""
    valid = np.ones((H, W, 4), bool)
    valid[..., 3] = False
    return np.zeros((H, W, 4), np.float64), valid


def mix(*pairs):
    """A reference that is the weighted sum of stamped plates: pairs of (stamp value, weight)."""
    def expected(fx):
        out = np.zeros((H, W, 4), np.float64)
        for value, weight in pairs:
            out += weight * stamped(value)
        return out, np.ones((H, W, 4), bool)
    return expected


def cases():
    from nuke_fidelity_cases import Case

    out = []

    def case(id, nuke, nuke_knobs, nb_params, stamp, expected, source, statement, unscored="", nuke_class=None):
        params = dict(nb_params, _stamp=dict(stamp))
        out.append(Case(id=id, nuke=nuke, nuke_class=nuke_class or nuke, nuke_knobs=tuple(nuke_knobs), nuke_inputs=("stamp",), nb_kind=nuke,
                        nb_params=params, nb_inputs=(), expected=expected, source=source, statement=statement, unscored=unscored))

    # ----------------------------------------------------------------------------------------------------------------- TimeOffset
    page = GUIDE + "timeoffset.html"
    offset_note = (f"TimeOffset ({page}): 'Enter a positive value to add frames to the head of the clip. Enter a negative value to subtract "
                   "frames from the head of the clip', read as: a positive offset delays the clip, so the picture at frame f is the input's frame f - offset")
    for offset, frame, picks in ((3.0, 10, 7), (-3.0, 10, 13), (10.0, 25, 15)):
        case(f"timeoffset_offset_{offset:g}_at_{frame}", "TimeOffset", (("time_offset", offset),), {"time_offset": offset}, {"frame": frame},
             shows(picks), "guide", f"TimeOffset by {offset:g} frames shows input frame {picks} at frame {frame}. " + offset_note)
    rounding = (f"TimeOffset ({page}): 'Whole frames - round all times down to whole frames. Fractional time - calculate exact times which "
                "may result in times between frames'")
    for offset, frame, picks in ((0.5, 10, 9), (-0.5, 10, 10), (0.25, 10, 9)):
        case(f"timeoffset_whole_frames_round_down_{offset:g}", "TimeOffset", (("rounding_mode", "Whole frames"), ("time_offset", offset)),
             {"time_offset": offset, "rounding_mode": "Whole frames"}, {"frame": frame}, shows(picks), "guide",
             f"Whole frames: offset {offset:g} at frame {frame} asks for the time {frame - offset:g}, which rounds down to {picks}. " + rounding)
    for offset, frame in ((0.5, 10), (-0.25, 10)):
        case(f"timeoffset_fractional_time_{offset:g}", "TimeOffset", (("rounding_mode", "Fractional time"), ("time_offset", offset)),
             {"time_offset": offset, "rounding_mode": "Fractional time"}, {"frame": frame}, shows(frame - offset), "guide",
             f"Fractional time: offset {offset:g} at frame {frame} asks for the exact time {frame - offset:g}. " + rounding)
    reverse_note = (f"TimeOffset ({page}): 'Reverse Input: TimeOffset inverts the clip (making the last frame the first, and so on)'. The guide does not say "
                    "whose first and last frame; the clip's own range, the one an input states (a FrameRange sets it), is the reading used, "
                    "so frame t becomes first + last - t")
    for frame, picks in ((1, 20), (5, 16), (20, 1)):
        case(f"timeoffset_reverse_input_at_{frame}", "TimeOffset", (("reverse_input", True),), {"reverse_input": 1}, {"frame": frame, "chain": CLIP20},
             shows(picks), "convention", f"Reverse input on a 1 to 20 clip shows input frame {picks} at frame {frame}. " + reverse_note)
    hold_note = (f"TimeOffset ({page}): 'Nuke fills the empty frames at the head or tail of the clip by holding the first or last frame'. "
                 "The clip is a 1 to 20 frame range, which holds outside itself")
    for offset, frame, picks in ((5.0, 3, 1), (-5.0, 18, 20), (5.0, 12, 7)):
        case(f"timeoffset_clip_edges_hold_{offset:g}_at_{frame}", "TimeOffset", (("time_offset", offset),), {"time_offset": offset},
             {"frame": frame, "chain": CLIP20}, shows(picks), "guide",
             f"Offset {offset:g} on a 1 to 20 clip at frame {frame} shows input frame {picks}. " + hold_note)
    case("timeoffset_reverse_with_offset", "TimeOffset", (("time_offset", 3.0), ("reverse_input", True)), {"time_offset": 3.0, "reverse_input": 1},
         {"frame": 5, "chain": CLIP20}, shows(19), "convention", "Reverse input together with an offset of 3 on a 1 to 20 clip",
         unscored="the guide does not say whether the offset or the reversal is applied first; the two orders pick different frames")

    # ------------------------------------------------------------------------------------------------------------------ FrameHold
    page = GUIDE + "framehold.html"
    for frame in (3, 12, 40):
        case(f"framehold_first_frame_at_{frame}", "FrameHold", (("firstFrame", 7.0), ("increment", 0.0)), {"first_frame": 7.0, "increment": 0.0},
             {"frame": frame}, shows(7), "guide",
             f"FrameHold ({page}): 'The first frame to use. If increment is set to 0, first frame is used at every frame of the input clip': "
             f"first frame 7 shows input frame 7 at frame {frame}")
    for frame in (6, 11, 16):
        case(f"framehold_increment_5_uses_frame_{frame}", "FrameHold", (("firstFrame", 1.0), ("increment", 5.0)),
             {"first_frame": 1.0, "increment": 5.0}, {"frame": frame}, shows(frame), "guide",
             f"FrameHold ({page}): 'if this is set to 5 and first frame is set to 1, Nuke only uses frames 1, 6, 11, 16, and so on from the input': "
             f"at frame {frame}, one of those frames, the input's own frame {frame} shows")
    for frame in (2, 8):
        case(f"framehold_increment_5_between_frame_{frame}", "FrameHold", (("firstFrame", 1.0), ("increment", 5.0)),
             {"first_frame": 1.0, "increment": 5.0}, {"frame": frame}, shows(1 + 5 * ((frame - 1) // 5)), "convention",
             f"FrameHold with first frame 1 and increment 5 at frame {frame}, between two listed frames", unscored=(
                 "the guide lists the frames used (1, 6, 11, 16) but not which of them an output frame between two of them shows; "
                 "'the latest listed frame at or before it' is the natural reading and no source states it"))
    for frame, first in ((3, 10.0), (0, 10.0), (-4, 1.0)):
        case(f"framehold_increment_outside_{frame}_first_{first:g}", "FrameHold", (("firstFrame", first), ("increment", 5.0)),
             {"first_frame": first, "increment": 5.0}, {"frame": frame}, shows(first), "convention",
             f"FrameHold at frame {frame} (zero or negative, or before the first frame {first:g})",
             unscored="the guide states nothing for frame 0, negative frames or frames before the first frame")
    case("framehold_negative_increment", "FrameHold", (("firstFrame", 20.0), ("increment", -5.0)), {"first_frame": 20.0, "increment": -5.0},
         {"frame": 25}, shows(20), "convention", "FrameHold with first frame 20 and a negative increment of 5",
         unscored="the guide gives no rule for a negative increment")
    rounding = (f"FrameHold ({page}): 'Whole frames - round all times down to whole frames. Fractional time - calculate exact times which "
                "may result in times between frames'")
    case("framehold_whole_frames_round_down", "FrameHold", (("rounding_mode", "Whole frames"), ("firstFrame", 7.75)),
         {"first_frame": 7.75, "increment": 0.0, "rounding_mode": "Whole frames"}, {"frame": 12}, shows(7), "guide",
         "Whole frames: first frame 7.75 rounds down to 7. " + rounding)
    case("framehold_fractional_time", "FrameHold", (("rounding_mode", "Fractional time"), ("firstFrame", 7.75)),
         {"first_frame": 7.75, "increment": 0.0, "rounding_mode": "Fractional time"}, {"frame": 12}, shows(7.75), "guide",
         "Fractional time: first frame 7.75 is kept as the exact time. " + rounding)

    # -------------------------------------------------------------------------------------------------------------------- Retime
    page = GUIDE + "retime.html"
    map_note = ("the guide states no formula for the mapping; the one used puts output.first on input.first and moves the input frame by "
                f"speed for each output frame, which is what its lock note ('the output frames move to accommodate the speed') and speed text "
                f"('values higher than 1 increase playback speed') describe ({page})")
    none_note = f"Retime filter none: 'passes fractional frame numbers to the input' ({page})"
    retime = {"retime_filter": "none"}
    for speed, frame, picks in ((2.0, 4, 7.0), (2.0, 1, 1.0), (0.5, 4, 2.5), (0.5, 7, 4.0), (1.5, 3, 4.0)):
        case(f"retime_speed_{speed:g}_none_at_{frame}", "Retime", (("speed", speed), ("filter", "none")), dict(retime, speed=speed),
             {"frame": frame, "chain": CLIP20}, shows(picks), "convention",
             f"Retime at speed {speed:g}, filter none, shows input frame {picks:g} at output frame {frame} (1 + {frame - 1} x {speed:g}). {map_note}. {none_note}")
    near_note = f"Retime filter nearest: 'rounds the center of the range to nearest integer frame' ({page})"
    for speed, frame, centre, picks in ((0.4, 4, 2.2, 2), (0.7, 4, 3.1, 3), (0.7, 3, 2.4, 2), (0.9, 5, 4.6, 5)):
        case(f"retime_speed_{speed:g}_nearest_at_{frame}", "Retime", (("speed", speed), ("filter", "nearest")),
             {"retime_filter": "nearest", "speed": speed}, {"frame": frame, "chain": CLIP20}, shows(picks), "convention",
             f"Retime at speed {speed:g}, filter nearest, centre {centre:g} rounds to input frame {picks} at output frame {frame}. {map_note}. {near_note}")
    case("retime_nearest_half_way", "Retime", (("speed", 0.5), ("filter", "nearest")), {"retime_filter": "nearest", "speed": 0.5},
         {"frame": 4, "chain": CLIP20}, shows(2), "convention", "Retime filter nearest on a centre exactly half way between two frames (2.5)",
         unscored="the guide does not say which way a half rounds")
    locked = {"input_range_start": 1, "input_range_start_lock": 1, "input_range_end": 20, "input_range_end_lock": 1,
              "output_range_start": 1, "output_range_start_lock": 1, "output_range_end": 10, "output_range_end_lock": 1, "retime_filter": "none"}
    locked_knobs = (("input.first", 1.0), ("input.first_lock", True), ("input.last", 20.0), ("input.last_lock", True), ("output.first", 1.0),
                    ("output.first_lock", True), ("output.last", 10.0), ("output.last_lock", True), ("filter", "none"))
    lock_note = (f"Retime ({page}): 'If both output range fields are locked, the retime is calculated to make the frame ranges match': "
                 "input 1 to 20 locked, output 1 to 10 locked")
    for frame, picks in ((1, 1.0), (10, 20.0)):
        case(f"retime_both_ranges_locked_ends_at_{frame}", "Retime", locked_knobs, dict(locked), {"frame": frame, "chain": CLIP20}, shows(picks), "guide",
             f"With both ranges locked output frame {frame} shows input frame {picks:g}. " + lock_note)
    case("retime_both_ranges_locked_middle", "Retime", locked_knobs, dict(locked), {"frame": 5, "chain": CLIP20}, shows(1 + 4 * 19 / 9), "convention",
         "With both ranges locked the output frames sit evenly on the input range: output frame 5 shows input frame 1 + 4 x 19/9. " + lock_note)
    rev = dict(retime, reverse=1)
    for frame, picks in ((1, 20.0), (2, 19.0), (20, 1.0)):
        case(f"retime_reverse_at_{frame}", "Retime", (("reverse", True), ("filter", "none")), rev, {"frame": frame, "chain": CLIP20}, shows(picks), "guide",
             f"Retime reverse ('When enabled, the input frames run backwards', {page}) on a 1 to 20 clip: output frame {frame} shows input frame {picks:g}, "
             "the first output frame being the clip's last")
    before = {"retime_filter": "none", "output_range_start": 5, "output_range_start_lock": 1}
    before_knobs = (("output.first", 5.0), ("output.first_lock", True), ("filter", "none"))
    before_note = (f"Retime ({page}), the frames before output.first, for a 20 frame sequence with output.first 5: hold '1, 1, 1, 1, 1, 2, 3, 4', "
                   "loop '17, 18, 19, 20, 1, 2, 3, 4', bounce '5, 4, 3, 2, 1, 2, 3, 4'; output frames 1 to 8 in that order")
    for policy, picks in (("hold", {1: 1, 3: 1, 5: 1, 6: 2}), ("continue", {1: 1, 3: 1, 5: 1, 6: 2}),
                          ("loop", {1: 17, 4: 20, 5: 1, 6: 2}), ("bounce", {1: 5, 3: 3, 4: 2, 5: 1, 6: 2})):
        for frame, value in picks.items():
            case(f"retime_before_{policy}_at_{frame}", "Retime", before_knobs + (("before", policy),), dict(before, retime_before=policy),
                 {"frame": frame, "chain": CLIP20}, shows(float(value)), "guide",
                 f"Retime before = {policy}, output.first 5, 1 to 20 clip: output frame {frame} shows input frame {value}. {before_note}"
                 + (". 'continue' reads in the guide exactly as hold, and the clip's range holds the frame" if policy == "continue" else ""))
    case("retime_before_black_at_3", "Retime", before_knobs + (("before", "black"),), dict(before, retime_before="black"),
         {"frame": 3, "chain": CLIP20}, black, "guide", f"Retime before = black: 'frames are black until the output.first frame is reached' ({page}), output frame 3")
    case("retime_before_black_keeps_first_frame", "Retime", before_knobs + (("before", "black"),), dict(before, retime_before="black"),
         {"frame": 5, "chain": CLIP20}, shows(1.0), "guide", f"Retime before = black: the output.first frame itself (5) shows input frame 1 ({page})")
    after = {"retime_filter": "none"}
    after_knobs = (("filter", "none"),)
    after_note = (f"Retime ({page}), the frames after the last output frame, for a 20 frame sequence: hold '16, 17, 18, 19, 20, 20, 20, 20', "
                  "loop '16, 17, 18, 19, 20, 1, 2, 3, 4', bounce '16, 17, 18, 19, 20, 19, 18, 17'; the clip's output frames 1 to 20 show input 1 to 20")
    for policy, picks in (("hold", {20: 20, 21: 20, 25: 20}), ("continue", {21: 20, 25: 20}), ("loop", {20: 20, 21: 1, 25: 5}),
                          ("bounce", {20: 20, 21: 19, 22: 18, 25: 15})):
        for frame, value in picks.items():
            case(f"retime_after_{policy}_at_{frame}", "Retime", after_knobs + (("after", policy),), dict(after, retime_after=policy),
                 {"frame": frame, "chain": CLIP20}, shows(float(value)), "guide",
                 f"Retime after = {policy}, 1 to 20 clip at speed 1: output frame {frame} shows input frame {value}. {after_note}"
                 + (". 'continue' reads in the guide exactly as hold, and the clip's range holds the frame" if policy == "continue" else ""))
    case("retime_after_black_at_23", "Retime", after_knobs + (("after", "black"),), dict(after, retime_after="black"), {"frame": 23, "chain": CLIP20}, black,
         "guide", f"Retime after = black: 'frames are black from output.last until the end of the sequence is reached' ({page}), output frame 23")
    case("retime_filter_box", "Retime", (("speed", 2.0), ("filter", "box"), ("shutter", 1.0)), {"speed": 2.0, "retime_filter": "box", "shutter": 1.0},
         {"frame": 4, "chain": CLIP20}, mix((6, 0.25), (7, 0.5), (8, 0.25)), "convention", "Retime filter box at speed 2, output frame 4: the weighted average of the frames the box covers",
         unscored="the guide says only 'a weighted average of several frames together to cover the output range' and gives no weights")
    case("retime_shutter_blending", "Retime", (("speed", 2.0), ("filter", "box"), ("shutter", 0.5)), {"speed": 2.0, "retime_filter": "box", "shutter": 0.5},
         {"frame": 4, "chain": CLIP20}, shows(7.0), "convention", "Retime shutter 0.5 'generates less frame-blending'",
         unscored="the guide gives no width or weights for the blend, only that a lower shutter blends less")

    # ----------------------------------------------------------------------------------------------------------------- FrameRange
    page = GUIDE + "framerange.html"
    for frame in (25, 30, 50):
        case(f"framerange_inside_at_{frame}", "FrameRange", (("first_frame", 25.0), ("last_frame", 50.0)), {"first_frame": 25, "last_frame": 50},
             {"frame": frame}, shows(frame), "guide",
             f"FrameRange ({page}): 'if your original clip is 50 frames but you want to use only the last 25 frames in your composite, you would "
             f"enter 25 as the first frame and leave the last frame at 50': the frames 25 to 50 are the input's own, so frame {frame} shows frame {frame}")
    for frame in (10, 60):
        case(f"framerange_outside_at_{frame}", "FrameRange", (("first_frame", 25.0), ("last_frame", 50.0)), {"first_frame": 25, "last_frame": 50},
             {"frame": frame}, shows(25 if frame < 25 else 50), "convention", f"FrameRange 25 to 50 at frame {frame}, outside its range",
             unscored="the guide says FrameRange sets which frames the Viewer and flipbook play and what AppendClip joins, and states nothing about the "
                      "picture at a frame outside the range")

    # ------------------------------------------------------------------------------------------------------------------- TimeClip
    page = GUIDE + "timeclip.html"
    clip = {"frame_range_type": "custom", **NO_FADE}
    clip_knobs = (("fadeIn", 0.0), ("fadeOut", 0.0))
    for expr, frame, picks, note in (
            ("frame+49", 1, 50, "the guide's frame+499 example ('At frame 1, image.0500.rgb is read in; at frame 2, image.0501.rgb'), scaled to 49"),
            ("frame+49", 2, 51, "the same example at frame 2"),
            ("frame*2", 3, 6, "the guide's frame*2 example ('At frame 1, image.0002.rgb is read in; at frame 2, image.0004.rgb; at frame 3, image.0006.rgb')"),
            ("frame*2", 1, 2, "the same example at frame 1")):
        case(f"timeclip_expression_{expr.replace('+', 'plus').replace('*', 'times')}_at_{frame}", "TimeClip", (("frame_mode", "expression"),) + clip_knobs,
             dict(clip, frame_mode="expression", frame_expr=expr, first=1, last=100), {"frame": frame}, shows(picks), "guide",
             f"TimeClip frame = expression {expr}: frame {frame} reads input frame {picks}; {note} ({page})")
    for offset, frame, picks, note in ((49, 1, 50, "the guide's offset 499 example scaled to 49"), (-10, 20, 10, "the guide's -10 example ('At frame 20, image.0010.rgb is read in')"),
                                       (-10, 21, 11, "the same example at frame 21")):
        case(f"timeclip_offset_{offset}_at_{frame}", "TimeClip", (("frame_mode", "offset"),) + clip_knobs,
             dict(clip, frame_mode="offset", frame_expr=str(offset), first=1, last=100), {"frame": frame}, shows(picks), "guide",
             f"TimeClip frame = offset {offset}: frame {frame} reads input frame {picks}; {note} ({page})")
    for start, frame, picks in ((1, 1, 50), (1, 4, 53), (20, 20, 50), (20, 25, 55)):
        case(f"timeclip_start_at_{start}_shows_{frame}", "TimeClip", (("frame_mode", "start at"),) + clip_knobs,
             dict(clip, frame_mode="start at", frame_expr=str(start), first=50, last=90), {"frame": frame}, shows(picks), "guide",
             f"TimeClip frame = start at {start}: the clip, which begins at frame 50, is read in from frame {start}, so frame {frame} reads input frame {picks} "
             f"('all frames are offset so that the clip starts at the specified frame', {page})")
    before_note = (f"TimeClip ({page}), frames before the first frame (first 5, a 20 frame sequence): hold 'the first frame in the sequence is held until "
                   "the first frame is reached', loop 'substitutes an equal number of frames', black 'frames are black until the first frame is reached'. "
                   "The guide's loop example '17, 18, 19, 20, 1, 2, 3, 4' is read for the frames before first only: the range 5 to 20 is 16 frames, so frames "
                   "1 to 4 show 17 to 20")
    for policy, picks in (("hold", {1: 5, 4: 5}), ("loop", {1: 17, 4: 20, 5: 5})):
        for frame, value in picks.items():
            case(f"timeclip_before_{policy}_at_{frame}", "TimeClip", (("first", 5.0), ("before", policy)) + clip_knobs,
                 dict(clip, first=5, last=20, before=policy), {"frame": frame}, shows(float(value)), "guide",
                 f"TimeClip before = {policy}, range 5 to 20: frame {frame} shows input frame {value}. {before_note}")
    case("timeclip_before_black_at_3", "TimeClip", (("first", 5.0), ("before", "black")) + clip_knobs, dict(clip, first=5, last=20, before="black"),
         {"frame": 3}, black, "guide", f"TimeClip before = black, range 5 to 20: frame 3 is black. {before_note}")
    case("timeclip_before_bounce", "TimeClip", (("first", 5.0), ("before", "bounce")) + clip_knobs, dict(clip, first=5, last=20, before="bounce"),
         {"frame": 3}, shows(7.0), "convention", "TimeClip before = bounce, range 5 to 20, frame 3",
         unscored="the guide's bounce example '5, 4, 3, 2, 1, 2, 3, 4' is the Retime example copied (it counts down from 5 to 1, not from the range's "
                  "first frame), so it does not state what the frames before first show")
    after_note = (f"TimeClip ({page}), frames after the last frame (a 20 frame sequence): hold '16, 17, 18, 19, 20, 20, 20, 20', loop '16, 17, 18, 19, 20, "
                  "1, 2, 3, 4', bounce '16, 17, 18, 19, 20, 19, 18, 17', black 'frames are black from last until the end of the sequence is reached'")
    for policy, picks in (("hold", {20: 20, 21: 20, 30: 20}), ("loop", {20: 20, 21: 1, 24: 4}), ("bounce", {20: 20, 21: 19, 22: 18})):
        for frame, value in picks.items():
            case(f"timeclip_after_{policy}_at_{frame}", "TimeClip", (("last", 20.0), ("after", policy)) + clip_knobs,
                 dict(clip, first=1, last=20, after=policy), {"frame": frame}, shows(float(value)), "guide",
                 f"TimeClip after = {policy}, range 1 to 20: frame {frame} shows input frame {value}. {after_note}")
    case("timeclip_after_black_at_21", "TimeClip", (("last", 20.0), ("after", "black")) + clip_knobs, dict(clip, first=1, last=20, after="black"),
         {"frame": 21}, black, "guide", f"TimeClip after = black, range 1 to 20: frame 21 is black. {after_note}")
    for frame, picks in ((5, 20.0), (6, 19.0), (20, 5.0)):
        case(f"timeclip_reverse_at_{frame}", "TimeClip", (("first", 5.0), ("last", 20.0), ("reverse", True)) + clip_knobs,
             dict(clip, first=5, last=20, reverse=1), {"frame": frame}, shows(picks), "guide",
             f"TimeClip reverse ('Reverse the clip within the specified frame range', {page}): range 5 to 20, frame {frame} shows input frame {picks:g}")
    for frame in (6, 10, 15):
        case(f"timeclip_fade_leaves_the_middle_alone_at_{frame}", "TimeClip", (("first", 1.0), ("last", 20.0), ("fadeIn", 3.0), ("fadeOut", 3.0)),
             dict(clip, first=1, last=20, fade_in=3.0, fade_out=3.0), {"frame": frame}, shows(float(frame)), "guide",
             f"TimeClip fade in 3 and fade out 3 ('The number of frames to fade from black' and 'to black', {page}): frame {frame} of the 1 to 20 clip is "
             "in neither fade and shows the input unchanged")
    case("timeclip_fade_in_curve", "TimeClip", (("first", 1.0), ("last", 20.0), ("fadeIn", 4.0)), dict(clip, first=1, last=20, fade_in=4.0, fade_out=0.0),
         {"frame": 2}, mix((2.0, 0.4)), "convention", "TimeClip fade in 4: the picture at frame 2",
         unscored="the guide gives the length of the fade and no curve; (k + 1) / (fade + 1) is our reading")

    # ------------------------------------------------------------------------------------------------------------------ AppendClip
    page = GUIDE + "appendclip.html"
    two = [[("FrameRange", {"first_frame": 1, "last_frame": 5})], [("FrameRange", {"first_frame": 11, "last_frame": 15})]]
    blue0, blue1 = 0.2, 0.3
    join = (f"AppendClip ({page}): 'Splicing refers to joining clips head-to-tail', two clips cut with FrameRange: frames 1 to 5 of the first, "
            "then frames 11 to 15 of the second")
    for frame, value, blue, which in ((1, 1, blue0, "first"), (3, 3, blue0, "first"), (5, 5, blue0, "first"), (6, 11, blue1, "second"), (10, 15, blue1, "second")):
        case(f"appendclip_head_to_tail_at_{frame}", "AppendClip", (), {}, {"frame": frame, "clips": two}, shows(float(value), blue), "guide",
             f"AppendClip output frame {frame} shows frame {value} of the {which} clip. {join}")
    slip = (f"AppendClip ({page}): First Frame 'Sets the amount of frames to slip the start of the first clip in the sequence. For example, a value "
            "of 5 causes the first clip to start playing at frame 5 instead of frame 1'")
    for frame, value, blue in ((5, 1, blue0), (9, 5, blue0), (10, 11, blue1)):
        case(f"appendclip_first_frame_5_at_{frame}", "AppendClip", (("firstFrame", 5.0),), {"first_frame": 5}, {"frame": frame, "clips": two},
             shows(float(value), blue), "guide", f"AppendClip first frame 5: output frame {frame} shows frame {value}. {slip}")
    for frame, value, blue in ((2, 2, blue0), (4, 4, blue0)):
        case(f"appendclip_fade_in_leaves_the_rest_alone_at_{frame}", "AppendClip", (("fadeIn", 1.0), ("fadeOut", 1.0)),
             {"fade_in": 1, "fade_out": 1}, {"frame": frame, "clips": two}, shows(float(value), blue), "guide",
             f"AppendClip fade in 1 and fade out 1: 'the number of frames to fade in from black on the first clip' and 'out to black on the last clip' "
             f"({page}); output frame {frame} is in neither")
    case("appendclip_fade_in_curve", "AppendClip", (("fadeIn", 4.0),), {"fade_in": 4}, {"frame": 1, "clips": two}, mix((1.0, 0.2)), "convention",
         "AppendClip fade in 4: the picture at the first frame",
         unscored="the guide gives the number of frames and no curve; (k + 1) / (fade + 1) is our reading")
    for frame, value, blue in ((1, 1, blue0), (2, 2, blue0)):
        case(f"appendclip_dissolve_before_the_overlap_at_{frame}", "AppendClip", (("dissolve", 2.0),), {"dissolve": 2}, {"frame": frame, "clips": two},
             shows(float(value), blue), "guide",
             f"AppendClip cross dissolve 2 ('the number of frames to use to dissolve between clips', {page}): output frame {frame} is before any dissolve "
             "and shows the first clip's own frame")
    case("appendclip_dissolve_weights", "AppendClip", (("dissolve", 2.0),), {"dissolve": 2}, {"frame": 4, "clips": two}, mix((4.0, 0.5)), "convention",
         "AppendClip cross dissolve 2: the blend inside the overlap",
         unscored="the guide gives only the number of frames: not whether they overlap the clips or lengthen the edit, nor the blend weights")

    # ------------------------------------------------------------------------------------------------------------------- TimeEcho
    page = GUIDE + "timeecho.html"

    def echo(method, n, fade, frame):
        """The guide's echo: the n frames ending at `frame`, the oldest `fade` of them faded 1/fade, 2/fade, ... of the way up (the last stays at full)."""
        frames = [frame - n + 1 + i for i in range(n)]                      # oldest first
        weights = [min(1.0, (i + 1) / fade) for i in range(n)]
        values = [w * f for w, f in zip(weights, frames)]
        if method == "plus":
            red = sum(values)
        elif method == "max":
            red = max(values)
        else:
            red = sum(values) / n
        green = {"plus": sum(w * GREEN for w in weights), "max": max(w * GREEN for w in weights), "average": sum(w * GREEN for w in weights) / n}[method]
        blue = {"plus": sum(w * BLUE for w in weights), "max": max(w * BLUE for w in weights), "average": sum(w * BLUE for w in weights) / n}[method]

        def expected(fx):
            ref = np.empty((H, W, 4), np.float64)
            ref[..., 0], ref[..., 1], ref[..., 2], ref[..., 3] = red, green, blue, 1.0
            valid = np.ones((H, W, 4), bool)
            valid[..., 3] = False                                            # the guide speaks of pixel values, not of how alpha adds up
            return ref, valid
        return expected
    names = {"plus": "Plus", "max": "Max", "average": "Average"}
    mode_note = {"plus": "Plus 'uses the sum of all frames'", "max": "Max 'uses the highest value from the frames'",
                 "average": "Average 'averages the frames (uses the sum divided by the number of frames)'"}
    for method in ("plus", "max", "average"):
        case(f"timeecho_{method}_4_frames_at_10", "TimeEcho", (("mode", names[method]), ("framesbehind", 4.0), ("framesfade", 1.0)),
             {"method": method, "frames": 4, "falloff": 1.0, "frames_fade": 1.0}, {"frame": 10}, echo(method, 4, 1, 10), "guide",
             f"TimeEcho {names[method]} over 4 frames at frame 10 merges frames 7, 8, 9 and 10: 'if the current frame is 10, and you set this value to 4, "
             f"TimeEcho merges frames 7, 8, 9, and 10'; {mode_note[method]} ({page})")
    for frame in (4, 12):
        case(f"timeecho_one_frame_at_{frame}", "TimeEcho", (("mode", "Max"), ("framesbehind", 1.0)), {"method": "max", "frames": 1, "falloff": 1.0, "frames_fade": 1.0},
             {"frame": frame}, shows(float(frame)), "guide", f"TimeEcho looking at 1 frame shows the current frame ({frame}) ({page})")
    fade_note = (f"TimeEcho ({page}): 'the fade will gradually happen over the first five frames used, with the fifth frame remaining at maximum: the first "
                 "image is reduced to 20%, the second to 40%, the third to 60%, the fourth to 80%, the fifth remains at 100%'; 'the further back a frame is in "
                 "time, the less effect it has', so the first image is the furthest back")
    for method in ("plus", "max", "average"):
        case(f"timeecho_fade_5_{method}_at_10", "TimeEcho", (("mode", names[method]), ("framesbehind", 5.0), ("framesfade", 5.0)),
             {"method": method, "frames": 5, "falloff": 1.0, "frames_fade": 5.0}, {"frame": 10}, echo(method, 5, 5, 10), "guide",
             f"TimeEcho {names[method]}, 5 frames faded over 5 at frame 10: frames 6 to 10 at 20%, 40%, 60%, 80% and 100%. {fade_note}; {mode_note[method]}")
    case("timeecho_fade_shorter_than_the_frames", "TimeEcho", (("mode", "Plus"), ("framesbehind", 6.0), ("framesfade", 3.0)),
         {"method": "plus", "frames": 6, "falloff": 1.0, "frames_fade": 3.0}, {"frame": 10}, echo("plus", 6, 3, 10), "guide",
         "TimeEcho Plus, 6 frames faded over 3 at frame 10: frames 5, 6 and 7 at 33%, 67% and 100%, and frames 8, 9 and 10 at full (the fade covers "
         "the first three frames used, the rest stay at maximum). " + fade_note)
    case("timeecho_fade_longer_than_the_frames", "TimeEcho", (("mode", "Plus"), ("framesbehind", 3.0), ("framesfade", 5.0)),
         {"method": "plus", "frames": 3, "falloff": 1.0, "frames_fade": 5.0}, {"frame": 10}, echo("plus", 3, 5, 10), "convention",
         "TimeEcho with a fade over more frames than it looks at",
         unscored="the guide's fade example uses as many frames as the fade; with fewer frames the ramp is cut short and which part of it is used is not stated")

    # ------------------------------------------------------------------------------------------------------------------- TimeBlur
    page = GUIDE + "timeblur.html"
    window_note = (f"TimeBlur ({page}): 'sampling its input at divisions times over shutter frames starting at the current frame plus offset'; shutter offset "
                   "centred 'if you set the shutter value to 1 and your current frame is 30, the shutter stays open from frame 29,5 to 30,5', start 'from frame "
                   "30 to 31', end 'from frame 29 to 30', custom 'open the shutter at the time you specify' (current frame plus the value). The guide "
                   "does not place the samples inside the window; they sit at the middle of equal parts of it, so the stamp's average is the window's centre")

    def blur(shutter, offset, custom, centre, frame=30, divisions=4):
        case(f"timeblur_{offset}_shutter_{shutter:g}_custom_{custom:g}", "TimeBlur",
             (("divisions", float(divisions)), ("shutter", shutter), ("shutteroffset", offset), ("shuttercustomoffset", custom)),
             {"divisions": divisions, "shutter": shutter, "shutter_offset": offset, "custom_offset": custom}, {"frame": frame}, shows(centre), "convention",
             f"TimeBlur shutter {shutter:g}, offset {offset}" + (f" {custom:g}" if offset == "custom" else "") + f", at frame {frame}: the window's centre is "
             f"{centre:g}. {window_note}")
    blur(1.0, "centred", 0.0, 30.0)
    blur(1.0, "start", 0.0, 30.5)
    blur(1.0, "end", 0.0, 29.5)
    blur(0.5, "start", 0.0, 30.25)
    blur(0.5, "end", 0.0, 29.75)
    blur(1.0, "custom", -0.5, 30.0)
    blur(1.0, "custom", 2.0, 32.5)
    blur(2.0, "custom", 0.0, 31.0)

    # ------------------------------------------------------------------------------------------------------------------- TimeWarp
    page = GUIDE + "timewarp.html"
    look = (f"TimeWarp ({page}): input frame 'Sets which frame of the input sequence appears at the current frame. For example, you could set frame 10 of "
            "the input to output at frame 5'")
    straight = {"lookup": ("linear", ((1, 1), (5, 10)))}
    case("timewarp_lookup_key_at_5", "TimeWarp", (("filter", "none"),), {"lookup_filter": "none"}, {"frame": 5, "curves": straight}, shows(10.0), "guide",
         "TimeWarp with a lookup key of 10 at frame 5 shows input frame 10 at frame 5. " + look)
    case("timewarp_lookup_key_at_1", "TimeWarp", (("filter", "none"),), {"lookup_filter": "none"}, {"frame": 1, "curves": straight}, shows(1.0), "guide",
         "TimeWarp with a lookup key of 1 at frame 1 shows input frame 1 at frame 1. " + look)
    none_note = f"TimeWarp filter none: 'passes fractional frame numbers to the input' ({page})"
    for frame, picks in ((3, 5.5), (2, 3.25), (4, 7.75)):
        case(f"timewarp_straight_curve_none_at_{frame}", "TimeWarp", (("filter", "none"),), {"lookup_filter": "none"}, {"frame": frame, "curves": straight},
             shows(picks), "guide",
             f"TimeWarp with a straight lookup curve from (1, 1) to (5, 10), filter none: frame {frame} reads the fractional input frame {picks:g}. {look}. {none_note}")
    near_note = f"TimeWarp filter nearest: 'rounds the center of the range to nearest integer frame' ({page})"
    for frame, centre, picks in ((4, 7.75, 8), (2, 3.25, 3)):
        case(f"timewarp_straight_curve_nearest_at_{frame}", "TimeWarp", (("filter", "nearest"),), {"lookup_filter": "nearest"},
             {"frame": frame, "curves": straight}, shows(float(picks)), "guide",
             f"TimeWarp with a straight lookup curve from (1, 1) to (5, 10), filter nearest: frame {frame} reads {centre:g}, "
             f"which rounds to input frame {picks}. {look}. {near_note}")
    case("timewarp_nearest_half_way", "TimeWarp", (("filter", "nearest"),), {"lookup_filter": "nearest"}, {"frame": 3, "curves": straight}, shows(6.0),
         "convention", "TimeWarp filter nearest on a lookup exactly half way between two frames (5.5)",
         unscored="the guide does not say which way a half rounds")
    held = {"lookup": ("constant", ((1, 3), (10, 8)))}
    for frame, picks in ((1, 3.0), (5, 3.0), (9, 3.0), (10, 8.0), (14, 8.0)):
        case(f"timewarp_hold_curve_at_{frame}", "TimeWarp", (("filter", "none"),), {"lookup_filter": "none"}, {"frame": frame, "curves": held}, shows(picks),
             "guide", f"TimeWarp with a held lookup curve (input frame 3 from frame 1, input frame 8 from frame 10): frame {frame} shows input frame {picks:g}. {look}")
    case("timewarp_box_filter", "TimeWarp", (("filter", "box"), ("length", 1.0)), {"lookup_filter": "blend"}, {"frame": 3, "curves": straight}, mix((5, 0.5), (6, 0.5)),
         "convention", "TimeWarp filter box on a fractional lookup", unscored=(
             "the guide says only 'a weighted average of several frames together to cover the output range' and gives no weights; "
             "NodeBased's blend of the two nearest frames is a unit-wide box and nothing states that is Nuke's"))
    return out
