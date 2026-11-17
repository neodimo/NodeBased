"""Behaviour cases for the Image, Merge, Metadata and Other nodes the review book listed as unchecked (lane 11, step B3, November 2026).

Most of these nodes leave the picture alone: Dot and NoOp say so in the guide's words, the metadata nodes change only metadata, the
viewing and analysis nodes (Viewer, PostageStamp, CurveTool, Profile) show or measure their input, and Write and Output hand their
input on. Each such case states that the output pixels equal the input's. ZMerge, CopyBBox, ContactSheet and AddTimeCode have
cases of their own. A node whose behaviour is not a picture (Read reads a file, Backdrop and Group organise the graph, Root is the
project's settings) or that NodeBased does not have is listed `unscored` with the reason, so the review book shows it as checked
and the reason instead of an empty page.

Where the sources come from
* `guide`: the Nuke 17.0 Reference Guide page (learn.foundry.com/nuke/17.0/content/reference_guide/<group>_nodes/<node>.html)
  states the behaviour in words.
* `convention`: the page describes what the node is for (metadata, viewing, measuring, writing a file) without a sentence on the
  pixels it passes on; the note names the page and the reading.

Only names defined above `CASES = _cases()` in `nuke_fidelity_cases` are imported (this module is imported by that line's hook).
"""
from __future__ import annotations

import numpy as np

GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/"


def cases():
    from nuke_fidelity_cases import Case, _all_valid

    out = []

    def case(id, nuke, nb_kind, nb_params, expected, source, statement, nuke_knobs=(), nuke_inputs=("A",), nb_inputs=("A",),
             unscored="", inexpressible="", nuke_class=None):
        out.append(Case(id=id, nuke=nuke, nuke_class=nuke_class or nuke, nuke_knobs=nuke_knobs, nuke_inputs=nuke_inputs,
                        nb_kind=nb_kind, nb_params=nb_params, nb_inputs=nb_inputs, expected=expected, source=source,
                        statement=statement, unscored=unscored, inexpressible=inexpressible))

    def same_as(name):
        def expected(fx):
            ref = fx[name].astype(np.float64)
            return ref, _all_valid(ref)
        return expected

    # ------------------------------------------------------------------------------------------------ pass-through nodes
    stated = (
        ("Dot", "other_nodes/dot.html", "The Dot node does not alter the input image in any way."),
        ("NoOp", "other_nodes/noop.html", "NoOp passes the input image through unchanged."),
    )
    for node, page, words in stated:
        case(f"{node.lower()}_passes_input_unchanged", node, node, {}, same_as("A"), "guide",
             f"{words} ({GUIDE}{page})")

    read_as = (
        ("ViewMetaData", "metadata_nodes/viewmetadata.html", "shows the input's metadata in a table; the picture is not touched"),
        ("ModifyMetaData", "metadata_nodes/modifymetadata.html", "adds, edits or removes the metadata associated with a file; the pixels are not touched"),
        ("CopyMetaData", "metadata_nodes/copymetadata.html", "copies metadata from its Meta input onto the Image stream, whose pixels go on unchanged"),
        ("CompareMetaData", "metadata_nodes/comparemetadata.html", "compares two inputs' metadata and shows the differences; the first input's pixels go on unchanged"),
        ("Viewer", "image_nodes/viewer.html", "displays its input; the node computes nothing of its own"),
        ("PostageStamp", "other_nodes/postagestamp.html", "shows a thumbnail of its input on the node and passes the input on"),
        ("CurveTool", "image_nodes/curvetool.html", "analyses the input (crop, intensity, exposure, max luma) into knob values and passes the image on"),
        ("Profile", "image_nodes/profile.html", "graphs the input's values along a line or box for inspection and passes the image on"),
        ("Write", "image_nodes/write.html", "renders its input to a file; its output in the graph is its input"),
        ("Output", "other_nodes/output.html", "marks a group's output; what it receives is what the group outputs"),
    )
    for node, page, reading in read_as:
        case(f"{node.lower()}_passes_input_unchanged", node, node, {}, same_as("A"), "convention",
             f"{node} {reading}. Reading of the guide page {GUIDE}{page}, which describes what the node is for without a sentence on the "
             f"pixels it passes on.")

    # ------------------------------------------------------------------------------------------------------- AddTimeCode
    case("addtimecode_passes_input_unchanged", "AddTimeCode", "AddTimeCode", {"_frame": 1}, same_as("A"), "convention",
         f"AddTimeCode writes a timecode into the metadata; the pixels go on unchanged. Reading of {GUIDE}metadata_nodes/addtimecode.html, "
         "which describes only the timecode it adds. Run at frame 1.",
         nuke_knobs=(("startcode", "01:00:00:00"), ("fps", 24)))

    # ------------------------------------------------------------------------------------------------------------ ZMerge
    zpage = f"{GUIDE}merge_nodes/zmerge.html"
    zdepth = (("A", 0, 10.0), ("B", 1, 10.0))       # input 0's depth is 10 x its red, input 1's 10 x the other plate's green

    def nearest(fx, larger_is_near=False):
        a, b = fx["A"].astype(np.float64), fx["B"].astype(np.float64)
        az, bz = a[..., 0] * 10.0, b[..., 1] * 10.0
        a_near = az > bz if larger_is_near else az < bz
        return a, b, a_near

    def pick_nearest(larger_is_near):
        def expected(fx):
            a, b, a_near = nearest(fx, larger_is_near)
            ref = np.where(a_near[..., None], a, b)
            return ref, _all_valid(ref)
        return expected

    def near_over_far(fx):
        a, b, a_near = nearest(fx)
        near = np.where(a_near[..., None], a, b)
        far = np.where(a_near[..., None], b, a)
        ref = near + far * (1.0 - near[..., 3:4])
        return ref, _all_valid(ref)

    zbase = {"depth_layer": "depth.Z", "depth_math": "depth", "smoothing": 0.0, "mix": 1.0, "_depth": zdepth}
    case("zmerge_smaller_z_is_nearer", "ZMerge", "ZMerge", dict(zbase, backwards=False, alpha_channel="none"), pick_nearest(False), "guide",
         f"'It does this by setting each output pixel to the input with the closest Z depth value ... By default, ZMerge considers smaller "
         f"Z depth values ... to be closer to the camera' ({zpage}). Alpha channel off, so each pixel is one input's.",
         nuke_knobs=(("z_channel", "depth.Z"), ("backwards", False), ("alpha_channel", "none")), nuke_inputs=("A", "B"), nb_inputs=("A", "B"))
    case("zmerge_backwards_larger_z_is_nearer", "ZMerge", "ZMerge", dict(zbase, backwards=True, alpha_channel="none"), pick_nearest(True),
         "guide", f"'smaller Z = further away (backwards): When enabled, darker Z values are further away and hidden by lighter Z values' "
         f"({zpage}).",
         nuke_knobs=(("z_channel", "depth.Z"), ("backwards", True), ("alpha_channel", "none")), nuke_inputs=("A", "B"), nb_inputs=("A", "B"))
    case("zmerge_alpha_blends_nearer_over_farther", "ZMerge", "ZMerge", dict(zbase, backwards=False, alpha_channel="rgba.alpha"),
         near_over_far, "convention",
         f"'When enabled, ZMerge uses the inputs' alpha channels to blend between the foreground and background' ({zpage}); the blend is "
         "read as the nearer pixel over the farther one (premultiplied over: near + far * (1 - near alpha)), the guide giving no formula.",
         nuke_knobs=(("z_channel", "depth.Z"), ("backwards", False), ("alpha_channel", "rgba.alpha")), nuke_inputs=("A", "B"),
         nb_inputs=("A", "B"))

    # ---------------------------------------------------------------------------------------------------------- CopyBBox
    case("copybbox_keeps_the_b_stream_pixels", "CopyBBox", "CopyBBox", {}, same_as("B"), "guide",
         f"'Copies the bounding box from the A input onto the B stream' ({GUIDE}merge_nodes/copybbox.html): the pixels are B's. Both "
         "fixtures cover the whole frame, so A's bounding box crops nothing. NodeBased's sockets run the other way round (its A is the "
         "stream, its B gives the box), so the NodeBased side is wired with the stream on A.",
         nuke_inputs=("A", "B"), nb_inputs=("B", "A"))

    # ------------------------------------------------------------------------------------------------------ ContactSheet
    cpage = f"{GUIDE}merge_nodes/contactsheet.html"
    sheet = {"rows": 1, "gap": 0, "labels": "none", "fit": "fit", "height": 12, "roworder": "BottomTop", "colorder": "LeftRight",
             "center": 0, "splitinputs": 0, "startframe": 1, "endframe": 1, "_frame": 1}

    def side_by_side(fx):
        a, b = fx["A"].astype(np.float64), fx["B"].astype(np.float64)
        ref = np.concatenate([a, b], axis=1)
        return ref, _all_valid(ref)

    case("contactsheet_one_cell_is_the_input", "ContactSheet", "ContactSheet", dict(sheet, columns=1, width=16), same_as("A"), "convention",
         f"ContactSheet lays its inputs out in a grid of rows x columns cells on a sheet of width x height ({cpage}); one input on a "
         "one-cell sheet of its own size is that input.",
         nuke_knobs=(("width", 16), ("height", 12), ("rows", 1), ("columns", 1)))
    case("contactsheet_left_right_column_order", "ContactSheet", "ContactSheet", dict(sheet, columns=2, width=32), side_by_side,
         "convention",
         f"With column order LeftRight the first input fills the left cell and the second the right ({cpage}); two 16 x 12 inputs on a "
         "32 x 12 sheet of one row and two columns sit side by side unscaled.",
         nuke_knobs=(("width", 32), ("height", 12), ("rows", 1), ("columns", 2), ("colorder", "LeftRight")), nuke_inputs=("A", "B"),
         nb_inputs=("A", "B"))

    # ------------------------------------------------------------------------------------------------- not scored, with why
    def none(fx):
        ref = fx["A"].astype(np.float64)
        return ref, _all_valid(ref)

    unscored_existing = (
        ("Read", "image_nodes/read.html", "Read reads a file from disk; the audit's fixture is an in-memory picture, so there is no file to "
         "compare (Read's decoding is checked against files by the Read test modules)"),
        ("Backdrop", "other_nodes/backdrop.html", "Backdrop is a box behind nodes on the graph with no input and no output"),
        ("Group", "other_nodes/group.html", "a Group's picture is whatever the nodes inside it make; it has no behaviour of its own to state"),
        ("Input", "other_nodes/input.html", "Input is a group's input socket: it has no input of its own in a case, and its picture is the "
         "group's caller's"),
        ("Precomp", "other_nodes/precomp.html", "Precomp renders another script; a case would need a second document on disk"),
        ("BurnIn", "other_nodes/burnin.html", "BurnIn draws text in a font; the guide states no glyph shapes, so the drawn pixels cannot be "
         "written down without guessing"),
        ("Assert", "other_nodes/assert.html", "the guide states 'The Assert node does not alter the input image in any way', but NodeBased "
         "runs Assert's check in the graph evaluator, outside the per-node kernel the audit drives"),
    )
    for node, page, why in unscored_existing:
        case(f"{node.lower()}_not_a_pixel_behaviour", node, node, None, none, "guide", f"See {GUIDE}{page}.", unscored=why,
             inexpressible=why)

    unpaired = (
        ("UDIM Import", "UDIM_Import", "image_nodes/udimimport.html"),
        ("BlinkScript", "BlinkScript", "other_nodes/blinkscript.html"),
        ("AudioRead", "AudioRead", "other_nodes/audioread.html"),
        ("Link", "Link", "other_nodes/link.html"),
        ("LiveGroup", "LiveGroup", "other_nodes/livegroup.html"),
        ("LiveInput", "LiveInput", "other_nodes/liveinput.html"),
        ("VariableGroup", "VariableGroup", "other_nodes/variablegroup.html"),
    )
    for node, cls, page in unpaired:
        why = f"NodeBased has no {node} node"
        case(f"{cls.lower()}_not_in_nodebased", node, "NoOp", None, none, "guide", f"See {GUIDE}{page}.", unscored=why,
             inexpressible=why, nuke_class=cls)
    root_why = ("Root is the project's settings (frame range, fps, formats, proxy); they live in the document, not in a node with pixels, "
                "and stay as they are this step")
    case("root_project_settings", "Root", "NoOp", None, none, "guide", f"See {GUIDE}other_nodes/root.html.", unscored=root_why,
         inexpressible=root_why, nuke_class="Root")
    return out
