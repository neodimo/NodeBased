"""The defaults four nodes started with before they followed the guide (2D behaviour lane 2, step P).

A new AddTimeCode starts at 01:00:00:00, a new TimeEcho looks at one frame and takes the maximum, a new TimeBlur has a half shutter that opens at the
frame, and a new ContactSheet is 3072 x 2048 with three rows and no gap, as the Nuke 17.0 Reference Guide lists. A document saved before keeps the older
value for every param it does not store (`core.upgrade_document`, schema 32), so its pixels and metadata do not change.
"""
OLD = {
    "AddTimeCode": {"timecode": "00:00:00:00"},
    "TimeEcho": {"frames": 3, "method": "average"},
    "TimeBlur": {"shutter": 1.0, "shutter_offset": "centred"},
    "ContactSheet": {"rows": 4, "gap": 4, "width": 1920, "height": 1080},
}
