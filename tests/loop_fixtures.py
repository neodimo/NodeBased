"""Deterministic CPU-only operations for bounded loop tests."""
import json
import time


def generate(options, inputs, progress, cancelled):
    deadline=time.monotonic()+float(options.get("sleep",0))
    while time.monotonic()<deadline:
        if cancelled.is_set(): raise RuntimeError("cancelled by fixture")
        time.sleep(.005)
    if options.get("fail_attempt")==options["attempt"]:
        log = f"; worker log artifact {options['worker_log_id']}" if options.get("worker_log_id") else ""
        raise RuntimeError("fixture provider failure" + log)
    return json.dumps({"inputs":inputs,"attempt":options["attempt"],"provider":options["provider_id"],
                       "version":options["provider_version"],"options":options},sort_keys=True).encode()


def verify(options, inputs, progress, cancelled):
    if cancelled.is_set(): raise RuntimeError("cancelled by fixture")
    attempt=options["attempt"]
    verdict="PASS" if attempt>int(options.get("fail_verifications",0)) else "FAIL"
    return json.dumps({"score_card":{"verdict":verdict},"attempt":attempt},sort_keys=True).encode()
