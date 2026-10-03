import time

from nodebased.artifacts import ArtifactStore


def succeeds(options, progress, cancelled):
    progress((0.5, "fixture working"))
    return ArtifactStore().put(b"worker fixture result", "generated_sequence",
                               {"producer": "fixture", "version": 1, "inputs": []})


def raises(options, progress, cancelled):
    print("fixture provider log line", flush=True)
    raise ValueError("fixture provider exploded")


def allocates(options, progress, cancelled):
    return bytearray(int(options["allocation_bytes"]))


def hangs(options, progress, cancelled):
    while True:
        time.sleep(0.1)


def cancellable(options, progress, cancelled):
    for index in range(1000):
        if cancelled.is_set():
            raise RuntimeError("fixture cancelled")
        progress((index / 1000, "fixture long job"))
        time.sleep(0.01)
    return ArtifactStore().put(b"should not complete", "generated_sequence",
                               {"producer": "fixture", "version": 1, "inputs": []})
