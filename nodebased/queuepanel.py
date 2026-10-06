"""Qt view and controls for the persistent conditioning job queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QTableWidget, QTableWidgetItem, QFormLayout)

from .artifacts import ArtifactStore
from .jobs import Queue
from .loops import LoopRun, LoopError
from .generative import ProviderDescription


def create_default_queue():
    root = Path(os.environ.get("NODEBASED_CACHE", Path.home() / ".cache" / "nodebased"))
    workers = json.loads(os.environ.get("NODEBASED_REMOTE_WORKERS", "[]"))
    secret = os.environ.get("NODEBASED_WORKER_SECRET")
    return Queue(root / "jobs.sqlite", ArtifactStore(root / "artifacts"),
                 max_workers=max(1, len(workers) + 1), secret=secret, remote_workers=workers)


class QueuePanel(QWidget):
    COLUMNS = ("Chain / link", "State", "Progress", "Worker", "Elapsed", "Artifact ID")

    def __init__(self, queue, parent=None):
        super().__init__(parent)
        self.queue = queue
        self.runner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conditioning-queue")
        self.loop_runs = LoopRun(queue.path.parent / "conditioning-loops.sqlite", queue)
        self.selected_loop = ""
        self.use_result_callback = None
        layout = QVBoxLayout(self)
        self.status = QLabel("Queue is ready")
        layout.addWidget(self.status)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(self.COLUMNS))
        self.tree.setHeaderLabels(self.COLUMNS)
        self.tree.setUniformRowHeights(True)
        layout.addWidget(self.tree)
        form = QFormLayout()
        self.scene_id = QLineEdit(); self.scene_id.setPlaceholderText("SceneState artifact ID")
        self.bundle_id = QLineEdit(); self.bundle_id.setPlaceholderText("ControlBundle artifact ID")
        self.provider = QComboBox(); self.provider.addItems(["reproject", "null"])
        self.max_attempts = QSpinBox(); self.max_attempts.setRange(1, 100); self.max_attempts.setValue(3)
        self.spend_cap = QDoubleSpinBox(); self.spend_cap.setRange(0.01, 1_000_000); self.spend_cap.setValue(3.0)
        self.spend_per_attempt = QDoubleSpinBox(); self.spend_per_attempt.setRange(0.0, 1_000_000); self.spend_per_attempt.setValue(1.0)
        self.spend_unit = QLineEdit("credits (estimate)")
        self.feedback_reference = QCheckBox("Use previous result as reference frames")
        self.feedback_text = QLineEdit(); self.feedback_text.setPlaceholderText("Optional text feedback")
        for label, widget in (("SceneState",self.scene_id),("ControlBundle",self.bundle_id),
                              ("Provider",self.provider),("Maximum attempts",self.max_attempts),
                              ("Spend cap",self.spend_cap),("Estimate per attempt",self.spend_per_attempt),
                              ("Estimate units",self.spend_unit),("Feedback",self.feedback_reference),
                              ("Feedback text",self.feedback_text)):
            form.addRow(label, widget)
        layout.addLayout(form)
        self.loop_capability = QLabel("Provider control support appears here before launch.")
        self.loop_capability.setWordWrap(True); self.loop_capability.setObjectName("loopCapabilities")
        layout.addWidget(self.loop_capability)
        self.loop_start = QPushButton("Start bounded loop")
        self.loop_cancel = QPushButton("Cancel selected loop")
        self.loop_result = QPushButton("Use selected result in ConditionedRead")
        loop_buttons = QHBoxLayout()
        for button in (self.loop_start, self.loop_cancel, self.loop_result): loop_buttons.addWidget(button)
        layout.addLayout(loop_buttons)
        self.attempts = QTableWidget(0, 8)
        self.attempts.setHorizontalHeaderLabels(("Attempt", "State", "Controls", "Generated sequence",
            "Verification", "Cumulative estimate", "Stop reason", "Queue links / provenance"))
        self.attempts.setObjectName("loopAttemptHistory")
        layout.addWidget(self.attempts)
        self.loop_start.clicked.connect(self.start_loop)
        self.loop_cancel.clicked.connect(self.cancel_loop)
        self.loop_result.clicked.connect(self.use_selected_result)
        self.provider.currentTextChanged.connect(self.refresh_capabilities)
        self.tree.itemSelectionChanged.connect(self.refresh_loop_history)
        self.attempts.itemDoubleClicked.connect(self.show_attempt_details)
        self.refresh_capabilities()
        controls = QHBoxLayout()
        self.import_button = QPushButton("Import chain…")
        self.run_button = QPushButton("Run queue")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.cancel_button = QPushButton("Cancel chain")
        self.up_button = QPushButton("Priority ↑")
        self.down_button = QPushButton("Priority ↓")
        for button in (self.import_button, self.run_button, self.pause_button, self.resume_button,
                       self.cancel_button, self.up_button, self.down_button): controls.addWidget(button)
        layout.addLayout(controls)
        self.import_button.clicked.connect(self.import_chain)
        self.run_button.clicked.connect(self.run_queue)
        self.pause_button.clicked.connect(self.queue.pause)
        self.resume_button.clicked.connect(self.queue.resume)
        self.cancel_button.clicked.connect(self.cancel_selected)
        self.up_button.clicked.connect(lambda: self.bump_priority(1))
        self.down_button.clicked.connect(lambda: self.bump_priority(-1))
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def refresh(self):
        snapshot = self.queue.snapshot()
        self.tree.clear()
        by_chain = {}
        for chain in snapshot["chains"]:
            item = QTreeWidgetItem([chain["name"], chain["state"], "", "", "", ""])
            item.setData(0, Qt.ItemDataRole.UserRole, chain["id"])
            item.setToolTip(0, f"Priority {chain['priority']}")
            self.tree.addTopLevelItem(item)
            by_chain[chain["id"]] = item
        for link in snapshot["links"]:
            parent = by_chain.get(link["chain_id"])
            if parent is None: continue
            elapsed = f"{link['elapsed']:.1f}s" if link["started"] else "—"
            child = QTreeWidgetItem([link["name"], link["state"], f"{100*link['progress']:.0f}%",
                link["worker"] or "—", elapsed, link["artifact_id"] or "—"])
            child.setData(0, Qt.ItemDataRole.UserRole, link["chain_id"])
            parent.addChild(child)
        self.tree.expandAll()
        self.status.setText("Queue paused" if snapshot["paused"] else "Queue running" if any(
            row["state"] == "running" for row in snapshot["links"]) else "Queue is ready")
        self.refresh_loop_history()

    def refresh_capabilities(self):
        try:
            provider = ProviderDescription.load(self.provider.currentText())
            supported = [name for name in ("text", "reference_frames") if provider.accepted(name)]
            unsupported = [name for name in ("text", "reference_frames") if not provider.accepted(name)]
            self.loop_capability.setText(f"{provider.name} {provider.version} · supports: " +
                (", ".join(supported) if supported else "none") + " · unsupported: " +
                (", ".join(unsupported) if unsupported else "none"))
        except (OSError, ValueError, KeyError) as error:
            self.loop_capability.setText(f"Provider unavailable: {error}")

    def start_loop(self):
        try:
            options = {"estimated_spend": self.spend_per_attempt.value(),
                       "spend_unit": self.spend_unit.text().strip()}
            if self.feedback_text.text().strip(): options["text"] = self.feedback_text.text().strip()
            controls = (["reference_frames"] if self.feedback_reference.isChecked() else [])
            if options.get("text"): controls.append("text")
            loop_id = self.loop_runs.create(scene_state_id=self.scene_id.text().strip(),
                control_bundle_id=self.bundle_id.text().strip(), provider_id=self.provider.currentText(),
                provider_options=options, max_attempts=self.max_attempts.value(),
                max_estimated_spend=self.spend_cap.value(),
                estimated_spend_per_attempt=self.spend_per_attempt.value(), spend_unit=options["spend_unit"],
                feedback_controls=controls)
            self.selected_loop = loop_id
            self.runner.submit(self.loop_runs.run, loop_id)
            self.status.setText(f"Loop running · {loop_id[:8]}")
        except (LoopError, OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "Cannot start conditioning loop", str(error))
        self.refresh_loop_history()

    def cancel_loop(self):
        if not self.selected_loop: return
        self.loop_runs.cancel(self.selected_loop)
        self.status.setText("Loop cancellation requested")
        self.refresh_loop_history()

    def refresh_loop_history(self):
        snap = self.queue.snapshot()
        chain_id = self.selected_chain()
        chain = next((c for c in snap["chains"] if c["id"] == chain_id), None)
        if chain and chain["name"].startswith("Generate VerifyConditioning "):
            self.selected_loop = chain["name"].split("Generate VerifyConditioning ", 1)[1].split(" attempt ", 1)[0]
        if not self.selected_loop:
            history = self.loop_runs.list_snapshots()
            if history: self.selected_loop = history[0]["id"]
        self.attempts.setRowCount(0)
        if not self.selected_loop: return
        try: data = self.loop_runs.snapshot(self.selected_loop)
        except LoopError: return
        cumulative = 0.0
        for attempt in data["attempts"]:
            cumulative += float(attempt["spend"])
            row = self.attempts.rowCount(); self.attempts.insertRow(row)
            generated = attempt["outputs"].get("generated_sequence", "")
            controls = ", ".join(attempt["inputs"])
            fields = (str(attempt["number"]), attempt["state"], controls, generated,
                      attempt["verdict"] or "—", f"{attempt['spend']} {attempt['spend_unit']}",
                      attempt["error"] or data["error"] or data["state"], attempt["queue_chain"])
            fields = fields[:5] + (f"{cumulative:g} {attempt['spend_unit']}",) + fields[6:]
            for col, value in enumerate(fields):
                item = QTableWidgetItem(value); item.setData(Qt.ItemDataRole.UserRole, attempt["number"])
                self.attempts.setItem(row, col, item)

    def _selected_attempt(self):
        item = self.attempts.item(self.attempts.currentRow(), 0) if self.attempts.currentRow() >= 0 else None
        if not item or not self.selected_loop: return None
        data = self.loop_runs.snapshot(self.selected_loop)
        return next((a for a in data["attempts"] if a["number"] == int(item.text())), None)

    def show_attempt_details(self, *_):
        attempt = self._selected_attempt()
        if not attempt: return
        links = [row for row in self.queue.snapshot()["links"] if row["chain_id"] == attempt["queue_chain"]]
        provenance = []
        for aid in attempt["outputs"].values():
            if aid:
                try: provenance.append(f"{aid}: {self.queue.store.provenance(aid)}")
                except (ValueError, KeyError): provenance.append(f"{aid}: provenance unavailable")
        worker_log = ""
        match = __import__("re").search(r"worker log artifact ([0-9a-f]{64})", attempt["error"])
        if match:
            try: worker_log = self.queue.store.get(match.group(1)).decode("utf-8", "replace")[-4000:]
            except (OSError, ValueError): worker_log = "Worker log artifact is unavailable."
        detail = "\n".join([f"Attempt {attempt['number']} · {attempt['state']}",
            f"Inputs: {', '.join(attempt['inputs'])}", f"Provider: {attempt['provider']} {attempt['version']}",
            f"Options: {json.dumps(attempt['options'], sort_keys=True)}", f"Verdict: {attempt['verdict'] or '—'}",
            f"Estimated spend: {attempt['spend']} {attempt['spend_unit']}",
            f"Failure: {attempt['error'] or '—'}", f"Worker log: {worker_log or '—'}", "Queue links: " + "; ".join(
                f"{x['name']}={x['state']} ({x['id']}) {x['error']}" for x in links),
            "Provenance: " + "\n".join(provenance)])
        QMessageBox.information(self, "Attempt details", detail)

    def use_selected_result(self):
        attempt = self._selected_attempt()
        if not attempt or attempt["state"] not in ("pass", "failed_verification"):
            self.status.setText("Select a completed attempt with a generated result")
            return
        artifact_id = attempt["outputs"].get("generated_sequence")
        data = self.loop_runs.snapshot(self.selected_loop)
        if artifact_id and self.use_result_callback:
            try:
                self.use_result_callback(data["config"], artifact_id)
                self.status.setText(f"ConditionedRead uses selected result · {artifact_id[:12]}")
            except (OSError, ValueError, KeyError) as error:
                self.status.setText(f"Cannot load selected result: {error}")

    def selected_chain(self):
        item = self.tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def cancel_selected(self):
        chain_id = self.selected_chain()
        if chain_id: self.queue.cancel(chain_id)
        self.refresh()

    def bump_priority(self, delta):
        chain_id = self.selected_chain()
        if not chain_id: return
        chain = next((row for row in self.queue.snapshot()["chains"] if row["id"] == chain_id), None)
        if chain: self.queue.reorder(chain_id, chain["priority"] + delta)
        self.refresh()

    def import_chain(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import conditioning chain", "", "JSON files (*.json)")
        if not path: return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            chain_id = self.queue.add_chain(data["name"], data["links"], priority=data.get("priority", 0))
        except Exception as exc:
            QMessageBox.warning(self, "Cannot import chain", str(exc))
            return
        self.refresh()
        self.status.setText(f"Queued {data['name']} · {chain_id[:8]}")

    def run_queue(self):
        self.runner.submit(self.queue.run_until_idle)
        self.status.setText("Queue worker started")

    def closeEvent(self, event):
        self.timer.stop()
        self.runner.shutdown(wait=True, cancel_futures=True)
        super().closeEvent(event)
