"""Qt view and controls for the persistent conditioning job queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .artifacts import ArtifactStore
from .jobs import Queue


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
        layout = QVBoxLayout(self)
        self.status = QLabel("Queue is ready")
        layout.addWidget(self.status)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(self.COLUMNS))
        self.tree.setHeaderLabels(self.COLUMNS)
        self.tree.setUniformRowHeights(True)
        layout.addWidget(self.tree)
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
        self.runner.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)
