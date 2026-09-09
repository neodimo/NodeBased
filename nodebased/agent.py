"""JSON-lines command CLI and optional OS-user-local live desktop bridge."""
from __future__ import annotations

import argparse
import json
import sys

from PySide6.QtCore import QObject, QCoreApplication
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from .core import Dispatcher, load_document

MAX_REQUEST = 1024 * 1024


def response(dispatch, request):
    try:
        result = dispatch(request)
        return {"id": request.get("request_id"), "ok": True, "result": result}
    except Exception as error:
        return {"id": request.get("request_id") if isinstance(request, dict) else None, "ok": False, "error": str(error)}


class LocalBridge(QObject):
    def __init__(self, name, dispatch, parent=None):
        super().__init__(parent)
        self.dispatch = dispatch
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self.buffers = {}
        self.server.newConnection.connect(self.accept)
        # Never remove an existing endpoint: another application may own it.
        if not self.server.listen(name):
            raise ValueError(f"Cannot open local agent endpoint: {self.server.errorString()}")

    def accept(self):
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            socket.setReadBufferSize(MAX_REQUEST + 1)
            self.buffers[socket] = bytearray()
            socket.readyRead.connect(lambda s=socket: self.read(s))
            socket.disconnected.connect(lambda s=socket: self.drop(s))

    def drop(self, socket):
        self.buffers.pop(socket, None)
        socket.deleteLater()

    def read(self, socket):
        buffer = self.buffers[socket]
        buffer.extend(bytes(socket.readAll()))
        if len(buffer) > MAX_REQUEST:
            socket.abort()
            return
        while b"\n" in buffer:
            line, _, remainder = buffer.partition(b"\n")
            buffer[:] = remainder
            try:
                request = json.loads(line)
                result = response(self.dispatch, request)
            except (ValueError, UnicodeError) as error:
                result = {"ok": False, "error": str(error)}
            socket.write(json.dumps(result, allow_nan=False).encode() + b"\n")

    def close(self):
        for socket in list(self.buffers):
            socket.abort()
        self.server.close()


def main():
    parser = argparse.ArgumentParser(description="NodeBased JSON-lines agent interface")
    parser.add_argument("--connect", help="connect to a running GUI's --agent local endpoint")
    parser.add_argument("--project", help="headless initial project")
    args = parser.parse_args()
    app = QCoreApplication.instance() or QCoreApplication([])
    socket = None
    if args.connect:
        socket = QLocalSocket()
        socket.connectToServer(args.connect)
        if not socket.waitForConnected(5000):
            print(json.dumps({"ok": False, "error": socket.errorString()}))
            return 1
    dispatcher = Dispatcher(load_document(args.project) if args.project else None)
    evaluator = None
    for line in sys.stdin.buffer:
        if len(line) > MAX_REQUEST:
            print(json.dumps({"ok": False, "error": "Request exceeds 1 MiB"}), flush=True)
            continue
        if socket:
            socket.write(line.rstrip(b"\n") + b"\n")
            socket.flush()
            data = bytearray()
            while b"\n" not in data:
                if not socket.bytesAvailable() and not socket.waitForReadyRead(30000):
                    print(json.dumps({"ok": False, "error": "Agent response timed out or disconnected"}), flush=True)
                    return 1
                data.extend(bytes(socket.readAll()))
            print(data.decode().strip(), flush=True)
            continue
        try:
            request = json.loads(line)
            def dispatch(cmd):
                nonlocal evaluator
                if cmd.get("op") == "render":
                    from .imaging import Evaluator, write_png
                    evaluator = evaluator or Evaluator()
                    frame = evaluator.evaluate(dispatcher.document, cmd.get("id"))
                    write_png(cmd["path"], frame)
                    return {"path": cmd["path"], "width": frame.shape[1], "height": frame.shape[0]}
                return dispatcher.execute(cmd)
            result = response(dispatch, request)
        except (ValueError, UnicodeError) as error:
            result = {"ok": False, "error": str(error)}
        print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
