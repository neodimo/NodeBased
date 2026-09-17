"""Helpers for filing and recording NodeBased GitHub issues."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths

from . import __version__


_SETTINGS_KEY = "agent/file_issue_enabled"
_REPO = "neodimo/NodeBased"


def _settings():
    return QSettings("NodeBased", "NodeBased")


def issue_filing_enabled() -> bool:
    value = _settings().value(_SETTINGS_KEY, True)
    return value not in (False, "false", "0", 0)


def set_issue_filing_enabled(enabled: bool):
    settings = _settings()
    settings.setValue(_SETTINGS_KEY, bool(enabled))
    settings.sync()


def build_footer(agent_name):
    name = agent_name or "unknown"
    return f"NodeBased {__version__} | {platform.system()} {platform.release()} | {name} | filed by agent"


def _default_log_path():
    return Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation)) / "filed_issues.jsonl"


def _issue_number(url):
    match = re.search(r"/issues/(\d+)(?:/|$)", url or "")
    return int(match.group(1)) if match else None


def _github_token():
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    gh = shutil.which("gh")
    if not gh:
        return None
    try:
        result = subprocess.run([gh, "auth", "token"], capture_output=True, text=True,
                                timeout=30, check=False)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _file_issue_rest(title, body, labels, repo):
    token = _github_token()
    if not token:
        raise RuntimeError("No GitHub token available for issue filing")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({"title": title, "body": body, "labels": labels}).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "nodebased-agent",
            "Content-Type": "application/json",
        }, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API error {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"GitHub API request failed: {exc}") from exc
    if not 200 <= status < 300:
        raise RuntimeError(f"GitHub API error {status}: {response_body}")
    payload = json.loads(response_body)
    return payload.get("html_url"), payload.get("number")


def file_issue(title, body, labels=None, agent_name=None, repo=_REPO, log_path=None):
    if not issue_filing_enabled():
        raise RuntimeError("Issue filing is disabled")
    if not isinstance(title, str) or not title:
        raise ValueError("title must be a non-empty string")
    if body is None:
        body = ""
    elif not isinstance(body, str):
        raise ValueError("body must be a string")
    labels = list(labels or [])
    body = body + "\n\n" + build_footer(agent_name)

    url = None
    number = None
    gh = shutil.which("gh")
    if gh:
        command = [gh, "issue", "create", "--repo", repo, "--title", title, "--body", body]
        for label in labels:
            command.extend(["--label", label])
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            if result.returncode == 0:
                url = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None
                number = _issue_number(url)
        except Exception:
            pass
    if not url:
        url, number = _file_issue_rest(title, body, labels, repo)
        number = number if number is not None else _issue_number(url)

    entry = {"timestamp": time.time(), "title": title, "url": url, "number": number,
             "repo": repo, "agent_name": agent_name}
    destination = Path(log_path) if log_path is not None else _default_log_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")
    return {"url": url, "number": number}


def read_issue_log(log_path=None):
    destination = Path(log_path) if log_path is not None else _default_log_path()
    if not destination.exists():
        return []
    with destination.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
