"""
Startup check for a newer build.

config.json:
    "update": {"manifest": "\\\\server\\share\\TIF_Viewer\\latest.json"}   (file path or http(s) URL)

Manifest (published with each release):
    {"version": "1.1.0", "download": "\\\\server\\share\\TIF_Viewer\\1.1.0", "notes": "..."}

"download" may be a folder, a file or a URL; it is opened when the user chooses to update.
The check runs in a background thread and never blocks or fails the start.
"""
import json
import os
import threading
import urllib.request

from PySide6.QtCore import QObject, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox


def parse_version(text):
    """'1.2.10' -> (1, 2, 10). Non-numeric parts count as 0."""
    parts = []
    for p in str(text).strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_newer(latest, current):
    a, b = parse_version(latest), parse_version(current)
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


def read_manifest(location, timeout=5.0):
    if location.lower().startswith(("http://", "https://")):
        with urllib.request.urlopen(location, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    with open(location, "r", encoding="utf-8") as f:
        return json.load(f)


class UpdateChecker(QObject):
    update_available = Signal(dict)

    def __init__(self, current_version, manifest_location, parent=None):
        super().__init__(parent)
        self.current_version = current_version
        self.location = manifest_location

    def start(self):
        if not self.location:
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            manifest = read_manifest(self.location)
            if is_newer(manifest.get("version", "0"), self.current_version):
                self.update_available.emit(manifest) # Delivered to the GUI thread (queued)
        except Exception as e:
            print(f"Update check skipped: {e}")


def show_update_dialog(parent, current_version, manifest):
    latest = manifest.get("version", "?")
    notes = manifest.get("notes", "")
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle("Update Available")
    box.setText(f"A newer version is available.\n\nCurrent: {current_version}\nLatest: {latest}")
    if notes:
        box.setInformativeText(notes)
    btn_update = box.addButton("Update", QMessageBox.AcceptRole)
    box.addButton("Later", QMessageBox.RejectRole)
    box.exec()
    if box.clickedButton() is btn_update:
        target = manifest.get("download", "")
        if target.lower().startswith(("http://", "https://")):
            QDesktopServices.openUrl(QUrl(target))
        elif target:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(target)))
