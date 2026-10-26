import os
import sys


def app_dir():
    """
    Folder holding the editable deployment files (config.json, products.json):
    next to TIF_Viewer.exe when built with PyInstaller, else the source root.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def app_file(name):
    return os.path.join(app_dir(), name)
