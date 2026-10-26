"""
Where the program finds its declaration files (config.json, products.json).

- Local mode (default): next to TIF_Viewer.exe (or the source root).
- Shared mode: deploy.json next to the exe names a central shared folder
      {"shared_dir": "\\\\server\\share\\TIF_Viewer"}
  (or the environment variable TIF_VIEWER_SHARED_DIR). At start the declarations are copied
  from there into a per-user cache and loaded from it; when the share is unreachable the last
  cached copy is used, and only without a cache the local files.
"""
import json
import os
import shutil
import sys
import threading

DECLARATIONS = ("config.json", "products.json")
SHARE_TIMEOUT = 5.0 # Seconds to wait for the share before falling back to the cache


def app_dir():
    """Folder of TIF_Viewer.exe when built with PyInstaller, else the source root."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def app_file(name):
    return os.path.join(app_dir(), name)


def deploy_settings():
    try:
        with open(app_file("deploy.json"), "r", encoding="utf-8") as f:
            settings = json.load(f)
    except FileNotFoundError:
        settings = {}
    except Exception as e:
        print(f"deploy.json ignored: {e}")
        settings = {}
    env = os.environ.get("TIF_VIEWER_SHARED_DIR")
    if env:
        settings["shared_dir"] = env
    return settings


def shared_dir():
    return deploy_settings().get("shared_dir") or None


def cache_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "TIF_Viewer", "shared_cache")


def sync_declarations():
    """
    Copies the declarations from the shared folder into the cache (shared mode only).
    Returns a short status text: "local", "shared: <dir>", or "cached copy (...)".
    """
    share = shared_dir()
    if not share:
        return "local"

    result = {}

    def copy():
        try:
            if not os.path.isdir(share):
                raise OSError("folder not found")
            os.makedirs(cache_dir(), exist_ok=True)
            copied = []
            for name in DECLARATIONS:
                src = os.path.join(share, name)
                if os.path.isfile(src):
                    tmp = os.path.join(cache_dir(), name + ".tmp")
                    shutil.copyfile(src, tmp)
                    os.replace(tmp, os.path.join(cache_dir(), name)) # Never leave a half-written cache
                    copied.append(name)
            result["copied"] = copied
        except Exception as e:
            result["error"] = str(e)

    # A dead network share can block for a long time; never hold up the start for it
    t = threading.Thread(target=copy, daemon=True)
    t.start()
    t.join(SHARE_TIMEOUT)

    if "copied" in result:
        missing = [n for n in DECLARATIONS if n not in result["copied"]]
        if missing:
            print(f"Not found on the share (using cache/local): {', '.join(missing)}")
        return f"shared: {share}"
    reason = result.get("error", "timeout")
    print(f"Shared folder unreachable ({reason}); using the last cached declarations")
    return f"cached copy (share unreachable: {share})"


def declaration_path(name):
    """Path to load a declaration file from (shared cache in shared mode, else local)."""
    if shared_dir():
        cached = os.path.join(cache_dir(), name)
        if os.path.isfile(cached):
            return cached
    return app_file(name)


def default_update_manifest():
    """latest.json in the shared folder, when one is configured."""
    share = shared_dir()
    return os.path.join(share, "latest.json") if share else None
