"""
Updates this install to a GitHub release, without git: downloads TIF_Viewer-<version>.zip, checks
its sha256 and unpacks it over the program folder. The site's own files (config.json,
products.json, deploy.json, plugins/) are in the zip only under defaults/, so they are kept;
files that are not in the zip (data, .venv, models/) are not touched.

    update.bat              latest release
    update.bat 1.1.2        that version (also to go back to an older one)
    update.bat --check      only show the installed and the latest version

Close the viewer first. Repository: env UPDATE_REPO, else config.json "update": {"github":
"owner/repo"}, else DEFAULT_REPO. Proxy: the system settings (or HTTPS_PROXY), as urllib uses them.
A git checkout is not overwritten (that is updated with git): --force does it anyway.
"""
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPO = "tiger4432/TIF_VIWER"


def installed_version():
    with open(os.path.join(ROOT, "sat_widgets", "version.py"), encoding="utf-8") as f:
        return re.search(r'APP_VERSION\s*=\s*"([^"]+)"', f.read()).group(1)


def repo_name():
    if os.environ.get("UPDATE_REPO"):
        return os.environ["UPDATE_REPO"]
    try:
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
            return (json.load(f).get("update") or {}).get("github") or DEFAULT_REPO
    except (OSError, ValueError):
        return DEFAULT_REPO


def retried(fn, tries=4):
    """Network errors and 5xx (e.g. 504 Gateway Timeout from a proxy) are retried after 2, 4, 8 s."""
    for i in range(tries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == tries - 1:
                raise
            print(f"  {e} - retrying")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if i == tries - 1:
                raise
            print(f"  {e} - retrying")
        time.sleep(2 ** (i + 1))


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "TIF_Viewer-update",
                                               "Accept": "application/vnd.github+json"})
    def once():
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    return retried(once)


def release(repo, version=None):
    api = f"https://api.github.com/repos/{repo}/releases/"
    try:
        return json.loads(get(api + (f"tags/v{version.lstrip('vV')}" if version else "latest")))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"no release {'v' + version.lstrip('vV') if version else ''} in {repo} "
                               f"(https://github.com/{repo}/releases)") from None
        raise


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "TIF_Viewer-update"})
    def once():
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    retried(once)


def unpack(zip_path, top):
    """Writes the files under top/ of the zip over ROOT. Returns the number of files."""
    root = os.path.realpath(ROOT)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or not name.startswith(top + "/"):
                continue
            dest = os.path.realpath(os.path.join(root, name[len(top) + 1:]))
            if not dest.startswith(root + os.sep): # No writing outside the program folder
                raise RuntimeError(f"unsafe path in the zip: {info.filename}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(info) as src, open(dest + ".new", "wb") as out:
                out.write(src.read())
            os.replace(dest + ".new", dest)
            n += 1
    return n


def main(argv):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace") # Notes may hold characters the console lacks
    args = [a for a in argv if not a.startswith("--")]
    check, force = "--check" in argv, "--force" in argv
    repo, current = repo_name(), installed_version()
    rel = release(repo, args[0] if args else None)
    version = rel["tag_name"].lstrip("vV")
    print(f"Installed: {current}    Release: {version}    ({rel.get('html_url', repo)})")
    if check:
        return 0
    if version == current and not args:
        print("Already up to date.")
        return 0
    if os.path.isdir(os.path.join(ROOT, ".git")) and not force:
        print("This folder is a git checkout: update it with git (or run with --force).")
        return 1
    assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
    top = f"TIF_Viewer-{version}"
    if top + ".zip" not in assets:
        print(f"The release has no {top}.zip.")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, top + ".zip")
        print(f"Downloading {top}.zip ...")
        download(assets[top + ".zip"], zip_path)
        if top + ".zip.sha256" in assets:
            want = get(assets[top + ".zip.sha256"]).decode("ascii", "replace").split()[0].lower()
            with open(zip_path, "rb") as f:
                have = hashlib.sha256(f.read()).hexdigest()
            if have != want:
                print(f"sha256 mismatch (download broken?): {have} != {want}. Nothing changed.")
                return 1
            print("sha256 OK")
        n = unpack(zip_path, top)
    print(f"Updated {current} -> {version} ({n} files). config.json / products.json / deploy.json / plugins kept.")
    notes = (rel.get("body") or "").split("\n---\n")[0].strip()
    if notes:
        print("\n" + notes)
    print("\nStart the viewer with run.bat (it installs changed requirements).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e: # Network / proxy / GitHub errors: one line, no traceback
        print(f"Update failed: {type(e).__name__}: {e}")
        sys.exit(1)
