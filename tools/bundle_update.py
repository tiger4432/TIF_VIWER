"""
Updates this git checkout from release bundles (no network, no git pull).

    python tools/bundle_update.py                 apply the newest bundle in releases\\ (if newer than this)
    python tools/bundle_update.py FILE.bundle     apply that bundle
    python tools/bundle_update.py --watch [SEC]   keep looking into releases\\ every SEC seconds (default 30)

Drop the bundles of the GitHub releases into releases\\ next to the program: TIF_Viewer-<ver>.bundle (whole
history, always applies) or TIF_Viewer-<ver>-since-<prev>.bundle (small; needs <prev> here already). The newest
one that applies is fetched and merged (fast-forward only, local changes put aside and put back: git
--autostash); applied and older bundles are moved to releases\\applied\\. dev.bat / run.bat do this at start
(dev.bat watch: also while running, jurigged reloads the changed code). Exit code 3 = updated: the
.bat restarts itself (a .bat changed by the update must not go on from its old position).
"""
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP = os.path.join(ROOT, "releases")
NAME = re.compile(r"TIF_Viewer-(\d+(?:\.\d+)*)(?:-since-(\d+(?:\.\d+)*))?\.bundle$", re.I)


def vkey(v):
    return tuple(int(x) for x in v.split("."))


def installed():
    with open(os.path.join(ROOT, "sat_widgets", "version.py"), encoding="utf-8") as f:
        return re.search(r'APP_VERSION\s*=\s*"([^"]+)"', f.read()).group(1)


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r


def apply(path):
    """Fetches the bundle's main and fast-forwards to it. Returns the new HEAD line."""
    git("bundle", "verify", path)
    git("fetch", path, "main:refs/remotes/bundle/main", "--force")
    before = git("log", "--oneline", "-1", "HEAD").stdout.strip()
    git("merge", "--ff-only", "--autostash", "refs/remotes/bundle/main")
    after = git("log", "--oneline", "-1", "HEAD").stdout.strip()
    return before, after


def candidates():
    """[(version, since, path)] of the bundles in releases\\, newest first (a full bundle before a since-one)."""
    if not os.path.isdir(DROP):
        return []
    out = []
    for f in os.listdir(DROP):
        m = NAME.match(f)
        if m:
            out.append((m.group(1), m.group(2), os.path.join(DROP, f)))
    return sorted(out, key=lambda c: (vkey(c[0]), c[1] is None), reverse=True)


def move_done(cur):
    """Bundles not newer than the installed version go to releases\\applied\\."""
    done = os.path.join(DROP, "applied")
    for ver, _since, path in candidates():
        if vkey(ver) <= vkey(cur):
            os.makedirs(done, exist_ok=True)
            shutil.move(path, os.path.join(done, os.path.basename(path)))


def auto(quiet=False):
    """Applies the newest bundle in releases\\ that is newer and applies. Returns True when updated."""
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        if not quiet:
            print("Not a git checkout: bundles cannot be applied here (use update.bat or unpack the zip).")
        return False
    cur = installed()
    for ver, since, path in candidates():
        if vkey(ver) <= vkey(cur):
            continue
        if since and vkey(since) > vkey(cur):
            print(f"{os.path.basename(path)}: needs {since} first (installed {cur}) - skipped")
            continue
        try:
            before, after = apply(path)
        except RuntimeError as e:
            print(f"{os.path.basename(path)}: not applied - {e}")
            continue
        print(f"Updated {cur} -> {installed()} from {os.path.basename(path)}\n  before: {before}\n  now:    {after}")
        move_done(installed())
        return True
    move_done(cur)
    if not quiet:
        print(f"No newer bundle in {DROP} (installed {cur}).")
    return False


def main(argv):
    os.makedirs(DROP, exist_ok=True)
    if argv and argv[0] == "--watch":
        sec = float(argv[1]) if len(argv) > 1 else 30.0
        print(f"Watching {DROP} every {sec:g} s (Ctrl+C or close the window to stop)")
        while True:
            try:
                auto(quiet=True)
            except Exception as e: # Keep watching
                print(f"bundle update: {e}")
            time.sleep(sec)
    if argv:
        path = os.path.abspath(argv[0])
        if not os.path.exists(path):
            print(f"No such file: {path}")
            return 1
        before, after = apply(path)
        print(f"before: {before}\nnow:    {after}")
        return 0
    return 3 if auto() else 0 # 3: updated (the .bat that called this starts again, reading its new version)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:
        print(f"bundle update FAILED: {e}\n\"git status\" shows local changes or commits in the way.")
        sys.exit(1)
