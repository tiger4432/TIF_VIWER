"""
Release bundle: a zip to unpack and run on a PC without git (no git pull at the site).

    python tools/make_release.py [OUTDIR]          -> OUTDIR/TIF_Viewer-<version>.zip (+ .sha256)

Contents: the committed files (the working tree must be clean), top folder TIF_Viewer-<version>/.
The files a site edits are NOT at their place in the zip, so unpacking a new release over an
installed one keeps them: config.json, products.json, deploy.json and plugins/ go to defaults/,
and run.bat / build.bat copy them into place only when missing. Left out: dist/ (old builds) and
models/ (optional PatchCore models, 35 MB: copy them separately when needed).
Version: sat_widgets/version.py APP_VERSION; notes: docs/RELEASE_NOTES.md.
"""
import hashlib
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_FILES = ("config.json", "products.json", "deploy.json") # Edited per site: to defaults/
SITE_DIRS = ("plugins/",)
LEFT_OUT = ("dist/", "models/")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "release"))
    if git("status", "--porcelain", "--untracked-files=no").strip():
        sys.exit("The working tree has uncommitted changes: commit them first (the release is what is committed).")
    version = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', open(os.path.join(ROOT, "sat_widgets", "version.py")).read()).group(1)
    commit = git("rev-parse", "--short", "HEAD").strip()
    top = f"TIF_Viewer-{version}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, top + ".zip")
    files = [f for f in git("ls-files").splitlines() if not f.startswith(LEFT_OUT)]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            src = os.path.join(ROOT, f)
            if not os.path.isfile(src):
                continue
            site = f in SITE_FILES or f.startswith(SITE_DIRS)
            arc = f"{top}/defaults/{f}" if site else f"{top}/{f}"
            if f.lower().endswith(".bat"): # cmd.exe needs CRLF (.gitattributes), whatever this checkout has
                data = open(src, "rb").read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                z.writestr(zipfile.ZipInfo.from_file(src, arc), data, zipfile.ZIP_DEFLATED)
            else:
                z.write(src, arc)
        z.writestr(f"{top}/VERSION.txt", f"{version}\ncommit {commit}\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(path)}\n")
    print(f"{path}  ({os.path.getsize(path) / 1e6:.1f} MB, {len(files)} files, commit {commit})\nsha256 {digest}")


if __name__ == "__main__":
    main()
