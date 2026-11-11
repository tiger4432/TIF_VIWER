# PyInstaller build definition: pyinstaller --noconfirm --clean TIF_Viewer.spec
# Output: dist/TIF_Viewer/TIF_Viewer.exe (+ _internal/). config.json and products.json are
# copied next to the exe by build.bat so they can be edited per deployment.
import glob
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

# conda Python (and a venv made from it) keeps the DLLs of its standard extension modules in
# <conda env>\Library\bin: libffi for _ctypes, OpenSSL for _ssl (requests), liblzma, bz2, sqlite.
# That folder is on PATH only in an activated conda prompt and a venv does not look like conda to
# PyInstaller, so they were left out and the exe stopped with "DLL load failed while importing
# _ctypes". The folder goes on PATH for the dependency search and those DLLs are taken explicitly.
# (python.org Python has them in DLLs\, which PyInstaller finds itself.)
extra_binaries = []
conda_bin = os.path.join(sys.base_prefix, "Library", "bin")
if os.path.isdir(conda_bin):
    os.environ["PATH"] = conda_bin + os.pathsep + os.environ.get("PATH", "")
    for pattern in ("ffi*.dll", "libffi*.dll", "libssl*.dll", "libcrypto*.dll", "liblzma*.dll", "libbz2*.dll",
                    "sqlite3*.dll"):
        extra_binaries += [(f, ".") for f in glob.glob(os.path.join(conda_bin, pattern))]
    print("[spec] conda DLLs taken from", conda_bin, ":", ", ".join(os.path.basename(f) for f, _ in extra_binaries) or "none")

hiddenimports = collect_submodules("imagecodecs") # tifffile imports codecs lazily
# Standard library for plugins/results_api.py (loaded from file at run time, so PyInstaller does
# not see its imports). Another package used by the plugin must be added here and to requirements.
hiddenimports += ["urllib.request", "urllib.parse", "urllib.error", "http.client", "ssl", "json", "base64",
                  "hashlib", "hmac", "uuid", "csv", "socket", "getpass", "email", "zipfile", "gzip"]
hiddenimports.append("threadpoolctl") # Imported lazily by sat_widgets/void_layer_change.py
hiddenimports += collect_submodules("requests") + ["urllib3", "idna", "charset_normalizer", "certifi"] # plugins/results_api.py
try: # Optional PatchCore scorer (build with WITH_PATCHCORE=1, see docs/PATCHCORE.md)
    import onnxruntime # noqa: F401
    hiddenimports.append("onnxruntime")
except ImportError:
    pass

a = Analysis(
    ["tif_view_GL.py"],
    pathex=[],
    binaries=extra_binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Not used by the app. setuptools/pkg_resources would add a runtime hook that fails in the
    # frozen exe (setuptools 7x: "No module named 'backports'").
    excludes=["tkinter", "setuptools", "pkg_resources", "distutils", "_distutils_hack"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TIF_Viewer",
    debug=False,
    strip=False,
    upx=False,
    # Windowed (the released exe; errors in dialogs). set BUILD_CONSOLE=1 before build.bat: a console
    # window that keeps the full error text (DLL / import errors at start)
    console=os.environ.get("BUILD_CONSOLE") == "1",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="TIF_Viewer",
)
