# PyInstaller build definition: pyinstaller --noconfirm --clean TIF_Viewer.spec
# Output: dist/TIF_Viewer/TIF_Viewer.exe (+ _internal/). config.json and products.json are
# copied next to the exe by build.bat so they can be edited per deployment.
import os

from PyInstaller.utils.hooks import collect_submodules

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
    binaries=[],
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
