# PyInstaller build definition: pyinstaller --noconfirm --clean TIF_Viewer.spec
# Output: dist/TIF_Viewer/TIF_Viewer.exe (+ _internal/). config.json and products.json are
# copied next to the exe by build.bat so they can be edited per deployment.
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("imagecodecs") # tifffile imports codecs lazily
# Standard library for plugins/results_api.py (loaded from file at run time, so PyInstaller does
# not see its imports). Another package used by the plugin must be added here and to requirements.
hiddenimports += ["urllib.request", "urllib.parse", "urllib.error", "http.client", "ssl", "json", "base64",
                  "hashlib", "hmac", "uuid", "csv", "socket", "getpass", "email", "zipfile", "gzip"]
hiddenimports.append("threadpoolctl") # Imported lazily by sat_widgets/void_layer_change.py
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
    console=False, # Windowed (same as the released exe); errors are shown in dialogs
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="TIF_Viewer",
)
