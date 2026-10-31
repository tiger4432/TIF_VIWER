"""
Automatic void marking of measured wafers without a window (sat_widgets/pipeline.py; the viewer's
Auto Run panel does the same in the window), see docs/AUTO_RUN.md.

    python tools/auto_run.py --once FILE                one file (TIF, or a project folder / patches.json)
    python tools/auto_run.py --watch [GLOB|FOLDER ...]  new files until Ctrl+C (default: config "auto_run.watch")

Settings: config.json "auto_run" (read only), overridden by the options below.
A file is taken when its size and mtime did not change for --stable seconds; finished files are
recorded in a state file (default <output_root>/auto_run_state.json) and not processed again
unless they change (size / mtime). One failing file does not stop the watcher.
"""
import argparse
import os
import signal
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sat_widgets import pipeline # noqa: E402
from sat_widgets.app_paths import declaration_path # noqa: E402


def load_settings(args):
    config_path = args.config or os.environ.get("TIF_VIEWER_CONFIG") or declaration_path("config.json")
    config = pipeline.read_json(config_path, {}) or {}
    s = dict(config.get("auto_run") or {})
    s["config_file"] = config_path
    s["products_file"] = args.products or os.environ.get("TIF_VIEWER_PRODUCTS") or s.get("products_file")
    for key in ("output_root", "plugin", "method", "void_type", "threshold", "start_layer", "end_layer",
                "stable_seconds", "poll_seconds", "state_file"):
        v = getattr(args, key)
        if v is not None:
            s[key] = v
    if args.no_send:
        s["send_results"] = False
    if args.watch: # A folder -> its TIFs
        s["watch"] = []
        for w in args.watch:
            s["watch"] += [os.path.join(w, "*.tif"), os.path.join(w, "*.tiff")] if os.path.isdir(w) else [w]
    return s


def run_one(path, settings):
    try:
        return pipeline.run_wafer(path, settings)
    except Exception as e: # run_wafer reports its own errors; this is a last guard for the watcher
        traceback.print_exc()
        return {"status": "error", "reason": f"{type(e).__name__}: {e}", "source_file": path}


def watch(settings):
    s = pipeline.merged_settings(settings)
    w = pipeline.FileWatcher.for_settings(s)
    if not w.globs:
        sys.exit('Nothing to watch: give --watch GLOB or set config.json "auto_run": {"watch": [...]}')
    print(f"[auto_run] watching {'; '.join(w.globs)} (stable {w.stable} s, poll {s['poll_seconds']} s); "
          "Ctrl+C stops", flush=True)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1)) # Service stop: after the current file
    signal.signal(signal.SIGINT, signal.default_int_handler) # Ctrl+C, also when started in the background
    try:
        while not stop:
            for p in w.poll():
                print(f"[auto_run] new file: {p}", flush=True)
                w.finished(p, run_one(p, settings))
                sys.stdout.flush()
                if stop:
                    break
            for _ in range(int(max(1, float(s["poll_seconds"]) * 10))):
                if stop:
                    break
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print("[auto_run] stopped", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", metavar="FILE")
    mode.add_argument("--watch", metavar="GLOB", nargs="*", help="glob patterns or folders (default: config)")
    ap.add_argument("--config", help="config.json to read (default: the viewer's)")
    ap.add_argument("--products", help="products.json to read (default: the viewer's)")
    ap.add_argument("--output-root", dest="output_root")
    ap.add_argument("--plugin", help="Results API plugin file (default: config.json results_api.plugin)")
    ap.add_argument("--method", choices=["layer_change", "die_to_die", "patchcore", "segmentation"])
    ap.add_argument("--void-type", dest="void_type")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--start-layer", dest="start_layer", type=int)
    ap.add_argument("--end-layer", dest="end_layer", type=int)
    ap.add_argument("--no-send", action="store_true", help="do not send to the Results API")
    ap.add_argument("--stable", dest="stable_seconds", type=float)
    ap.add_argument("--poll", dest="poll_seconds", type=float)
    ap.add_argument("--state", dest="state_file")
    args = ap.parse_args()
    settings = load_settings(args)
    if args.once:
        summary = run_one(args.once, settings)
        sys.exit(0 if summary.get("status") == "ok" else 1)
    watch(settings)


if __name__ == "__main__":
    main()
