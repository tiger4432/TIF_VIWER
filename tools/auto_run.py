"""
Unattended void marking of measured wafers (sat_widgets/pipeline.py), see docs/AUTO_RUN.md.

    python tools/auto_run.py --once FILE              one file (TIF, or a project folder / patches.json)
    python tools/auto_run.py --watch FOLDER           new files in FOLDER until Ctrl+C

Settings: config.json "auto_run" (read only), overridden by the options below.
Watch: a file is taken when its size and mtime did not change for --stable seconds; finished files
are recorded in a state file (default <output_root>/auto_run_state.json) and not processed again
unless they change (size / mtime). One failing file does not stop the watcher.
"""
import argparse
import fnmatch
import json
import os
import signal
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sat_widgets import pipeline # noqa: E402
from sat_widgets.app_paths import declaration_path # noqa: E402

WATCH_DEFAULTS = {
    "patterns": ["*.tif", "*.tiff"], # Matched on the file name (case-insensitive); "patches.json" for projects
    "recursive": False,
    "stable_seconds": 10.0,
    "poll_seconds": 5.0,
    "state_file": None, # Default: <output_root>/auto_run_state.json
}


def load_settings(args):
    config_path = args.config or os.environ.get("TIF_VIEWER_CONFIG") or declaration_path("config.json")
    config = pipeline.read_json(config_path, {}) or {}
    s = dict(config.get("auto_run") or {})
    s["config_file"] = config_path
    s["products_file"] = args.products or os.environ.get("TIF_VIEWER_PRODUCTS") or s.get("products_file")
    for key in ("output_root", "plugin", "method", "void_type", "threshold", "start_layer", "end_layer"):
        v = getattr(args, key)
        if v is not None:
            s[key] = v
    if args.no_send:
        s["send_results"] = False
    s["watch"] = dict(WATCH_DEFAULTS, **(s.get("watch") or {}))
    if args.pattern:
        s["watch"]["patterns"] = args.pattern
    for key in ("stable_seconds", "poll_seconds", "state_file"):
        v = getattr(args, key)
        if v is not None:
            s["watch"][key] = v
    if args.recursive:
        s["watch"]["recursive"] = True
    s.setdefault("output_root", pipeline.DEFAULTS["output_root"])
    return s


def run_one(path, settings):
    try:
        return pipeline.run_wafer(path, settings)
    except Exception as e: # run_wafer reports its own errors; this is a last guard for the watcher
        traceback.print_exc()
        return {"status": "error", "reason": f"{type(e).__name__}: {e}", "source_file": path}


class State:
    """Finished files: path -> {size, mtime, status, output_dir, finished_at}. Written atomically."""

    def __init__(self, path):
        self.path = path
        self.data = pipeline.read_json(path, {}) or {}

    def done(self, path, size, mtime):
        e = self.data.get(path)
        return e is not None and e.get("size") == size and e.get("mtime") == mtime

    def record(self, path, size, mtime, summary):
        self.data[path] = {"size": size, "mtime": mtime, "status": summary.get("status"),
                           "reason": summary.get("reason", ""), "output_dir": summary.get("output_dir"),
                           "finished_at": summary.get("finished_at")}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)


def candidates(folder, watch):
    pats = [p.lower() for p in watch["patterns"]]
    if watch.get("recursive"):
        walk = ((d, files) for d, _, files in os.walk(folder))
    else:
        walk = [(folder, [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])]
    for d, files in walk:
        for f in files:
            if any(fnmatch.fnmatch(f.lower(), p) for p in pats):
                yield os.path.abspath(os.path.join(d, f))


def watch(folder, settings):
    w = settings["watch"]
    state = State(w.get("state_file") or os.path.join(settings["output_root"], "auto_run_state.json"))
    seen = {} # path -> (size, mtime, unchanged since)
    print(f"[auto_run] watching {os.path.abspath(folder)} for {', '.join(w['patterns'])} "
          f"(stable {w['stable_seconds']} s, poll {w['poll_seconds']} s); Ctrl+C stops")
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1)) # Service stop: after the current file
    signal.signal(signal.SIGINT, signal.default_int_handler) # Ctrl+C, also when started in the background
    try:
        while not stop:
            try:
                paths = list(candidates(folder, w))
            except OSError as e: # Network share away for a moment
                print(f"[auto_run] folder not readable: {e}")
                paths = []
            now = time.time()
            for p in paths:
                try:
                    st = os.stat(p)
                except OSError:
                    continue # Removed / renamed meanwhile
                key = (st.st_size, st.st_mtime)
                if state.done(p, *key):
                    continue
                prev = seen.get(p)
                if prev is None or prev[:2] != key:
                    seen[p] = key + (now,)
                    continue
                if now - prev[2] < float(w["stable_seconds"]) or st.st_size == 0:
                    continue
                print(f"[auto_run] new file: {p}", flush=True)
                summary = run_one(p, settings)
                sys.stdout.flush()
                state.record(p, st.st_size, st.st_mtime, summary)
                seen.pop(p, None)
                if stop:
                    break
            for _ in range(int(max(1, float(w["poll_seconds"]) * 10))):
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
    mode.add_argument("--watch", metavar="FOLDER")
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
    ap.add_argument("--pattern", action="append", help="file name pattern to watch (repeatable)")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--stable", dest="stable_seconds", type=float)
    ap.add_argument("--poll", dest="poll_seconds", type=float)
    ap.add_argument("--state", dest="state_file")
    args = ap.parse_args()
    settings = load_settings(args)
    if args.once:
        summary = run_one(args.once, settings)
        sys.exit(0 if summary.get("status") == "ok" else 1)
    if not os.path.isdir(args.watch):
        ap.error(f"not a folder: {args.watch}")
    watch(args.watch, settings)


if __name__ == "__main__":
    main()
