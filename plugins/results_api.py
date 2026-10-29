"""
Results API plugin: sends the marking results to another system, and loads markings from one.
Edit this file for your system. It is loaded from the program folder every time a button is
pressed (Extraction -> Results API), so changes apply without restarting or rebuilding.

Settings: config.json "results_api" (passed as `settings`), e.g.
    "results_api": {
        "plugin": "plugins/results_api.py",        # this file (relative to the program folder or absolute)
        "send":    {"url": "http://server/api/voids", "method": "POST", "timeout": 30,
                    "headers": {"Authorization": "Bearer ..."}},
        "receive": {"url": "http://server/api/voids?wafer={source_name}", "timeout": 30,
                    "replace": true}
    }

The four functions below are called by the program:
    build_payload(ctx)          -> payload              GUI thread   what "Send" sends ("Preview" shows it)
    send(payload, settings)     -> (ok, message)        background   the request
    fetch(ctx, settings)        -> data                 background   the request of "Load"
    to_voids(data, ctx)         -> voids.json content   GUI thread   converts the response

ctx (read only) describes the current state:
    app_version, timestamp (ISO 8601), user, host
    product          selected WF product name ("" if none)
    source_file      opened TIF / project folder;  source_name: its name without folder / extension
    layers           loaded absolute layer numbers;  current_layer
    grid             {"start_x", "start_y", "pitch_x", "pitch_y", "rows", "cols", "angle"}
    um_per_px        Scale (0 = not set)
    bonding_map      {"text": pasted map, "reject_values": [...], "rows": [[LEG per cell]]} or None
    voids_json       exactly what "Save JSON" writes (version, layer_numbering, um_per_px, types, voids);
                     field names of each void follow config.json "output_columns"
    records          one dict per void with the internal names (see sat_widgets/output_columns.py):
                     layer, row, col, label (LEG), type, center_x/y, radius_x/y, p, global_x/y,
                     bbox_rel_x/y, *_um, ...
    types            declared void types {name: {color, shape, ...}}
    last_export_dir  folder of the last export in this session, or None
    settings         config.json "results_api"

to_voids must return voids.json content: {"voids": [...]} (optionally "types", "um_per_px",
"layer_numbering": "absolute") or a list of voids. Per void, either global_x / global_y (raw image
pixels, as "globalCX" / "globalCY") or col / row + center_x / center_y (relative to the cell),
plus layer, type, radius_x / radius_y; names as in voids.json. Loading is one undo step
(Ctrl+Z); "replace": false adds to the current markings.

Only the Python standard library is available in the built exe (json, urllib, http.client, ssl,
base64, hashlib, csv, ...). Another package (e.g. requests) must be added to requirements.txt and
TIF_Viewer.spec (hiddenimports) and the exe rebuilt.
"""
import json
import urllib.parse
import urllib.request


def build_payload(ctx):
    """What is sent. Default: the voids.json content plus where it comes from."""
    return {
        "source": {
            "file": ctx["source_file"],
            "name": ctx["source_name"],
            "product": ctx["product"],
            "layers": ctx["layers"],
        },
        "sent_at": ctx["timestamp"],
        "sent_by": ctx["user"],
        "app_version": ctx["app_version"],
        "um_per_px": ctx["um_per_px"] or None,
        "grid": ctx["grid"],
        "voids": ctx["voids_json"]["voids"],
    }


def _request(url, settings, data=None):
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    headers.update(settings.get("headers") or {})
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=settings.get("method") or ("POST" if data is not None else "GET"))
    with urllib.request.urlopen(req, timeout=float(settings.get("timeout", 30))) as resp:
        return resp.status, resp.read()


def send(payload, settings):
    """Sends the payload. Returns (ok, message shown to the user)."""
    cfg = settings.get("send") or {}
    url = cfg.get("url")
    if not url:
        return False, 'Set config.json "results_api": {"send": {"url": ...}}'
    status, body = _request(url, cfg, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    text = body[:300].decode("utf-8", "replace")
    return 200 <= status < 300, f"HTTP {status}: {text}"


def fetch(ctx, settings):
    """Requests the markings. Returns the parsed response (passed to to_voids)."""
    cfg = settings.get("receive") or {}
    url = cfg.get("url")
    if not url:
        raise ValueError('Set config.json "results_api": {"receive": {"url": ...}}')
    # {source_name}, {product}, {current_layer} in the URL are filled in (URL-encoded)
    url = url.format(**{k: urllib.parse.quote(str(ctx[k])) for k in ("source_name", "product", "current_layer")})
    status, body = _request(url, cfg)
    if not 200 <= status < 300:
        raise RuntimeError(f"HTTP {status}: {body[:300].decode('utf-8', 'replace')}")
    return json.loads(body.decode("utf-8"))


def to_voids(data, ctx):
    """Converts the response to voids.json content. Default: the response already is one."""
    if isinstance(data, dict) and "voids" not in data and "data" in data:
        data = data["data"] # e.g. {"data": {...voids.json...}}
    return data
