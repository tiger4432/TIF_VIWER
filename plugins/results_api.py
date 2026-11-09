"""
Results API plugin: sends the marking results to another system, and loads markings from one.
Edit this file for your system. It is loaded from the program folder every time a button is
pressed (Extraction -> Results API), so changes apply without restarting or rebuilding.

Settings: config.json "results_api" (passed as `settings`), e.g.
    "results_api": {
        "plugin": "plugins/results_api.py",        # this file (relative to the program folder or absolute)
        "send":    {"url": "http://server/api/voids", "method": "POST", "timeout": 30,
                    "headers": {"Authorization": "Bearer ..."}},
        "receive": {"url": "http://server/api/voids?wafer={wafer_id}", "timeout": 30,
                    "replace": true},
        "bonding_map": {"url": "http://mapserver/api/bondingmap/{wafer_id}", "timeout": 30,
                        "auto_fetch": true},   # fetch when a file is opened (wafer ID parsed)
                        # GET filters: "params": {"wafer": "{wafer_id}", "type": "csv"} -> ?wafer=...&type=csv
                        # JSON filter: "params": {"filters": {"waferId": "{wafer_id}", "layer": [1, 2]}}
                        #   -> ?filters=%7B%22waferId%22%3A%22W07%22%2C%22layer%22%3A%5B1%2C2%5D%7D
        "notify":  {"url": "http://mailserver/send?to={to}&title={title}&body={body}",
                    "recipients": ["a@x.com", "b@x.com"], "recipients_sep": ",", "timeout": 30}
    }

The functions below are called by the program:
    build_payload(ctx)          -> payload              GUI thread   what "Send" sends ("Preview" shows it)
    send(payload, settings)     -> (ok, message)        background   the request
    fetch(ctx, settings)        -> data                 background   the request of "Load"
    to_voids(data, ctx)         -> voids.json content   GUI thread   converts the response
    fetch_bonding_map(ctx, settings) -> map             background   "Fetch Map" (wafer ID -> bonding map)
    parse_wafer_id(file_name, settings) -> str | dict   GUI thread   optional: wafer ID from the opened
                                                                     file name (else config.json
                                                                     "wafer_id": {"pattern": ...})
    product_for(wafer_id, file_name, settings) -> str   Auto Run     optional: product of a measured wafer
                                                                     (settings = config.json "auto_run")
    notify_review(title, body, recipients, ctx, settings) -> (ok, message)
                                                        Auto Run     review request of a finished wafer
                                                                     (docs/AUTO_RUN.md)

ctx (read only) describes the current state:
    app_version, timestamp (ISO 8601), user, host
    product          selected WF product name ("" if none)
    source_file      opened TIF / project folder;  source_name: its name without folder / extension
    wafer_id         the Wafer ID box (parsed from the file name, editable)
    layers           loaded absolute layer numbers;  current_layer
    grid             {"start_x", "start_y", "pitch_x", "pitch_y", "rows", "cols", "angle"}
    um_per_px        Scale (0 = not set)
    bonding_map      {"text": pasted map, "reject_values": [...], "rows": [[LEG per cell]]} or None
    voids_json       what "Save JSON" writes: meta.json (version, layer_numbering, um_per_px, pixel_scale,
                     grid, types) with the voids.json list as "voids";
                     field names of each void follow config.json "output_columns"
    records          one dict per void with the internal names (see sat_widgets/output_columns.py):
                     layer, row, col, label (LEG), type, center_x/y, radius_x/y, p, global_x/y,
                     bbox_rel_x/y, *_um, ...
    types            declared void types {name: {color, shape, ...}}
    last_export_dir  folder of the last export in this session, or None
    settings         config.json "results_api"

fetch_bonding_map returns the map as text pasted from Excel (tab or comma separated rows, row /
column headers allowed), a list of rows, or {"text" | "rows": ..., "reject_values": [...]}; the map
is put into the Bonding Map box and applied (reject values too, when given).

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
        "wafer_id": ctx["wafer_id"],
        "sent_at": ctx["timestamp"],
        "sent_by": ctx["user"],
        "app_version": ctx["app_version"],
        "um_per_px": ctx["um_per_px"] or None,
        "grid": ctx["grid"],
        "voids": ctx["voids_json"]["voids"],
    }


def _request(url, settings, data=None, ctx=None):
    """
    data: bytes are sent as is; a dict / list is sent as JSON, a str as UTF-8. urlopen itself takes
    only bytes: a dict or list there fails with "can't concat str to bytes", a str with "POST data
    should be bytes". For a form (key=value&...) pass urllib.parse.urlencode(d).encode() and set
    "headers": {"Content-Type": "application/x-www-form-urlencoded"}.
    A GET has no body (servers ignore one): filters go into the URL. settings "params" (dict; its
    strings, also nested ones, may hold {wafer_id} etc. when ctx is given) and a dict passed as data
    with a GET are added as ?key=value&... (see _query: a dict / list value goes as URI-encoded JSON,
    e.g. "params": {"filters": {"waferId": "{wafer_id}"}} -> ?filters=%7B%22waferId%22%3A%22W07%22%7D).
    """
    method = (settings.get("method") or ("POST" if data is not None else "GET")).upper()
    params = _fill_value(dict(settings.get("params") or {}), ctx)
    if method in ("GET", "HEAD", "DELETE") and isinstance(data, dict):
        params.update(data)
        data = None
    if params:
        url += ("&" if "?" in url else "?") + _query(params)
    if isinstance(data, (dict, list)):
        data = json.dumps(data, ensure_ascii=False)
    if isinstance(data, str):
        data = data.encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    headers.update(settings.get("headers") or {})
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=float(settings.get("timeout", 30))) as resp:
        return resp.status, resp.read()


def _query(params):
    """
    Query string, encoded like JavaScript encodeURIComponent (space %20, / %2F). A dict / list value
    is sent as compact JSON ({"a":1}), true / false / null as in JSON, other values as text.
    """
    def text(v):
        if isinstance(v, (dict, list, bool)) or v is None:
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        return str(v)
    return urllib.parse.urlencode({k: text(v) for k, v in params.items()}, quote_via=urllib.parse.quote)


def _fill_value(value, ctx):
    """{wafer_id} etc. filled into the strings of a value (nested dicts / lists too), not URL-encoded."""
    if ctx is None:
        return value
    if isinstance(value, dict):
        return {k: _fill_value(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_value(v, ctx) for v in value]
    return _fill(value, ctx, quote=False) if isinstance(value, str) else value


def send(payload, settings):
    """Sends the payload. Returns (ok, message shown to the user)."""
    cfg = settings.get("send") or {}
    url = cfg.get("url")
    if not url:
        return False, 'Set config.json "results_api": {"send": {"url": ...}}'
    status, body = _request(url, cfg, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    text = body[:300].decode("utf-8", "replace")
    return 200 <= status < 300, f"HTTP {status}: {text}"


def _fill(url, ctx, quote=True):
    """{wafer_id}, {source_name}, {product}, {current_layer} in a URL are filled in (URL-encoded)."""
    enc = urllib.parse.quote if quote else (lambda v: v)
    return url.format(**{k: enc(str(ctx[k])) for k in ("wafer_id", "source_name", "product", "current_layer")})


def fetch(ctx, settings):
    """Requests the markings. Returns the parsed response (passed to to_voids)."""
    cfg = settings.get("receive") or {}
    url = cfg.get("url")
    if not url:
        raise ValueError('Set config.json "results_api": {"receive": {"url": ...}}')
    status, body = _request(_fill(url, ctx), cfg, ctx=ctx)
    if not 200 <= status < 300:
        raise RuntimeError(f"HTTP {status}: {body[:300].decode('utf-8', 'replace')}")
    return json.loads(body.decode("utf-8"))


def fetch_bonding_map(ctx, settings):
    """Bonding map of ctx["wafer_id"]. Default: GET the URL; JSON ({"text" | "rows", "reject_values"}) or text
    (CSV / tab separated, UTF-8 or cp949; the response bytes may also be returned as they are)."""
    cfg = settings.get("bonding_map") or {}
    url = cfg.get("url")
    if not url:
        raise ValueError('Set config.json "results_api": {"bonding_map": {"url": ".../{wafer_id}"}}')
    status, body = _request(_fill(url, ctx), cfg, ctx=ctx)
    if not 200 <= status < 300:
        raise RuntimeError(f"HTTP {status}: {body[:300].decode('utf-8', 'replace')}")
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = body.decode("cp949") # CSV saved by Korean Excel
    try:
        return json.loads(text)
    except ValueError:
        return text # Plain text map: CSV (comma / semicolon, quoted fields) or tab separated


# Optional: wafer ID from the opened file name (overrides config.json "wafer_id": {"pattern": ...}).
# A dict also gives the lot (Auto Run picks the product from lot / wafer ID).
# def parse_wafer_id(file_name, settings):
#     # e.g. "LOT1234_W07_SAT.tif" -> {"lot": "LOT1234", "wafer_id": "LOT1234-07"}
#     lot, wafer = file_name.split("_")[:2]
#     return {"lot": lot, "wafer_id": f"{lot}-{wafer[1:]}"}


# Optional, Auto Run (docs/AUTO_RUN.md): product of a measured wafer (overrides config.json
# "auto_run": {"products": [{"lot", "wafer_id", "file", "product"}]}). None: the rules decide.
# settings: config.json "auto_run".
# def product_for(wafer_id, file_name, settings):
#     # e.g. from the lot prefix, or ask the MES: GET http://mes/api/wafer/{wafer_id} -> {"product": ...}
#     return {"LOTA": "WF_A", "LOTB": "WF_B"}.get(wafer_id[:4])


def notify_review(title, body, recipients, ctx, settings):
    """
    Automatic run (Auto Run / tools/auto_run.py): asks a reviewer to check a finished wafer.
    title, body: text built by the program; recipients: config "results_api": {"notify": {"recipients"}}.
    ctx: as above plus "summary" (summary.json content) and "review_queue" (path of review_queue.jsonl).
    Returns (ok, message). Default: GET notify.url with {to}, {title}, {body} filled in (URL-encoded,
    recipients joined by notify.recipients_sep, default ","); no url: printed and appended to
    review_queue.jsonl.
    """
    cfg = settings.get("notify") or {}
    url = cfg.get("url")
    if not url:
        entry = {"title": title, "body": body, "recipients": recipients, "wafer_id": ctx["wafer_id"],
                 "product": ctx["product"], "status": ctx["summary"]["status"], "output_dir": ctx["last_export_dir"],
                 "queued_at": ctx["timestamp"]}
        print(f"[review] {title}\n{body}")
        with open(ctx["review_queue"], "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True, "no notify.url: written to review_queue.jsonl"
    q = lambda v: urllib.parse.quote(str(v), safe="")
    full = url.format(to=q(cfg.get("recipients_sep", ",").join(recipients)), title=q(title), body=q(body))
    status, resp = _request(full, dict(cfg, method="GET"))
    return 200 <= status < 300, f"HTTP {status}: {resp[:300].decode('utf-8', 'replace')}"


def to_voids(data, ctx):
    """Converts the response to voids.json content. Default: the response already is one."""
    if isinstance(data, dict) and "voids" not in data and "data" in data:
        data = data["data"] # e.g. {"data": {...voids.json...}}
    return data
