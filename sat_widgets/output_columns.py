"""
Output field / column names of the void exports (voids.json, voids.csv).

Keys are the internal names used by the code; values are the names written to the files.
Edit the values below, or override them per deployment (no rebuild needed) in config.json:

    "output_columns": {
        "csv":  {"bbox_rel_x": "ValidX", "bbox_rel_y": "ValidY", "radius_x": null},
        "json": {"bbox_rel_x": "validX"}
    }

- A value of null leaves the column / field out.
- CSV column order follows CSV_COLUMNS.
- voids.json is read back with the same names (files written with the default names stay
  readable). Leaving out global_x/global_y makes reloading depend on the grid settings.

p: exponent of "curve" (superellipse) voids, empty for other shapes.
bbox_rel_x / bbox_rel_y: void center relative to the top-left corner of the cell's reference
bbox (the "bbox" type marking on the topmost layer of that cell), in grid-aligned pixels.
Empty when the cell has no bbox.
*_um: the same values in um (pixels x Scale). Empty while the scale is not set.
layer0: Detect Voids "Layer change" found the void already visible on layer 0 (bubble?), until
the reviewer clears it (B). JSON true / false, CSV 1 / 0. Files without it read as false.
auto_state: "detected" (detected void) / "estimated" (Detect BBOX) while not reviewed (edited)
yet, else empty. Read back so the review can continue after loading.
snr: signal-to-noise of a detected void on its layer (Detect Voids); hand markings 100; empty for
bboxes. Voids below the SNR filter (Void Detection panel, meta.json "snr_min") are not written.

Pixel values (positions and sizes: PIXEL_KEYS) are written scaled as if the grid's x pitch were
NORM_PITCH_X (value x 300 / pitch_x), so files of scans with another resolution compare
directly; the actual pixel values go to the *_raw fields. Change or switch off (null = actual
pixels in both) in config.json:

    "output_scale": {"pitch_x": 300}

meta.json (next to voids.json) records the scale ("pixel_scale"); reading prefers the *_raw fields, else divides by
the recorded scale (files without either are actual pixels).
"""

NORM_PITCH_X = 300.0
PIXEL_KEYS = ("center_x", "center_y", "radius_x", "radius_y", "global_x", "global_y", "bbox_rel_x", "bbox_rel_y")

# voids.json, per void
JSON_FIELDS = {
    "key": "key",
    "col": "x",
    "row": "y",
    "layer": "layer",
    "void_index": "voidIndex",
    "type": "type",
    "center_x": "centerX",
    "center_y": "centerY",
    "radius_x": "radiusX",
    "radius_y": "radiusY",
    "p": "p",
    "created_at": "createdAt",
    "patch_label": "patchLabel",
    "global_x": "globalCX",
    "global_y": "globalCY",
    "bbox_rel_x": "bboxRelX",
    "bbox_rel_y": "bboxRelY",
    "center_x_raw": "centerX_raw",
    "center_y_raw": "centerY_raw",
    "radius_x_raw": "radiusX_raw",
    "radius_y_raw": "radiusY_raw",
    "global_x_raw": "globalCX_raw",
    "global_y_raw": "globalCY_raw",
    "bbox_rel_x_raw": "bboxRelX_raw",
    "bbox_rel_y_raw": "bboxRelY_raw",
    "center_x_um": "centerXUm",
    "center_y_um": "centerYUm",
    "radius_x_um": "radiusXUm",
    "radius_y_um": "radiusYUm",
    "bbox_rel_x_um": "bboxRelXUm",
    "bbox_rel_y_um": "bboxRelYUm",
    "layer0": "layer0",
    "auto_state": "autoState",
    "snr": "snr",
}

# voids.csv
CSV_COLUMNS = {
    "layer": "Layer",
    "row": "CellRow",
    "col": "CellCol",
    "label": "Label",
    "type": "VoidType",
    "global_x": "GlobalX",
    "global_y": "GlobalY",
    "center_x": "RelativeX",
    "center_y": "RelativeY",
    "radius_x": "RadiusX",
    "radius_y": "RadiusY",
    "p": "P",
    "bbox_rel_x": "BBoxRelX",
    "bbox_rel_y": "BBoxRelY",
    "global_x_raw": "GlobalX_raw",
    "global_y_raw": "GlobalY_raw",
    "center_x_raw": "RelativeX_raw",
    "center_y_raw": "RelativeY_raw",
    "radius_x_raw": "RadiusX_raw",
    "radius_y_raw": "RadiusY_raw",
    "bbox_rel_x_raw": "BBoxRelX_raw",
    "bbox_rel_y_raw": "BBoxRelY_raw",
    "center_x_um": "RelativeX_um",
    "center_y_um": "RelativeY_um",
    "radius_x_um": "RadiusX_um",
    "radius_y_um": "RadiusY_um",
    "bbox_rel_x_um": "BBoxRelX_um",
    "bbox_rel_y_um": "BBoxRelY_um",
    "layer0": "Layer0",
    "auto_state": "AutoState",
    "snr": "SNR",
}


def resolve(config=None):
    """Returns (json_fields, csv_columns) with config.json 'output_columns' overrides applied."""
    json_fields = dict(JSON_FIELDS)
    csv_columns = dict(CSV_COLUMNS)
    overrides = (config or {}).get("output_columns") or {}
    for target, names in (("json", json_fields), ("csv", csv_columns)):
        for key, name in (overrides.get(target) or {}).items():
            if key not in names:
                print(f"[output_columns] Unknown {target} key '{key}' ignored. Valid keys: {', '.join(names)}")
                continue
            names[key] = name
    return json_fields, csv_columns


def norm_pitch(config=None):
    """Grid x pitch the pixel values are scaled to (config.json "output_scale"), None = no scaling."""
    cfg = (config or {}).get("output_scale", {"pitch_x": NORM_PITCH_X})
    value = (cfg or {}).get("pitch_x") if isinstance(cfg, dict) else None
    return float(value) if value else None
