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
"""

# voids.json, per void
JSON_FIELDS = {
    "key": "key",
    "col": "x",
    "row": "y",
    "layer": "layer",
    "void_index": "voidIndex",
    "type": "type",
    "type_id": "type_id",
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
