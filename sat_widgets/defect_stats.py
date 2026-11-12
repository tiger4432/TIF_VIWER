"""
Defect statistics of a wafer, written with Export (CSV option): defect rate per bonding LEG and
defect type / group, spec-out judgement per type and a bin per chip.

Chips: the grid cells with a LEG (bonding map value; rejected values are not chips), as the exported
images. A chip has a defect of a type when it holds a void of that type on any layer (bboxes and voids
below the SNR filter do not count; a chip counts once whatever the number of voids or layers).

Spec out: plugins/defect_spec.py (config.json "defect_stats": {"plugin": ...}) may judge a type per
chip: spec_out(type_name, voids, chip) -> True (spec out) / False (in spec) / None (no rule). A type
without a rule (or without the plugin) is spec out whenever it occurs.

Bins: one per chip.
    plugin chip_bin(chip, spec_out_types, defect_types) -> bin, when the plugin defines it; else
    config.json "defect_stats": {"bins": [{"bin": "B2", "types": ["Crack", "Edge"]}, ...],
                                 "good_bin": "GOOD", "other_bin": "FAIL"}
    in priority order: the first entry naming a spec-out type (or its group) of the chip. Without
    "bins" every spec-out type is its own bin (the first in the declared type order). A chip without
    a spec-out defect: good_bin (default "GOOD"); spec out but in no listed bin: other_bin ("FAIL").

Files:
    defect_rate.csv  LEG, Kind (type / group / any / bin), Name, DefectChips, SpecOutChips, TotalChips,
                     DefectRate(%), SpecOutRate(%); per LEG and ALL. Group rows when void types declare
                     "group". Bin rows: the chips of the bin (both counts).
    chip_bins.csv    Row, Col, LEG, Bin, then a column per defect type: 0 none, 1 defect in spec,
                     2 spec out (single values, no lists)
    bin_map.txt      the bins as a grid (tab separated, row / column numbers as headers), like a
                     pasted bonding map
"""
import csv
import importlib.util
import os

from .app_paths import app_file

DEFAULT_PLUGIN = "plugins/defect_spec.py"


def load_plugin(settings):
    """The spec plugin (fresh import, edits apply at the next export), or None when the file is missing."""
    path = (settings or {}).get("plugin") or DEFAULT_PLUGIN
    path = path if os.path.isabs(path) else app_file(path)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("defect_spec_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leg_order(legs):
    return sorted(set(legs), key=lambda x: (len(x), x))


def compute(vm, cfg, bmap, records, settings=None, ctx=None):
    """
    records: VoidManager.void_records (SNR-filtered, internal names). Returns {"rate_rows",
    "chips": [{row, col, leg, bin, state: {type: 0/1/2}}], "types", "bins", "errors"}.
    """
    settings = settings or {}
    ctx = ctx or {}
    errors = []
    try:
        plugin = load_plugin(settings)
    except Exception as e: # A broken plugin: every type counts when it occurs
        plugin = None
        errors.append(f"defect_spec plugin not loaded: {type(e).__name__}: {e}")
    judge = getattr(plugin, "spec_out", None) if plugin else None
    binner = getattr(plugin, "chip_bin", None) if plugin else None

    legs = {}
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            leg = bmap.get_key(r, c) if bmap is not None else None
            if leg:
                legs[(r, c)] = str(leg)
    types = [n for n in vm.types if not vm.is_bbox({"type": n})]
    group_of = lambda t: str((vm.types.get(t) or {}).get("group") or t)
    by_chip = {}
    for rec in records:
        cell = (rec["row"], rec["col"])
        t = rec.get("type") or "unknown"
        if cell not in legs or vm.is_bbox({"type": t}):
            continue
        if t not in types:
            types.append(t)
        by_chip.setdefault(cell, {}).setdefault(t, []).append(rec)

    # Chip size for the rules: the bbox type's size (product chip size / Scale), in px and um
    s = vm.um_per_px if vm.um_per_px and vm.um_per_px > 0 else None
    bt = next((n for n in vm.types if vm.is_bbox({"type": n})), None)
    size_px = ((vm.types[bt].get("def_w") or 0.0), (vm.types[bt].get("def_h") or 0.0)) if bt else (0.0, 0.0)
    size_um = tuple(ctx["chip_size_um"]) if ctx.get("chip_size_um") else (
        (size_px[0] * s, size_px[1] * s) if s and size_px[0] else None)

    bins_cfg = settings.get("bins") or []
    good_bin = str(settings.get("good_bin", "GOOD"))
    other_bin = str(settings.get("other_bin", "FAIL"))
    chips = []
    for cell in sorted(legs):
        found = by_chip.get(cell, {})
        chip = {"row": cell[0], "col": cell[1], "leg": legs[cell], "size_px": size_px, "size_um": size_um,
                "um_per_px": s, "wafer_id": ctx.get("wafer_id", ""), "product": ctx.get("product", ""),
                "voids": [v for vs in found.values() for v in vs]}
        state = {}
        for t in types:
            if t not in found:
                state[t] = 0
                continue
            out = None
            if judge is not None:
                try:
                    out = judge(t, found[t], chip)
                except Exception as e:
                    errors.append(f"spec_out({t!r}) at row {cell[0]}, col {cell[1]}: {type(e).__name__}: {e}")
            state[t] = 1 if out is False else 2 # No rule (None) or an error: spec out when it occurs
        spec = [t for t in types if state[t] == 2]
        defect = [t for t in types if state[t] >= 1]
        b = None
        if binner is not None:
            try:
                b = binner(chip, spec, defect)
            except Exception as e:
                errors.append(f"chip_bin at row {cell[0]}, col {cell[1]}: {type(e).__name__}: {e}")
        if b is None:
            if not spec:
                b = good_bin
            elif bins_cfg:
                names = set(spec) | {group_of(t) for t in spec}
                b = next((str(e["bin"]) for e in bins_cfg if names & set(e.get("types") or [])), other_bin)
            else:
                b = spec[0]
        chips.append({"row": cell[0], "col": cell[1], "leg": legs[cell], "bin": str(b), "state": state})

    has_groups = any((vm.types.get(t) or {}).get("group") for t in types)
    groups = list(dict.fromkeys(group_of(t) for t in types)) if has_groups else []
    bin_names = list(dict.fromkeys([good_bin] + [str(e["bin"]) for e in bins_cfg] + [c["bin"] for c in chips]))
    rate_rows = []
    for leg in _leg_order(legs.values()) + ["ALL"]:
        sel = [c for c in chips if leg == "ALL" or c["leg"] == leg]
        total = len(sel)
        pct = lambda n: round(100.0 * n / total, 2) if total else 0.0
        def row(kind, name, n_def, n_out):
            rate_rows.append([leg, kind, name, n_def, n_out, total, pct(n_def), pct(n_out)])
        for t in types:
            row("type", t, sum(c["state"][t] >= 1 for c in sel), sum(c["state"][t] == 2 for c in sel))
        for g in groups:
            ts = [t for t in types if group_of(t) == g]
            row("group", g, sum(any(c["state"][t] >= 1 for t in ts) for c in sel),
                sum(any(c["state"][t] == 2 for t in ts) for c in sel))
        row("any", "any defect", sum(any(v >= 1 for v in c["state"].values()) for c in sel),
            sum(any(v == 2 for v in c["state"].values()) for c in sel))
        for b in bin_names:
            n = sum(c["bin"] == b for c in sel)
            row("bin", b, n, n)
    return {"rate_rows": rate_rows, "chips": chips, "types": types, "bins": bin_names, "errors": errors,
            "rows": cfg.rows, "cols": cfg.cols}


def write(output_dir, res):
    """defect_rate.csv, chip_bins.csv, bin_map.txt (UTF-8 with BOM: Excel reads it)."""
    with open(os.path.join(output_dir, "defect_rate.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["LEG", "Kind", "Name", "DefectChips", "SpecOutChips", "TotalChips", "DefectRate(%)",
                    "SpecOutRate(%)"])
        w.writerows(res["rate_rows"])
    with open(os.path.join(output_dir, "chip_bins.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Row", "Col", "LEG", "Bin"] + res["types"])
        for c in res["chips"]:
            w.writerow([c["row"], c["col"], c["leg"], c["bin"]] + [c["state"][t] for t in res["types"]])
    grid = [[""] * res["cols"] for _ in range(res["rows"])]
    for c in res["chips"]:
        grid[c["row"]][c["col"]] = c["bin"]
    with open(os.path.join(output_dir, "bin_map.txt"), "w", newline="", encoding="utf-8") as f:
        f.write("\t".join([""] + [str(i) for i in range(res["cols"])]) + "\r\n")
        for i, r in enumerate(grid):
            f.write("\t".join([str(i)] + r) + "\r\n")
