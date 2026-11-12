"""
Defect spec plugin: decides per chip whether a defect type is out of spec, and optionally the chip's
bin. Edit it for your products. Used by Export (CSV option) for defect_rate.csv / chip_bins.csv /
bin_map.txt (see sat_widgets/defect_stats.py); read again at every export, no restart needed.
config.json "defect_stats": {"plugin": "plugins/defect_spec.py", "bins": [...], "good_bin": "GOOD"}

spec_out(type_name, voids, chip) -> True (spec out) / False (in spec) / None (no rule: spec out
                                    whenever the type occurs)
    voids: the chip's voids of that type (all layers), dicts with the internal names of
           sat_widgets/output_columns.py: layer, type, radius_x / radius_y (px), radius_x_um /
           radius_y_um (None while the Scale is not set), center_x_um, bbox_rel_x_um, snr, ...
           (the *_raw fields are actual image pixels)
    chip:  row, col, leg, size_px (w, h), size_um (w, h) or None, um_per_px or None, wafer_id,
           product, voids (every defect void of the chip, all types)

chip_bin(chip, spec_out_types, defect_types) -> bin name, or None for the config rule (optional;
    delete or rename it to use config.json "defect_stats" "bins")

The helpers below work in um when the Scale is set, else in actual pixels.
"""
import math


def diameter(v):
    """Largest diameter of a void (um, else px)."""
    rx = v.get("radius_x_um") if v.get("radius_x_um") is not None else v.get("radius_x_raw", v.get("radius_x"))
    ry = v.get("radius_y_um") if v.get("radius_y_um") is not None else v.get("radius_y_raw", v.get("radius_y"))
    return 2.0 * max(rx or 0.0, ry or 0.0)


def area(v):
    """Ellipse area of a void (um2, else px2)."""
    rx = v.get("radius_x_um") if v.get("radius_x_um") is not None else v.get("radius_x_raw", v.get("radius_x"))
    ry = v.get("radius_y_um") if v.get("radius_y_um") is not None else v.get("radius_y_raw", v.get("radius_y"))
    return math.pi * (rx or 0.0) * (ry or 0.0)


def chip_area(chip):
    """Chip (bbox) area (um2, else px2)."""
    w, h = chip["size_um"] if chip.get("size_um") else chip["size_px"]
    return w * h


def spec_out(type_name, voids, chip):
    # Examples (uncomment / adapt; type names as declared in config.json void_types):
    # if type_name == "Randomo":    # spec out: a void larger than 50 um, or all together > 1 % of the chip
    #     return max(diameter(v) for v in voids) > 50 or sum(area(v) for v in voids) > 0.01 * chip_area(chip)
    # if type_name == "Edge":       # spec out from 3 voids on
    #     return len(voids) >= 3
    return None # No rule: spec out whenever the type occurs


# def chip_bin(chip, spec_out_types, defect_types):
#     if not spec_out_types:
#         return "GOOD"
#     if "Edge" in spec_out_types:
#         return "B3"
#     return "B2"
