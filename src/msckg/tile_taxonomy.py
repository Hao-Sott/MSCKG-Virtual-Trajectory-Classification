from pathlib import Path

import numpy as np
import pandas as pd


def _three_level(values, low, high, zero_separate=False):
    values = np.asarray(values, dtype=float)
    if zero_separate:
        return np.where(values == 0, "C", np.where(values <= high, "B", "A"))
    return np.where(values <= low, "C", np.where(values <= high, "B", "A"))


def grade_tile_components(frame):
    result = frame.copy()
    result["major_road_level"] = np.where(result["major_road"] > 0, "A", "B")
    result["minor_road_level"] = np.where(result["minor_road"] > 0, "A", "B")
    result["functional_level"] = _three_level(result["functional_area"], 0, 0.50, True)
    result["building_level"] = _three_level(result["building"], 0, 0.25, True)
    result["green_level"] = _three_level(result["green_land"], 0.25, 0.75)
    result["water_level"] = _three_level(result["water"], 0.25, 0.75)
    return result


def assign_taxonomy(pixel_counts, taxonomy):
    pixels = pd.read_csv(pixel_counts)
    required = {"tile_id", "expressway", "national_road", "provincial_road", "county_road", "other_road", "water", "green_land", "functional_area", "building"}
    missing = required.difference(pixels.columns)
    if missing:
        raise ValueError(f"Tile pixel table is missing {sorted(missing)}")
    denominator = pixels.get("pixel_total", pd.Series(65536, index=pixels.index)).astype(float)
    identifiers = {"tile_id": pixels["tile_id"]}
    for column in ("tile_row", "tile_col"):
        if column in pixels.columns:
            identifiers[column] = pixels[column]
    proportions = pd.DataFrame(
        {
            **identifiers,
            "major_road": pixels[["expressway", "national_road", "provincial_road"]].sum(axis=1) / denominator,
            "minor_road": pixels[["county_road", "other_road"]].sum(axis=1) / denominator,
            "water": pixels["water"] / denominator,
            "green_land": pixels["green_land"] / denominator,
            "functional_area": pixels["functional_area"] / denominator,
            "building": pixels["building"] / denominator,
        }
    )
    graded = grade_tile_components(proportions)
    rules = pd.read_csv(taxonomy)
    keys = ["major_road_level", "minor_road_level", "functional_level", "building_level", "green_level", "water_level"]
    missing = set(keys + ["tile_category"]).difference(rules.columns)
    if missing:
        raise ValueError(f"Tile taxonomy is missing {sorted(missing)}")
    return graded.merge(rules, on=keys, how="left", validate="many_to_one")


def run_tile_taxonomy(pixel_counts, taxonomy, output):
    result = assign_taxonomy(pixel_counts, taxonomy)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    valid = result[result["tile_category"].notna()].copy()
    invalid = result[result["tile_category"].isna()].copy()
    valid.to_csv(output, index=False, encoding="utf-8-sig")
    if len(invalid):
        invalid.to_csv(output.with_name(f"{output.stem}_invalid{output.suffix}"), index=False, encoding="utf-8-sig")
    return valid
