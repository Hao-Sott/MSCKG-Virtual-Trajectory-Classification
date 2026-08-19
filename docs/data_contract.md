# Private data contract

The repository does not contain data. Column names can be changed in `configs/experiment.yaml`; common legacy aliases are detected for trajectory features.

## Tile pixel composition

`paths.tile_index` is a CSV containing `tile_id`, `tile_row`, `tile_col`, and the configured relative image-path field. `paths.tile_image_root` is the private PNG root. `paths.color_templates` is a CSV with `feature`, `r`, `g`, and `b`. Multiple RGB rows may map to the same feature.

`msckg extract-tile-pixels` produces the CSV configured by `paths.tile_pixels`:

CSV configured by `paths.tile_pixels`:

| Field | Meaning |
|---|---|
| `tile_id` | Unique level-18 tile identifier |
| `tile_row`, `tile_col` | Level-18 tile indices |
| `pixel_total` | Tile pixel count; optional, default 65,536 |
| `expressway` | Expressway pixel count |
| `national_road` | National-road pixel count |
| `provincial_road` | Provincial-road pixel count |
| `county_road` | County-road pixel count |
| `other_road` | Other-road pixel count |
| `water` | Water pixel count |
| `green_land` | Green-land pixel count |
| `functional_area` | Functional-area pixel count |
| `building` | Building pixel count |

## Tile taxonomy

CSV configured by `paths.tile_taxonomy`. It contains every valid combination of `major_road_level`, `minor_road_level`, `functional_level`, `building_level`, `green_level`, and `water_level`, plus `tile_category`. Labels use A for the highest content level. Invalid combinations are omitted.

## POIs

CSV configured by `paths.pois`:

| Configured role | Default field |
|---|---|
| POI identifier | `poi_id` |
| First-level category | `pc1` |
| Second-level category | `pc2` |
| Containing tile | `tile_id` |
| Longitude | `lon` |
| Latitude | `lat` |

## Knowledge graph

Each graph directory contains:

| File | Format |
|---|---|
| `entity2id.txt` | `entity_name<TAB>zero_based_id` |
| `relation2id.txt` | `relation_name<TAB>zero_based_id` |
| `train.txt` | `head_name<TAB>tail_name<TAB>relation_name` |

Entity IDs are contiguous. The full graph uses internal prefixes `w_` for T, `wc_` for TC, `p_` for P, `pc1_` for first-level POI categories, and `pc2_` for second-level POI categories.

The six relations are tile adjacency, tile-category membership, POI-tile containment, POI proximity within 50 m, POI-second-level-category membership, and second-level-to-first-level POI-category membership.

## Trajectories

CSV configured by `paths.trajectory`:

| Configured role | Default field | Meaning |
|---|---|---|
| Group | `gr` | Trajectory sample identifier |
| Label | `class` | One of 0, 2, 3, 4, 6, 12 |
| Longitude | `lon` | Point longitude |
| Latitude | `lat` | Point latitude |
| Tile entity | `T_entity_id` | Full-graph entity ID |
| POI entities | `P_entity_ids` | Semicolon-separated entity IDs or `NoPOI` |
| Tile category | `TC_entity_id` | Full-graph entity ID |
| POI categories | `PC_entity_ids` | Semicolon-separated second-level category entity IDs or `NoPOI` |

All rows sharing `gr` must have the same domain label. T and TC are averaged over the tile records in a trajectory group. P and PC are averaged over all matched entities in the group; if the group has no POI, both vectors are all −1.

## Temporal trajectories

The temporal CSV has the trajectory fields plus `restored_time` or the configured timestamp field. Rows are sorted within each trajectory. Simultaneous observations are mean-collapsed. Augmented points retain the source timestamp.

## Embeddings and feature archives

Full-graph entity embeddings are `.npy` arrays with shape `N × 100`; row `i` is entity ID `i`. Feature archives are `.npz` files containing `gr`, `classes`, and the available matrices among T, P, TC, and PC.

## Tile context for spatial expansion

The private table passed to `expand-training` contains `tile_entity`, `neighbor_tile_entity`, `neighbor_tile_category`, `neighbor_poi_entities`, and `neighbor_poi_categories`. Neighbours follow the level-18 8-neighbourhood.

## Expert annotations

The Excel workbook configured by `paths.expert_annotations` contains a tile-category name and two expert ratings. Ratings are 1 for complete match, 0 for partial match, and −1 for complete mismatch. Configure the three column names under `columns.expert_category`, `columns.expert_1`, and `columns.expert_2`.

## Optional Word2Vec matrix

A supplied `.npy` matrix must have one row per feature-archive trajectory in exactly the same group order. If omitted, Word2Vec is fitted from training-partition PC sequences using the settings in the configuration file.
