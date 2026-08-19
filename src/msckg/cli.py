import argparse
import json
from pathlib import Path

from .analysis import domain_indicators, expert_validation
from .baselines import run_classifier_comparison, run_semantic_baselines
from .config import load_config, resolve_path
from .constants import ABLATIONS, FEATURE_GROUPS, KGE_MODELS
from .evaluation import repeated_splits, run_feature_combinations, select_xgboost_parameters
from .experiments import component_ablation, embedding_comparison, spatial_expansion_comparison
from .features import build_feature_archive
from .gru import run_gru_experiment
from .io import load_feature_archive
from .kg import build_knowledge_graph, make_ablation_graphs
from .kge import align_embedding, train_dglke
from .plots import ablation_heatmap, class_accuracy_heatmap, domain_scatter, embedding_heatmap, gain_importance, grouped_bars, gru_line, poi_hit, repeated_uncertainty, ripley_curves, spatial_expansion
from .spatial import expand_training_trajectories, run_spatial_validation
from .tile_taxonomy import run_tile_taxonomy
from .tile_pixels import extract_tile_pixels


def xgb(config):
    return config["classification"]["xgboost"]


def output(config, name):
    return resolve_path(config, "outputs") / name


def command_validate(config, _):
    required = ["trajectory", "tile_index", "tile_image_root", "color_templates", "tile_taxonomy", "pois"]
    optional = ["temporal_trajectory", "expert_annotations"]
    report = {key: {"path": str(resolve_path(config, key)), "exists": resolve_path(config, key).exists(), "required": key in required} for key in required + optional}
    print(json.dumps(report, indent=2))
    if not all(report[key]["exists"] for key in required):
        raise SystemExit(2)


def command_tiles(config, args):
    target = Path(args.output or resolve_path(config, "tiles"))
    run_tile_taxonomy(resolve_path(config, "tile_pixels"), resolve_path(config, "tile_taxonomy"), target)


def command_tile_pixels(config, _):
    extract_tile_pixels(resolve_path(config, "tile_index"), resolve_path(config, "tile_image_root"), resolve_path(config, "color_templates"), resolve_path(config, "tile_pixels"), config["columns"]["tile_image_path"], config["project"]["jobs"])


def command_build_kg(config, _):
    result = build_knowledge_graph(resolve_path(config, "tiles"), resolve_path(config, "pois"), resolve_path(config, "kg_full"), config["columns"], config["knowledge_graph"]["poi_proximity_m"])
    print(json.dumps(result, indent=2))


def command_ablate_kg(config, _):
    make_ablation_graphs(resolve_path(config, "kg_full"), resolve_path(config, "kg_variants"))


def command_train_kge(config, args):
    settings = config["knowledge_graph"]
    models = KGE_MODELS if args.model == "all" else (args.model,)
    variants = tuple(ABLATIONS) if args.variant == "all" else (args.variant,)
    for variant in variants:
        kg_dir = resolve_path(config, "kg_full") if variant == "full" else resolve_path(config, "kg_variants") / variant
        for model in models:
            if variant != "full" and model != "DistMult":
                continue
            target = resolve_path(config, "embeddings") / ("models" if variant == "full" else "ablations") / (model if variant == "full" else variant)
            existing = target / "entity_embeddings_100d.npy"
            if existing.exists() and not args.force and not args.dry_run:
                if variant != "full" and not (target / "entity_embeddings_full_id_order.npy").exists():
                    align_embedding(existing, kg_dir / "entity2id.txt", resolve_path(config, "kg_full") / "entity2id.txt", target / "entity_embeddings_full_id_order.npy")
                print(json.dumps({"variant": variant, "model": model, "status": "skipped_existing"}, indent=2))
                continue
            command, metadata = train_dglke(kg_dir, target, model, settings, args.gpu, config["project"]["jobs"], args.dry_run)
            print(json.dumps({"variant": variant, "model": model, **metadata, "command": command}, indent=2))
            if not args.dry_run and variant != "full":
                align_embedding(target / "entity_embeddings_100d.npy", kg_dir / "entity2id.txt", resolve_path(config, "kg_full") / "entity2id.txt", target / "entity_embeddings_full_id_order.npy")


def command_features(config, args):
    if args.variant == "full":
        embedding = resolve_path(config, "embeddings") / "models" / args.model / "entity_embeddings_100d.npy"
        target = resolve_path(config, "features") / "models" / f"{args.model}.npz"
        active = FEATURE_GROUPS
    else:
        embedding = resolve_path(config, "embeddings") / "ablations" / args.variant / "entity_embeddings_full_id_order.npy"
        target = resolve_path(config, "features") / "ablations" / f"{args.variant}.npz"
        active = {
            "no_t_tc": ("P", "PC"),
            "no_tc": ("T", "P", "PC"),
            "no_p_pc": ("T", "TC"),
            "no_pc": ("T", "P", "TC"),
        }[args.variant]
    build_feature_archive(resolve_path(config, "trajectory"), embedding, target, config["columns"], config["knowledge_graph"]["embedding_dim"], active)
    print(target)


def default_features(config):
    return resolve_path(config, "features") / "models" / "DistMult.npz"


def command_main(config, args):
    archive = Path(args.features or default_features(config))
    run_feature_combinations(load_feature_archive(archive), xgb(config), output(config, "main"), config["project"]["seed"], config["classification"]["test_size"], config["project"]["jobs"])


def command_select_parameters(config, args):
    archive = Path(args.features or default_features(config))
    select_xgboost_parameters(load_feature_archive(archive), xgb(config), output(config, "parameter_selection"), config["project"]["seed"], config["classification"]["test_size"], config["project"]["jobs"])


def command_embeddings(config, _):
    embedding_comparison(resolve_path(config, "features") / "models", xgb(config), output(config, "embedding_comparison"), config["project"]["seed"], config["classification"]["test_size"], config["project"]["jobs"])


def command_repeated(config, args):
    archive = Path(args.features or default_features(config))
    repeated_splits(load_feature_archive(archive), xgb(config), output(config, "repeated_splits"), config["classification"]["repeated_seeds"], config["classification"]["test_size"], config["project"]["jobs"])


def command_baselines(config, args):
    archive = Path(args.features or default_features(config))
    run_semantic_baselines(resolve_path(config, "trajectory"), load_feature_archive(archive), config["columns"], xgb(config), output(config, "semantic_baselines"), args.word2vec, config["baselines"]["word2vec"], config["project"]["seed"], config["classification"]["test_size"], config["project"]["jobs"])


def command_classifiers(config, args):
    archive = Path(args.features or default_features(config))
    combination = None if args.combination == "all" else args.combination
    run_classifier_comparison(load_feature_archive(archive), xgb(config), output(config, "classifier_comparison"), combination, config["project"]["seed"], config["classification"]["test_size"], config["project"]["jobs"])


def command_spatial(config, args):
    archive = Path(args.features or default_features(config))
    run_spatial_validation(resolve_path(config, "trajectory"), load_feature_archive(archive), config["columns"], xgb(config), output(config, "spatial_validation"), args.combination, config["project"]["jobs"], config["spatial"]["block_quantiles"], config["spatial"]["grid_size_degrees"])


def command_ablation(config, _):
    component_ablation(resolve_path(config, "features") / "ablations", xgb(config), output(config, "component_ablation"), config["project"]["seed"], config["classification"]["test_size"], config["project"]["jobs"])


def command_expand(config, args):
    groups = [int(value) for value in Path(args.train_groups).read_text(encoding="utf-8").splitlines() if value.strip()]
    expand_training_trajectories(resolve_path(config, "trajectory"), args.tile_context, groups, config["columns"], args.output, args.target_per_class, config["project"]["seed"])


def command_expansion_comparison(config, args):
    spatial_expansion_comparison(args.expanded, args.unexpanded, xgb(config), output(config, "spatial_expansion"), config["classification"]["repeated_seeds"], config["classification"]["test_size"], config["project"]["jobs"])


def command_expert(config, _):
    expert_validation(resolve_path(config, "expert_annotations"), config["columns"], output(config, "expert_validation"))


def command_gru(config, args):
    embedding = resolve_path(config, "embeddings") / "models" / "DistMult" / "entity_embeddings_100d.npy"
    run_gru_experiment(resolve_path(config, "temporal_trajectory"), embedding, config["columns"], xgb(config), config["gru"], output(config, "gru"), config["project"]["seed"], config["project"]["jobs"], config["classification"]["test_size"], args.device)


def command_indicators(config, args):
    domain_indicators(resolve_path(config, "trajectory"), config["columns"], output(config, "domain_indicators"), args.class_accuracy, args.feature_combination)


def command_plot(_, args):
    functions = {
        "embedding-heatmap": embedding_heatmap,
        "class-heatmap": class_accuracy_heatmap,
        "ablation-heatmap": ablation_heatmap,
        "bars": grouped_bars,
        "domain-scatter": domain_scatter,
        "gain": gain_importance,
        "ripley": ripley_curves,
        "spatial-expansion": spatial_expansion,
    }
    if args.kind == "gru-line":
        gru_line(args.input, args.second_input, args.output)
    elif args.kind == "repeated":
        repeated_uncertainty(args.input, args.second_input, args.output, args.feature_combination)
    elif args.kind == "poi-hit":
        poi_hit(args.input, args.second_input, args.output)
    elif args.kind == "bars":
        grouped_bars(args.input, args.output, args.name_column)
    elif args.kind == "gain":
        gain_importance(args.input, args.output, args.feature_combination)
    else:
        functions[args.kind](args.input, args.output)


def parser():
    root = argparse.ArgumentParser(prog="msckg")
    root.add_argument("--config", default="configs/experiment.yaml")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").set_defaults(handler=command_validate)
    commands.add_parser("extract-tile-pixels").set_defaults(handler=command_tile_pixels)
    tile = commands.add_parser("classify-tiles")
    tile.add_argument("--output")
    tile.set_defaults(handler=command_tiles)
    commands.add_parser("build-kg").set_defaults(handler=command_build_kg)
    commands.add_parser("ablate-kg").set_defaults(handler=command_ablate_kg)
    train = commands.add_parser("train-kge")
    train.add_argument("--model", choices=("all",) + KGE_MODELS, default="DistMult")
    train.add_argument("--variant", choices=("all",) + tuple(ABLATIONS), default="full")
    train.add_argument("--gpu", type=int)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--force", action="store_true")
    train.set_defaults(handler=command_train_kge)
    feature = commands.add_parser("prepare-features")
    feature.add_argument("--model", choices=KGE_MODELS, default="DistMult")
    feature.add_argument("--variant", choices=tuple(ABLATIONS), default="full")
    feature.set_defaults(handler=command_features)
    for name, handler in (("select-parameters", command_select_parameters), ("main", command_main), ("repeated-splits", command_repeated), ("semantic-baselines", command_baselines), ("classifier-comparison", command_classifiers), ("spatial-validation", command_spatial)):
        item = commands.add_parser(name)
        item.add_argument("--features")
        if name == "semantic-baselines":
            item.add_argument("--word2vec")
        if name in {"classifier-comparison", "spatial-validation"}:
            item.add_argument("--combination", default="all" if name == "classifier-comparison" else "T+PC")
        item.set_defaults(handler=handler)
    commands.add_parser("embedding-comparison").set_defaults(handler=command_embeddings)
    commands.add_parser("component-ablation").set_defaults(handler=command_ablation)
    expand = commands.add_parser("expand-training")
    expand.add_argument("--tile-context", required=True)
    expand.add_argument("--train-groups", required=True)
    expand.add_argument("--output", required=True)
    expand.add_argument("--target-per-class", type=int)
    expand.set_defaults(handler=command_expand)
    comparison = commands.add_parser("spatial-expansion")
    comparison.add_argument("--expanded", required=True)
    comparison.add_argument("--unexpanded", required=True)
    comparison.set_defaults(handler=command_expansion_comparison)
    commands.add_parser("expert-validation").set_defaults(handler=command_expert)
    gru = commands.add_parser("gru")
    gru.add_argument("--device")
    gru.set_defaults(handler=command_gru)
    indicators = commands.add_parser("domain-indicators")
    indicators.add_argument("--class-accuracy")
    indicators.add_argument("--feature-combination", default="T+PC")
    indicators.set_defaults(handler=command_indicators)
    plot = commands.add_parser("plot")
    plot.add_argument("--kind", choices=("embedding-heatmap", "class-heatmap", "ablation-heatmap", "bars", "gru-line", "domain-scatter", "gain", "repeated", "poi-hit", "ripley", "spatial-expansion"), required=True)
    plot.add_argument("--input", required=True)
    plot.add_argument("--second-input")
    plot.add_argument("--output", required=True)
    plot.add_argument("--name-column", default="method")
    plot.add_argument("--feature-combination", default="T+PC")
    plot.set_defaults(handler=command_plot)
    return root


def main():
    args = parser().parse_args()
    config = load_config(args.config)
    args.handler(config, args)


if __name__ == "__main__":
    main()
