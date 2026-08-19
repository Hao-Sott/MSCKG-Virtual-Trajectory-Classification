import argparse
import subprocess
import sys


STAGES = (
    ("validate",),
    ("extract-tile-pixels",),
    ("classify-tiles",),
    ("build-kg",),
    ("ablate-kg",),
    ("train-kge", "--model", "all", "--variant", "full"),
    *(("train-kge", "--model", "DistMult", "--variant", variant) for variant in ("no_t_tc", "no_tc", "no_p_pc", "no_pc")),
    *(("prepare-features", "--model", model, "--variant", "full") for model in ("TransE_L1", "TransE_L2", "TransR", "ComplEx", "RotatE", "DistMult", "RESCAL")),
    *(("prepare-features", "--model", "DistMult", "--variant", variant) for variant in ("no_t_tc", "no_tc", "no_p_pc", "no_pc")),
    ("select-parameters",),
    ("main",),
    ("embedding-comparison",),
    ("repeated-splits",),
    ("semantic-baselines",),
    ("classifier-comparison",),
    ("spatial-validation",),
    ("component-ablation",),
    ("expert-validation",),
    ("gru",),
    ("domain-indicators",),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=len(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    for index, stage in enumerate(STAGES[args.start : args.stop], args.start):
        command = [sys.executable, "-m", "msckg.cli", "--config", args.config, *stage]
        if stage[0] == "train-kge" and args.gpu is not None:
            command.extend(("--gpu", str(args.gpu)))
        print(index, subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
