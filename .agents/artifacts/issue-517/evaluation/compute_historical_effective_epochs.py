"""Calculate retained effective row epochs for issue #517 comparisons."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


TRAIN_STEPS = 5_000
GLOBAL_BATCH_SIZE = 8_192
SEQUENCE_PRESENTATIONS = TRAIN_STEPS * GLOBAL_BATCH_SIZE
OUTPUT_PATH = Path(
    ".agents/artifacts/issue-517/evaluation/"
    "issue517_historical_effective_epochs.csv"
)


@dataclass(frozen=True)
class TrainingDataset:
    experiment: str
    arm: str
    train_rows: int
    source: str


DATASETS = (
    TrainingDataset("issue187_v3", "CDS", 78_387_828, "issue #187 body"),
    TrainingDataset("issue187_v3", "3-prime UTR", 10_305_306, "issue #187 body"),
    TrainingDataset("issue187_v3", "ncRNA exon", 15_277_064, "issue #187 body"),
    TrainingDataset("issue187_v3", "TSS / 5-prime UTR", 8_124_514, "issue #187 body"),
    TrainingDataset("issue187_v3", "Enhancer", 96_639_800, "issue #187 body"),
    TrainingDataset("issue187_v3", "Background", 15_145_768, "issue #187 body"),
    TrainingDataset("issue232_v4", "CDS", 57_591_204, "issue #232 body"),
    TrainingDataset("issue232_v4", "3-prime UTR", 12_586_492, "issue #232 body"),
    TrainingDataset("issue232_v4", "ncRNA exon", 16_319_886, "issue #232 body"),
    TrainingDataset("issue232_v4", "TSS / 5-prime UTR", 11_281_780, "issue #232 body"),
    TrainingDataset("issue232_v4", "Enhancer", 88_030_162, "issue #232 body"),
    TrainingDataset("issue232_v4", "Background", 38_070_756, "issue #232 body"),
    TrainingDataset(
        "issue517_annotation_first",
        "CDS",
        46_882_278,
        "issue #517 logbook FAS-517-009",
    ),
    TrainingDataset(
        "issue517_annotation_first",
        "3-prime UTR",
        11_364_040,
        "issue #517 logbook FAS-517-009",
    ),
    TrainingDataset(
        "issue517_annotation_first",
        "ncRNA exon",
        6_209_692,
        "issue #517 logbook FAS-517-009",
    ),
    TrainingDataset(
        "issue517_annotation_first",
        "TSS / 5-prime UTR",
        7_577_794,
        "issue #517 logbook FAS-517-009",
    ),
    TrainingDataset(
        "issue517_annotation_first",
        "Enhancer",
        25_364_652,
        "issue #517 logbook FAS-517-009",
    ),
    TrainingDataset(
        "issue517_gpn_uniform",
        "CDS",
        71_002_636,
        "config/gpn_star_p_publication.yaml",
    ),
    TrainingDataset(
        "issue517_gpn_uniform",
        "3-prime UTR",
        20_538_748,
        "config/gpn_star_p_publication.yaml",
    ),
    TrainingDataset(
        "issue517_gpn_uniform",
        "ncRNA exon",
        20_026_822,
        "config/gpn_star_p_publication.yaml",
    ),
    TrainingDataset(
        "issue517_gpn_uniform",
        "TSS / 5-prime UTR",
        14_650_866,
        "config/gpn_star_p_publication.yaml",
    ),
    TrainingDataset(
        "issue517_gpn_uniform",
        "Enhancer",
        131_467_840,
        "config/gpn_star_p_publication.yaml",
    ),
    TrainingDataset(
        "issue517_gpn_uniform",
        "Background",
        77_330_858,
        "config/gpn_star_p_publication.yaml",
    ),
    TrainingDataset(
        "issue517_phylop_uniform",
        "CDS",
        69_483_774,
        "config/phylop_uniform_publication.yaml",
    ),
    TrainingDataset(
        "issue517_phylop_uniform",
        "3-prime UTR",
        14_496_656,
        "config/phylop_uniform_publication.yaml",
    ),
    TrainingDataset(
        "issue517_phylop_uniform",
        "ncRNA exon",
        20_790_530,
        "config/phylop_uniform_publication.yaml",
    ),
    TrainingDataset(
        "issue517_phylop_uniform",
        "TSS / 5-prime UTR",
        11_580_082,
        "config/phylop_uniform_publication.yaml",
    ),
    TrainingDataset(
        "issue517_phylop_uniform",
        "Enhancer",
        79_725_424,
        "config/phylop_uniform_publication.yaml",
    ),
    TrainingDataset(
        "issue517_phylop_uniform",
        "Background",
        52_119_732,
        "config/phylop_uniform_publication.yaml",
    ),
    TrainingDataset(
        "issue326_targeted",
        "Enhancer Arm A",
        76_710_830,
        "HF dataset revision 45073f2ed8d7edbc25d089d51dd0b835ee52fc7c",
    ),
    TrainingDataset(
        "issue326_targeted",
        "Enhancer Arm B",
        62_734_288,
        "HF dataset revision 8b224bb7becc1de4210bc574f34d6c4ed1db3084",
    ),
    TrainingDataset(
        "issue351_targeted",
        "Enhancer tiled",
        10_992_626,
        "HF dataset revision 9ea662689ef520a956669f2bbdaf3301f19b957b",
    ),
    TrainingDataset(
        "issue351_targeted",
        "Enhancer centered",
        4_237_620,
        "HF dataset revision 3e01749d83ad69ad6b1f3d8ebe12315ad17d60f4",
    ),
)


def main() -> None:
    assert TRAIN_STEPS > 0
    assert GLOBAL_BATCH_SIZE > 0
    assert len(DATASETS) == 33
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "experiment",
                "arm",
                "train_rows",
                "optimizer_steps",
                "global_batch_size",
                "sequence_presentations",
                "effective_row_epochs",
                "source",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for dataset in DATASETS:
            assert dataset.train_rows > 0
            writer.writerow(
                {
                    "experiment": dataset.experiment,
                    "arm": dataset.arm,
                    "train_rows": dataset.train_rows,
                    "optimizer_steps": TRAIN_STEPS,
                    "global_batch_size": GLOBAL_BATCH_SIZE,
                    "sequence_presentations": SEQUENCE_PRESENTATIONS,
                    "effective_row_epochs": (
                        f"{SEQUENCE_PRESENTATIONS / dataset.train_rows:.6f}"
                    ),
                    "source": dataset.source,
                }
            )


if __name__ == "__main__":
    main()
