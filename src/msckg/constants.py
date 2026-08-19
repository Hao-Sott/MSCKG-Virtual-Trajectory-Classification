from collections import OrderedDict

CLASS_LABELS = (0, 2, 3, 4, 6, 12)
CLASS_NAMES = OrderedDict(
    (
        (0, "food supervision"),
        (2, "tourism"),
        (3, "mechanical service"),
        (4, "weather monitoring"),
        (6, "hotel"),
        (12, "sport"),
    )
)
FEATURE_GROUPS = ("T", "P", "TC", "PC")
FEATURE_COMBINATIONS = OrderedDict(
    (
        ("PC", ("PC",)),
        ("TC", ("TC",)),
        ("P", ("P",)),
        ("T", ("T",)),
        ("T+P", ("T", "P")),
        ("TC+PC", ("TC", "PC")),
        ("P+TC", ("P", "TC")),
        ("T+PC", ("T", "PC")),
        ("T+P+TC", ("T", "P", "TC")),
        ("T+P+PC", ("T", "P", "PC")),
        ("T+TC+PC", ("T", "TC", "PC")),
        ("P+TC+PC", ("P", "TC", "PC")),
        ("T+P+TC+PC", ("T", "P", "TC", "PC")),
    )
)
KGE_MODELS = ("TransE_L1", "TransE_L2", "TransR", "ComplEx", "RotatE", "DistMult", "RESCAL")
ABLATIONS = OrderedDict(
    (
        ("full", ()),
        ("no_t_tc", ("w_", "wc_")),
        ("no_tc", ("wc_",)),
        ("no_p_pc", ("p_", "pc1_", "pc2_")),
        ("no_pc", ("pc1_", "pc2_")),
    )
)
ABLATION_LABELS = {
    "full": "Full",
    "no_t_tc": "No T/TC",
    "no_tc": "No TC",
    "no_p_pc": "No P/PC",
    "no_pc": "No PC",
}

