"""Separate diagnostics without redefining the existing checkpoint objective."""

GROUPS = {
    "rhythm": ("onset", "beat", "density", "tempo"),
    "geometry": ("geometry",),
    "structure": ("kind", "duration", "repeats", "combo"),
    "movement": ("motion", "flow", "compatibility"),
    "pattern": ("gap",),
}


def grouped_losses(terms):
    return {
        name: sum(terms.get(key, 0) for key in keys)
        for name, keys in GROUPS.items()
        if any(key in terms for key in keys)
    }


def update_quality_bests(metadata, metric):
    bests = metadata.setdefault("quality_bests", {})
    for group, value in metric.get("validation_groups", {}).items():
        if group not in bests or value < bests[group]["loss"]:
            bests[group] = {"epoch": metric["epoch"], "loss": value}


def training_summary(metadata, control, epoch, stop=False):
    return {
        "best_epoch": metadata["best_epoch"],
        "best_loss": None if metadata["best_epoch"] == 0 else control.best,
        "last_epoch": epoch,
        "next_learning_rate": control.lr,
        "stale_epochs": control.stale,
        "patience": control.patience,
        "stop_reason": "early_stopping" if stop else None,
        "training_songs": metadata["training_songs"],
        "validation_songs": metadata["validation_songs"],
        "test_songs": metadata["test_songs"],
    }
