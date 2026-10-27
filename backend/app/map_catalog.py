"""The same inclusive star/search filters for pagination and selection of all matches."""

import math
from . import store
from .ml.split import song_groups


def star_rating(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(value, 2) if math.isfinite(value) and value >= 0 else None


def matching_maps(query="", minimum=None, maximum=None, per_song="all"):
    if per_song not in ("all", "easiest", "middle", "hardest"):
        raise ValueError("Unbekannte Auswahl pro Song.")
    for value in (minimum, maximum):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Sternegrenzen müssen endliche, nicht negative Zahlen sein.")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("Sterne von darf nicht größer sein als Sterne bis.")
    query = query.strip().casefold()
    records = store.all_records("map")
    # Group before filtering: a filtered-out difficulty can connect two encodings.
    groups = (
        song_groups(
            [
                {**item, "audio_hash": item.get("audio_hash") or f"missing:{item['id']}"}
                for item in records
            ]
        )
        if per_song != "all"
        else [None] * len(records)
    )
    result, candidates = [], {}
    for item, group in zip(records, groups):
        title = " ".join(
            str(item.get(key, "")) for key in ("title", "artist", "creator", "version")
        )
        rating = star_rating(item.get("star_rating"))
        if query not in title.casefold():
            continue
        if minimum is not None and (rating is None or rating < minimum):
            continue
        if maximum is not None and (rating is None or rating > maximum):
            continue
        normalized = {**item, "star_rating": rating}
        if per_song != "all":
            if rating is None:
                continue
            candidates.setdefault(group, []).append(normalized)
        result.append(normalized)
    if per_song != "all":
        chosen = set()
        for maps in candidates.values():
            ordered = sorted(maps, key=lambda item: (item["star_rating"], item["id"]))
            index = {"easiest": 0, "middle": (len(ordered) - 1) // 2, "hardest": len(ordered) - 1}[
                per_song
            ]
            chosen.add(ordered[index]["id"])
        result = [item for item in result if item["id"] in chosen]
    return result
