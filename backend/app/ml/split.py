import re
import unicodedata


def identity(record):
    def normalize(value):
        return re.sub(r"\W+", "", unicodedata.normalize("NFKC", value).casefold())

    artist, title = normalize(record.get("artist", "")), normalize(record.get("title", ""))
    if not artist or not title or artist == "unknown" or title == "untitled":
        return None
    return f"{artist}:{title}"


def song_groups(records):
    """Union by audio bytes OR artist/title to keep alternate encodings/difficulties together."""
    parents = list(range(len(records)))
    seen = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, record in enumerate(records):
        keys = [("hash", record["audio_hash"])]
        name = identity(record)
        if name:
            keys.append(("name", name))
        for key in keys:
            if key in seen:
                parents[root(index)] = root(seen[key])
            seen[key] = index
    return [str(root(index)) for index in range(len(records))]
