import json
import pytest
from app import difficulty, store


class Context:
    def check(self):
        pass

    def report(self, *args, **kwargs):
        pass


def fake_calculator(monkeypatch, tmp_path, results):
    tool = tmp_path / "BobDifficulty.dll"
    tool.write_bytes(b"tool")
    monkeypatch.setattr(difficulty, "DIFFICULTY", tool)
    calls = []

    class Done:
        returncode = 0
        stderr = ""

    def run(command, **kwargs):
        paths = open(command[2], encoding="utf-8").read().split("\n")
        paths = [path for path in paths if path]
        calls.append(paths)
        with open(command[3], "w", encoding="utf-8") as output:
            for path in paths:
                output.write(json.dumps({"path": path, "calculator": "test 1", **results(path)}) + "\n")
        return Done()

    monkeypatch.setattr(difficulty.subprocess, "run", run)
    return calls


def test_calculator_results_keep_order_and_failures(monkeypatch, tmp_path):
    calls = fake_calculator(
        monkeypatch,
        tmp_path,
        lambda path: {"error": "taiko"} if path.endswith("b.osu") else {"star_rating": 6.1, "aim_difficulty": 3.2, "speed_difficulty": 2.5, "mods": []},
    )
    first, second = difficulty.calculate([tmp_path / "a.osu", tmp_path / "b.osu"])
    assert first["star_rating"] == 6.1 and first["calculator"] == "test 1" and "mods" not in first
    assert second is None and len(calls) == 1


def test_library_annotation_fills_only_missing_and_keeps_lazer_rating(monkeypatch, tmp_path):
    # Maps saved by other tests in this session stay untouched (the fake fails for them).
    fake_calculator(
        monkeypatch,
        tmp_path,
        lambda path: {"star_rating": 5.0, "aim_difficulty": 2.6, "speed_difficulty": 2.1}
        if path.startswith(str(tmp_path))
        else {"error": "not part of this test"},
    )
    store.save("map", dict(id="stars-a", map_path=str(tmp_path / "a.osu"), star_rating=4.8, star_source="lazer"))
    store.save(
        "map",
        dict(
            id="stars-b", map_path=str(tmp_path / "b.osu"), star_rating=3.0, star_source="calculated",
            difficulty_attributes={"star_rating": 3.0, "calculator": "test 1"},
        ),
    )  # fmt: skip
    result = difficulty.annotate_library(Context())
    assert result["calculated"] >= 1
    rated = store.get("map", "stars-a")
    assert rated["star_rating"] == 5.0 and rated["lazer_star_rating"] == 4.8
    assert rated["star_source"] == "calculated"
    assert store.get("map", "stars-b")["star_rating"] == 3.0


def test_missing_calculator_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(difficulty, "DIFFICULTY", tmp_path / "missing.dll")
    assert not difficulty.available()
    with pytest.raises(ValueError, match="setup"):
        difficulty.calculate([tmp_path / "a.osu"])
