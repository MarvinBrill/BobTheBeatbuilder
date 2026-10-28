from dataclasses import asdict

import numpy as np

from app.beatmaps import HitObject
from app.ml.candidate_selection import Candidate
from app.ml.continuation_search import choose_continuation
from app.ml.generation_trace import GenerationTrace


def candidate(obj, score, name, repair=0):
    return Candidate(obj, np.zeros(8), np.zeros(8), score, repair, name)


def test_lookahead_avoids_bad_later_connection_using_final_repeat_endpoint():
    # Both slider heads are at x=100. A repeat returns to the head; a single traversal ends at x=300.
    repeated = candidate(
        HitObject(100, 100, 1000, True, 500, 100, 100, repeats=2), 0.1, "local-winner"
    )
    forward = candidate(
        HitObject(100, 100, 1000, True, 500, 300, 100, repeats=1), 0, "good-continuation"
    )
    history = [HitObject(80, 100, 500, end_x=80, end_y=100)]
    before = [asdict(obj) for obj in history]
    branches = []

    def following(branch_history, times, branch):
        branches.append(branch_history)
        distance = abs(300 - branch_history[-1].end_x)
        next_object = HitObject(300, 100, 1600, end_x=300, end_y=100)
        return [candidate(next_object, -distance / 100, f"next-{branch}", repair=distance / 512)]

    choices = [(item.score, item, [1000, 1600], None) for item in [repeated, forward]]
    trace = GenerationTrace()
    chosen = choose_continuation(choices, history, following, trace)
    assert choices[0][1] is repeated and chosen[1] is forward
    assert len(branches) == 2 and branches[0] is not branches[1]
    assert [asdict(obj) for obj in history] == before
    assert branches[0][-1].end_x == 100 and branches[1][-1].end_x == 300
    assert all(event["next_object"]["time"] == 1600 for event in trace.events)
    assert trace.events[0]["mean_action_score"] == (0.1 - 2) / 2


def test_search_compares_at_most_three_continuations_and_returns_original_choice():
    choices = [
        (
            float(4 - index),
            candidate(HitObject(index, 0, 1000), float(4 - index), index),
            [1000],
            None,
        )
        for index in range(8)
    ]
    visited = []

    def following(history, times, branch):
        visited.append(branch)
        return []

    chosen = choose_continuation(choices, [], following)
    assert visited == [0, 1, 2]
    assert chosen is choices[0]
