"""Optional depth-two search, with at most three fully repaired branch histories."""

from .generation_trace import object_snapshot

LOOKAHEAD_WIDTH = 3


def choose_continuation(choices, history, evaluate_next, trace=None):
    ordered = sorted(choices, key=lambda choice: -choice[0])[:LOOKAHEAD_WIDTH]
    compared = []
    for branch, choice in enumerate(ordered):
        score, candidate, times, _skipped_time = choice
        branch_history = history + [candidate.obj]
        # The callback sees the actual slider tail and repeat parity of this branch.
        following = evaluate_next(branch_history, times, branch)
        continuation = following[0] if following else None
        normalized = (score + continuation.score) / 2 if continuation is not None else score
        compared.append((normalized, choice))
        if trace is not None:
            trace.record(
                "continuation_score",
                time=candidate.obj.time,
                candidate_id=candidate.identifier,
                branch=branch,
                first_score=score,
                second_score=continuation.score if continuation is not None else None,
                mean_action_score=normalized,
                history_tail=object_snapshot(candidate.obj),
                next_object=object_snapshot(continuation.obj) if continuation is not None else None,
                next_repair=continuation.repair if continuation is not None else None,
            )
    return max(compared, key=lambda item: item[0])[1]
