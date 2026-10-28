import numpy as np
import pytest
from app.beatmaps import HitObject
from app.ml.audio import FRAME_MS
from app.ml.motif_memory import spectral_signature, link_spans, transform_geometry, MotifMemory


def test_ordered_spectrum_not_energy_identifies_distant_repeat():
    rng = np.random.default_rng(8)
    phrase = rng.normal(-0.3, 0.2, (160, 64)).astype(np.float32)
    different = phrase[:, ::-1].copy()
    # Means/energy are identical, but the ordered spectral bands differ.
    assert np.allclose(phrase.mean(1), different.mean(1))
    mel = np.concatenate([phrase, different, np.full((1600, 64), -1), phrase])
    spans = [
        (0, 160 * FRAME_MS),
        (160 * FRAME_MS, 320 * FRAME_MS),
        (1920 * FRAME_MS, 2080 * FRAME_MS),
    ]
    links = link_spans(mel, spans)
    assert len(links) == 1
    assert links[0].reference_start == 0
    assert links[0].similarity == pytest.approx(1)
    assert spectral_signature(np.zeros((160, 64))) is None


@pytest.mark.parametrize("repeats", [1, 2, 3, 4])
def test_whole_slider_transform_preserves_controls_and_repeat_endpoint(repeats):
    curve = [[100, 100], [160, 80], [210, 160], [270, 120]]
    obj = HitObject(100, 100, 1000, True, 500, curve=curve, repeats=repeats)
    points, tail = transform_geometry(obj, [100, 100], [250, 200], np.pi / 2, 0.8)
    distances = np.linalg.norm(np.array(curve)[:, None] - np.array(curve)[None, :], axis=-1)
    assert np.allclose(np.linalg.norm(points[:, None] - points[None, :], axis=-1), distances * 0.8)
    assert np.allclose(tail, points[-1] if repeats % 2 else points[0])
    assert obj.curve == curve


def test_different_rhythm_remains_a_valid_alternative():
    from app.ml.motif_memory import MotifLink

    memory = MotifMemory([MotifLink(10000, 14000, 0, 4000, 1)])
    history = [
        HitObject(x, 100, time, end_x=x, end_y=100)
        for time, x in [(100, 100), (600, 160), (1100, 220), (10100, 200), (10600, 260)]
    ]
    assert memory.proposals(11200, history) == []
    proposals = memory.proposals(11100, history)
    assert len(proposals) == 1
    assert np.allclose(proposals[0][:2] * [512, 384], [320, 100])
    matching = HitObject(320, 100, 11100, end_x=320, end_y=100)
    assert memory.cost(matching, history) == pytest.approx(0)
    alternate = HitObject(320, 100, 11100, True, repeats=2)
    assert memory.cost(alternate, history) == 0


def test_motif_spans_cross_model_windows_without_inventing_meter():
    from app.timing import TimingPoint
    from app.ml.motif_memory import analysis_spans

    spans = analysis_spans(np.zeros((4000, 64)), [TimingPoint(0, 500)])
    assert (8000, 12000) in spans  # Crosses the 11.89 s central model window.
    assert analysis_spans(np.zeros((4000, 64)), [TimingPoint(0, 500, beat_confidence=0.3)]) == []


def test_motif_proposal_shares_candidate_budget_and_ordinary_guards():
    import torch
    from app.ml.candidate_selection import rank_candidates
    from app.ml.motif_memory import MotifLink
    from test_generation_trace import fake_model, decoder_output, SETTINGS

    memory = MotifMemory([MotifLink(10000, 14000, 0, 4000, 1)])
    history = [
        HitObject(x, 100, time, end_x=x, end_y=100)
        for time, x in [(100, 100), (600, 160), (1100, 220), (10100, 200), (10600, 260)]
    ]
    settings = {**SETTINGS, "slider_bias": 0}
    candidates = rank_candidates(
        fake_model(),
        decoder_output(),
        11100,
        12000,
        500,
        settings,
        history,
        np.random.default_rng(42),
        torch.device("cpu"),
        motif_memory=memory,
    )
    assert len(candidates) <= settings["candidates"]
    assert any(candidate.obj.x == 320 and candidate.obj.y == 100 for candidate in candidates)
    assert all(
        0 <= candidate.obj.x <= 512 and 0 <= candidate.obj.y <= 384 for candidate in candidates
    )


def test_empty_motif_memory_preserves_all_candidate_decisions_and_random_state():
    import torch
    from dataclasses import asdict
    from app.ml.candidate_selection import rank_candidates
    from test_generation_trace import fake_model, decoder_output, SETTINGS

    plain_rng, motif_rng = np.random.default_rng(42), np.random.default_rng(42)
    args = (fake_model(), decoder_output(), 1000, 2000, 500, SETTINGS, [])
    plain = rank_candidates(*args, plain_rng, torch.device("cpu"))
    motif = rank_candidates(*args, motif_rng, torch.device("cpu"), motif_memory=MotifMemory([]))
    assert [asdict(candidate.obj) for candidate in plain] == [
        asdict(candidate.obj) for candidate in motif
    ]
    assert [candidate.score for candidate in plain] == [candidate.score for candidate in motif]
    assert plain_rng.bit_generator.state == motif_rng.bit_generator.state
