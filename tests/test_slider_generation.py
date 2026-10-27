from types import SimpleNamespace
import numpy as np
import pytest
import torch
from app.ml.slider_geometry import geometry_candidates, fit_curve_bounds, curve_diagnostics
from app.ml.playability import candidate_object
from app.ml.sequence_generation import choose_candidate
from app.ml.architecture import DURATION_BEATS

SCALE = np.array([512, 384])
CURVE = np.array([[100, 150], [150, 80], [230, 220], [300, 150]], dtype=float)
LOOP = np.array([[180, 180], [450, 30], [40, 30], [260, 180]], dtype=float)
SETTINGS = dict(cs=4, ar=9, od=7, slider_bias=1, temperature=0.8, candidates=12)


def test_uncertain_slider_sampling_preserves_the_whole_learned_shape():
    mean = (CURVE / SCALE).reshape(1, 8)
    samples = list(
        geometry_candidates(
            mean, np.full((1, 8), 0.6), np.ones(1), 100, 1.5, np.random.default_rng(8), True
        )
    )
    assert np.array_equal(samples[0][0], mean[0])
    original_distances = np.linalg.norm(CURVE[:, None] - CURVE[None, :], axis=-1)
    for geometry, _ in samples[1:]:
        points = geometry.reshape(4, 2) * SCALE
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        factor = distances[0, 3] / original_distances[0, 3]
        assert np.allclose(distances, original_distances * factor, atol=1e-5)
        assert np.exp(-0.2) - 1e-6 <= factor <= np.exp(0.2) + 1e-6
        assert not curve_diagnostics(points, 36)["unsafe"]


def test_bounds_adjustment_preserves_curvature_instead_of_clipping_handles():
    assert np.array_equal(fit_curve_bounds(CURVE, 38), CURVE)
    original = CURVE * 3 - [200, 300]
    fitted = fit_curve_bounds(original, 38)
    assert np.all(fitted >= 38) and np.all(fitted <= SCALE - 38)
    distances = np.linalg.norm(original[:, None] - original[None, :], axis=-1)
    actual = np.linalg.norm(fitted[:, None] - fitted[None, :], axis=-1)
    assert np.allclose(actual, distances * (actual[0, 3] / distances[0, 3]))


@pytest.mark.parametrize(
    "curve",
    [
        LOOP,
        np.array([[100, 180], [440, 180], [40, 180], [200, 180]]),
        np.array([[100, 180], [180, 100], [180, 100], [300, 180]]),
        np.array([[150, 150], [165, 120], [125, 120], [140, 150]]),
    ],
)
def test_pathological_slider_shapes_are_rejected(curve):
    assert curve_diagnostics(curve, 36)["unsafe"]
    assert (
        candidate_object(
            (curve / SCALE).reshape(-1), 1000, 3000, 500, True, 1, 1, False, SETTINGS, []
        )
        is None
    )


@pytest.mark.parametrize(
    "curve",
    [
        CURVE,
        np.array([[100, 200], [130, 80], [270, 80], [300, 200]]),
        np.array([[100, 180], [160, 180], [220, 180], [280, 180]]),
    ],
)
def test_ordinary_straight_arc_and_s_sliders_remain_available(curve):
    result = candidate_object(
        (curve / SCALE).reshape(-1), 1000, 3000, 500, True, 1, 1, False, SETTINGS, []
    )
    assert result is not None and result[0].slider
    assert np.array_equal(result[0].curve, curve)


def decoder_output(curve):
    raw = np.zeros((2, 17), np.float32)
    probabilities = ((curve / SCALE).reshape(-1) + 0.5) / 2
    raw[:, 1:9] = np.log(probabilities / (1 - probabilities))
    raw[:, 9:] = -0.5  # Extremely uncertain prediction.
    duration = torch.full((1, 1, len(DURATION_BEATS)), -100.0)
    duration[..., DURATION_BEATS.index(1)] = 100
    return {
        "geometry": torch.tensor(raw.reshape(1, 1, -1)),
        "kind": torch.tensor([[[-100.0, 100.0]]]),
        "duration": duration,
        "repeats": torch.tensor([[[100.0, -100.0, -100.0, -100.0]]]),
        "combo": torch.zeros(1, 1),
        "motion": torch.zeros(1, 1, 8),
        "hidden": torch.zeros(1, 1, 4),
    }


def test_full_candidate_selection_preserves_curves_and_has_a_safe_fallback():
    model = SimpleNamespace(
        config=SimpleNamespace(mixtures=2),
        compatibility=lambda hidden, geometry, motion: torch.zeros(len(geometry)),
    )
    for seed in range(10):
        obj, _ = choose_candidate(
            model,
            decoder_output(CURVE),
            1000,
            3000,
            500,
            SETTINGS,
            [],
            np.random.default_rng(seed),
            torch.device("cpu"),
        )
        assert obj.slider and not curve_diagnostics(obj.curve, 36)["unsafe"]
        # An actual curve survives; the guard must not silently flatten every slider.
        chord = np.array(obj.curve)[-1] - obj.curve[0]
        handle = np.array(obj.curve)[1] - obj.curve[0]
        assert abs(chord[0] * handle[1] - chord[1] * handle[0]) > 1000
    obj, repair = choose_candidate(
        model,
        decoder_output(LOOP),
        1000,
        3000,
        500,
        SETTINGS,
        [],
        np.random.default_rng(42),
        torch.device("cpu"),
    )
    assert not obj.slider and repair >= 1


def test_circle_position_does_not_depend_on_unused_slider_handles():
    geometry = (CURVE / SCALE).reshape(-1)
    first = candidate_object(geometry, 1000, 2000, 500, False, 0, 1, False, SETTINGS, [])[0]
    geometry[2:] = 1.5
    second = candidate_object(geometry, 1000, 2000, 500, False, 0, 1, False, SETTINGS, [])[0]
    assert (first.x, first.y) == (second.x, second.y) == (100, 150)
