import numpy as np
import pytest

from ML_LC_Classifier import (
    bayesian_smooth,
    classify_from_probs,
    local_mean_variance,
    local_variance,
    logits_to_probs,
    probs_to_logits,
)


def test_logit_probability_conversion_clips_probability_endpoints():
    probabilities = np.array([[[0.0, 0.25]], [[1.0, 0.75]]])

    logits = probs_to_logits(probabilities)
    restored = logits_to_probs(logits)

    assert np.all(np.isfinite(logits))
    np.testing.assert_allclose(restored.sum(axis=0), 1.0)
    assert restored[0, 0, 0] > 0
    assert restored[1, 0, 0] < 1


def test_bayesian_smooth_preserves_constant_probabilities():
    probabilities = np.empty((2, 5, 4), dtype=np.float64)
    probabilities[0] = 0.7
    probabilities[1] = 0.3

    result = bayesian_smooth(probabilities, window_size=3)

    np.testing.assert_allclose(result[0], 0.7)
    np.testing.assert_allclose(result[1], 0.3)
    np.testing.assert_allclose(result.sum(axis=0), 1.0)


def test_smoothing_accepts_class_last_arrays_and_per_class_smoothness():
    probabilities = np.full((5, 5, 2), 0.5, dtype=np.float64)
    probabilities[2, 2] = [0.95, 0.05]

    result = bayesian_smooth(
        probabilities,
        smoothness={"forest": 2.0, "water": 0.5},
        class_bands=["forest", "water"],
        window_size=3,
        class_axis=-1,
    )
    variance = local_variance(probabilities, window_size=3, class_axis=-1)

    assert result.shape == probabilities.shape
    assert variance.shape == probabilities.shape
    np.testing.assert_allclose(result.sum(axis=-1), 1.0)
    assert not np.allclose(result[2, 2], probabilities[2, 2])


def test_local_mean_variance_uses_input_for_empty_selected_neighborhood():
    logits = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    mean, variance = local_mean_variance(logits, window_size=3)

    assert np.isnan(mean).all()
    np.testing.assert_array_equal(variance, np.ones_like(variance))


def test_classification_maps_values_and_marks_all_invalid_pixels():
    probabilities = np.array(
        [
            [[0.2, np.nan], [np.nan, 0.7]],
            [[0.8, np.nan], [np.nan, 0.3]],
        ]
    )

    result = classify_from_probs(probabilities, class_values=[4, 9])

    np.testing.assert_array_equal(result, [[9, -1], [-1, 4]])


@pytest.mark.parametrize("window_size", [0, 2, -3])
def test_smoothing_rejects_invalid_window_sizes(window_size):
    with pytest.raises(ValueError, match="positive odd"):
        bayesian_smooth(np.full((2, 3, 3), 0.5), window_size=window_size)
