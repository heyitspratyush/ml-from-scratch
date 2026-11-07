
import numpy as np
import pytest

from src.gradients import calculate_gradients


def test_multiple_features():

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4]
    ])

    y = np.array([
        [3],
        [5],
        [7]
    ])

    y_pred = np.array([
        [1],
        [2],
        [3]
    ])

    dw, db = calculate_gradients(X, y, y_pred)

    expected_dw = np.array([
        [-40 / 3],
        [-58 / 3]
    ])

    expected_db = -6

    np.testing.assert_allclose(dw, expected_dw)
    assert db == pytest.approx(expected_db)

    assert dw.shape == (2, 1)
    assert np.isscalar(db)


def test_mismatched_shapes():

    X = np.array([[1], [2], [3]])
    y = np.array([[3], [5], [7]])
    y_pred = np.array([[1, 2, 3]])

    with pytest.raises(ValueError):
        calculate_gradients(X, y, y_pred)


def test_observation_count_mismatch():

    X = np.array([[1, 2], [2, 3]])
    y = np.array([[3], [5], [7]])
    y_pred = np.array([[1], [2], [3]])

    with pytest.raises(ValueError):
        calculate_gradients(X, y, y_pred)


def test_invalid_X_dimensions():

    X = np.array([1, 2, 3])
    y = np.array([[3], [5], [7]])
    y_pred = np.array([[1], [2], [3]])

    with pytest.raises(ValueError):
        calculate_gradients(X, y, y_pred)


def test_invalid_y_dimensions():

    X = np.array([[1], [2], [3]])
    y = np.array([3, 5, 7])
    y_pred = np.array([[1], [2], [3]])

    with pytest.raises(ValueError):
        calculate_gradients(X, y, y_pred)
