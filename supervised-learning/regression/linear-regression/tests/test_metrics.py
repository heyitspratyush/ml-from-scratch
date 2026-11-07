
import numpy as np
import pytest

from src.metrics import mean_squared_error


# TEST 1: Actual and predicted values are identical
def test_identical_values():

    y = np.array([10, 20])
    y_pred = np.array([10, 20])

    assert mean_squared_error(y, y_pred) == 0.0


# TEST 2: Multiple values with nonzero errors
def test_nonzero_errors():

    y = np.array([10, 20])
    y_pred = np.array([8, 18])

    assert mean_squared_error(y, y_pred) == 4.0


# TEST 3: Different array shapes should raise ValueError
def test_mismatched_shapes():

    y = np.array([[10], [20]])
    y_pred = np.array([10, 20])

    with pytest.raises(ValueError):
        mean_squared_error(y, y_pred)
