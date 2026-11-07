
import numpy as np

from src.predict import predict


def test_predictions():

    X = np.array([
        [500],
        [700],
        [900],
        [1100],
        [1300]
    ])

    w = 0.05
    b = 0

    predictions = predict(X, w, b)

    expected_predictions = np.array([
        [25],
        [35],
        [45],
        [55],
        [65]
    ])

    np.testing.assert_allclose(predictions, expected_predictions)

    assert predictions.shape == (5, 1)
