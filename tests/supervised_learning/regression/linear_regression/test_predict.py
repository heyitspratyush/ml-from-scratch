
import numpy as np

from ml_from_scratch.supervised_learning.regression.linear_regression.components.predict import predict


def test_predictions():

    X = np.array([
        [500],
        [700],
        [900],
        [1100],
        [1300]
    ])

    w = np.array([[0.05]])
    b = np.array([[0]])

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

def test_predictions_multiple_features():

    X = np.array([
        [500, 1],
        [700, 2],
        [900, 3]
    ])

    w = np.array([
        [0.05],
        [2.0]
    ])

    b = 0.0

    predictions = predict(X, w, b)

    expected_predictions = np.array([
        [27.0],
        [39.0],
        [51.0]
    ])

    np.testing.assert_allclose(predictions, expected_predictions)

    assert predictions.shape == (3, 1)