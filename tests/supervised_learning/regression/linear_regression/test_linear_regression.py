import numpy as np
import pytest
from ml_from_scratch.supervised_learning.regression.linear_regression.models.linear_regression import LinearRegression



def test_model_initialization():
    model = LinearRegression(learning_rate=0.01,n_iterations=1000)
    np.testing.assert_allclose(model.learning_rate, 0.01)
    assert model.n_iterations == 1000
    assert model.w is None
    assert model.b is None

def test_prediction_before_training():
    model = LinearRegression(learning_rate=0.01,n_iterations=1000)
    with pytest.raises(RuntimeError):
        model.predict(np.array([[3],[4]]))


def test_model_training():
    X = np.array([[500],[700],[900],[1100],[1300]])
    y = np.array([[25],[35],[45],[55],[65]])
    model = LinearRegression(1e-7,1000)
    model.fit(X,y)
    predictions = model.predict(X)
    np.testing.assert_allclose(predictions,y,rtol = 1e-4,atol=1e-4)


