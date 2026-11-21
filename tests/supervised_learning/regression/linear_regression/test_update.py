from ml_from_scratch.supervised_learning.regression.linear_regression.components.update import update_parameters
import numpy as np 
import pytest

def test_update_parameters():
    w = 2.0
    b = 1.0
    dw = -4.0
    db = -2.0
    alpha = 0.1
    new_w,new_b = update_parameters(w,b,dw,db,alpha)
    np.testing.assert_allclose(new_w,2.4)
    np.testing .assert_allclose(new_b,1.2)  

def test_update_multiple_features():
    w = np.array([[1.0],[2.0]])
    b = 1.0
    dw = np.array([[-0.5],[-1.0]])
    db = -0.5
    alpha = 0.1
    new_w,new_b = update_parameters(w,b,dw,db,alpha)
    expected_w = np.array([[1.05],[2.1]])
    expected_b = 1.05
    np.testing.assert_allclose(new_w,expected_w)
    np.testing.assert_allclose(new_b,expected_b)

def test_update_does_not_modify_original_weights():
    w = np.array([[1.0],[2.0]])
    b = 1.0
    dw = np.array([[-0.5],[-1.0]])
    db = -0.5
    alpha = 0.1
    original_w = w.copy()
    original_b = b
    new_w,new_b = update_parameters(w,b,dw,db,alpha)
    assert np.array_equal(w,original_w)
    assert b == original_b

def test_update_with_zero_gradients():
    w = np.array([[1.0],[2.0]])
    b = 1.0
    dw = np.array([[0.0],[0.0]])
    db = 0.0
    alpha = 0.1
    new_w,new_b = update_parameters(w,b,dw,db,alpha)
    assert np.array_equal(new_w,w)
    assert new_b == b

def test_update_with_large_learning_rate():
    w  =np.array([[1.0],[2.0]])
    b = 1.0
    dw = np.array([[-0.5],[-1.0]])
    db = -0.5
    alpha = 0
    new_w,new_b = update_parameters(w,b,dw,db,alpha)
    assert np.array_equal(new_w,w)
    assert new_b == b


def test_update_rejects_negative_learning_rate():
    w = np.array([[1.0], [2.0]])
    b = 1.0

    dw = np.array([[0.5], [1.0]])
    db = 0.5

    alpha = -0.1

    with pytest.raises(ValueError):
        update_parameters(w, b, dw, db, alpha)