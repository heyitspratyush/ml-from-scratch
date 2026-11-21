
def update_parameters(w, b, dw, db, alpha):
    if alpha <0:
        raise ValueError("Learning rate must be non-negative")  
    w = w - alpha * dw
    b = b - alpha * db
    return w, b
