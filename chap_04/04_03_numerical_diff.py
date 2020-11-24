# %% numeric diff (bad version)
def numerical_diff(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h


# %% numeric diff
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)
