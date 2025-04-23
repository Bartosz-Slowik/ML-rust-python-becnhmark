from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

def train():
    X, y = load_breast_cancer(return_X_y=True)
    return LogisticRegression(max_iter=1000).fit(X, y)

def predict():
    # Fit once then predict repeatedly
    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    return model.predict(X)

def test_train(benchmark):
    result = benchmark(train)
    assert result is not None

def test_predict(benchmark):
    result = benchmark(predict)
    assert result.shape[0] > 0
