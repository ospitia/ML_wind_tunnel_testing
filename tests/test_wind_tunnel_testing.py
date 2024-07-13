import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from wind_tunnel_testing.config import RAW_DATA_DIR, RANDOM_SEED
from wind_tunnel_testing.dataset import load_ml_test_data
from wind_tunnel_testing.modeling import (
    feature_selection,
    generate_grid_sample_points,
    optimization,
    optimize_wind_tunnel_visit_points,
    predict,
    train_rf,
)

np.random.seed(RANDOM_SEED)


def test_load_ML_test_data(pytester):
    data = load_ml_test_data(xlsx_data_filename="ML_test", directory=RAW_DATA_DIR)

    assert isinstance(data, dict), "output is not a dictionary"
    assert len(data) == 6, "unexpected nuber of inputs"


def test_train_rf(pytester):
    x = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(x).ravel() + np.random.randn(80) * 0.1
    model = train_rf(x, np.expm1(y))

    assert isinstance(model, RandomForestRegressor), "unexpected model type"


def test_predict(pytester):
    x = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(x).ravel() + np.random.randn(80) * 0.1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = train_rf(x_train, np.expm1(y_train))

    y_pred = predict(model, x_test)
    mse = mean_squared_error(y_test, np.log1p(y_pred))

    assert isinstance(y_pred, np.ndarray), "Unexpected predictions type"
    assert y_pred.shape == y_test.shape, "Predictions shape mismatch"
    assert mse <= 0.1, "MSE is too high"


def test_optimization(pytester):
    x = np.sort(5 * np.random.rand(80, 2), axis=0)
    y = np.sin(x[:, 0]) + np.cos(x[:, 1]) + np.random.randn(80) * 0.1  # Adding some noise

    model = train_rf(x, np.expm1(y))
    optimal = optimization(model, bounds=[(0.01, 0.2), (0.01, 0.344)])

    assert isinstance(optimal, np.ndarray), "Not expected dict type"
    assert len(optimal) == 2, "not expected number of coordinates"


def test_feature_selection(pytester):
    x = np.sort(5 * np.random.rand(44, 100), axis=0)
    y = np.sin(x[:, 0]) + np.cos(x[:, 1]) + np.random.randn(44) * 0.1  # Adding some noise

    model = train_rf(x, np.expm1(y))
    number_features = 20
    selected_features = feature_selection(model, number_features)

    assert len(selected_features) <= number_features, "Unexpected number of features"
    assert np.issubdtype(
        type(selected_features), np.integer
    ), "Selected features indices should be integers"
    assert np.all(
        selected_features < x.shape[1]
    ), "Selected feature indices exceed number of features"


def test_generate_grid_sample_points(pytester):
    n_samples = 20
    bounds = [(0.01, 0.2), (0.01, 0.344)]
    samples = generate_grid_sample_points(bounds=bounds, n_samples=n_samples)

    assert samples.shape == (n_samples, len(bounds)), "Sample shape mismatch"
    assert samples[:, 0].min() >= bounds[0][0], "Variable 1 out of min bound"
    assert samples[:, 0].max() <= bounds[0][1], "Variable 1 out of max bound"
    assert samples[:, 1].min() >= bounds[1][0], "Variable 2 out of min bound"
    assert samples[:, 1].max() <= bounds[1][1], "Variable 2 out of max bound"


def test_optimize_wind_tunnel_visit_points(pytester):
    n_samples = 20
    distance_matrix = np.random.uniform(low=0, high=100, size=(n_samples, n_samples))
    np.fill_diagonal(distance_matrix, 0)
    route = optimize_wind_tunnel_visit_points(distance_matrix)

    assert len(route) == n_samples, "Route length mismatch"
    assert len(set(route)) == n_samples, "Route contains duplicates"
    assert all(0 <= node < n_samples for node in route), "Route contains invalid node indices"
