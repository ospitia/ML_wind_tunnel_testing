import random
from typing import List, Tuple, Union, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

from wind_tunnel_testing.config import RANDOM_SEED, RANDOM_FOREST_CONFIG

random.seed(RANDOM_SEED)


def train_rf(
    x: Union[pd.DataFrame, npt.NDArray],
    y: Union[pd.Series, npt.ArrayLike],
    apply_log: Optional[bool] = None,
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on the provided data.

    Args:
        x (Union[pd.DataFrame, npt.NDArray]): Training data features.
        y (Union[pd.Series, npt.NDArray]): Training data target.
        apply_log (Optional[bool]): Applying a log transformation can help in stabilizing the variance of the output
                                    distribution and making the data more suitable for regression, if the target
                                    variable has a wide range or follows a skewed distribution. Defaults to None

    Returns:
        RandomForestRegressor: Trained RandomForestRegressor model.

    """

    if apply_log:
        y = np.log1p(y)

    model = RandomForestRegressor(
        n_estimators=RANDOM_FOREST_CONFIG["n_estimators"], random_state=RANDOM_SEED
    )
    model.fit(x, y)
    return model


def predict(
    model, data: Union[pd.DataFrame, npt.NDArray], apply_exp: Optional[bool] = None
) -> npt.NDArray:
    """
    Generate predictions using the provided model.

    Args:
        model (RandomForestRegressor): Trained model.
        data (Union[pd.DataFrame, npt.NDArray]): Data for which predictions are to be made.
        apply_exp (Optional[bool]): use True, if the model was trained with log(y), the predictions are transformed
                                    back with exp. Defaults to None.

    Returns:
        npt.NDArray: Model predictions.
    """
    predictions = model.predict(data)

    if apply_exp:
        # Transform back the predictions
        predictions = np.expm1(predictions)

    return predictions


def evaluate_rf_accuracy(x, y) -> float:
    """
    Evaluate the accuracy of a RandomForestRegressor model using cross-validation.

    This function trains a RandomForestRegressor model with the provided features `x` and
    target `y`. It uses k-fold cross-validation to compute the mean squared error (MSE)
    and returns mean of the MSE scores.

    Args:
        x (array-like): The feature matrix (e.g., H1, H2 values).
        y (array-like): The target values (e.g., c_total).

    Returns:
        float: The mean squared error (MSE) of the model, averaged over the cross-validation folds.
    """
    model = RandomForestRegressor(
        n_estimators=RANDOM_FOREST_CONFIG["n_estimators"], random_state=RANDOM_SEED
    )

    # Perform k-fold cross-validation to compute the negative mean squared error
    # metrics which measure the distance between the model and the data,
    # like metrics.mean_squared_error, are available as neg_mean_squared_error
    # which return the negated value of the metric.
    scores = cross_val_score(model, x, y, cv=2, scoring="neg_mean_squared_error")
    return -scores.mean()


def generate_grid_sample_points(
    n_samples: int = 20, bounds: List[Tuple] | Tuple[Tuple] = ((0.01, 0.2), (0.01, 0.344))
) -> npt.NDArray:
    """
    Generate sample points using Latin Hypercube sampling within given bounds.

    Args:
        n_samples (int, optional): Number of samples to generate. Defaults to 20.
        bounds (List[Tuple[float, float]], optional): Bounds for each variable.
                Defaults to [(0.01, 0.2), (0.01, 0.344)].

    Returns:
        npt.NDArray: Array of sampled points.
    """
    sampler = qmc.LatinHypercube(d=len(bounds), seed=RANDOM_SEED)
    lhs_sampled_points = sampler.random(n=n_samples)

    H1_min, H1_max = bounds[0][0], bounds[0][1]
    H2_min, H2_max = bounds[1][0], bounds[1][1]

    # applying bounds scale
    H1_samples = H1_min + lhs_sampled_points[:, 0] * (H1_max - H1_min)
    H2_samples = H2_min + lhs_sampled_points[:, 1] * (H2_max - H2_min)

    return np.column_stack((H1_samples, H2_samples))


def optimize_wind_tunnel_visit_points(distance_matrix: npt.ArrayLike) -> List:
    """
    Solve the Traveling Salesman Problem (TSP) to find the optimal route for visiting points.
    code adapted from
    https://github.com/google/or-tools/blob/stable/ortools/constraint_solver/samples/tsp_cities.py

    Args:
        distance_matrix (npt.ArrayLike): Matrix with distances between points.

    Returns:
        List[int]: Optimal route as a list of point indices.
    """
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route
    return []


def optimization(
    model, bounds: List[Tuple] | Tuple[Tuple] = ((0.01, 0.2), (0.01, 0.344))
) -> npt.ArrayLike:
    """
    Optimize the model predictions within the given bounds.

    Args:
        model (RandomForestRegressor): Trained model.
        bounds (List[Tuple[float, float]], optional): Bounds for the variables.
                Defaults to [(0.01, 0.2), (0.01, 0.344)].

    Returns:
        result (ArrayLike): the optimal values of variables.
    """

    def objective(H: npt.ArrayLike) -> float:
        """Objective function to be minimized."""
        H1, H2 = H
        return model.predict([[H1, H2]])[0]

    # initial guesses
    H1 = np.arange(bounds[0][0], bounds[0][1], 0.01)
    H2 = np.arange(bounds[1][0], bounds[1][1], 0.01)
    x0 = np.array([random.choice(H1), random.choice(H2)])

    # Perform optimization using the Powell method
    # Powell's method is a conjugate direction method, which helps finding the minimum of a function
    # without gradients - since the model is not differentiable
    result = minimize(
        objective, x0=x0, bounds=bounds, method="Powell", options={"xtol": 1e-8, "disp": True}
    )

    return result.x


def feature_selection(model: RandomForestRegressor, max_features: int = 20) -> npt.ArrayLike:
    """
    Perform feature selection using the trained model.

    Args:
        model (RandomForestRegressor): Trained model.
        max_features (int, optional): Maximum number of features to select. Defaults to 20.

    Returns:
        npt.NDArray: Indices of selected features.
    """
    selector = SelectFromModel(model, max_features=max_features, prefit=True)
    selected_features = selector.get_support(indices=True)
    return selected_features
