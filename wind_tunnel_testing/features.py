import random
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from scipy.spatial.distance import pdist, squareform

from wind_tunnel_testing.config import RANDOM_SEED
from wind_tunnel_testing.dataset import RECORDED_POINTS, VARIABLES
from wind_tunnel_testing.modeling import (
    feature_selection,
    generate_grid_sample_points,
    optimization,
    optimize_wind_tunnel_visit_points,
    predict,
    train_rf,
    evaluate_rf_accuracy,
)

random.seed(RANDOM_SEED)

from sklearn.ensemble import IsolationForest


def remove_outliers_isolation_forest(df, columns, contamination=0.01, random_state=RANDOM_SEED):
    """
    Remove outliers from the dataframe using Isolation Forest.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns (list): The list of columns to check for outliers.
    - contamination (float): The proportion of outliers in the data set.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: The dataframe with outliers removed.
    """
    # Initialize Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)

    # Fit the model and predict outliers
    outliers = iso_forest.fit_predict(df[columns])

    # Filter the dataframe to remove outliers
    cleaned_df = df[outliers != -1]

    return cleaned_df


def compute_gt_ctotal(cx: float, cy: float) -> float:
    """Compute the total ground truth C value from its components."""
    return ((cx**2) + (cy**2)) ** 0.5


def train_helper(data: Dict, input_data_cols: List = VARIABLES) -> any:
    """
    Trains a random forest with modeling.train_rf() using the given data and input columns
    and the compute_gt_ctotal() as output var

    Args:
        data (Dict[str, Dict]): The input data dictionary.
        input_data_cols (List[str]): List of columns to use as input features.

    Returns:
        model: The trained random forest model.
    """

    # Remove outliers from 'cx' and 'cy' columns using Isolation Forest
    cleaned_data = remove_outliers_isolation_forest(
        data["Data"], columns=["cx", "cy"], contamination=0.01
    )
    x = cleaned_data[input_data_cols]
    y = compute_gt_ctotal(cleaned_data["cx"], cleaned_data["cy"])
    model = train_rf(x, y)

    return model


def modeling(
    data: Dict, input_data_cols: List = VARIABLES, inference_data_source: str = "Q1"
) -> npt.ArrayLike:
    """
    Performs modeling by training a model and making predictions.

    Args:
        data (Dict[str, Dict]): The input data dictionary.
        input_data_cols (List[str]): List of columns to use as input features.
        inference_data_source (str): The source key for inference data.

    Returns:
        ctotal (ArrayLike): The predicted values.
    """
    model = train_helper(data=data, input_data_cols=input_data_cols)

    inference_data = data[inference_data_source]
    ctotal = predict(model, inference_data[input_data_cols])

    return ctotal


def get_optimal_H(data: Dict, input_data_cols: List = VARIABLES) -> Dict:
    """
    Gets the optimal H1 and H2 values using the given data and input columns.

    Args:
        data (Dict[str, Dict]): The input data dictionary.
        input_data_cols (List[str]): List of columns to use as input features.

    Returns:
        Dict: Dictionary containing The optimal H1 and H2 values and the corresponding prediction for ctotal.
    """
    model = train_helper(data, input_data_cols)

    result = optimization(model, bounds=[(0.01, 0.2), (0.01, 0.344)])
    optimal_H1, optimal_H2 = result
    ctotal = model.predict(result.reshape(1, -1))

    return {"H1": optimal_H1, "H2": optimal_H2, "ctotal": ctotal[0]}


def model_optimization(
    data: Dict, input_data_cols: List = RECORDED_POINTS, inference_data_source: str = "Q4"
) -> npt.ArrayLike:
    """
    Optimizes the model by selecting features and makes predictions with the optimized model

    Args:
        data (Dict[str, Dict]): The input data dictionary.
        input_data_cols (List[str]): List of columns to use as input features.
        inference_data_source (str): The source key for inference data.

    Returns:
        ctotal (ArrayLike): The predicted values by the optimized model.
    """
    model = train_helper(data, input_data_cols)
    selected_features = [f"p{i}" for i in feature_selection(model)]

    # train another model with the selected features
    model_selected = train_helper(data, selected_features)

    # compute predictions for the selected features
    inference_data = data[inference_data_source]
    inference_data_selected = inference_data[selected_features]
    ctotal = predict(model_selected, inference_data_selected)

    return ctotal


def wind_tunnel_time_optimization(
    n_samples: int = 20, bounds: List[Tuple] | Tuple[Tuple] = ((0.01, 0.2), (0.01, 0.344))
) -> List[Dict[str, float]]:
    """
    Optimizes wind tunnel visit times using downsampled variable space.

    Args:
        n_samples (int): Number of samples to generate for the grid.
        bounds (Union[List[Tuple[float, float]], Tuple[Tuple[float, float]]]): Bounds for the grid sampling.

    Returns:
        List[Dict[str, Union[float, int]]]: The optimal visit sequence with time and H values.
    """

    def duration(point1, point2):
        H1_a, H2_a = point1
        H1_b, H2_b = point2
        time = abs(H1_b - H1_a) / 0.01 + abs(H2_b - H2_a) / 0.015
        return time

    downsampled_variables_space = generate_grid_sample_points(n_samples=n_samples, bounds=bounds)
    distances = pdist(downsampled_variables_space, metric=duration)

    route = optimize_wind_tunnel_visit_points(squareform(distances))

    optimal_visit_sequence = []
    current_time = 0
    for i in range(1, len(route)):
        optimal_visit_sequence.append(
            {
                "time": current_time,
                "H1": downsampled_variables_space[route[i], 0],
                "H2": downsampled_variables_space[route[i], 1],
            }
        )
        current_time += duration(
            downsampled_variables_space[route[i - 1]], downsampled_variables_space[route[i]]
        )

    return optimal_visit_sequence


def find_n_samples_iteratively(
    data: Dict,
    input_data_cols: List = RECORDED_POINTS,
    bounds: List[Tuple] | Tuple[Tuple] = ((0.01, 0.2), (0.01, 0.344)),
    n_samples: int = 20,
    growth_factor: float = 1.2,
    error_threshold: float = 0.1,
    max_iterations: int = 5,
) -> int:
    """
    Iteratively find the optimal number of samples required to achieve the desired model accuracy.

    This function trains a surrogate model with an initial number of samples and
    increases the sample size until the model error is below a specified threshold or the maximum
    number of iterations is reached.

    Args:
        data (Dict): The input data used to train the model.
        input_data_cols (List): The columns of the input data to be used for model training.
        bounds (List[Tuple] | Tuple[Tuple]): The bounds for generating sample points.
        n_samples (int): Initial number of samples.
        growth_factor (float): The factor by which to increase the sample size in each iteration.
        error_threshold (float): The desired model accuracy measured in Mean Squared Error (MSE).
        max_iterations (int): The maximum number of iterations to perform.

    Returns:
        int: The optimal number of samples required to achieve the desired model accuracy.
    """
    reference_model = train_helper(data, input_data_cols)

    # Iterate to find optimal n_samples
    for _ in range(max_iterations):
        sample_points = generate_grid_sample_points(n_samples, bounds)

        c_total_values = predict(reference_model, sample_points)

        x, y = sample_points, c_total_values

        model_error = evaluate_rf_accuracy(x, y)

        print(f"Number of samples: {n_samples}, Model Error (MSE): {model_error}")

        if model_error < error_threshold:
            break

        n_samples = int(n_samples * growth_factor)  # Increase sample size

    return n_samples
