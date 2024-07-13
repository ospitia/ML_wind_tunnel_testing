from pathlib import Path
from typing import Dict

import pandas as pd
import typer

from wind_tunnel_testing.config import RAW_DATA_DIR

app = typer.Typer()

# Constants for the variables and recorded points in the dataset
VARIABLES = ["H1", "H2"]
RECORDED_POINTS = [f"p{i}" for i in range(0, 112)]  # Generates 'p0', 'p1', ..., 'p111'


def load_ml_test_data(
    xlsx_data_filename: str = "ML_test", directory: Path = RAW_DATA_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Load machine learning test data from an Excel file.

    Args:
        xlsx_data_filename (str): The name of the Excel file (without extension) to load. Defaults to 'ML_test'.
        directory (Path): The directory where the Excel file is located. Defaults to `RAW_DATA_DIR` from the config.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are
                                sheet names and values are DataFrames containing the sheets data.
    """
    file_path = Path(directory) / f"{xlsx_data_filename}.xlsx"

    # Load the Excel file into a dictionary of DataFrames
    data = pd.read_excel(file_path, sheet_name=None)

    return data
