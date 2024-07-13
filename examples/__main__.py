import shutil

from examples.wind_tunnel_testing_campaign import (
    H1_H2_for_lowest_ctotal,
    minimize_wind_tunnel_time,
    surrogate_model,
    surrogate_model_from_discrete_points,
    surrogate_model_optimal_pressure_sensors,
)
from wind_tunnel_testing.config import RAW_DATA_DIR, REPORTS_DIR

if __name__ == "__main__":
    print("\n *** Running wind_tunnel_testing_campaign examples *** \n")
    input_file_name = "ML_test.xlsx"
    shutil.copy(RAW_DATA_DIR / input_file_name, REPORTS_DIR / input_file_name)
    minimize_wind_tunnel_time()
    surrogate_model()
    H1_H2_for_lowest_ctotal()
    surrogate_model_from_discrete_points()
    surrogate_model_optimal_pressure_sensors()
    print(f"\n *** Check reports @ {REPORTS_DIR / input_file_name} *** \n")
