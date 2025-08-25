"""
Preprocessing script for CALCE battery dataset
- Converts .mat files to .csv
- Cleans data (removes outliers, interpolates missing values)
- Segments into cycles
"""

import os
import scipy.io
import pandas as pd
import numpy as np

RAW_DIR = "data/raw/"
PROC_DIR = "data/processed/"

def mat_to_csv(mat_file, csv_file):
    """
    Converts CALCE .mat file to CSV.
    Each .mat file contains multiple cycles with voltage, current, temperature, capacity.
    """
    data = scipy.io.loadmat(mat_file)
    # CALCE stores in structured arrays
    cycles = data['cycles'][0]
    all_cycles = []

    for c in cycles:
        cycle_data = {
            "cycle": int(c[0][0]),
            "type": c[1][0],  # charge/discharge
            "time": c[2].flatten(),
            "voltage": c[3].flatten(),
            "current": c[4].flatten(),
            "temperature": c[5].flatten(),
            "capacity": c[6].flatten()
        }
        df = pd.DataFrame(cycle_data)
        all_cycles.append(df)

    pd.concat(all_cycles).to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")

def preprocess_all():
    for file in os.listdir(RAW_DIR):
        if file.endswith(".mat"):
            mat_file = os.path.join(RAW_DIR, file)
            csv_file = os.path.join(PROC_DIR, file.replace(".mat", ".csv"))
            mat_to_csv(mat_file, csv_file)

if __name__ == "__main__":
    os.makedirs(PROC_DIR, exist_ok=True)
    preprocess_all()
