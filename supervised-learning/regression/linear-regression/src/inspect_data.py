
from pathlib import Path
import pandas as pd

# Locate the root ML project directory
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Construct the dataset path
DATASET_PATH = PROJECT_ROOT / "datasets" / "raw" / "house_prices_toy.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(DATASET_PATH)

# Display the dataset
print("Dataset:")
print(df)

# Display basic information
print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)
