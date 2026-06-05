import pandas as pd
from pathlib import Path

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "YearBuilt",
    "GarageArea",
    "SalePrice"     # Target variable
] 


# path: Path = DATA_PATH → parameter with type + default value
# -> pd.DataFrame → return type hint
def load_data(path: Path = Path("data/train.csv")) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[FEATURES]
    return df


if __name__ == "__main__":
    df = load_data()
    print(f'\n{df.head()}')
    print(f"\nShape:{df.shape}")

    