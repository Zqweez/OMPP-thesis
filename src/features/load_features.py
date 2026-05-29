import pandas as pd
from pathlib import Path
from typing import List, Tuple

def load_and_prepare_data(data_path: str, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Load the dataset and prepare features
    
    Returns
    -------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target variable.
    feature_cols : List[str]
        List of feature column names.
    df : pd.DataFrame
        The entire dataframe loaded from the CSV file with all columns.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    # Load the dataset
    df = pd.read_csv(data_path)

    # Exclude all target columns and sequence-related columns from the features
    exclude_cols = ["Sequence", "OMPP nr","Peptide","Sequence",
                    "Potentiation_Mean","MIC_Mean","NPN_Mean","DISC_Mean","Cytotoxicity_Mean","Hemolysis_Mean"]

    # Define feature columns (excluding sequence, target, and any additional excluded columns)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[target]

    # print(f"{Fore.GREEN}Data loaded successfully with {X.shape[0]} samples and {X.shape[1]} features{Style.RESET_ALL}")
    
    return X, y, feature_cols, df

