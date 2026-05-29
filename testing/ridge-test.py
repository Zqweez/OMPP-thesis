from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib
from colorama import Fore, Style


def load_and_explore_data():
    """Load the peptide features data and display some basic information"""
    # Load the dataset
    df = pd.read_csv('prestudy/peptide_features.csv')
    
    print("Dataset Loaded")
    print("Dataset shape:", df.shape)

    print("\nDataset info:")
    print(df.info())
        
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def create_ridge_pipeline():
    """Create a Ridge regression pipeline with preprocessing."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])
    return pipe

def prepare_features_target(df):
    """Prepare features and target variable."""
    # Define target variable
    target = 'Hydrophobic_Moment'
    print(Fore.CYAN + f"\nTarget variable: {target}" + Style.RESET_ALL)
    
    # Define feature columns (excluding sequence and target)
    feature_cols = [col for col in df.columns if col not in ['Sequence', target]]
    
    print(Fore.YELLOW + f"\nFeature columns ({len(feature_cols)}):" + Style.RESET_ALL)

    # Print feature columns
    for col in feature_cols:
        print(f"  - {col}")
    
    X = df[feature_cols]
    y = df[target]
    
    return X, y, feature_cols

def fit_model(pipeline, X_train, y_train):
    """ Use GridSearchCV to find the best hyperparameters for Ridge regression """
    alpha_grid = np.logspace(-3, 3, 15) # from 0.001 to 1000
    param_grid = {
        "ridge__alpha": alpha_grid,
    }

    # Create GridSearchCV object with 5-fold cross-validation
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error', # Use negative MSE as scoring
        n_jobs=-1, # Use all available cores
        refit=True,  # after selecting best alpha, refit on full training set
        return_train_score=True
    )
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    print(Fore.GREEN + f"\n Best parameters: {grid_search.best_params_}" + Style.RESET_ALL)
    print(Fore.LIGHTBLUE_EX + f"Best CV score (neg MSE): {grid_search.best_score_}" + Style.RESET_ALL)

    # Return the fitted pipeline 
    return grid_search

def save_model(model, coef_df, grid, results):
    """Save the trained model to a file."""
    folder_path = os.path.join(os.path.dirname(__file__), "ridge-results")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    model_path = os.path.join(folder_path, "ridge_model.joblib")
    # Save the model using joblib
    joblib.dump(model, model_path)

    # Save coefficients table
    coef_path = os.path.join(folder_path, "ridge_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)

    # Save CV results (full table of alphas and scores)
    cv_path = os.path.join(folder_path, "ridge_gridsearch_results.csv")
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(cv_path, index=False)

    # Save evaluation metrics
    results_path =  os.path.join(folder_path, "ridge_evaluation_metrics.csv")
    results_df = pd.DataFrame([results])
    results_df.to_csv(results_path, index=False)

    print(Fore.GREEN + f"Model, coefficients, grid search results, and evaluation metrics saved." + Style.RESET_ALL)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the trained model."""
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    print("\n" + "="*50)
    print(Fore.CYAN + "MODEL EVALUATION RESULTS" + Style.RESET_ALL)
    print("="*50)

    print("\nTraining set performance:")
    print(f"RMSE: {np.sqrt(train_mse):.4f}")
    print(f"MAE : {train_mae:.4f}")
    print(f"R^2 : {train_r2:.4f}")
    
    print("\nTest set performance:")
    print(f"RMSE: {np.sqrt(test_mse):.4f}")
    print(f"MAE : {test_mae:.4f}")
    print(f"R^2 : {test_r2:.4f}")

    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    results = {
        "y_test": y_test, "y_test_pred": y_test_pred,
        "test_r2": test_r2, "train_r2": train_r2,
        "test_mse": test_mse, "train_mse": train_mse,
        "test_mae": test_mae, "train_mae": train_mae,
        "cv_r2_mean": cv_scores.mean(), "cv_r2_std": cv_scores.std()
    }
    
    return results

def plot_results(results, coef_df=None):
    """Plot model evaluation results."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(results['y_test'], results['y_test_pred'], alpha=0.6)
    plt.plot([results['y_test'].min(), results['y_test'].max()], 
             [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
    plt.xlabel('Actual Hydrophobic Moment')
    plt.ylabel('Predicted Hydrophobic Moment')
    plt.title(f'Actual vs Predicted\nR² = {results["test_r2"]:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(1, 3, 2)
    residuals = results['y_test'] - results['y_test_pred']
    plt.scatter(results['y_test_pred'], residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Hydrophobic Moment')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Feature Importance (if available)
    ax3 = plt.subplot(1, 3, 3)
    if coef_df is not None and len(coef_df) > 0:
        # Ensure we have the expected columns
        if not {"feature", "coef"}.issubset(coef_df.columns):
            raise ValueError("coef_df must contain columns: 'feature' and 'coef'")

        # Take top_n by absolute coefficient magnitude
        imp = coef_df.copy()
        imp["importance"] = np.abs(imp["coef"])
        imp = imp.sort_values("importance", ascending=False)

        # Plot horizontal bar chart (reverse so largest is at top)
        imp = imp.sort_values("importance", ascending=True)

        bars = ax3.barh(imp["feature"], imp["importance"])
        ax3.set_xlabel("|Coefficient| (after scaling)")
        ax3.set_title(f"Feature Importance (Top {len(imp)})")
        ax3.grid(True, alpha=0.3)

        # Add value labels
        x_max = float(imp["importance"].max()) if len(imp) else 1.0
        for bar in bars:
            width = bar.get_width()
            ax3.text(
                width + 0.01 * x_max,  # small offset to the right
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                ha="left",
                va="center",
                fontsize=9
            )
    else:
        ax3.axis("off")
        ax3.set_title("Feature Importance (not provided)")
    
    
    plt.tight_layout()
    plt.show()

def main():
    print(Fore.GREEN + "Ridge Regression Model" + Style.RESET_ALL)
    print(Fore.YELLOW + "="*40 + Style.RESET_ALL)

    # 1. Load and explore data
    df = load_and_explore_data()

    # 2. Make a pipeline
    pipeline = create_ridge_pipeline()

    # 3. Prepare features and target
    X, y, feature_cols = prepare_features_target(df)

    # 4. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(Fore.CYAN + f"\nTraining set size: {X_train.shape[0]} samples" + Style.RESET_ALL)
    print(Fore.CYAN + f"Testing set size: {X_test.shape[0]} samples" + Style.RESET_ALL)

    # 5. Create and fit model
    grid = fit_model(pipeline, X_train, y_train)
    best_model = grid.best_estimator_

    # Extract some info about the best model
    coefs = best_model.named_steps["ridge"].coef_
    coef_df = pd.DataFrame({"feature": X.columns, "coef": coefs})
    coef_df["abs_coef"] = np.abs(coef_df["coef"])
    coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])

    # 6. Evaluate model
    results = evaluate_model(best_model, X_train, X_test, y_train, y_test)

    # 7. Save the model and results
    save_model(best_model, coef_df, grid, results)

    # 8. Plot true vs predicted
    plot_results(results, coef_df)
    

if __name__ == "__main__":
    main()