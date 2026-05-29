from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
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

def nested_cross_validation(X, y, outer_cv=5, inner_cv=5, random_state=42):
    """ Perform nested cross-validation for Ridge regression. """
    
    # Define alpha grid for hyperparameter search
    alpha_grid = np.logspace(-3, 3, 15)  # from 0.001 to 1000
    param_grid = {"ridge__alpha": alpha_grid}
    
    # Create CV objects
    outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    inner_kf = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
    
    # Storage for results
    best_params_per_fold = []
    fold_details = []
    
    print(Fore.GREEN + f"\nStarting Nested Cross-Validation:" + Style.RESET_ALL)
    print(f"Outer CV: {outer_cv} folds, Inner CV: {inner_cv} folds")
    print("="*50)
    
    # Outer CV loop
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(X, y)):
        # outer_train_idx, outer_test_idx = train/test indices for this outer fold
        print(f"\nOuter Fold {fold_idx + 1}/{outer_cv}")
        print("-" * 30)
        
        # Split data for this outer fold
        # iloc is used to get rows by index
        X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]
        
        print(f"Outer train size: {len(X_outer_train)}, Outer test size: {len(X_outer_test)}")
        
        # Inner CV: Find best hyperparameters
        pipeline = create_ridge_pipeline()
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_kf,  # Use inner CV object
            scoring='neg_mean_squared_error', # Might test changing to r2 or neg_mean_absolute_error
            n_jobs=-1,
            refit=True
        )
        
        # Fit grid search on outer training set
        grid_search.fit(X_outer_train, y_outer_train)
        best_alpha = grid_search.best_params_['ridge__alpha']
        best_inner_score = grid_search.best_score_
        
        print(f"Best alpha from inner CV: {best_alpha:.4f}")
        print(f"Best inner CV score (neg MSE): {best_inner_score:.4f}")
        
        # Train final model with best parameters on full outer training set
        best_model = grid_search.best_estimator_
        
        # Evaluate on outer test set
        y_outer_pred = best_model.predict(X_outer_test)
        outer_mse = mean_squared_error(y_outer_test, y_outer_pred)
        outer_r2 = r2_score(y_outer_test, y_outer_pred)
        outer_mae = mean_absolute_error(y_outer_test, y_outer_pred)
        
        print(f"Outer test performance:")
        print(f"  MSE: {outer_mse:.4f}")
        print(f"  RMSE: {np.sqrt(outer_mse):.4f}")
        print(f"  R²: {outer_r2:.4f}")
        print(f"  MAE: {outer_mae:.4f}")
        
        best_params_per_fold.append(best_alpha)

        # Store results
        fold_details.append({
            'fold': fold_idx + 1,
            'best_alpha': best_alpha,
            'inner_cv_score': best_inner_score,
            'outer_test_mse': outer_mse,
            'outer_test_rmse': np.sqrt(outer_mse),
            'outer_test_r2': outer_r2,
            'outer_test_mae': outer_mae,
            'outer_train_size': len(X_outer_train),
            'outer_test_size': len(X_outer_test)
        })
    
    # Calculate summary statistics
    mse_scores = [detail['outer_test_mse'] for detail in fold_details]
    rmse_scores = [detail['outer_test_rmse'] for detail in fold_details]
    r2_scores = [detail['outer_test_r2'] for detail in fold_details]
    mae_scores = [detail['outer_test_mae'] for detail in fold_details]
    
    results = {
        'best_params_per_fold': best_params_per_fold,
        'fold_details': fold_details,
        'summary': {
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'best_alpha_mean': np.mean(best_params_per_fold),
            'best_alpha_std': np.std(best_params_per_fold)
        }
    }
    
    return results

def print_nested_cv_results(results):
    """Print formatted results from nested cross-validation."""
    print("\n" + "="*50)
    print(Fore.CYAN + "NESTED CROSS-VALIDATION RESULTS" + Style.RESET_ALL)
    print("="*50)
    
    summary = results['summary']
    
    print(f"\nPerformance Estimates (Outer CV - Unbiased):")
    print(f"MSE:  {summary['mse_mean']:.4f} ± {summary['mse_std']:.4f}")
    print(f"RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
    print(f"R²:   {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    print(f"MAE:  {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    
    print(f"\nHyperparameter Stability:")
    print(f"Best Alpha: {summary['best_alpha_mean']:.4f} ± {summary['best_alpha_std']:.4f}")
    
    print(f"\nPer-Fold Details:")
    for detail in results['fold_details']:
        print(f"Fold {detail['fold']}: α={detail['best_alpha']:.4f}, "
              f"R²={detail['outer_test_r2']:.4f}, "
              f"RMSE={np.sqrt(detail['outer_test_mse']):.4f}")

def train_final_model_on_full_data(X, y):
    """
    Train a final model on the full dataset using the most commonly selected alpha.
    This is for deployment purposes after nested CV evaluation.
    """
    print(f"\nTraining final model on full dataset...")
    
    # Use inner CV to find best alpha on full data
    pipeline = create_ridge_pipeline()
    alpha_grid = np.logspace(-3, 3, 15)
    param_grid = {"ridge__alpha": alpha_grid}
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # 5-fold CV on full data
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        refit=True
    )
    
    grid_search.fit(X, y)
    
    print(f"Final model best alpha: {grid_search.best_params_['ridge__alpha']:.4f}")
    print(f"Final model CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_nested_cv_results(results, final_model, feature_cols):
    """Save nested CV results and final model."""
    folder_path = os.path.join(os.path.dirname(__file__), "ridge-nested-cv-results")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save nested CV results summary
    summary_path = os.path.join(folder_path, "nested_cv_summary.csv")
    summary_df = pd.DataFrame([results['summary']])
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed fold results
    fold_details_path = os.path.join(folder_path, "nested_cv_fold_details.csv")
    fold_details_df = pd.DataFrame(results['fold_details'])
    fold_details_df.to_csv(fold_details_path, index=False)
    
    # Save final model
    model_path = os.path.join(folder_path, "final_ridge_model.joblib")
    joblib.dump(final_model, model_path)
    
    # Save final model coefficients
    coefs = final_model.named_steps["ridge"].coef_
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": coefs})
    coef_df["abs_coefficient"] = np.abs(coef_df["coefficient"])
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    
    coef_path = os.path.join(folder_path, "final_model_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    
    print(Fore.GREEN + f"Nested CV results and final model saved to {folder_path}" + Style.RESET_ALL)

def plot_nested_cv_results(results):
    """Plot nested CV results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    folds = [d['fold'] for d in results['fold_details']]
    r2_scores = [d['outer_test_r2'] for d in results['fold_details']]
    rmse_scores = [np.sqrt(d['outer_test_mse']) for d in results['fold_details']]
    alphas = [d['best_alpha'] for d in results['fold_details']]
    
    # Plot R² scores per fold
    axes[0, 0].bar(folds, r2_scores, alpha=0.7)
    axes[0, 0].axhline(y=results['summary']['r2_mean'], color='red', linestyle='--', 
                       label=f"Mean = {results['summary']['r2_mean']:.3f}")
    axes[0, 0].set_title('R² Score per Outer Fold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot RMSE scores per fold
    axes[0, 1].bar(folds, rmse_scores, alpha=0.7, color='orange')
    axes[0, 1].axhline(y=results['summary']['rmse_mean'], color='red', linestyle='--',
                       label=f"Mean = {results['summary']['rmse_mean']:.3f}")
    axes[0, 1].set_title('RMSE per Outer Fold')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot best alpha per fold
    axes[1, 0].bar(folds, alphas, alpha=0.7, color='green')
    axes[1, 0].axhline(y=results['summary']['best_alpha_mean'], color='red', linestyle='--',
                       label=f"Mean = {results['summary']['best_alpha_mean']:.3f}")
    axes[1, 0].set_title('Best Alpha per Outer Fold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Alpha')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot performance distribution
    axes[1, 1].boxplot([r2_scores, rmse_scores], labels=['R²', 'RMSE'])
    axes[1, 1].set_title('Performance Distribution Across Folds')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print(Fore.GREEN + "Ridge Regression with Nested Cross-Validation" + Style.RESET_ALL)
    print(Fore.YELLOW + "="*50 + Style.RESET_ALL)

    # 1. Load and explore data
    df = load_and_explore_data()

    # 2. Prepare features and target
    X, y, feature_cols = prepare_features_target(df)

    # 3. Perform nested cross-validation
    nested_cv_results = nested_cross_validation(X, y, outer_cv=5, inner_cv=5)

    # 4. Print results
    print_nested_cv_results(nested_cv_results)

    # 5. Train final model on full data (for deployment)
    final_model = train_final_model_on_full_data(X, y)

    # 6. Save results
    save_nested_cv_results(nested_cv_results, final_model, feature_cols)

    # 7. Plot results
    plot_nested_cv_results(nested_cv_results)

    print(f"\n{Fore.GREEN}Nested cross-validation complete!{Style.RESET_ALL}")
    print(f"Unbiased performance estimate: R² = {nested_cv_results['summary']['r2_mean']:.4f} ± {nested_cv_results['summary']['r2_std']:.4f}")

if __name__ == "__main__":
    main()