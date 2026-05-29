from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _add_footer(fig, settings: Dict[str, Any], y_offset: float = 0.01) -> None:
	"""Add compact reproducibility footer to figures."""
	footer = (
		f"target={settings.get('target_arg', 'N/A')}, "
		f"model={settings.get('model_name', 'N/A')}, "
		f"scale_target={settings.get('scale_target', 'N/A')}, "
		f"selection_mode={settings.get('selection_mode', 'N/A')}, "
		f"n_trials={settings.get('n_trials', 'N/A')}, "
		f"selected_features={settings.get('selected_feature_count', 'N/A')}"
	)
	fig.text(0.5, y_offset, footer, ha="center", va="bottom", fontsize=9, color="gray")


def plot_training_fit_scatter(
	*,
	y_true: pd.Series,
	y_pred: np.ndarray,
	output_path: str | Path,
	settings: Dict[str, Any],
) -> None:
	"""Create a simple predicted-vs-true scatter plot with in-sample metrics."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	y_true_values = np.asarray(y_true, dtype=float)
	y_pred_values = np.asarray(y_pred, dtype=float)
	if y_true_values.size == 0 or y_pred_values.size == 0:
		return

	residuals = y_true_values - y_pred_values
	denom = float(np.sum((y_true_values - np.mean(y_true_values)) ** 2))
	rmse = float(np.sqrt(np.mean(residuals ** 2)))
	mae = float(np.mean(np.abs(residuals)))
	r2 = float(1.0 - np.sum(residuals ** 2) / denom) if denom > 0 else float("nan")

	fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
	ax.scatter(y_true_values, y_pred_values, alpha=0.75, color="#3454D1", edgecolor="white", linewidth=0.5)

	min_val = float(min(np.min(y_true_values), np.min(y_pred_values)))
	max_val = float(max(np.max(y_true_values), np.max(y_pred_values)))
	ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", alpha=0.6)

	ax.set_title("Training Fit: Predicted vs True", fontweight="bold")
	ax.set_xlabel("True values")
	ax.set_ylabel("Predicted values")
	ax.grid(True, alpha=0.25)

	fig.text(
		0.5,
		1.02,
		f"RMSE = {rmse:.4f} | MAE = {mae:.4f} | R2 = {r2:.4f}",
		ha="center",
		va="bottom",
		fontsize=10,
		color="gray",
	)
	_add_footer(fig, settings, y_offset=-0.02)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
