import json
import optuna
from pathlib import Path

from src.trajectory_prediction.predictors import predict_future_weeks_with_metrics

import numpy as np


# Folder where Optuna study DBs and trial summaries are saved.
TUNING_DIR = Path("optuna_tuning_results")
TUNING_DIR.mkdir(parents=True, exist_ok=True)


def score_reliable_horizon_summary(reliable_horizon_summary_df, max_horizon):
    """
    Convert the reliable horizon summary into one scalar Optuna score.

    Higher is better:
    - more reliable horizons
    - higher cosine similarity
    - lower RMSE
    - lower displacement magnitude error
    - better displacement direction
    """
    if reliable_horizon_summary_df.empty:
        return 0.0

    row = reliable_horizon_summary_df.iloc[0]

    n_horizon_score = row.get("n_reliable_horizons", 0) / max_horizon
    cosine_score = row.get("weighted_mean_cosine_similarity_original", 0)
    rmse_score = 1 / (1 + row.get("weighted_mean_rmse_original", np.inf))

    displacement_error = row.get(
        "weighted_mean_displacement_magnitude_error_original",
        row.get("mean_displacement_magnitude_error_original", np.inf),
    )
    displacement_error_score = 1 / (1 + displacement_error)

    displacement_cosine_score = row.get(
        "weighted_mean_displacement_cosine_similarity_original",
        row.get("mean_displacement_cosine_similarity_original", 0),
    )

    score = (
        n_horizon_score
        + cosine_score
        + rmse_score
        + displacement_error_score
        + displacement_cosine_score
    )

    return float(score)


def run_optuna_tuning(
    model_name,
    suggest_model_config,
    train_model_df,
    validation_model_df,
    test_model_df,
    n_weeks,
    min_cosine,
    min_rmse_improvement,
    min_displacement_cosine,
    n_trials=10,
    study_name=None,
    tuning_dir=TUNING_DIR,
    save_trials_csv=True,
    save_best_params_json=True,
):
    """
    Simple Optuna wrapper for all statistical models.

    This saves each study into a local SQLite DB so tuning can be resumed later.

    suggest_model_config is a small function that receives an Optuna trial and
    returns the model-specific config to test.
    """
    tuning_dir = Path(tuning_dir)
    tuning_dir.mkdir(parents=True, exist_ok=True)

    study_name = study_name or f"{model_name}_tuning"
    db_path = tuning_dir / f"{study_name}.db"
    storage = f"sqlite:///{db_path}"

    def objective(trial):
        print(f"Started {model_name} trial #{trial.number}")

        trial_model_config = suggest_model_config(trial)

        _, _, reliable_horizon_summary_df, _ = predict_future_weeks_with_metrics(
            train_model_df,
            validation_df=validation_model_df,
            test_df=test_model_df,
            n_weeks=n_weeks,
            model_name=model_name,
            model_config=trial_model_config,
            min_cosine=min_cosine,
            min_rmse_improvement=min_rmse_improvement,
            min_displacement_cosine=min_displacement_cosine,
            PROJECT_NAME=None,
        )

        score = score_reliable_horizon_summary(
            reliable_horizon_summary_df,
            max_horizon=n_weeks,
        )

        print(f"Finished {model_name} trial #{trial.number} with score={score:.4f}")
        return score

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("Study DB:", db_path)
    print("Best score:", study.best_value)
    print("Best params:", study.best_params)

    if save_trials_csv:
        trials_df = study.trials_dataframe()
        trials_csv_path = tuning_dir / f"{study_name}_trials.csv"
        trials_df.to_csv(trials_csv_path, index=False)
        print("Trials CSV:", trials_csv_path)

    if save_best_params_json:
        best_params_path = tuning_dir / f"{study_name}_best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        print("Best params JSON:", best_params_path)

    return study
