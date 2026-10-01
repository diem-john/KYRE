from config import parse_config
from pathlib import Path
import pandas as pd
import os

from src.trajectory_prediction.utils.logger import Logger
from src.trajectory_prediction.utils.preprocessing import load_centroid_weekly_embeddings, preprocess_centroids, split_train_validation_test_sequences
from src.trajectory_prediction.utils.pca_preprocessing import add_pca_features
from src.trajectory_prediction.predictors import predict_future_weeks_with_metrics
from src.trajectory_prediction.statistical_predictors import save_stat_model
from src.trajectory_prediction.metrics import print_prediction_fit_metrics, check_historical_and_prediction_centroid_overlap, evaluate_decoded_behavior_similarity, print_decoded_behavior_prediction_fit_metrics
from src.trajectory_prediction.centroid_anomaly_visualization_forecast import visualize_centroid_forecasts
from src.behavioral_anomaly_detector.anomaly_detector import detect_centroid_behavioral_anomaly
from src.bca_regression_decoder.pipeline import run_bca_centroid_decoder
from src.behavioral_anomaly_detector.utils import build_decoded_behavior_column, scale_and_fit_centroid_embeddings, prepare_historical_predicted_embeddings_for_decoding

from config import (EMB_PREFIX, CLUSTER_COL, PCA_PREFIX, USER_WEIGHT_COL,
                    MAX_BACKTEST_WINDOWS, MODEL_NAMES, PREDICTIVE_MODEL,
                    TOP_K, ROLLING_WINDOW, MIN_PERIODS, DECODED_Z_THRESHOLD,
                    MOVEMENT_Z_THRESHOLD, TOP_K_JACCARD_THRESHOLD, DIRECTIONAL_EFFICIENCY_THRESHOLD, SEED,
                    ANOMALY_SCORE_THRESHOLD, MAX_HISTORY_PER_CLUSTER, IS_SAVE_SPLIT_CSV, IS_DECODE_AND_VISUALIZE_ALL_PREDICTION_APPROACH)


"""
    Date Last Edited: July 23, 2026
    % Developer: 
    
    Gieo Coronado

"""

def _standardize_run_modes(run_mode_input):
    # Ensures that if the mode is end-to-end, stages would not be ran twice
    if 0 in run_mode_input: 
        run_mode_input = [1, 2, 3, 4, 5]
        
    # Ensures that ordering is followed (observing the different run mode dependencies)        
    order = [1, 2, 3, 4, 5]
    modes = [mode for mode in order if mode in set(run_mode_input)]
    return modes

def _load_and_prepare_decoded_centroid_emb(
    decoded_centroid_dir
):
    """
    Load decoded centroid embeddings when they are not already available,
    add prediction indicators, and return all three DataFrames.
    """

    try:
        decoded_historical_weekly_centroid_emb = pd.read_csv(
            f"{decoded_centroid_dir}/decoded_historical_weekly_centroid_emb.csv"
        )
        decoded_predicted_weekly_centroid_emb = pd.read_csv(
            f"{decoded_centroid_dir}/decoded_predicted_weekly_centroid_emb.csv"
        )

        decoded_historical_weekly_centroid_emb["is_prediction"] = False
        decoded_predicted_weekly_centroid_emb["is_prediction"] = True

        decoded_weekly_centroid_emb = pd.concat(
            [
                decoded_historical_weekly_centroid_emb,
                decoded_predicted_weekly_centroid_emb,
            ],
            ignore_index=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "The decoded centroid embeddings were not available in memory "
            "and could not be loaded from file."
        ) from exc

    return (
        decoded_historical_weekly_centroid_emb,
        decoded_predicted_weekly_centroid_emb,
        decoded_weekly_centroid_emb,
    )
    
def main():
    try:
        # Handles user inputs from the CLI
        user_config = parse_config()
        
        PROJECT_NAME = user_config.general.project_name
        PROJECT_DIR = Path("projects") / PROJECT_NAME
        CENTROID_DIR = PROJECT_DIR / "cluster_embeddings" / "centroids_with_usage_labels"
        
        logger = Logger(PROJECT_DIR)
        
        run_mode = _standardize_run_modes(user_config.general.mode)
        
        # DFs that are needed to proceed with the trajectory prediction and visualization
        train_model_df = validation_model_df = test_model_df = historical_behavior_change_df = predicted_behavior_change_df = decoded_historical_weekly_centroid_emb = decoded_predicted_weekly_centroid_emb = decoded_weekly_centroid_emb = None 
        
        logger.log(msg='Cluster Trajectory Prediction Pipeline Started', level="STAGE")
        
        logger.log(msg='Loading weekly centroid embeddings...', level="INFO")
        centroid_df = load_centroid_weekly_embeddings(CENTROID_DIR)
        
        logger.log(msg='Preprocessing weekly centroid embeddings...', level="INFO")
        clean_df, _ = preprocess_centroids(
            centroid_df,
            min_weeks=user_config.general.min_weeks,
            max_gap=user_config.general.max_gap,
            max_staleness_weeks=user_config.general.max_staleness_weeks,
            emb_prefix=EMB_PREFIX,
            cluster_col=CLUSTER_COL,
            user_weight_col=USER_WEIGHT_COL,
            )
        
        logger.log(msg='Splitting weekly centroid embeddings...', level="INFO")
        train_df, validation_df, test_df, _ = split_train_validation_test_sequences(
            clean_df,
            validation_weeks=user_config.general.n_validation,
            test_weeks=user_config.general.max_horizon,
            min_train_weeks=user_config.general.min_weeks,
            cluster_col=CLUSTER_COL,
            project_name=PROJECT_NAME,
            is_save_split_csv=IS_SAVE_SPLIT_CSV
        )

        logger.log(msg='Adding PCA features to weekly centroid embeddings...', level="INFO")
        train_model_df, pca_model, scaler_model, pca_variance_df = add_pca_features(
            train_df,
            n_dims=user_config.general.pca_dims,
            input_prefix=EMB_PREFIX,
            output_prefix=PCA_PREFIX,
            seed=user_config.general.seed
            )
        
        validation_model_df, _, _, _ = add_pca_features(
            validation_df,
            pca_model=pca_model,
            scaler_model=scaler_model,
            n_dims=user_config.general.pca_dims,
            input_prefix=EMB_PREFIX,
            output_prefix=PCA_PREFIX,
            seed=user_config.general.seed
        )
        test_model_df, _, _, _ = add_pca_features(
            test_df,
            pca_model=pca_model,
            scaler_model=scaler_model,
            n_dims=user_config.general.pca_dims,
            input_prefix=EMB_PREFIX,
            output_prefix=PCA_PREFIX,
            seed=user_config.general.seed
        )
        
        # Kalman Filter Approach
        if 1 in run_mode:
            
            logger.log(msg='Started cluster trajectory prediction using Kalman Filter approach...', level="STAGE")
            _, _, kalman_reliable_horizon_summary_df, _ = predict_future_weeks_with_metrics(
                train_model_df,
                validation_df=validation_model_df,
                test_df=test_model_df,
                min_train_weeks=user_config.general.min_weeks,
                n_weeks=user_config.general.max_horizon,
                forecast_n_known_weeks=user_config.general.forecast_n_known_weeks,
                model_name="kalman",
                min_cosine=user_config.reliable_horizon.min_cosine,
                min_rmse_improvement=user_config.reliable_horizon.min_rmse_improvement,
                min_displacement_cosine=user_config.reliable_horizon.min_displacement_cosine,
                PROJECT_NAME=PROJECT_NAME,
                cluster_col=CLUSTER_COL,
                emb_prefix=PCA_PREFIX,
                user_weight_col=USER_WEIGHT_COL,
                q_pos=user_config.statistical_approach.q_pos,
                q_vel=user_config.statistical_approach.q_vel,
                r=user_config.statistical_approach.r,
                velocity_decay=user_config.statistical_approach.velocity_decay,
                anchor_n_weeks=user_config.statistical_approach.anchor_n_weeks,
                anchor_weight=user_config.statistical_approach.anchor_weight,
                clean_df=clean_df,
                max_backtest_windows=MAX_BACKTEST_WINDOWS,
                pca=pca_model,
                scaler=scaler_model,
            )
            print_prediction_fit_metrics(logger, kalman_reliable_horizon_summary_df)
            

            kalman_config = {
                "dim": user_config.general.pca_dims,
                "q_pos": user_config.statistical_approach.q_pos,
                "q_vel": user_config.statistical_approach.q_vel,
                "r": user_config.statistical_approach.r,
                "velocity_decay": user_config.statistical_approach.velocity_decay,
                "anchor_n_weeks": user_config.statistical_approach.anchor_n_weeks,
                "anchor_weight": user_config.statistical_approach.anchor_weight,
                "n_weeks": user_config.general.max_horizon,
                "min_cosine": user_config.reliable_horizon.min_cosine,
            }

            logger.log(msg='Saving prediction hyperparameters using the Kalman Filter approach...', level="STAGE")
            save_stat_model(
                model_name="kalman",
                model_config=kalman_config,
                scaler=scaler_model,
                pca=pca_model,
                input_prefix=EMB_PREFIX,
                model_prefix=PCA_PREFIX,
                extra={
                    "pca_dims": user_config.general.pca_dims,
                    "pca_prefix": PCA_PREFIX,
                },
                project_name=PROJECT_NAME,
                cluster_col=CLUSTER_COL,
                seed=user_config.general.seed
            )

        # Multi-lag Drift Approach
        if 2 in run_mode:
            
            logger.log(msg='Started cluster trajectory prediction using Multi-lag Drift approach...', level="STAGE")
            _, _, drift_reliable_horizon_summary_df, _ = predict_future_weeks_with_metrics(
                train_model_df,
                validation_df=validation_model_df,
                test_df=test_model_df,
                min_train_weeks=user_config.general.min_weeks,
                n_weeks=user_config.general.max_horizon,
                forecast_n_known_weeks=user_config.general.forecast_n_known_weeks,
                model_name="drift",
                min_cosine=user_config.reliable_horizon.min_cosine,
                min_rmse_improvement=user_config.reliable_horizon.min_rmse_improvement,
                min_displacement_cosine=user_config.reliable_horizon.min_displacement_cosine,
                PROJECT_NAME=PROJECT_NAME,
                cluster_col=CLUSTER_COL,
                emb_prefix=PCA_PREFIX,
                user_weight_col=USER_WEIGHT_COL,
                lag=user_config.statistical_approach.lag,
                clean_df=clean_df,
                max_backtest_windows=MAX_BACKTEST_WINDOWS,
                pca=pca_model,
                scaler=scaler_model,
            )
            print_prediction_fit_metrics(logger, drift_reliable_horizon_summary_df)
            
            drift_config = {
                "dim": user_config.general.pca_dims,
                "lag": user_config.statistical_approach.lag,
                "n_weeks": user_config.general.max_horizon,
                "min_cosine": user_config.reliable_horizon.min_cosine,
            }

            logger.log(msg='Saving prediction hyperparameters using the Multi-lag Drift approach...', level="STAGE")
            save_stat_model(
                model_name="drift",
                model_config=drift_config,
                scaler=scaler_model,
                pca=pca_model,
                project_name=PROJECT_NAME,
                cluster_col=CLUSTER_COL,
                seed=user_config.general.seed,
                input_prefix=EMB_PREFIX,
                model_prefix=PCA_PREFIX,
                extra=None,
            )

        # Polynomial Regression Approach
        if 3 in run_mode:
            
            logger.log(msg='Started cluster trajectory prediction using Polynomial Regression approach...', level="STAGE")
            _, _, polynomial_reliable_horizon_summary_df, _ = predict_future_weeks_with_metrics(
                train_model_df,
                validation_df=validation_model_df,
                test_df=test_model_df,
                min_train_weeks=user_config.general.min_weeks,
                n_weeks=user_config.general.max_horizon,
                forecast_n_known_weeks=user_config.general.forecast_n_known_weeks,
                model_name="polynomial",
                min_cosine=user_config.reliable_horizon.min_cosine,
                min_rmse_improvement=user_config.reliable_horizon.min_rmse_improvement,
                min_displacement_cosine=user_config.reliable_horizon.min_displacement_cosine,
                PROJECT_NAME=PROJECT_NAME,
                cluster_col=CLUSTER_COL,
                emb_prefix=PCA_PREFIX,
                user_weight_col=USER_WEIGHT_COL,
                degree=user_config.statistical_approach.degree,
                clean_df=clean_df,
                max_backtest_windows=MAX_BACKTEST_WINDOWS,
                pca=pca_model,
                scaler=scaler_model,
            )
            print_prediction_fit_metrics(logger, polynomial_reliable_horizon_summary_df)

            polynomial_config = {
                "dim": user_config.general.pca_dims,
                "degree": user_config.statistical_approach.degree,
                "n_weeks": user_config.general.max_horizon,
                "min_cosine": user_config.reliable_horizon.min_cosine,
            }

            logger.log(msg='Saving prediction hyperparameters using the Polynomial Regression approach...', level="STAGE")
            save_stat_model(
                model_name="polynomial",
                model_config=polynomial_config,
                scaler=scaler_model,
                pca=pca_model,
                project_name=PROJECT_NAME,
                cluster_col=CLUSTER_COL,
                seed=user_config.general.seed,
                input_prefix=EMB_PREFIX,
                model_prefix=PCA_PREFIX,
                extra=None
            )
        
        # Behavioral Decoder and Anomaly Detection
        if 4 in run_mode:
            logger.log(msg='Started Behavioral Decoder and Anomaly Detection...', level="STAGE")
            
            logger.log(msg=f'Preparing historical and predicted centroid embeddings inputs before behavioral decoder and anomaly detection...', level="INFO")
            
            # Update predictive model list relative to existing directories
            base_dir = Path(
                f"projects/{PROJECT_NAME}/centroid_trajectory_predictions"
            )
            model_names = [
                model
                for model in MODEL_NAMES
                if (base_dir / model).is_dir()
            ]
            
            if not IS_DECODE_AND_VISUALIZE_ALL_PREDICTION_APPROACH:
                model_names = [PREDICTIVE_MODEL.lower()]
            
            for model_approach in model_names:
                logger.log(msg=f'Processing behavior decode and anomaly detection for {model_approach.capitalize()} approach...', level="STAGE")
                # Loading of historical embeddings
                historical_weekly_centroid_emb = centroid_df.copy()
                
                # Prepares the centroid inputs (historical and predicted centroid embeddings for decoding.)
                historical_weekly_centroid_emb, predicted_weekly_centroid_emb, concat_weekly_centroid_emb = prepare_historical_predicted_embeddings_for_decoding(
                    historical_weekly_centroid_emb,
                    model_approach,
                    PROJECT_NAME,
                    PROJECT_DIR
                )

                logger.log(msg=f'Decoding centroid behavior...', level="INFO")
                
                # Decode historical and predicted centroid embeddings
                decoder_result = run_bca_centroid_decoder(
                    project_name=PROJECT_NAME,
                    historical_centroid_df=historical_weekly_centroid_emb,
                    predicted_centroid_df=predicted_weekly_centroid_emb,
                    emb_prefix=EMB_PREFIX,
                    cluster_col=CLUSTER_COL,
                    top_k=user_config.decoder.decoder_top_k,
                    batch=user_config.decoder.decoder_batch_size,
                    logger=logger,
                    model_name=model_approach,
                )
                
                logger.log(msg=f'Started anomaly detection...', level="INFO")
                
                DECODED_CENTROID_DIR = PROJECT_DIR / "centroid_trajectory_predictions" / "behavioral_decoder" / "decoded_centroids" / model_approach
                decoded_historical_weekly_centroid_emb, decoded_predicted_weekly_centroid_emb, decoded_weekly_centroid_emb = _load_and_prepare_decoded_centroid_emb(
                    DECODED_CENTROID_DIR
                )
                
                FLAGGED_AND_DECODED_CENTROID_DIR = PROJECT_DIR / "centroid_trajectory_predictions" / "flagged_decoded_centroids" / model_approach
                # Detect which centroid-week shows behavioral anomalies
                behavior_change_weeks_df, historical_behavior_change_df, predicted_behavior_change_df = detect_centroid_behavioral_anomaly(
                    decoded_weekly_centroid_emb=decoded_weekly_centroid_emb,
                    concat_weekly_centroid_emb=concat_weekly_centroid_emb,
                    cluster_col="final_cluster_label",
                    year_col="year",
                    week_col="week",
                    is_prediction_col="is_prediction",
                    timeline_col="timeline_type",
                    feature_col="feature_id",
                    prob_col="pred_used_prob",
                    elapsed_col="pred_elapsed",
                    emb_prefix="emb_",
                    top_k=TOP_K,
                    rolling_window=ROLLING_WINDOW,
                    min_periods=MIN_PERIODS,
                    decoded_z_threshold=DECODED_Z_THRESHOLD,
                    movement_z_threshold=MOVEMENT_Z_THRESHOLD,
                    top_k_jaccard_threshold=TOP_K_JACCARD_THRESHOLD,
                    directional_efficiency_threshold=DIRECTIONAL_EFFICIENCY_THRESHOLD,
                    save_dir=FLAGGED_AND_DECODED_CENTROID_DIR
                )
                
                # Evaluates the decoded elapsed time accurary and feature used matching
                if check_historical_and_prediction_centroid_overlap(predicted_behavior_change_df, historical_behavior_change_df):
                    compact_overall_summary_df = evaluate_decoded_behavior_similarity(
                        logger,
                        historical_behavior_change_df=historical_behavior_change_df,
                        predicted_behavior_change_df=predicted_behavior_change_df,
                        decoded_weekly_centroid_emb=decoded_weekly_centroid_emb,
                        project_name=PROJECT_NAME,
                        model_name=model_approach
                    )
                    print_decoded_behavior_prediction_fit_metrics(logger=logger,
                                                                  compact_overall_summary_df=compact_overall_summary_df,
                                                                  model_name=model_approach)
            
        # Visualization of centroid predictions
        if 5 in run_mode:
            logger.log(msg='Started cluster trajectory prediction visualization...', level="STAGE")
            
            # Update predictive model list relative to existing directories
            base_dir = Path(
                f"projects/{PROJECT_NAME}/centroid_trajectory_predictions"
            )
            model_names = [
                model
                for model in MODEL_NAMES
                if (base_dir / model).is_dir()
            ]
            
            if not IS_DECODE_AND_VISUALIZE_ALL_PREDICTION_APPROACH:
                model_names = [PREDICTIVE_MODEL.lower()]
            
            for model_approach in model_names:
                logger.log(msg=f'Visualizing centroids for the {model_approach.capitalize()} approach...', level="STAGE")
                # Checks if `historical_behavior_change_df` or `predicted_behavior_change_df` (Anomaly detection output) is loaded in memory. Otherwise, reload or raise error.

                FLAGGED_AND_DECODED_CENTROID_DIR = PROJECT_DIR / "centroid_trajectory_predictions" / "flagged_decoded_centroids" / model_approach
                historical_behavior_change_df = pd.read_csv(f'{FLAGGED_AND_DECODED_CENTROID_DIR}/historical_behavior_change.csv')
                predicted_behavior_change_df = pd.read_csv(f'{FLAGGED_AND_DECODED_CENTROID_DIR}/predicted_behavior_change.csv')

                
                DECODED_CENTROID_DIR = PROJECT_DIR / "centroid_trajectory_predictions" / "behavioral_decoder" / "decoded_centroids" / model_approach
                decoded_historical_weekly_centroid_emb, decoded_predicted_weekly_centroid_emb, decoded_weekly_centroid_emb = _load_and_prepare_decoded_centroid_emb(
                    DECODED_CENTROID_DIR
                )
                
                logger.log(msg=f'Adding decoded behavior column for visualization...', level="INFO")
                # Adds a decoded behavior column for visualization
                historical_behavior_change_df, predicted_behavior_change_df = build_decoded_behavior_column(
                    historical_behavior_change_df,
                    predicted_behavior_change_df,
                    decoded_weekly_centroid_emb
                )
                
                logger.log(msg=f'Scaling and fitting the centroid embeddings using the same Scaler and PCA model...', level="INFO")
                # Scales and fit the centroid embeddings using the same Scaler and PCA model.
                historical_behavior_change_df, predicted_behavior_change_df = scale_and_fit_centroid_embeddings(
                    historical_behavior_change_df,
                    predicted_behavior_change_df,
                    seed=SEED,
                )
                
                VISUALIZED_CENTROID_ANOMALY_DIR = PROJECT_DIR / "centroid_trajectory_predictions" / "anomaly_centroid_visualization"
                os.makedirs(VISUALIZED_CENTROID_ANOMALY_DIR, exist_ok=True)
                
                fig2d, fig3d = visualize_centroid_forecasts(
                    historical_behavior_change_df,
                    predicted_behavior_change_df,
                    anomaly_score_threshold=ANOMALY_SCORE_THRESHOLD,
                    max_history_weeks_per_cluster=MAX_HISTORY_PER_CLUSTER,
                    saving_dir=VISUALIZED_CENTROID_ANOMALY_DIR,
                    model=model_approach,
                )
        
        logger.log(msg='Cluster Trajectory Prediction Pipeline Finished.', level="STAGE")
        
    except RuntimeError as e:
        logger.log(msg=e, level="ERROR")
if __name__ == "__main__":
    main()