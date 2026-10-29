import argparse
from dataclasses import dataclass
from typing import List

# -----------------------------
# Configuration
# -----------------------------
PROJECT_NAME = "0604_hitachi_v6_supcon_w_user_days_trainobj_t1_04adl08supcon"
PREDICTIVE_MODEL = "Kalman"

EMB_PREFIX = "emb_"
CLUSTER_COL = "final_cluster_label"
USER_WEIGHT_COL = "n_users"

IS_SAVE_SPLIT_CSV=False

PCA_PREFIX = "z_"

MODEL_NAMES = ["drift", "polynomial", "kalman"]

MAX_BACKTEST_WINDOWS = 5

SEED = 42

# VISUALIZATION
ANOMALY_SCORE_THRESHOLD = 3
MAX_HISTORY_PER_CLUSTER = 52 

# Anomaly Detection thresholds
TOP_K = 10
ROLLING_WINDOW = 8
MIN_PERIODS = 4
DECODED_Z_THRESHOLD = 2.5
MOVEMENT_Z_THRESHOLD = 2.5
TOP_K_JACCARD_THRESHOLD = 0.85
DIRECTIONAL_EFFICIENCY_THRESHOLD = 0.4

IS_DECODE_AND_VISUALIZE_ALL_PREDICTION_APPROACH = False # only decodes and visualizes the final prediction approach as stated in `PREDICTIVE_MODEL`

@dataclass
class GeneralConfig:
    project_name: str
    mode: List[int]
    seed: int
    pca_dims: int
    min_weeks: int
    max_gap: int
    n_validation: int
    max_horizon: int
    forecast_n_known_weeks: int
    max_staleness_weeks: int

@dataclass  
class ReliableHorizonConfig:
    min_cosine: float
    min_rmse_improvement: float
    min_displacement_cosine: float
    
@dataclass
class StatisticalApproachConfig:
    lag: int
    degree: int
    q_pos: float
    q_vel: float
    r: float
    velocity_decay: float
    anchor_n_weeks: int
    anchor_weight: float

@dataclass
class DecoderConfig:
    decoder_batch_size: int
    decoder_top_k: int | None
    
@dataclass
class Config:
    general: GeneralConfig
    reliable_horizon: ReliableHorizonConfig
    statistical_approach: StatisticalApproachConfig
    decoder: DecoderConfig
    
def parse_config():
    parser = argparse.ArgumentParser()
    
    # General Hyperparameters
    general = parser.add_argument_group("General configuration")
    general.add_argument("--project_name", type=str, default=PROJECT_NAME, help='The project folder name.')
    general.add_argument("--mode", type=int, nargs='+', required=True, choices=[0, 1, 2, 3, 4, 5], help='Run Mode {0:End-to-End, 1:Kalman Filter, 2:Drift, 3:Polynomial Regression, 4:Anomaly Detection and Decoder, 5:Visualization}')
    general.add_argument("--seed", type=int, default=42, help='The seed used for reproducible results.')
    general.add_argument("--pca_dims", type=int, default=32, help='The target number of embedding dimensions after applying PCA to the original centroid embeddings.')
    general.add_argument("--min_weeks", type=int, default=8, help='The minimum number of historical centroid position weeks to use for centroid prediction.')
    general.add_argument("--max_gap", type=int, default=2, help='The maximum gap, in weeks, that the pipeline can fill using linear interpolation.')
    general.add_argument("--n_validation", type=int, default=12, help='Number of weeks used for validation.')
    general.add_argument("--max_horizon", type=int, default=24, help='The number of weeks in the future to forecast.')
    general.add_argument("--forecast_n_known_weeks", type=int, default=None, help='The number of known weeks in the future to forecast. Used for development checking. (Default: None)')
    general.add_argument("--max_staleness_weeks", type=int, default=3, help='Determines which clusters receive predictions based on the recency of their weekly data. A value of 0 includes only clusters with data from the latest week, while a value of 1 includes clusters with data from the latest week or the previous week, etc.')
    
    # parameters for reliable horizon testing
    reliable_horizon  = parser.add_argument_group("Reliable horizon testing parameters")
    
    reliable_horizon.add_argument("--min_cosine", type=float, default=0.95, help='The minimum horizon cosine similarity score to be classified as reliable.')
    reliable_horizon.add_argument("--min_rmse_improvement", type=float, default=0.0, help='The minimum horizon RMSE improvement vs persistent approach score to be classified as reliable.')
    reliable_horizon.add_argument("--min_displacement_cosine", type=float, default=0.25, help='The minimum horizon displacement cosine similarity score to be classified as reliable.')
    
    # parameters for reliable horizon testing
    statistical_approach  = parser.add_argument_group("Statistical approache's hyperparameters")
    
    statistical_approach.add_argument("--lag", type=int, default=52, help='Defines how many latest weeks to consider when predicting centroid positions using the Multi-lag Drift approach.')
    statistical_approach.add_argument("--degree", type=int, default=1, help='Defines the number of degrees to use when using the Polynomial Regression approach.')
    statistical_approach.add_argument("--q_pos", type=float, default=2.089585121381213e-06, help='Defines how much the "true" centroid position can deviate from the motion model (next_centroid = current_centroid + prev_velocity).')
    statistical_approach.add_argument("--q_vel", type=float, default=1.7370454844366206e-06, help='Defines how much the centroid velocity can change. Lower values make velocity smoother and less reactive to week-to-week jitter.')
    statistical_approach.add_argument("--r", type=float, default=0.0013765290181605246, help='Defines how noisy the observed weekly centroids are. Higher values smoothes more.')
    statistical_approach.add_argument("--velocity_decay", type=float, default=0.8990708432926103, help='Defines how much velocity damping during filtering and forecasting to reduce overshoot.')
    statistical_approach.add_argument("--anchor_n_weeks", type=int, default=5, help='Number of recent historical centroid weeks averaged to anchor the first Kalman forecast.')
    statistical_approach.add_argument("--anchor_weight", type=float, default=0, help='Weight assigned to the recent average centroid when blending it with the Kalman filtered position. (Must be between 0 and 1.)')
    
    decoder = parser.add_argument_group("BCA regression decoder configuration")

    decoder.add_argument("--decoder_batch_size", type=int, default=2048, help="Number of candidate features decoded per batch.")
    decoder.add_argument("--decoder_top_k", type=int, default=None, help="Optional top-k decoded features to keep per centroid. Default keeps all above threshold.")

    args = parser.parse_args()
    
    general_cfg = GeneralConfig(
        project_name = args.project_name,
        mode = args.mode,
        seed = args.seed,
        pca_dims = args.pca_dims,
        min_weeks = args.min_weeks,
        max_gap = args.max_gap,
        n_validation = args.n_validation,
        max_horizon = args.max_horizon,
        forecast_n_known_weeks = args.forecast_n_known_weeks,
        max_staleness_weeks = args.max_staleness_weeks,
    )
    
    reliable_horizon_cfg = ReliableHorizonConfig(
        min_cosine = args.min_cosine,
        min_rmse_improvement = args.min_rmse_improvement,
        min_displacement_cosine = args.min_displacement_cosine,
    )
    
    statistical_approach_cfg= StatisticalApproachConfig(
        lag = args.lag,
        degree = args.degree,
        q_pos = args.q_pos,
        q_vel = args.q_vel,
        r = args.r,
        velocity_decay = args.velocity_decay,
        anchor_n_weeks = args.anchor_n_weeks,
        anchor_weight = args.anchor_weight,
    )

    decoder_cfg = DecoderConfig(
        decoder_batch_size=args.decoder_batch_size,
        decoder_top_k=args.decoder_top_k,
    )
    
    return Config(
        general=general_cfg,
        reliable_horizon=reliable_horizon_cfg,
        statistical_approach=statistical_approach_cfg,
        decoder=decoder_cfg
    )
