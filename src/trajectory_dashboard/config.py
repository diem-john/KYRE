PROJECT_NAME = (
    "0604_hitachi_v6_supcon_w_user_days_trainobj_t1_04adl08supcon"
)

MODEL_NAME = "kalman"

DEFAULT_HISTORY_WEEKS = 54

ANOMALY_SCORE_THRESHOLD = 3

ANOMALY_FLAG_MAPPING = {
    "multiple_movement_signals": "Multiple Movement Signals",
    "far_from_recent_baseline": "Centroid Movement From Recent Baseline Detected",
    "4_week_centroid_drift": "4 week Centroid Baseline Movement Detected",
    "8_week_centroid_drift": "8 week Centroid Baseline Movement Detected",
    "no_main_movement": "No Movement Changes",
    "elapsed_and_top_feature_change": "Features Used and Elapsed Time Changed",
    "top_feature_change": "Features Used Changed",
    "elapsed_time_change": "Elapsed Time Changed",
    "no_behavioral_change": "No behavioral change",
}