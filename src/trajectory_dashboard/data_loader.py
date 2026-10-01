import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_dashboard_paths(
    project_name: str,
    model_name: str,
) -> dict[str, Path]:
    """
    Build the paths to the project result files used by the trajectory prediction dashboard.
    """

    project_dir = PROJECT_ROOT / "projects" / project_name

    trajectory_dir = (
        project_dir
        / "centroid_trajectory_predictions"
    )

    anomaly_dir = (
        trajectory_dir
        / "flagged_decoded_centroids"
        / model_name
    )

    decoded_dir = (
        trajectory_dir
        / "behavioral_decoder"
        / "decoded_centroids"
        / model_name
    )

    return {
        "historical_anomalies": (
            anomaly_dir
            / "historical_behavior_change.csv"
        ),
        "predicted_anomalies": (
            anomaly_dir
            / "predicted_behavior_change.csv"
        ),
        "historical_decoded": (
            decoded_dir
            / "decoded_historical_weekly_centroid_emb.csv"
        ),
        "predicted_decoded": (
            decoded_dir
            / "decoded_predicted_weekly_centroid_emb.csv"
        ),
    }


def validate_dashboard_paths(
    paths: dict[str, Path],
) -> tuple[bool, list[Path]]:
    """
    Check whether all required dashboard CSV files exist.

    Returns:
        all_files_exist:
            True when every required file exists.

        missing_files:
            List of required files that were not found.
    """

    missing_files = [
        path
        for path in paths.values()
        if not path.is_file()
    ]

    all_files_exist = len(missing_files) == 0

    return all_files_exist, missing_files


def load_dashboard_data(
    paths: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    """
    Load all dashboard CSV files into pandas DataFrames.
    """

    return {
        file_label: pd.read_csv(file_path)
        for file_label, file_path in paths.items()
    }


def validate_dashboard_columns(
    dashboard_data: dict[str, pd.DataFrame],
) -> dict[str, list[str]]:
    """
    Check whether each loaded DataFrame contains its required columns.

    Returns:
        A dictionary containing only datasets with missing columns.
    """

    required_columns = {
        "historical_anomalies": [
            "final_cluster_label",
            "year",
            "week",
            "behavior_change_score",
            "behavior_change_type",
            "triggered_main_flags",
            "anomaly_priority_level",
            "anomaly_priority_rank",
            "anomaly_priority_label",
        ],
        "predicted_anomalies": [
            "final_cluster_label",
            "year",
            "week",
            "behavior_change_score",
            "behavior_change_type",
            "triggered_main_flags",
            "anomaly_priority_level",
            "anomaly_priority_rank",
            "anomaly_priority_label",
        ],
        "historical_decoded": [
            "final_cluster_label",
            "year",
            "week",
            "feature_id",
            "feature",
            "pred_used_prob",
            "pred_elapsed",
        ],
        "predicted_decoded": [
            "final_cluster_label",
            "year",
            "week",
            "feature_id",
            "feature",
            "pred_used_prob",
            "pred_elapsed",
        ],
    }

    missing_columns = {}

    for data_name, expected_columns in required_columns.items():
        dataframe = dashboard_data[data_name]

        missing = [
            column
            for column in expected_columns
            if column not in dataframe.columns
        ]

        if missing:
            missing_columns[data_name] = missing

    return missing_columns

def prepare_anomaly_data(
    dashboard_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine historical and predicted anomaly rows into one DataFrame.
    """

    historical = dashboard_data["historical_anomalies"].copy()
    predicted = dashboard_data["predicted_anomalies"].copy()

    historical["source"] = "Historical"
    predicted["source"] = "Predicted"

    anomaly_data = pd.concat(
        [historical, predicted],
        ignore_index=True,
    )

    anomaly_data["year"] = pd.to_numeric(
        anomaly_data["year"],
        errors="coerce",
    )

    anomaly_data["week"] = pd.to_numeric(
        anomaly_data["week"],
        errors="coerce",
    )

    anomaly_data["week_start"] = pd.to_datetime(
        anomaly_data["year"].astype("Int64").astype(str)
        + "-W"
        + anomaly_data["week"].astype("Int64").astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    anomaly_data["year_week"] = (
        anomaly_data["year"].astype("Int64").astype(str)
        + "-W"
        + anomaly_data["week"].astype("Int64").astype(str).str.zfill(2)
    )

    anomaly_data = anomaly_data.sort_values(
        [
            "source",
            "final_cluster_label",
            "week_start",
        ]
    ).reset_index(drop=True)

    return anomaly_data


def prepare_decoded_data(
    dashboard_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine historical and predicted decoded-feature rows.
    """

    historical = dashboard_data["historical_decoded"].copy()
    predicted = dashboard_data["predicted_decoded"].copy()

    historical["source"] = "Historical"
    predicted["source"] = "Predicted"

    decoded_data = pd.concat(
        [historical, predicted],
        ignore_index=True,
    )

    decoded_data["year"] = pd.to_numeric(
        decoded_data["year"],
        errors="coerce",
    )

    decoded_data["week"] = pd.to_numeric(
        decoded_data["week"],
        errors="coerce",
    )

    decoded_data["pred_used_prob"] = pd.to_numeric(
        decoded_data["pred_used_prob"],
        errors="coerce",
    )

    decoded_data["pred_elapsed"] = pd.to_numeric(
        decoded_data["pred_elapsed"],
        errors="coerce",
    )

    decoded_data["week_start"] = pd.to_datetime(
        decoded_data["year"].astype("Int64").astype(str)
        + "-W"
        + decoded_data["week"].astype("Int64").astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    decoded_data["year_week"] = (
        decoded_data["year"].astype("Int64").astype(str)
        + "-W"
        + decoded_data["week"].astype("Int64").astype(str).str.zfill(2)
    )

    decoded_data = decoded_data.sort_values(
        [
            "source",
            "final_cluster_label",
            "week_start",
            "pred_elapsed",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    return decoded_data


def add_dashboard_pca_coordinates(
    anomaly_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create shared 3D PCA coordinates for historical and predicted centroids.

    The scaler and PCA model are fitted using historical centroid embeddings.
    Predicted centroids are transformed with the same fitted scaler and PCA.
    """

    result = anomaly_data.copy()

    embedding_columns = sorted(
        [
            column
            for column in result.columns
            if column.startswith("emb_")
        ],
        key=lambda column: int(column.split("_")[-1]),
    )

    if not embedding_columns:
        raise ValueError(
            "No embedding columns beginning with 'emb_' were found."
        )

    historical_mask = result["source"] == "Historical"
    predicted_mask = result["source"] == "Predicted"

    historical_embeddings = (
        result.loc[
            historical_mask,
            embedding_columns,
        ]
        .apply(pd.to_numeric, errors="coerce")
    )

    predicted_embeddings = (
        result.loc[
            predicted_mask,
            embedding_columns,
        ]
        .apply(pd.to_numeric, errors="coerce")
    )

    if historical_embeddings.empty:
        raise ValueError(
            "No historical centroid embeddings were found. "
            "Historical rows are required to fit dashboard PCA."
        )

    if historical_embeddings.isna().any().any():
        raise ValueError(
            "Historical embedding columns contain missing "
            "or invalid values."
        )

    if (
        not predicted_embeddings.empty
        and predicted_embeddings.isna().any().any()
    ):
        raise ValueError(
            "Predicted embedding columns contain missing "
            "or invalid values."
        )

    if historical_embeddings.shape[0] < 3:
        raise ValueError(
            "At least three historical centroid rows are required "
            "to create 3D PCA coordinates."
        )

    if historical_embeddings.shape[1] < 3:
        raise ValueError(
            "At least three embedding dimensions are required "
            "to create 3D PCA coordinates."
        )

    scaler = StandardScaler()

    historical_scaled = scaler.fit_transform(
        historical_embeddings
    )

    pca = PCA(
        n_components=3,
        random_state=42,
    )

    historical_pca = pca.fit_transform(
        historical_scaled
    )

    pca_columns = [
        "pca_0",
        "pca_1",
        "pca_2",
    ]

    result.loc[
        historical_mask,
        pca_columns,
    ] = historical_pca

    if not predicted_embeddings.empty:
        predicted_scaled = scaler.transform(
            predicted_embeddings
        )

        predicted_pca = pca.transform(
            predicted_scaled
        )

        result.loc[
            predicted_mask,
            pca_columns,
        ] = predicted_pca

    return result