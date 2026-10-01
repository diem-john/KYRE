from pathlib import Path
import pandas as pd
import re

def load_historical_weekly_centroid_parquets(
    project_name
):
    dir=f'projects/{project_name}/cluster_embeddings/centroids_with_usage_labels'
    parquet_dir = Path(dir)
    pattern = re.compile(r"^(\d{4})_(\d{2})_centroid_emb_profiles\.parquet$")

    matched = []
    for fp in parquet_dir.glob("*.parquet"):
        m = pattern.match(fp.name)
        if m:
            year = int(m.group(1))
            week = int(m.group(2))
            matched.append((year, week, fp))

    dfs = []
    week_labels = []

    for year, week, fp in matched:
        df = pd.read_parquet(fp)

        if "year" not in df.columns:
            df["year"] = year
        if "week" not in df.columns:
            df["week"] = week

        dfs.append(df)
        week_labels.append(f"{year}_{week:02d}")
        
    df = pd.concat(dfs, ignore_index=True)

    return df, week_labels