import pandas as pd
import numpy as np

def preprocess(df):

    # -----------------------------
    # 1. Rename timestamp column
    # -----------------------------
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # -----------------------------
    # 2. Melt wide → long format
    # -----------------------------
    df_long = df.melt(
        id_vars=['timestamp'],
        var_name='sensor_id',
        value_name='speed'
    )

    # -----------------------------
    # 3. Convert speed to numeric
    # -----------------------------
    df_long['speed'] = pd.to_numeric(
        df_long['speed'],
        errors='coerce'
    )

    # -----------------------------
    # 4. Replace 0 with NaN
    # -----------------------------
    df_long['speed'] = df_long['speed'].replace(0, np.nan)

    # -----------------------------
    # 5. Forward fill missing values
    # -----------------------------
    df_long['speed'] = df_long.groupby(
        'sensor_id'
    )['speed'].transform(lambda x: x.ffill())

    # -----------------------------
    # 6. Outlier handling
    # -----------------------------
    df_long['is_anomaly'] = (
        (df_long['speed'] < 0) |
        (df_long['speed'] > 90)
    ).astype(int)

    # Cap values
    df_long['speed'] = df_long['speed'].clip(0, 90)

    # -----------------------------
    # 7. Drop remaining nulls
    # -----------------------------
    df_long = df_long.dropna()

    # -----------------------------
    # 8. Sort values
    # -----------------------------
    df_long = df_long.sort_values(
        by=['sensor_id', 'timestamp']
    )

    return df_long


if __name__ == "__main__":

    print("Loading dataset...")

    df = pd.read_csv("data/raw/dataset.csv")

    print("Preprocessing started...")

    processed_df = preprocess(df)

    processed_df.to_csv(
        "data/processed/cleaned_data.csv",
        index=False
    )

    print(" Advanced Preprocessing Done")
    print(processed_df.head())