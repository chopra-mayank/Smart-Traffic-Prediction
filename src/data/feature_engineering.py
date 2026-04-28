import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_features(df):

    # -----------------------------------
    # Time Features
    # -----------------------------------
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)

    df['minute_of_day'] = (
        df['hour'] * 60 +
        df['timestamp'].dt.minute
    )

    # -----------------------------------
    # Cyclical Encoding
    # -----------------------------------
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)

    # -----------------------------------
    # Time Category
    # -----------------------------------
    def get_time_category(hour):

        if 5 <= hour < 12:
            return 'morning'

        elif 12 <= hour < 17:
            return 'afternoon'

        elif 17 <= hour < 21:
            return 'evening'

        else:
            return 'night'

    df['time_of_day_category'] = df['hour'].apply(
        get_time_category
    )

    # -----------------------------------
    # Calendar Features
    # -----------------------------------
    df['is_weekend'] = (
        df['day'] >= 5
    ).astype(int)

    df['is_rush_morning'] = (
        df['hour'].between(7, 9)
    ).astype(int)

    df['is_rush_evening'] = (
        df['hour'].between(17, 19)
    ).astype(int)

    df['is_peak_hour'] = (
        df['is_rush_morning'] |
        df['is_rush_evening']
    ).astype(int)

    # -----------------------------------
    # Sensor Encoding
    # -----------------------------------
    le = LabelEncoder()

    df['sensor_id_encoded'] = le.fit_transform(
        df['sensor_id']
    )

    # -----------------------------------
    # Sensor Historical Average
    # -----------------------------------
    df['sensor_avg_speed'] = df.groupby(
        'sensor_id'
    )['speed'].transform('mean')

    # -----------------------------------
    # Lag Features
    # -----------------------------------
    df['speed_lag_1'] = df.groupby(
        'sensor_id'
    )['speed'].shift(1)

    df['speed_lag_3'] = df.groupby(
        'sensor_id'
    )['speed'].shift(3)

    df['speed_lag_6'] = df.groupby(
        'sensor_id'
    )['speed'].shift(6)

    df['speed_lag_12'] = df.groupby(
        'sensor_id'
    )['speed'].shift(12)

    # -----------------------------------
    # Rolling Statistics
    # -----------------------------------
    df['rolling_mean_6'] = df.groupby(
        'sensor_id'
    )['speed'].transform(
        lambda x: x.rolling(6).mean()
    )

    df['rolling_std_6'] = df.groupby(
        'sensor_id'
    )['speed'].transform(
        lambda x: x.rolling(6).std()
    )

    df['rolling_mean_12'] = df.groupby(
        'sensor_id'
    )['speed'].transform(
        lambda x: x.rolling(12).mean()
    )

    df['rolling_min_12'] = df.groupby(
        'sensor_id'
    )['speed'].transform(
        lambda x: x.rolling(12).min()
    )

    df['rolling_max_12'] = df.groupby(
        'sensor_id'
    )['speed'].transform(
        lambda x: x.rolling(12).max()
    )

    # -----------------------------------
    # Difference Features
    # -----------------------------------
    df['speed_diff_1'] = df.groupby(
        'sensor_id'
    )['speed'].diff(1)

    df['speed_diff_3'] = df.groupby(
        'sensor_id'
    )['speed'].diff(3)

    # -----------------------------------
    # Remove nulls from lag features
    # -----------------------------------
    df = df.dropna()

    return df


if __name__ == "__main__":

    print("Loading cleaned data...")

    df = pd.read_csv(
        "data/processed/cleaned_data.csv"
    )

    print("Creating advanced features...")

    feature_df = create_features(df)

    feature_df.to_csv(
        "data/processed/feature_data.csv",
        index=False
    )

    print(" Advanced Feature Engineering Done")
    print(feature_df.head())