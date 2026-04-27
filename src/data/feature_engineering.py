import pandas as pd

def create_features(df):
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek  # 0 = Monday
    df['is_weekend'] = df['day'].apply(lambda x: 1 if x >= 5 else 0)

    # Optional: peak traffic indicator (useful feature)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_data.csv")

    df = create_features(df)

    df.to_csv("data/processed/feature_data.csv", index=False)
    print("✅ Feature Engineering Done")
    print(df.head())