import pandas as pd

def preprocess(df):
    # Rename first column → timestamp
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Convert all other columns to numeric (important for METR-LA)
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create single traffic column (average of sensors)
    sensor_cols = df.columns.drop('timestamp')
    df['traffic'] = df[sensor_cols].mean(axis=1)

    # Keep only required columns
    df = df[['timestamp', 'traffic']]

    # Clean data
    df = df.dropna()
    df = df.sort_values(by='timestamp')

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/dataset.csv")
    df = preprocess(df)

    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print("✅ Preprocessing Done")