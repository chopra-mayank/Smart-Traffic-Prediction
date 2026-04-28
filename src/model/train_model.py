import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():

    print("Loading feature dataset...")

    df = pd.read_csv(
        "data/processed/feature_data.csv"
    )

    # -----------------------------
    # Features
    # -----------------------------
    feature_cols = [

        'hour',
        'day',
        'month',
        'week_of_year',
        'minute_of_day',

        'hour_sin',
        'hour_cos',
        'day_sin',
        'day_cos',

        'is_weekend',
        'is_peak_hour',
        'is_rush_morning',
        'is_rush_evening',

        'sensor_id_encoded',
        'sensor_avg_speed',

        'speed_lag_1',
        'speed_lag_3',
        'speed_lag_6',
        'speed_lag_12',

        'rolling_mean_6',
        'rolling_std_6',
        'rolling_mean_12',
        'rolling_min_12',
        'rolling_max_12',

        'speed_diff_1',
        'speed_diff_3',

        'is_anomaly'
    ]

    X = df[feature_cols]

    # -----------------------------
    # Target
    # -----------------------------
    y = df['speed']

    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")

    model.fit(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------
    predictions = model.predict(X_test)

    # -----------------------------
    # Evaluation
    # -----------------------------
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(" Model Training Done")
    print("MAE:", mae)
    print("R2 Score:", r2)

    # -----------------------------
    # Save Predictions
    # -----------------------------
    result = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': predictions
    })

    result.to_csv(
        "data/predictions/predictions.csv",
        index=False
    )

    print("✅ Predictions saved")


if __name__ == "__main__":
    train_model()