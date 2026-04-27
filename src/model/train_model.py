import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    # Load data
    df = pd.read_csv("data/processed/feature_data.csv")

    # Features & Target
    X = df[['hour', 'day', 'is_weekend', 'is_peak_hour']]
    y = df['traffic']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("✅ Model Training Done")
    print("MAE:", mae)
    print("R2 Score:", r2)

    # Save predictions
    result = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': predictions
    })

    result.to_csv("data/predictions/predictions.csv", index=False)
    print("✅ Predictions saved")


if __name__ == "__main__":
    train_model()