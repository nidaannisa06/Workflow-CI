import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
#imported library
def load_processed_data(data_dir="housing_preprocessing"):
    """Load processed data with correct relative paths"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_data_path = os.path.join(script_dir, data_dir)

        X_train = pd.read_csv(os.path.join(full_data_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(full_data_path, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(full_data_path, "y_train.csv")).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(full_data_path, "y_test.csv")).iloc[:, 0]

        print(f"Data loaded successfully from: {full_data_path}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading processed data from {full_data_path}: {str(e)}")
        raise

def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Housing Price Prediction - Project Local Test")

    run_id = os.environ.get("MLFLOW_RUN_ID")

    if run_id is None:
        print("WARNING: MLFLOW_RUN_ID not set. Starting new run.")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            client = MlflowClient()

            # Load data
            X_train, X_test, y_train, y_test = load_processed_data()

            # Log params
            client.log_param(run_id, "train_samples", len(X_train))
            client.log_param(run_id, "test_samples", len(X_test))

            # Train model
            model = train_model(X_train, y_train)

            # Evaluate
            rmse, mae, r2 = evaluate_model(model, X_test, y_test)
            client.log_metric(run_id, "rmse", rmse)
            client.log_metric(run_id, "mae", mae)
            client.log_metric(run_id, "r2_score", r2)

            # Save model artifact
            model_filename = "random_forest_model.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)
            client.log_artifact(run_id, model_filename, "model_artefacts")
            os.remove(model_filename)

            # Save plot artifact
            predictions = model.predict(X_test)
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.5)
            plt.xlabel("Actual Prices")
            plt.ylabel("Predicted Prices")
            plt.title("Actual vs. Predicted Prices")
            plt.grid(True)
            plot_filename = "predictions_vs_actuals.png"
            plt.savefig(plot_filename)
            client.log_artifact(run_id, plot_filename, "visualizations")
            plt.close()
            os.remove(plot_filename)

            print(f"Training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    else:
        print("MLFLOW_RUN_ID provided, but continuation mode not implemented.")
        raise NotImplementedError("Script does not support continuing existing runs yet.")

if __name__ == "__main__":
    main()
