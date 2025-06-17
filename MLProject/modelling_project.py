import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient # <--- PASTI ADA IMPOR INI

# Pastikan konfigurasi DagsHub dikomentari untuk pengujian lokal MLflow Project
# dagshub.init(repo_owner='nidaannisa06', repo_name='Membangun_Model', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/nidaannisa06/Membangun_Model.mlflow")

def load_processed_data(data_dir="housing_preprocessing"): # Ini sudah benar
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
    
    # ### PERBAIKAN KRUSIAL: Dapatkan run ID langsung dari variabel lingkungan ###
    run_id = os.environ.get("MLFLOW_RUN_ID") # <--- PASTI ADA BARIS INI
    
    if run_id is None:
        print("WARNING: MLFLOW_RUN_ID environment variable not set. Starting a new run.")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
    
    client = MlflowClient() # <--- PASTI ADA BARIS INI (INISIALISASI CLIENT)

    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()
        
    # Log data parameters (GUNAKAN client.log_param)
    client.log_param(run_id, "train_samples", len(X_train)) # <--- PASTI ADA PERUBAHAN INI
    client.log_param(run_id, "test_samples", len(X_test)) # <--- PASTI ADA PERUBAHAN INI
        
    # Train model
    model = train_model(X_train, y_train)
        
    # Evaluate
    rmse, mae, r2 = evaluate_model(model, X_test, y_test)
        
    # Log metrics (GUNAKAN client.log_metric)
    client.log_metric(run_id, "rmse", rmse) # <--- PASTI ADA PERUBAHAN INI
    client.log_metric(run_id, "mae", mae) # <--- PASTI ADA PERUBAHAN INI
    client.log_metric(run_id, "r2_score", r2) # <--- PASTI ADA PERUBAHAN INI
        
    # Log model (manual logging, GUNAKAN client.log_artifact)
    model_filename = "random_forest_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved locally to {model_filename}")
        
    client.log_artifact(run_id, model_filename, "model_artefacts") # <--- PASTI ADA PERUBAHAN INI
    print(f"Model logged as artifact to MLflow local at '{model_filename}'")
    os.remove(model_filename)

    # Tambahkan artefak kedua (plot, GUNAKAN client.log_artifact)
    predictions = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs. Predicted Prices")
    plt.grid(True)
    plot_filename = "predictions_vs_actuals.png"
    plt.savefig(plot_filename)
    print(f"Plot saved locally to {plot_filename}")

    client.log_artifact(run_id, plot_filename, "visualizations") # <--- PASTI ADA PERUBAHAN INI
    print(f"Plot logged as artifact to MLflow local at '{plot_filename}'")
    plt.close()
    os.remove(plot_filename)

    print(f"Training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    main()