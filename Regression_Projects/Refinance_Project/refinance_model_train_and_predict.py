from pathlib import Path

import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Base directory for all generated files
BASE_DIR = Path(r"C:\Users\DELL\Desktop\course_prac\ml\Regression_Projects\Refinance_Project")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def base_file_path(filename):
    return BASE_DIR / filename

def load_data(filepath, target_column, drop_columns=None):
    """
    Load data from an Excel file, filter future data, and prepare features and target variable.
    Args:
        filepath (str): Path to the Excel file.
        target_column (str): Name of the target column.
        drop_columns (list): List of columns to drop.
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
    """
    try:
        df = pd.read_excel(filepath)
        future_data = df[df[target_column].isna()]
        future_data.to_csv(base_file_path('future_data.csv'), index=False)
        if drop_columns:
            df = df.drop(columns=drop_columns)
        df = df[~df[target_column].isna()]
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logging.info(f"Loaded data from {filepath}. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    return X, y

def train_rf_model(X, y, params=None):
    """
    Train a RandomForestRegressor model with standardization.
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        params (dict): Model parameters.
    Returns:
        model: Trained RandomForestRegressor.
        scaler: Fitted StandardScaler.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestRegressor(**(params or {}))
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        logging.info(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}, R2: {r2_score(y_test, y_pred):.4f}")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
    return model, scaler

def predict(X):
    """
    Predict using the trained model and scaler.
    Args:
        X (pd.DataFrame): Features to predict.
    Returns:
        np.ndarray: Predictions.
    """
    try:
        model = load_model(base_file_path('RF_model.sav'))
        scaler = load_scaler(base_file_path('scaler.sav'))
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        logging.info(f"Prediction completed. Shape: {preds.shape}")
        return preds
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

def save_model(model, filepath):
    """Save the trained model to disk."""
    try:
        joblib.dump(model, filepath)
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def save_scaler(scaler, filepath):
    """Save the fitted scaler to disk."""
    try:
        joblib.dump(scaler, filepath)
        logging.info(f"Scaler saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving scaler: {e}")
        raise

def load_model(filepath):
    """Load a trained model from disk."""
    try:
        model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_scaler(filepath):
    """Load a fitted scaler from disk."""
    try:
        scaler = joblib.load(filepath)
        logging.info(f"Scaler loaded from {filepath}")
        return scaler
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")
        raise

if __name__ == "__main__":
    # Specify columns to drop
    drop_columns = ['Date','Week No','Year']  # Update as needed

    try:
        # Training
        X, y = load_data(BASE_DIR / 'Weekly_Refinance_Volumes_Data.xlsx', target_column='Refinance', drop_columns=drop_columns)
        model, scaler = train_rf_model(X, y, params={'n_estimators': 100, 'max_depth': 5})
        save_model(model, base_file_path('RF_model.sav'))
        save_scaler(scaler, base_file_path('scaler.sav'))
    except Exception as e:
        logging.error(f"Error in training workflow: {e}")

    try:
        # Prediction
        
        # Ensure future data columns match training columns and order
        train_columns = X.columns.tolist()
        print(f"Training columns: {train_columns}")
        X_future = pd.read_csv(base_file_path('future_data.csv'))
        X_future = X_future.drop(columns=drop_columns, errors='ignore')
        X_future = X_future[train_columns]
        predictions = predict(X_future)
        # Save future data with predictions for verification
        future_data_full = pd.read_csv(base_file_path('future_data.csv'))
        future_data_full = future_data_full.reset_index(drop=True)
        future_data_full['Prediction'] = predictions
        future_data_full.to_excel(base_file_path('predictions_on_future_data.xlsx'), index=False)
        logging.info(f"Predictions saved to {base_file_path('predictions_on_future_data.xlsx')}")
    except Exception as e:
        logging.error(f"Error in prediction workflow: {e}")


from sklearn.model_selection import train_test_split

print("sklearn imported successfully")