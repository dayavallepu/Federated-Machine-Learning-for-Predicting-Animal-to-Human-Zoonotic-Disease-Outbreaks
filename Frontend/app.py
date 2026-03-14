# ===============================
# FLASK APPLICATION FOR ZOONOTIC DISEASE PREDICTION
# ===============================
import os
import numpy as np
import pandas as pd
import joblib
import sqlite3
import hashlib
from datetime import datetime
from functools import wraps

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import xgboost as xgb

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP library imported successfully")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP library not available. Install with: pip install shap")

# ===============================
# CONFIGURATION
# ===============================
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'users.db'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model paths
BASE_DIR = "saved_local_models"
GLOBAL_MODEL_DIR = "saved_global_model"
DATA_DIR = "Datasets"

CLIENTS = ["Farm", "Hospital", "Wildlife"]

CLIENT_FEATURES = {
    'Farm': ['Animal_Population', 'Animal_Mortality', 'Vaccination_Rate',
             'Temperature', 'Humidity', 'Rainfall', 'Contact_Index'],
    'Hospital': ['Reported_Cases', 'Mortality_Rate', 'Symptom_Count',
                 'Admissions', 'Positivity_Rate', 'Contact_Index'],
    'Wildlife': ['Species_Count', 'Migration_Flag', 'Habitat_Proximity',
                 'Wildlife_Mortality', 'Exposure_Score']
}

GLOBAL_FEATURES = [
    'Animal_Population', 'Animal_Mortality', 'Vaccination_Rate',
    'Temperature', 'Humidity', 'Rainfall', 'Contact_Index',
    'Reported_Cases', 'Mortality_Rate', 'Symptom_Count',
    'Admissions', 'Positivity_Rate',
    'Species_Count', 'Migration_Flag', 'Habitat_Proximity',
    'Wildlife_Mortality', 'Exposure_Score'
]

# Initialize models
global_model = None
xgb_model = None
feature_extractor = None
local_models = {}
preprocessors = {}
shap_explainer = None

# ===============================
# DATABASE INITIALIZATION
# ===============================
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  full_name TEXT,
                  role TEXT DEFAULT 'user',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Predictions history
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  client_type TEXT,
                  prediction_type TEXT,
                  predicted_class INTEGER,
                  confidence REAL,
                  input_data TEXT,
                  explanation_image BLOB,
                  shap_plot BLOB,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # User sessions
    c.execute('''CREATE TABLE IF NOT EXISTS user_sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  session_token TEXT,
                  ip_address TEXT,
                  user_agent TEXT,
                  login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  logout_time TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

# ===============================
# HELPER FUNCTIONS
# ===============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    return hash_password(password) == password_hash

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_user_data(user_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    return user

def save_prediction(user_id, client_type, prediction_type, predicted_class, 
                   confidence, input_data, explanation_image=None, shap_plot=None):
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    # QUICK FIX: Force predicted_class to be an integer
    if predicted_class is not None:
        # Convert to int if it's a string/numpy int
        try:
            predicted_class_int = int(predicted_class)
        except (ValueError, TypeError):
            # If conversion fails, use 0 as default
            predicted_class_int = 0
            print(f"⚠️ Warning: Could not convert predicted_class '{predicted_class}' to int")
    else:
        predicted_class_int = 0
    
    # QUICK FIX: Handle image data - only store if it's actual binary data
    explanation_binary = None
    if explanation_image and isinstance(explanation_image, bytes):
        explanation_binary = explanation_image
    elif explanation_image and explanation_image.startswith('data:image'):
        # Skip storing base64 URLs to prevent mixing with predicted_class
        explanation_binary = None
    
    shap_binary = None
    if shap_plot and isinstance(shap_plot, bytes):
        shap_binary = shap_plot
    elif shap_plot and shap_plot.startswith('data:image'):
        # Skip storing base64 URLs
        shap_binary = None
    
    c.execute('''INSERT INTO predictions 
                 (user_id, client_type, prediction_type, predicted_class, 
                  confidence, input_data, explanation_image, shap_plot)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (user_id, client_type, prediction_type, predicted_class_int,
               confidence, str(input_data), explanation_binary, shap_binary))
    
    conn.commit()
    prediction_id = c.lastrowid
    conn.close()
    return prediction_id
@app.route('/fix_db_now')
def fix_db_now():
    """Quick fix for existing bad data"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    # Find and fix bad predicted_class values
    c.execute("SELECT id, predicted_class FROM predictions")
    rows = c.fetchall()
    
    fixed_count = 0
    for row in rows:
        pred_id = row[0]
        pred_class = row[1]
        
        # If it's not an integer or looks like binary data
        if not isinstance(pred_class, int):
            try:
                # Try to convert to int
                if pred_class and len(str(pred_class)) > 3:
                    # Likely binary/string data - set to 0
                    new_value = 0
                else:
                    new_value = int(float(pred_class)) if pred_class else 0
                
                c.execute("UPDATE predictions SET predicted_class = ? WHERE id = ?", 
                         (new_value, pred_id))
                fixed_count += 1
            except:
                # Set to 0 if can't convert
                c.execute("UPDATE predictions SET predicted_class = 0 WHERE id = ?", 
                         (pred_id,))
                fixed_count += 1
    
    conn.commit()
    conn.close()
    
    return f"Fixed {fixed_count} rows in database"

def get_user_predictions(user_id, limit=10):
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('''SELECT * FROM predictions 
                 WHERE user_id = ? 
                 ORDER BY created_at DESC 
                 LIMIT ?''', (user_id, limit))
    predictions = c.fetchall()
    conn.close()
    return predictions

# ===============================
# MODEL LOADING FUNCTIONS
# ===============================
def load_models():
    """Load all ML models"""
    global global_model, xgb_model, feature_extractor, local_models, preprocessors, shap_explainer
    
    try:
        # Load global model
        global_model_path = os.path.join(GLOBAL_MODEL_DIR, "global_fdnn_model.h5")
        if os.path.exists(global_model_path):
            global_model = load_model(global_model_path)
            print("✅ Global FDNN model loaded")
            
            # Create feature extractor
            feature_extractor = Model(
                inputs=global_model.input,
                outputs=global_model.layers[-2].output
            )
        
        # Load XGBoost model
        xgb_model_path = os.path.join(GLOBAL_MODEL_DIR, "xgb_model.pkl")
        if os.path.exists(xgb_model_path):
            xgb_model = joblib.load(xgb_model_path)
            print("✅ XGBoost model loaded")
            
            # Initialize SHAP explainer for XGBoost
            if SHAP_AVAILABLE and xgb_model is not None:
                try:
                    # Create TreeExplainer for XGBoost
                    shap_explainer = shap.TreeExplainer(xgb_model)
                    print("✅ SHAP explainer initialized for XGBoost")
                except Exception as e:
                    print(f"⚠️ Failed to initialize SHAP explainer: {e}")
                    shap_explainer = None
        
        # Load local models
        for client in CLIENTS:
            client_dir = os.path.join(BASE_DIR, client)
            model_path = os.path.join(client_dir, "fdnn_model.h5")
            
            if os.path.exists(model_path):
                local_models[client] = load_model(model_path)
                
                # Load preprocessors
                imputer = joblib.load(os.path.join(client_dir, "imputer.pkl"))
                scaler = joblib.load(os.path.join(client_dir, "scaler.pkl"))
                preprocessors[client] = (imputer, scaler)
                
                print(f"✅ {client} model loaded")
        
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

# ===============================
# DATA PREPROCESSING FUNCTIONS
# ===============================
def cap_outliers_iqr(df, factor=1.5):
    df_capped = df.copy()
    for col in df.columns:
        if col == "Target":
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df_capped[col] = np.clip(df[col], lower, upper)
    return df_capped

def preprocess_client_data(df, client):
    """Preprocess client data from dataframe (for training data)"""
    X = df[CLIENT_FEATURES[client]]
    y = df["Target"]

    # Missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Outlier capping
    X_capped = cap_outliers_iqr(X_imputed)

    # Scaling
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_capped), columns=X.columns)

    # Align features to global
    X_global = pd.DataFrame(0, index=df.index, columns=GLOBAL_FEATURES)
    X_global[CLIENT_FEATURES[client]] = X_scaled

    return X_global, y, imputer, scaler

def preprocess_client_data_for_prediction(client, input_data):
    """Preprocess input data for prediction"""
    try:
        # Get client features
        features = CLIENT_FEATURES[client]
        
        # Create dataframe from input
        X = pd.DataFrame([input_data], columns=features)
        
        # Load preprocessors
        imputer, scaler = preprocessors.get(client, (None, None))
        
        if imputer and scaler:
            # Impute missing values
            X_imputed = pd.DataFrame(imputer.transform(X), columns=features)
            
            # Cap outliers
            X_capped = cap_outliers_iqr(X_imputed)
            
            # Scale
            X_scaled = pd.DataFrame(scaler.transform(X_capped), columns=features)
            
            # Create global-aligned features with float dtype
            X_global = pd.DataFrame(0.0, index=[0], columns=GLOBAL_FEATURES, dtype=np.float64)
            X_global[features] = X_scaled.astype(np.float64)
            
            return X_global.values
        else:
            return None
    except Exception as e:
        print(f"Preprocessing error for {client}: {e}")
        return None

# ===============================
# PREDICTION FUNCTIONS
# ===============================
def predict_local(client, input_data):
    """Make prediction using local model"""
    try:
        # Preprocess input
        X_processed = preprocess_client_data_for_prediction(client, input_data)
        
        if X_processed is None:
            return None, None
        
        # Make prediction
        model = local_models.get(client)
        if model:
            preds = model.predict(X_processed, verbose=0)
            predicted_class = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))
            
            return predicted_class, confidence, X_processed
    except Exception as e:
        print(f"Local prediction error for {client}: {e}")
    
    return None, None, None

def predict_global(input_data_dict):
    """Make prediction using global model (combined features)"""
    try:
        # Create global dataframe with float dtype
        X_global = pd.DataFrame(0.0, index=[0], columns=GLOBAL_FEATURES, dtype=np.float64)
        
        # Process each client's features
        for client in CLIENTS:
            features = CLIENT_FEATURES[client]
            client_data = []
            
            # Extract features for this client from input
            for feature in features:
                if feature in input_data_dict:
                    client_data.append(float(input_data_dict[feature]))
                else:
                    # Use default value if feature not provided
                    client_data.append(0.0)
            
            # Preprocess client data
            X_processed = preprocess_client_data_for_prediction(client, client_data)
            
            if X_processed is not None and X_processed.size > 0:
                # Extract relevant features - ensure proper indexing
                for i, feature in enumerate(features):
                    if feature in GLOBAL_FEATURES:
                        try:
                            idx = GLOBAL_FEATURES.index(feature)
                            X_global.loc[0, feature] = float(X_processed[0, idx])
                        except (IndexError, ValueError) as e:
                            print(f"Error assigning feature {feature}: {e}")
                            X_global.loc[0, feature] = 0.0
        
        # Extract features from global FDNN
        if feature_extractor is not None and X_global is not None:
            features_nn = feature_extractor.predict(X_global, verbose=0)
            
            # Predict with XGBoost
            if xgb_model is not None:
                predicted_class = int(xgb_model.predict(features_nn)[0])
                pred_proba = xgb_model.predict_proba(features_nn)[0]
                confidence = float(np.max(pred_proba))
                
                return predicted_class, confidence, features_nn, X_global
    except Exception as e:
        print(f"Global prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None, None, None

# ===============================
# EXPLAINABILITY FUNCTIONS
# ===============================
def generate_local_shap_plot(client, X_processed, predicted_class):
    """Generate simplified SHAP plot for local model using gradient-based importance"""
    try:
        if X_processed is None:
            return None
        
        # Get model
        model = local_models.get(client)
        if model is None:
            return None
        
        # Convert to tensor for gradient computation
        X_tensor = tf.convert_to_tensor(X_processed, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            preds = model(X_tensor)
            pred_score = preds[0, predicted_class]
        
        # Compute gradients (similar to SHAP for neural networks)
        grads = tape.gradient(pred_score, X_tensor).numpy()[0]
        
        # Get only the features that belong to this client
        client_features = CLIENT_FEATURES[client]
        client_feature_indices = [GLOBAL_FEATURES.index(f) for f in client_features if f in GLOBAL_FEATURES]
        
        # Filter gradients for client features
        client_grads = grads[client_feature_indices]
        client_abs_grads = np.abs(client_grads)
        
        # Filter features with significant gradients (> 0.001)
        significant_indices = np.where(client_abs_grads > 0.001)[0]
        
        if len(significant_indices) == 0:
            # If no significant features, show top 5
            significant_indices = np.argsort(client_abs_grads)[-5:][::-1]
        
        # Get significant feature names and values
        significant_features = [client_features[i] for i in significant_indices]
        significant_grads = client_grads[significant_indices]
        significant_abs_grads = client_abs_grads[significant_indices]
        
        # Sort by absolute gradient value
        sorted_idx = np.argsort(significant_abs_grads)[::-1]
        sorted_features = [significant_features[i] for i in sorted_idx]
        sorted_grads = significant_grads[sorted_idx]
        sorted_abs_grads = significant_abs_grads[sorted_idx]
        
        # Create SHAP-like plot using gradients
        plt.figure(figsize=(12, 8))
        
        # Create subplot for gradient importance
        ax1 = plt.subplot(211)
        colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in sorted_grads]
        bars = ax1.barh(sorted_features, sorted_abs_grads, color=colors)
        
        ax1.set_xlabel('Absolute Gradient Value (Feature Importance)')
        ax1.set_title(f'{client} Model - Important Feature Contributions\nPredicted Class: {predicted_class}')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add gradient value annotations
        for i, (bar, grad_val) in enumerate(zip(bars, sorted_grads)):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{grad_val:.3f}', va='center', fontsize=9,
                    color='red' if grad_val > 0 else 'blue')
        
        # Create subplot for gradient contributions (positive/negative)
        ax2 = plt.subplot(212)
        bars2 = ax2.barh(sorted_features, sorted_grads, color=colors)
        
        ax2.set_xlabel('Gradient Value (Contribution to Prediction)')
        ax2.set_title('Direction of Feature Impact')
        ax2.invert_yaxis()
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add contribution annotations
        for i, (bar, grad_val) in enumerate(zip(bars2, sorted_grads)):
            width = bar.get_width()
            impact = "Increases Risk" if grad_val > 0 else "Decreases Risk"
            ax2.text(width + (0.001 if width >= 0 else -0.001), 
                    bar.get_y() + bar.get_height()/2, 
                    impact, va='center', fontsize=8,
                    ha='left' if width >= 0 else 'right',
                    fontweight='bold')
        
        plt.tight_layout()
        
        # Save to bytes
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        img_bytes.seek(0)
        plt.close()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"Local SHAP plot error for {client}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_local_explanation_plot(client, input_data, predicted_class, X_processed):
    """Generate feature importance plot for local model - showing only important features"""
    try:
        if X_processed is None:
            return None
        
        # Get only the features that belong to this client
        client_features = CLIENT_FEATURES[client]
        client_feature_indices = [GLOBAL_FEATURES.index(f) for f in client_features if f in GLOBAL_FEATURES]
        
        # Get input values for client features
        client_input_values = X_processed[0][client_feature_indices]
        
        # Use absolute input values as importance measure
        importance = np.abs(client_input_values)
        
        # Filter features with significant values (> 0.1)
        significant_indices = np.where(importance > 0.1)[0]
        
        if len(significant_indices) == 0:
            # If no significant features, show top 5
            significant_indices = np.argsort(importance)[-5:][::-1]
        
        # Get significant feature names and values
        significant_features = [client_features[i] for i in significant_indices]
        significant_values = client_input_values[significant_indices]
        significant_importance = importance[significant_indices]
        
        # Sort by importance
        sorted_idx = np.argsort(significant_importance)[::-1]
        sorted_features = [significant_features[i] for i in sorted_idx]
        sorted_values = [significant_values[i] for i in sorted_idx]
        sorted_importance = [significant_importance[i] for i in sorted_idx]
        
        # Create plot with only important features
        plt.figure(figsize=(12, 8))
        
        # Create subplot for feature importance
        ax1 = plt.subplot(211)
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
        bars = ax1.barh(sorted_features, sorted_importance, color=colors)
        
        ax1.set_xlabel('Absolute Value (Feature Importance)')
        ax1.set_title(f'{client} Model - Important Feature Contributions')
        ax1.invert_yaxis()
        
        # Add value annotations
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', va='center', fontsize=9)
        
        # Create subplot for feature values
        ax2 = plt.subplot(212)
        value_colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in sorted_values]
        bars2 = ax2.barh(sorted_features, sorted_values, color=value_colors)
        
        ax2.set_xlabel('Scaled Feature Value')
        ax2.set_title('Input Feature Values')
        ax2.invert_yaxis()
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value annotations
        for i, (bar, val) in enumerate(zip(bars2, sorted_values)):
            width = bar.get_width()
            ax2.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=9, 
                    ha='left' if width >= 0 else 'right')
        
        plt.tight_layout()
        
        # Save to bytes
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', dpi=120, bbox_inches='tight')
        img_bytes.seek(0)
        plt.close()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Local explanation plot error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_shap_summary_plot(features_nn, predicted_class):
    """Generate SHAP summary bar plot for global model - showing only important features"""
    try:
        if not SHAP_AVAILABLE or shap_explainer is None:
            print("SHAP not available or explainer not initialized")
            return None
        
        # Get SHAP values
        shap_values = shap_explainer.shap_values(features_nn)
        
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"Features shape: {features_nn.shape}")
        print(f"Predicted class: {predicted_class}")
        
        # SHAP values shape is (1, 32, 3) - 1 sample, 32 features, 3 classes
        # We need to extract SHAP values for the predicted class
        if len(shap_values.shape) == 3:
            # For multi-class, get SHAP values for the predicted class
            shap_values_class = shap_values[0, :, predicted_class]  # Shape: (32,)
        else:
            shap_values_class = shap_values[0]  # Shape might be (32,) or (32, 3)
        
        print(f"SHAP values for class {predicted_class} shape: {shap_values_class.shape}")
        
        # Calculate absolute SHAP values for feature importance
        shap_abs = np.abs(shap_values_class)
        
        # Get top important features (absolute SHAP > 0.001)
        significant_indices = np.where(shap_abs > 0.001)[0]
        
        if len(significant_indices) == 0:
            # If no significant features, show top 10
            significant_indices = np.argsort(shap_abs)[-10:][::-1]
        
        # Create feature names for FDNN features
        feature_names = [f'FDNN_Feature_{i+1}' for i in range(features_nn.shape[1])]
        
        # Get significant feature names and SHAP values
        significant_features = [feature_names[i] for i in significant_indices]
        significant_shap = shap_values_class[significant_indices]
        significant_abs_shap = shap_abs[significant_indices]
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(significant_abs_shap)[::-1]
        sorted_features = [significant_features[i] for i in sorted_idx]
        sorted_shap = significant_shap[sorted_idx]
        sorted_abs_shap = significant_abs_shap[sorted_idx]
        
        # Create SHAP summary bar plot
        plt.figure(figsize=(12, 8))
        
        # Create subplot for SHAP importance
        ax1 = plt.subplot(211)
        colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in sorted_shap]
        bars = ax1.barh(sorted_features, sorted_abs_shap, color=colors)
        
        ax1.set_xlabel('Absolute SHAP Value (Feature Importance)')
        ax1.set_title(f'Global Model - Important Feature Contributions (SHAP)\nPredicted Class: {predicted_class}')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add SHAP value annotations
        for i, (bar, shap_val) in enumerate(zip(bars, sorted_shap)):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{shap_val:.3f}', va='center', fontsize=9,
                    color='red' if shap_val > 0 else 'blue')
        
        # Create subplot for SHAP contributions (positive/negative)
        ax2 = plt.subplot(212)
        bars2 = ax2.barh(sorted_features, sorted_shap, color=colors)
        
        ax2.set_xlabel('SHAP Value (Contribution to Prediction)')
        ax2.set_title('Direction of Feature Impact')
        ax2.invert_yaxis()
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add contribution annotations
        for i, (bar, shap_val) in enumerate(zip(bars2, sorted_shap)):
            width = bar.get_width()
            impact = "Increases Risk" if shap_val > 0 else "Decreases Risk"
            ax2.text(width + (0.001 if width >= 0 else -0.001), 
                    bar.get_y() + bar.get_height()/2, 
                    impact, va='center', fontsize=8,
                    ha='left' if width >= 0 else 'right',
                    fontweight='bold')
        
        plt.tight_layout()
        
        # Save to bytes
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        img_bytes.seek(0)
        plt.close()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"SHAP summary plot error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_global_feature_importance_plot(input_data_dict, X_global):
    """Generate feature importance plot for global model based on input values"""
    try:
        # Convert input dictionary to arrays
        features = list(input_data_dict.keys())
        values = np.array(list(input_data_dict.values()))
        
        # Calculate absolute values as importance measure
        importance = np.abs(values)
        
        # Filter features with significant values (> 0.1)
        significant_indices = np.where(importance > 0.1)[0]
        
        if len(significant_indices) == 0:
            # If no significant values, show top 8 features
            significant_indices = np.argsort(importance)[-8:][::-1]
        
        # Get significant features and values
        significant_features = [features[i] for i in significant_indices]
        significant_values = values[significant_indices]
        significant_importance = importance[significant_indices]
        
        # Sort by importance
        sorted_idx = np.argsort(significant_importance)[::-1]
        sorted_features = [significant_features[i] for i in sorted_idx]
        sorted_values = [significant_values[i] for i in sorted_idx]
        sorted_importance = [significant_importance[i] for i in sorted_idx]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create subplot for feature values
        ax1 = plt.subplot(211)
        value_colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in sorted_values]
        bars1 = ax1.barh(sorted_features, sorted_values, color=value_colors)
        
        ax1.set_xlabel('Input Feature Value')
        ax1.set_title('Global Model - Important Input Features')
        ax1.invert_yaxis()
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value annotations
        for i, (bar, val) in enumerate(zip(bars1, sorted_values)):
            width = bar.get_width()
            ax1.text(width + (0.01 if width >= 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{val:.2f}', va='center', fontsize=9,
                    ha='left' if width >= 0 else 'right')
        
        # Create subplot for feature importance
        ax2 = plt.subplot(212)
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
        bars2 = ax2.barh(sorted_features, sorted_importance, color=colors)
        
        ax2.set_xlabel('Absolute Value (Importance)')
        ax2.set_title('Feature Impact Magnitude')
        ax2.invert_yaxis()
        
        # Add importance annotations
        for i, (bar, imp) in enumerate(zip(bars2, sorted_importance)):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{imp:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save to bytes
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', dpi=120, bbox_inches='tight')
        img_bytes.seek(0)
        plt.close()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"Global feature importance plot error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===============================
# FLASK ROUTES
# ===============================
@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        full_name = request.form.get('full_name', '')
        
        # Validate input
        if not username or not email or not password:
            flash('Please fill all required fields', 'danger')
            return redirect(url_for('register'))
        
        # Hash password
        password_hash = hash_password(password)
        
        # Save to database
        try:
            conn = sqlite3.connect(app.config['DATABASE'])
            c = conn.cursor()
            c.execute('''INSERT INTO users (username, email, password_hash, full_name)
                         VALUES (?, ?, ?, ?)''',
                     (username, email, password_hash, full_name))
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'danger')
        except Exception as e:
            flash(f'Registration error: {str(e)}', 'danger')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Verify credentials
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and verify_password(password, user[1]):
            session['user_id'] = user[0]
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    user_data = get_user_data(session['user_id'])
    predictions = get_user_predictions(session['user_id'], limit=5)
    
    # Get model status
    models_loaded = all([global_model is not None, 
                        xgb_model is not None, 
                        len(local_models) == len(CLIENTS)])
    
    return render_template('dashboard.html',
                         user=user_data,
                         predictions=predictions,
                         models_loaded=models_loaded,
                         clients=CLIENTS)

@app.route('/predict/local', methods=['POST'])
@login_required
def predict_local_route():
    """Make local prediction"""
    try:
        client = request.form.get('client')
        if client not in CLIENTS:
            return jsonify({'error': 'Invalid client type'}), 400
        
        # Extract features for this client
        features = CLIENT_FEATURES[client]
        input_data = []
        
        for feature in features:
            value = request.form.get(feature, 0)
            try:
                input_data.append(float(value))
            except:
                input_data.append(0.0)
        
        # Make prediction
        predicted_class, confidence, X_processed = predict_local(client, input_data)
        
        if predicted_class is not None:
            # Generate explanation plots - showing only important features
            explanation_plot = generate_local_explanation_plot(client, input_data, predicted_class, X_processed)
            shap_plot = generate_local_shap_plot(client, X_processed, predicted_class)
            
            # Save prediction to database
            input_data_str = {features[i]: input_data[i] for i in range(len(features))}
            shap_plot_base64 = shap_plot if shap_plot else None
            save_prediction(session['user_id'], client, 'local', 
                          predicted_class, confidence, str(input_data_str), 
                          explanation_plot, shap_plot_base64)
            
            # Class labels
            class_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            
            return jsonify({
                'success': True,
                'client': client,
                'predicted_class': int(predicted_class),
                'class_label': class_labels[predicted_class],
                'confidence': round(confidence * 100, 2),
                'explanation_plot': explanation_plot,
                'shap_plot': shap_plot,
                'input_features': input_data_str,
                'shap_available': SHAP_AVAILABLE,
                'local_shap_available': True  # Always available with gradient-based method
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/global', methods=['POST'])
@login_required
def predict_global_route():
    """Make global prediction"""
    try:
        # Extract all features from form
        input_data_dict = {}
        for feature in GLOBAL_FEATURES:
            value = request.form.get(feature, 0)
            try:
                input_data_dict[feature] = float(value)
            except:
                input_data_dict[feature] = 0.0
        
        print(f"Global prediction input: {input_data_dict}")
        
        # Make prediction
        predicted_class, confidence, features_nn, X_global = predict_global(input_data_dict)
        
        print(f"Global prediction result: class={predicted_class}, confidence={confidence}")
        
        if predicted_class is not None:
            # Generate explanation plots - showing only important features
            shap_plot = generate_shap_summary_plot(features_nn, predicted_class)
            feature_importance_plot = generate_global_feature_importance_plot(input_data_dict, X_global)
            
            # Save prediction to database
            shap_plot_base64 = shap_plot if shap_plot else None
            save_prediction(session['user_id'], 'global', 'global', 
                          predicted_class, confidence, str(input_data_dict), 
                          feature_importance_plot, shap_plot_base64)
            
            # Class labels
            class_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            
            response_data = {
                'success': True,
                'predicted_class': int(predicted_class),
                'class_label': class_labels[predicted_class],
                'confidence': round(confidence * 100, 2),
                'explanation_plot': feature_importance_plot,
                'shap_plot': shap_plot,
                'input_features': input_data_dict,
                'shap_available': SHAP_AVAILABLE,
                'shap_explainer_loaded': shap_explainer is not None
            }
            
            print(f"Response data prepared, SHAP plot: {shap_plot is not None}")
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Global prediction failed'}), 500
            
    except Exception as e:
        print(f"Global prediction route error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/shap_demo')
@login_required
def shap_demo():
    """Generate SHAP demo plots with sample data"""
    try:
        if not SHAP_AVAILABLE or shap_explainer is None:
            return jsonify({'error': 'SHAP not available'}), 400
        
        # Create sample data
        sample_data = {
            'Animal_Population': 1500,
            'Animal_Mortality': 23,
            'Vaccination_Rate': 85.5,
            'Temperature': 28.5,
            'Humidity': 65.2,
            'Rainfall': 120.5,
            'Contact_Index': 0.75,
            'Reported_Cases': 45,
            'Mortality_Rate': 3.2,
            'Symptom_Count': 5,
            'Admissions': 12,
            'Positivity_Rate': 15.7,
            'Species_Count': 8,
            'Migration_Flag': 1,
            'Habitat_Proximity': 0.65,
            'Wildlife_Mortality': 7,
            'Exposure_Score': 0.45
        }
        
        # Make prediction
        predicted_class, confidence, features_nn, _ = predict_global(sample_data)
        
        if predicted_class is not None:
            # Generate SHAP plots
            shap_plot = generate_shap_summary_plot(features_nn, predicted_class)
            
            return jsonify({
                'success': True,
                'shap_plot': shap_plot,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
        else:
            return jsonify({'error': 'Failed to generate demo'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
@login_required
def prediction_history():
    """View prediction history"""
    predictions = get_user_predictions(session['user_id'], limit=50)
    return render_template('history.html', predictions=predictions)

@app.route('/api/model_status')
def model_status():
    """Check model loading status"""
    return jsonify({
        'global_model_loaded': global_model is not None,
        'xgb_model_loaded': xgb_model is not None,
        'local_models_loaded': len(local_models),
        'total_clients': len(CLIENTS),
        'shap_available': SHAP_AVAILABLE,
        'shap_explainer_loaded': shap_explainer is not None
    })


@app.route('/api/prediction/<int:prediction_id>')
@login_required
def get_prediction_details(prediction_id):
    """Get detailed prediction information"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        
        c.execute('''SELECT * FROM predictions 
                     WHERE id = ? AND user_id = ?''', 
                  (prediction_id, session['user_id']))
        prediction = c.fetchone()
        
        if prediction:
            # Parse input data
            import json
            input_data = json.loads(prediction[6]) if prediction[6] else {}
            
            # Get explanation images if available
            explanation_image = None
            shap_plot = None
            
            if prediction[7]:  # explanation_image blob
                explanation_image = base64.b64encode(prediction[7]).decode('utf-8')
                explanation_image = f"data:image/png;base64,{explanation_image}"
            
            if prediction[8]:  # shap_plot blob
                shap_plot = base64.b64encode(prediction[8]).decode('utf-8')
                shap_plot = f"data:image/png;base64,{shap_plot}"
            
            return jsonify({
                'success': True,
                'prediction': {
                    'id': prediction[0],
                    'client_type': prediction[2],
                    'prediction_type': prediction[3],
                    'predicted_class': prediction[4],
                    'confidence': prediction[5],
                    'input_data': input_data,
                    'created_at': prediction[9],
                    'explanation_image': explanation_image,
                    'shap_plot': shap_plot
                }
            })
        else:
            return jsonify({'error': 'Prediction not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/delete_prediction/<int:prediction_id>', methods=['DELETE'])
@login_required
def delete_prediction(prediction_id):
    """Delete a prediction"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        
        c.execute('''DELETE FROM predictions 
                     WHERE id = ? AND user_id = ?''', 
                  (prediction_id, session['user_id']))
        
        conn.commit()
        conn.close()
        
        if c.rowcount > 0:
            return jsonify({'success': True, 'message': 'Prediction deleted'})
        else:
            return jsonify({'error': 'Prediction not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

 
# ===============================
# ERROR HANDLERS
# ===============================
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# ===============================
# APPLICATION INITIALIZATION
# ===============================
if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Load ML models
    if load_models():
        print("✅ All models loaded successfully")
    else:
        print("⚠️ Some models failed to load")
    
    # Check SHAP availability
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP library not installed. Install with: pip install shap")
    
    # Create uploads directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run application
    app.run(debug=True, host='0.0.0.0', port=5000)