import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS to allow requests from your webpage
CORS(app)

# --- Load the prediction bundle once when the server starts ---
try:
    prediction_bundle = joblib.load('prediction_bundle.joblib')
    print("Prediction bundle loaded successfully. Server is ready.")
except Exception as e:
    prediction_bundle = None
    print(f"FATAL ERROR: Could not load 'prediction_bundle.joblib'. Error: {e}")

# --- Physical Constants ---
H_MASS_U = 1.00782503223
N_MASS_U = 1.00866491595
U_TO_MEV = 931.49410242

def get_prediction(z, n, bundle):
    """This is your original Python prediction logic."""
    if bundle is None:
        return None, "Model bundle not loaded on the server."

    models = bundle['models']
    weights = bundle['weights']
    lookup_table = bundle['lookup_table']
    model_config = bundle['model_config']
    
    try:
        nucleus_data = lookup_table.loc[(z, n)]
    except KeyError:
        return None, f"Nucleus with (Z={z}, N={n}) not found."

    required_features = ['Z', 'N', 'A', 'A23', 'I', 'vz', 'vn', 'pf', 'Zeo', 'Neo', 'Zshell', 'Nshell']
    feature_data = nucleus_data.copy()
    feature_data['Z'], feature_data['N'] = z, n
    X_new = pd.DataFrame([feature_data[required_features]])
    
    all_corrected_masses = []
    for i, row in model_config.iterrows():
        model_key = f"{row['models'].replace('+', '_plus')}_{row['opt']}"
        loaded_model = models[model_key]
        predicted_correction = loaded_model.predict(X_new)[0]
        theoretical_mass = nucleus_data[f"MEth_{row['models']}"]
        all_corrected_masses.append(theoretical_mass + predicted_correction)

    final_mass = np.average(all_corrected_masses, weights=weights)
    A = z + n
    atomic_mass = A + (final_mass / U_TO_MEV)
    binding_energy = (z * H_MASS_U + n * N_MASS_U - atomic_mass) * U_TO_MEV
    
    return {
        'Z': z, 'N': n, 'A': A,
        'Corrected_Mass_Excess_MeV': final_mass,
        'Binding_Energy_MeV': binding_energy,
        'Binding_Energy_per_Nucleon_MeV': binding_energy / A if A > 0 else 0
    }, None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'z' not in data or 'n' not in data:
        return jsonify({'error': 'Invalid input.'}), 400

    prediction, error_msg = get_prediction(data['z'], data['n'], prediction_bundle)
    if error_msg:
        return jsonify({'error': error_msg}), 404
    
    return jsonify(prediction)

@app.route('/')
def index():
    return "Nuclear Predictor API is running."
