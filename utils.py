import pickle
import pandas as pd

def load_artifacts():
    folder = 'artifacts'
    with open(f'{folder}/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    with open(f'{folder}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(f'{folder}/model.pkl', 'rb') as f:
        model = pickle.load(f)

    return pipeline, scaler, model


def predict(data_dict: dict, pipeline, scaler, model):
    # Prediction Pipeline
    new_data = pd.DataFrame(data_dict)

    # Apply the preprocessing pipeline used during training
    new_data_preprocessed = pipeline.transform(new_data)

    # If required, scale the preprocessed data (if using standardized data for prediction)
    new_data_scaled = scaler.transform(new_data_preprocessed)

    # Predict using the best model (replace `best_model` with the actual model object)
    prediction = model.predict(new_data_scaled)  # For labels
    prediction_proba = model.predict_proba(new_data_scaled)[:, 1]  # For probabilities, if applicable

    # Output predictions # 0:good 1:bad
    return int(prediction[0]), float(prediction_proba[0])


#format :
""""{
    'Status_Checking_Account': ['A11'],
    'Duration': [12],
    'Credit_History': ['A34'],
    'Purpose': ['A43'],
    'Credit_Amount': [3000],
    'Savings_Account': ['A61'],
    'Employment': ['A73'],
    'Installment_Rate': [4],
    'Personal_Status_Sex': ['A92'],
    'Other_Debtors': ['A101'],
    'Residence_Since': [4],
    'Property': ['A124'],
    'Age': [32],
    'Other_Installment_Plans': ['A143'],
    'Housing': ['A152'],
    'Existing_Credits': [2],
    'Job': ['A173'],
    'Liable_People': [1],
    'Telephone': ['A192'],
    'Foreign_Worker': ['A201']
}"""