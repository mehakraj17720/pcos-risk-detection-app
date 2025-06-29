import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open("models/pcos_model.pkl", "rb"))

# Sample input data (ensure it matches the feature columns)
sample_data = pd.DataFrame([{
    "Age (yrs)": 25,
    "Weight (Kg)": 70,
    "Height(Cm)": 165,
    "BMI": 25.7,
    "AMH(ng/mL)": 4.5,
    "FSH(mIU/mL)": 6.2,
    "LH(mIU/mL)": 8.5,
    "FSH/LH": 6.2 / 8.5,
    "TSH (mIU/L)": 2.1,
    "PRL(ng/mL)": 18.3,
    "RBS(mg/dl)": 90,
    "BP _Systolic (mmHg)": 120,
    "BP _Diastolic (mmHg)": 80,
    "Hair loss(Y/N)": 1,
    "Pimples(Y/N)": 0,
    "Weight gain(Y/N)": 1,
    "Pregnant(Y/N)": 0,
    "No. of abortions": 0,
    "Cycle(R/I)": 1,  # Replace with actual value
    "Cycle length(days)": 30
}])

# Ensure the columns match the trained model
expected_features = model.feature_names_in_
sample_data = sample_data[expected_features]

# Make prediction
prediction = model.predict(sample_data)
print("Predicted PCOS (1=Yes, 0=No):", prediction[0])
