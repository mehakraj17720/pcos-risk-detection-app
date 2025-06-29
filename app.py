import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# Load trained model
model = pickle.load(open("models/pcos_model.pkl", "rb"))

# Define feature columns
feature_columns = [
    "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI",
    "AMH(ng/mL)", "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH",
    "TSH (mIU/L)", "PRL(ng/mL)", "RBS(mg/dl)", "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)", "Hair loss(Y/N)", "Pimples(Y/N)", "Weight gain(Y/N)",
    "Pregnant(Y/N)", "No. of abortions", "Cycle(R/I)", "Cycle length(days)"
]

app = Flask(__name__)

@app.route("/")
def welcome():
    return render_template("welcome.html")  # Welcome page

@app.route("/form")
def form():
    return render_template("home.html")  # Input form page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        input_data = [float(request.form[col]) for col in feature_columns if col in request.form]

        # Ensure input matches model expectation
        if len(input_data) < len(feature_columns):
            input_data.extend([0] * (len(feature_columns) - len(input_data)))  # Fill missing with 0

        # Convert to numpy array & reshape
        input_array = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)[0]

        # Render result page with prediction
        result = "Positive for PCOS" if prediction == 1 else "Negative for PCOS"
        return render_template("result.html", prediction=result)
    
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/cure")
def cure():
    return render_template("cure.html")  # Cure page for PCOS

@app.route("/nextpage")
def nextpage():
    return render_template("nextpage.html")  # Next page after prediction


if __name__ == "__main__":
    app.run(debug=True)
