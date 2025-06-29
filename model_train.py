import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle
import os

# ✅ Load dataset
file_path = "data/PCOS_data.csv"
df = pd.read_csv(file_path)

# ✅ Clean column names 
df.columns = df.columns.str.strip()

# ✅ Drop unnecessary columns if they exist
drop_cols = ["Sl. No", "Patient File No.", "Unnamed: 44"]       
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# ✅ Convert numeric columns properly
numeric_cols = ["AMH(ng/mL)", "FSH(mIU/mL)", "LH(mIU/mL)", "TSH (mIU/L)", "PRL(ng/mL)", "RBS(mg/dl)"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ✅ Ensure FSH/LH ratio exists
if "FSH(mIU/mL)" in df.columns and "LH(mIU/mL)" in df.columns:  
    df["FSH/LH"] = df["FSH(mIU/mL)"] / df["LH(mIU/mL)"]

# ✅ Convert categorical Yes/No columns to 1/0
yes_no_cols = ["Pregnant(Y/N)", "Weight gain(Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)"]
for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].map({"Y": 1, "N": 0})
        df[col] = df[col].fillna(0)

# ✅ Convert Cycle (R/I) to numeric values
if "Cycle(R/I)" in df.columns:
    df["Cycle(R/I)"] = df["Cycle(R/I)"].map({"R": 1, "I": 0})   
    df["Cycle(R/I)"] = df["Cycle(R/I)"].fillna(0)

# ✅ Handle missing values for numeric columns
num_imputer = SimpleImputer(strategy="median")
numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# ✅ Define feature columns (only keeping existing ones)
feature_columns = [
    "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI", "AMH(ng/mL)", "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH",
    "TSH (mIU/L)", "PRL(ng/mL)", "RBS(mg/dl)", "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)",
    "Hair loss(Y/N)", "Pimples(Y/N)", "Weight gain(Y/N)", "Pregnant(Y/N)", "No. of abortions",
    "Cycle(R/I)", "Cycle length(days)"
]

feature_columns = [col for col in feature_columns if col in df.columns]

# ✅ Select features and target
X = df[feature_columns]
y = df.get("PCOS (Y/N)")

# ✅ Drop any remaining NaN values
X = X.dropna()
y = y.loc[X.index]

# ✅ Balance the dataset using SMOTE (Only if there are multiple classes)
if len(y.value_counts()) > 1:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    X_resampled, y_resampled = X, y

# ✅ Split data (Check if there's enough data)
if not X_resampled.empty and len(X_resampled) > 5:
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

# ✅ Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

# ✅ Save model
    os.makedirs("models", exist_ok=True)
    with open("models/pcos_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved successfully!")

else:
    print("⚠️ Not enough data after preprocessing! Check missing values and dataset.")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Predict on test data
y_pred = model.predict(X_test)

# ✅ Evaluate performance
print("\n📊 Evaluation Results:")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
