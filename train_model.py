import pandas as pd
import sqlite3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DB_FILE = "bank_data.db"
MODEL_FILE = "credit_model.xgb"


def train():
    print("🚀 Connecting to Data Warehouse...")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM loan_applications", conn)
    conn.close()

    # 1. Preprocessing: Convert "A11", "A12" to 0, 1, 2...
    print("🧹 Encoding categorical data...")
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        # Save the encoder to reverse it later if needed
        label_encoders[col] = le

    # 2. Split Data
    X = df.drop('target', axis=1)
    y = df['target']  # 0 = Paid, 1 = Default

    print(f"📊 Original Class Distribution:\n{y.value_counts()}")

    # 3. Handle Imbalance (SMOTE)
    # This creates "Synthetic" examples of Defaults so the AI doesn't just guess "Paid"
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"⚖️  Balanced Class Distribution:\n{y_resampled.value_counts()}")

    # Split into Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # 4. Train XGBoost
    print("🧠 Training XGBoost Model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # 5. Evaluate
    print("\n✅ Model Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 6. Save the Brain
    # We save the model AND the column names (to ensure the dashboard input matches)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(X.columns.tolist(), "model_columns.pkl")
    print(f"💾 Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    train()
