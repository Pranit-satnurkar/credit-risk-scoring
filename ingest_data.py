import pandas as pd
import sqlite3
import os

# Define file paths
CSV_FILE = "raw_data.csv"
DB_FILE = "bank_data.db"
TABLE_NAME = "loan_applications"

# Standard UCI Column Names (Since the raw file doesn't have them)
COLUMN_NAMES = [
    "checking_account", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_status", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "people_liable", "telephone", "foreign_worker", "target"
]


def ingest_data():
    print("🚀 Starting Data Ingestion Pipeline...")

    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: File '{CSV_FILE}' not found.")
        return

    try:
        print(f"📄 Reading {CSV_FILE}...")

        # KEY FIX: The raw German data is space-separated, not comma-separated
        df = pd.read_csv(CSV_FILE, sep=' ', header=None, names=COLUMN_NAMES)

        print(f"✅ Data Loaded. Shape: {df.shape} (Rows, Columns)")

        # Clean up the Target column (1=Good, 2=Bad -> Convert to 0=Good, 1=Bad for AI)
        # In the original data: 1 = Good, 2 = Bad
        # We want: 0 = Good, 1 = Default (Bad)
        df['target'] = df['target'].map({1: 0, 2: 1})
        print("🔄 Target column encoded: 0=Paid, 1=Default")

    except Exception as e:
        print(f"❌ Failed to read Data: {e}")
        return

    try:
        print(f"💾 Saving to SQLite database: {DB_FILE}...")
        conn = sqlite3.connect(DB_FILE)
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        conn.close()
        print(f"✅ Success! Data stored in table: '{TABLE_NAME}'")

    except Exception as e:
        print(f"❌ Database Error: {e}")


def verify_db():
    print("\n🔍 Verifying Data inside SQL...")
    conn = sqlite3.connect(DB_FILE)
    try:
        query = f"SELECT * FROM {TABLE_NAME} LIMIT 5"
        sql_df = pd.read_sql(query, conn)
        print(sql_df)
    except Exception as e:
        print(e)
    finally:
        conn.close()


if __name__ == "__main__":
    ingest_data()
    verify_db()
