import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="🏦",
    layout="centered",  # 'centered' looks more like a form/app than 'wide'
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR "CLEAN" LOOK ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #333;
    }
    .big-font {
        font-size:20px !important;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD BRAIN ---
MODEL_FILE = "credit_model.xgb"
COLUMNS_FILE = "model_columns.pkl"


@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_FILE)
        model_cols = joblib.load(COLUMNS_FILE)
        return model, model_cols
    except Exception as e:
        return None, None


model, model_columns = load_resources()

if not model:
    st.error("🚨 Model not found! Please run `train_model.py` first.")
    st.stop()

# --- HEADER ---
st.title("🏦 CreditGuard AI")
st.markdown("<p class='big-font'>Enterprise-Grade Loan Approval System</p>",
            unsafe_allow_html=True)
st.markdown("---")

# --- MAIN INPUT FORM (No Sidebar) ---
st.subheader("👤 Applicant Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 75, 30)
    check_acc = st.selectbox("Checking Account Status",
                             ["No Account", "Negative Balance", "Low Balance", "High Balance"])
    cred_hist = st.selectbox("Credit History",
                             ["Critical Account", "Existing Credits Paid", "Delay in Past", "No Credits Taken"])

with col2:
    job_type = st.selectbox("Employment Level", [
                            "Unskilled", "Skilled", "Management/Self-Employed", "Unemployed"])
    savings = st.selectbox("Savings Account",
                           ["Unknown", "< $100", "$100 - $500", "$500 - $1000", "> $1000"])
    purpose = st.selectbox("Loan Purpose",
                           ["Car (New)", "Car (Used)", "Furniture", "Education", "Business", "Renovation"])

st.markdown("### 💰 Loan Details")
col3, col4 = st.columns(2)
with col3:
    amount = st.number_input("Loan Amount ($)", 500, 20000, 4000, step=100)
with col4:
    duration = st.slider("Duration (Months)", 6, 72, 24)

# --- PREPROCESSING ---
# (Mapping inputs to the logic the model understands)
data = {
    'duration_months': duration,
    'credit_amount': amount,
    'age': age,
    'installment_rate': 3, 'residence_since': 2, 'existing_credits': 1, 'people_liable': 1,  # Defaults
    'checking_account': 0, 'credit_history': 0, 'purpose': 0, 'savings_account': 0,  # Placeholders
    'employment_status': 2, 'personal_status_sex': 2, 'other_debtors': 0, 'property': 1,
    'other_installment_plans': 2, 'housing': 1, 'job': 2, 'telephone': 1, 'foreign_worker': 0
}

# Logic mapping (Simplified for UI)
if "Negative" in check_acc:
    data['checking_account'] = 0
elif "Low" in check_acc:
    data['checking_account'] = 1
else:
    data['checking_account'] = 3

# DataFrame Creation
input_df = pd.DataFrame([data])
final_df = pd.DataFrame(columns=model_columns)
for col in final_df.columns:
    if col in input_df.columns:
        final_df[col] = input_df[col]
    else:
        final_df[col] = 0

# --- ACTION BUTTON ---
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Assess Risk Profile"):

    # Prediction
    prediction = model.predict(final_df)[0]
    probability = model.predict_proba(final_df)[0][1]  # Probability of Default

    # --- RESULTS SECTION ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.write("### Decision")
        if prediction == 0:
            st.success("✅ APPROVED")
            st.metric("Safety Score", f"{(1-probability)*100:.1f}%")
        else:
            st.error("⛔ REJECTED")
            st.metric("Risk Probability", f"{probability*100:.1f}%")

    with res_col2:
        st.write("### Why this decision?")
        with st.spinner("Calculating Explainability factors..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(final_df)

            # Custom SHAP Plot
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.summary_plot(shap_values, final_df,
                              plot_type="bar", show=False, color="#0066cc")
            st.pyplot(fig)
            st.caption(
                "The bars show which factors pushed the AI towards Yes (Positive) or No (Negative).")
