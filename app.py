import streamlit as st
import pandas as pd
import joblib

# ---------- Page config ----------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Custom CSS (adapts to light & dark mode) ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    :root {
        --card-bg: rgba(99, 102, 241, 0.06);
        --card-border: rgba(99, 102, 241, 0.18);
        --accent-a: #6366f1;
        --accent-b: #8b5cf6;
        --header-a: #f59e0b;
        --header-b: #ec4899;
        --muted-text: rgba(120, 120, 140, 0.9);
        --approve-bg: rgba(16, 185, 129, 0.10);
        --approve-border: rgba(16, 185, 129, 0.45);
        --approve-text: #059669;
        --reject-bg: rgba(239, 68, 68, 0.10);
        --reject-border: rgba(239, 68, 68, 0.45);
        --reject-text: #dc2626;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --muted-text: #94a3b8;
            --approve-text: #34d399;
            --reject-text: #f87171;
        }
    }

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 780px;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .app-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--header-a), var(--header-b));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-header p {
        color: var(--muted-text);
        font-size: 0.95rem;
    }

    /* Section card */
    .section-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.3rem 1.5rem 0.6rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.0rem;
        font-weight: 600;
        color: var(--accent-a);
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Inputs */
    div[data-baseweb="select"] > div, .stNumberInput input {
        border-radius: 10px !important;
    }

    /* Predict button */
    div.stButton > button, .stFormSubmitButton > button {
        width: 100%;
        background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.7rem 0;
        border-radius: 12px;
        border: none;
        margin-top: 0.6rem;
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
        color: white;
    }

    /* Result cards */
    .result-approved {
        background: var(--approve-bg);
        border: 1px solid var(--approve-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-rejected {
        background: var(--reject-bg);
        border: 1px solid var(--reject-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-title-approved {
        color: var(--approve-text);
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-title-rejected {
        color: var(--reject-text);
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-sub {
        color: var(--muted-text);
        font-size: 0.95rem;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_pipeline():
    return joblib.load("loan model.pkl")

pipe = load_pipeline()

# ---------- Header ----------
st.markdown("""
<div class="app-header">
    <h1>Loan Approval Predictor</h1>
    <p>Fill in the applicant details below to check loan eligibility instantly</p>
</div>
""", unsafe_allow_html=True)

# ---------- Form ----------
with st.form("loan_form"):

    st.markdown('<div class="section-card"><div class="section-title">👤 Personal Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=28, step=1)
    with c2:
        person_gender = st.selectbox("Gender", ["male", "female"])
    with c3:
        person_education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">💼 Financial Background</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        person_income = st.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=60000, step=1000)
    with c2:
        person_emp_exp = st.number_input("Employment Experience (yrs)", min_value=0, max_value=50, value=3, step=1)
    with c3:
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
    with c2:
        cb_person_cred_hist_length = st.number_input("Credit History Length (yrs)", min_value=0, max_value=40, value=4, step=1)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🏦 Loan Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        loan_amnt = st.number_input("Loan Amount (PKR)", min_value=500, max_value=100000, value=10000, step=500)
    with c2:
        loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    c1, c2 = st.columns(2)
    with c1:
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=11.0, step=0.1)
    with c2:
        loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Predict Loan Status")

# ---------- Prediction ----------
if submitted:
    input_df = pd.DataFrame([{
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
    }])

    prediction = pipe.predict(input_df)[0]
    proba = pipe.predict_proba(input_df)[0]
    approval_prob = proba[1] * 100
    rejection_prob = proba[0] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-approved">
            <div class="result-title-approved">Loan Approved</div>
            <div class="result-sub">Approval confidence: <b>{approval_prob:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <div class="result-title-rejected">Loan Rejected</div>
            <div class="result-sub">Rejection confidence: <b>{rejection_prob:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.progress(int(approval_prob))
    st.caption(f"Approval probability: {approval_prob:.1f}% · Rejection probability: {rejection_prob:.1f}%")