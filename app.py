import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #0D9488 0%, #065A82 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f0fdfa;
        border-left: 5px solid #0D9488;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .churn-high {
        background: #fff5f5;
        border-left: 5px solid #F96167;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .churn-low {
        background: #f0fdfa;
        border-left: 5px solid #0D9488;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0D9488;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_model():
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle


bundle = load_model()
model = bundle["model"]
le_ff = bundle["le_ff"]
le_ai = bundle["le_ai"]
le_as = bundle["le_as"]
le_bh = bundle["le_bh"]


st.markdown(
    """
<div class="main-header">
    <h1>✈️ Customer Churn Prediction</h1>
    <p style="font-size:1.1rem; opacity:0.9;">
        Powered by Random Forest — Enter customer details to predict churn probability
    </p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.metric("Algorithm", "Random Forest")
    st.metric("Accuracy", "89.01%")
    st.metric("AUC Score", "0.9537")
    st.metric("Training Samples", "763")
    st.metric("Testing Samples", "191")

    st.markdown("---")
    st.markdown("## 📌 Feature Guide")
    st.info(
        """
    **Services Opted**: Number of travel services the customer uses (1 = few, 6 = many)

    **Frequent Flyer**: Whether the customer is registered as a frequent flyer

    **Account Synced**: Whether their account is linked to social media
    """
    )

st.markdown(
    '<div class="section-header">👤 Enter Customer Details</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=18, max_value=65, value=30, step=1)
    frequent_flyer = st.selectbox("Frequent Flyer", options=["Yes", "No", "No Record"])

with col2:
    annual_income = st.selectbox(
        "Annual Income Class", options=["Low Income", "Middle Income", "High Income"]
    )
    services_opted = st.slider(
        "Services Opted", min_value=1, max_value=6, value=3, step=1
    )

with col3:
    account_synced = st.selectbox(
        "Account Synced to Social Media", options=["Yes", "No"]
    )
    booked_hotel = st.selectbox("Booked Hotel or Not", options=["Yes", "No"])

st.markdown("---")


if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):

    ff_enc = le_ff.transform([frequent_flyer])[0]
    ai_enc = le_ai.transform([annual_income])[0]
    as_enc = le_as.transform([account_synced])[0]
    bh_enc = le_bh.transform([booked_hotel])[0]

    input_data = pd.DataFrame(
        [[age, ff_enc, ai_enc, services_opted, as_enc, bh_enc]],
        columns=[
            "Age",
            "FrequentFlyer",
            "AnnualIncomeClass",
            "ServicesOpted",
            "AccountSyncedToSocialMedia",
            "BookedHotelOrNot",
        ],
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    churn_prob = probability[1] * 100
    stay_prob = probability[0] * 100

    st.markdown(
        '<div class="section-header">📋 Prediction Result</div>', unsafe_allow_html=True
    )

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(
                f"""
            <div class="churn-high">
                <h2>⚠️ HIGH CHURN RISK</h2>
                <h1 style="color:#F96167; font-size:3rem;">{churn_prob:.1f}%</h1>
                <p>Probability of churning</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="churn-low">
                <h2>✅ LOW CHURN RISK</h2>
                <h1 style="color:#0D9488; font-size:3rem;">{churn_prob:.1f}%</h1>
                <p>Probability of churning</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with res_col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        categories = ["Stayed", "Churned"]
        values = [stay_prob, churn_prob]
        bar_colors = ["#0D9488", "#F96167"]
        bars = ax.barh(
            categories, values, color=bar_colors, height=0.5, edgecolor="white"
        )
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=13,
                fontweight="bold",
            )
        ax.set_xlim(0, 110)
        ax.set_xlabel("Probability (%)", fontsize=11)
        ax.set_title("Prediction Probabilities", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown(
        '<div class="section-header">📊 What Drives This Prediction?</div>',
        unsafe_allow_html=True,
    )

    feature_names = [
        "Age",
        "Frequent Flyer",
        "Annual Income Class",
        "Services Opted",
        "Account Synced",
        "Booked Hotel",
    ]
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    colors = ["#0D9488"] * len(feature_names)
    ax2.bar(
        [feature_names[i] for i in sorted_idx],
        [importances[i] for i in sorted_idx],
        color=colors,
        edgecolor="white",
    )
    for i, (idx, val) in enumerate(
        zip(sorted_idx, [importances[j] for j in sorted_idx])
    ):
        ax2.text(
            i, val + 0.003, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold"
        )
    ax2.set_title("Feature Importance (Global)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Importance Score")
    ax2.tick_params(axis="x", rotation=15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown(
        '<div class="section-header">💡 Business Recommendations</div>',
        unsafe_allow_html=True,
    )

    if prediction == 1:
        st.warning(
            """
        **Retention Actions Recommended:**
        - 🎁 Offer a personalised discount or loyalty points bonus within 48 hours
        - 📧 Send a targeted email campaign highlighting unused premium services
        - 📞 Assign a customer success manager for high-value at-risk customers
        - 🏨 Provide a complimentary hotel upgrade on their next booking
        """
        )
    else:
        st.success(
            """
        **Customer Retention Status: Healthy**
        - ✅ This customer shows strong loyalty indicators
        - 📈 Consider upselling additional service tiers (move from current services opted)
        - 🌟 Enrol in a premium frequent flyer tier if not already a member
        - 📣 Ask for a referral — satisfied customers are your best brand ambassadors
        """
        )


st.markdown("---")
with st.expander("📁 View Dataset Sample & Statistics"):
    try:
        df_sample = pd.read_csv("customer_travel_churn.csv")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**First 10 Rows**")
            st.dataframe(df_sample.head(10), use_container_width=True)
        with col_b:
            st.markdown("**Summary Statistics**")
            st.dataframe(df_sample.describe(), use_container_width=True)
        churn_rate = df_sample["Target"].mean() * 100
        st.metric(
            "Overall Churn Rate",
            f"{churn_rate:.1f}%",
            delta=f"{df_sample['Target'].sum()} churned out of {len(df_sample)} customers",
        )
    except FileNotFoundError:
        st.info(
            "Place customer_travel_churn.csv in the same folder to see dataset stats."
        )

st.markdown("---")
st.caption(
    "Customer Churn Prediction App | Built with Streamlit | Random Forest Classifier | B.Tech Gen AI Project"
)
