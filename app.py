import pandas as pd
import joblib
import streamlit as st

st.title("Customer Next Month Spend Prediction")

# load the tuned XGBoost model, the cleaned data (for input defaults) and the model's column order
model = joblib.load("models/xgboost_optuna.pkl")
# columns dropped in 03_feature_engineering.ipynb, so the app doesn't ask for them
dropped = ["spend_next_month", "gender", "avg_order_value_6m", "snapshot_month", "snapshot_quarter", "email_data_available"]
data = pd.read_csv("data/processed/cleaned.csv").drop(columns=dropped)
model_columns = pd.read_csv("data/processed/X_test.csv", nrows=0).columns
income_median = data.groupby(["region", "membership_tier"])["annual_income"].median()


def predict(df):
    df = df[data.columns].copy()

    # same missing value handling as 01_preprocessing.ipynb
    df["education"] = df["education"].fillna("Unknown")
    group_income = pd.Series(list(zip(df["region"], df["membership_tier"])), index=df.index).map(income_median)
    df["annual_income"] = df["annual_income"].fillna(group_income).fillna(data["annual_income"].median())
    df["spend_trend_ratio"] = df["spend_trend_ratio"].fillna(data["spend_trend_ratio"].median())
    df["email_open_rate"] = df["email_open_rate"].fillna(0)
    df["email_click_rate"] = df["email_click_rate"].fillna(0)

    # same feature engineering as 03_feature_engineering.ipynb
    df["spend_momentum"] = df["spend_last_1m"] / (df["avg_spend_6m"] + 1)
    df["spend_per_order_1m"] = df["spend_last_1m"] / (df["orders_last_1m"] + 1)
    df["recent_buyer"] = (df["days_since_last_purchase"] <= 7).astype(int)
    df["inactive_30d"] = (df["days_since_last_purchase"] > 30).astype(int)
    df["holiday_month"] = df["target_month"].isin([11, 12]).astype(int)
    df["promo_x_discount"] = df["promo_planned_next_month"] * df["discount_share"]
    df["total_engagement"] = df["app_sessions_30d"] + df["site_visits_30d"]
    df["income_per_member"] = df["annual_income"] / df["household_size"]

    # ordinal encoding for ordered categories (same as 03)
    df["education_unknown"] = (df["education"] == "Unknown").astype(int)
    df["education"] = df["education"].replace("Unknown", "Bachelor")
    df["education"] = df["education"].map({"High school": 0, "Bachelor": 1, "Master": 2, "PhD": 3})
    df["membership_tier"] = df["membership_tier"].map({"Basic": 0, "Silver": 1, "Gold": 2})

    # one-hot encode the rest and line up columns with what the model was trained on
    df = pd.get_dummies(df, dtype=int).reindex(columns=model_columns, fill_value=0)
    return model.predict(df).clip(min=0).astype(float).round(2)


tab1, tab2 = st.tabs(["Single customer", "Upload CSV"])

# ---------- one customer, typed in by hand ----------
with tab1:
    inputs = {}
    cols = st.columns(3)
    for i, col in enumerate(data.columns):
        with cols[i % 3]:
            if data[col].dtype == "object":
                inputs[col] = st.selectbox(col, sorted(data[col].unique()))
            else:
                inputs[col] = st.number_input(col, value=float(data[col].median()))

    if st.button("Predict"):
        prediction = predict(pd.DataFrame([inputs]))[0]
        st.success(f"Predicted next month spend: {prediction:,.2f}")

# ---------- many customers from a CSV file ----------
with tab2:
    st.write("Upload a CSV with the same columns as the dataset (extra columns like `customer_id` are fine).")
    file = st.file_uploader("Choose a CSV file", type="csv")

    if file is not None:
        uploaded = pd.read_csv(file)
        if "snapshot_year" not in uploaded.columns and "snapshot_date" in uploaded.columns:
            uploaded["snapshot_year"] = pd.to_datetime(uploaded["snapshot_date"]).dt.year
        missing = [c for c in data.columns if c not in uploaded.columns]

        if missing:
            st.error(f"These columns are missing from the file: {', '.join(missing)}")
        else:
            uploaded["predicted_spend_next_month"] = predict(uploaded)
            st.success(f"Predicted {len(uploaded)} customers")
            st.dataframe(uploaded)
            st.download_button("Download predictions", uploaded.to_csv(index=False),
                               file_name="predictions.csv", mime="text/csv")
