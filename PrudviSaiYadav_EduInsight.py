"""
EduInsight — Student Performance Decision Dashboard
Single-file Python application for data analytics, visualization and prediction.

Dataset:
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

Place StudentPerformanceFactors.csv in the same folder as this file.
Run:
    pip install -r requirements.txt
    streamlit run eduinsight.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="EduInsight | Student Performance BI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://www.kaggle.com/datasets/lainguyn123/student-performance-factors"
DATA_FILE = "StudentPerformanceFactors.csv"

EXPECTED = [
    "Hours_Studied","Attendance","Parental_Involvement","Access_to_Resources",
    "Extracurricular_Activities","Sleep_Hours","Previous_Scores","Motivation_Level",
    "Internet_Access","Tutoring_Sessions","Family_Income","Teacher_Quality",
    "School_Type","Peer_Influence","Physical_Activity","Learning_Disabilities",
    "Parental_Education_Level","Distance_from_Home","Gender","Exam_Score"
]

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        source = "Kaggle dataset"
    except Exception:
        # Small demo fallback so the application remains runnable.
        rng = np.random.default_rng(42)
        n = 250
        df = pd.DataFrame({
            "Hours_Studied": rng.integers(5, 36, n),
            "Attendance": rng.integers(55, 101, n),
            "Parental_Involvement": rng.choice(["Low","Medium","High"], n),
            "Access_to_Resources": rng.choice(["Low","Medium","High"], n),
            "Extracurricular_Activities": rng.choice(["Yes","No"], n),
            "Sleep_Hours": rng.integers(4, 11, n),
            "Previous_Scores": rng.integers(50, 101, n),
            "Motivation_Level": rng.choice(["Low","Medium","High"], n),
            "Internet_Access": rng.choice(["Yes","No"], n, p=[0.9,0.1]),
            "Tutoring_Sessions": rng.integers(0, 6, n),
            "Family_Income": rng.choice(["Low","Medium","High"], n),
            "Teacher_Quality": rng.choice(["Low","Medium","High"], n),
            "School_Type": rng.choice(["Public","Private"], n),
            "Peer_Influence": rng.choice(["Negative","Neutral","Positive"], n),
            "Physical_Activity": rng.integers(0, 7, n),
            "Learning_Disabilities": rng.choice(["Yes","No"], n, p=[0.1,0.9]),
            "Parental_Education_Level": rng.choice(["High School","College","Postgraduate"], n),
            "Distance_from_Home": rng.choice(["Near","Moderate","Far"], n),
            "Gender": rng.choice(["Male","Female"], n),
        })
        score = (
            38 + 0.65*df["Hours_Studied"] + 0.18*df["Attendance"]
            + 0.20*df["Previous_Scores"] + 0.35*df["Sleep_Hours"]
            + 0.55*df["Tutoring_Sessions"] + rng.normal(0, 3, n)
        )
        score += df["Motivation_Level"].map({"Low":-2,"Medium":1,"High":3})
        score += df["Access_to_Resources"].map({"Low":-1,"Medium":1,"High":2})
        df["Exam_Score"] = np.clip(score, 35, 100).round().astype(int)
        source = "Built-in demo data (download the Kaggle CSV for the full analysis)"

    # Basic cleaning
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].replace(["", "NA", "N/A", "na", "null"], np.nan)
    if "Exam_Score" in df:
        df["Exam_Score"] = pd.to_numeric(df["Exam_Score"], errors="coerce")
    df = df.dropna(subset=["Exam_Score"]).reset_index(drop=True)
    return df, source

def train_model(df):
    X = df.drop(columns=["Exam_Score"])
    y = df["Exam_Score"]
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = RandomForestRegressor(
        n_estimators=180, random_state=42, max_depth=10, n_jobs=-1
    )
    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return pipe, mae, r2, numeric, categorical

def risk_band(score):
    if score < 60:
        return "High Support Need", "Prioritize targeted academic support."
    if score < 70:
        return "Medium Support Need", "Monitor attendance, study habits and engagement."
    return "Lower Support Need", "Maintain current habits and monitor progress."

df, source = load_data()

st.markdown("""
<style>
.main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
.subtitle {font-size: 1.05rem; opacity: .75; margin-bottom: 1rem;}
.kpi {padding: 16px; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);}
.small {font-size: .85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 EduInsight</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Student Performance Business Intelligence & Decision Dashboard</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Dashboard Controls")
    if st.button("🔄 Refresh / Recalculate"):
        st.cache_data.clear()
        st.rerun()
    st.caption(source)
    st.divider()
    st.markdown("**Decision flow**")
    st.write("Data → KPIs → Trends → Drivers → Risks → Actions")

# Filters
st.subheader("1. Executive Overview")
fdf = df.copy()
with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        genders = sorted(fdf["Gender"].dropna().unique()) if "Gender" in fdf else []
        gender = st.multiselect("Gender", genders, default=genders)
    with c2:
        schools = sorted(fdf["School_Type"].dropna().unique()) if "School_Type" in fdf else []
        school = st.multiselect("School Type", schools, default=schools)
    with c3:
        motivation = sorted(fdf["Motivation_Level"].dropna().unique()) if "Motivation_Level" in fdf else []
        motiv = st.multiselect("Motivation", motivation, default=motivation)

if gender:
    fdf = fdf[fdf["Gender"].isin(gender)]
if school:
    fdf = fdf[fdf["School_Type"].isin(school)]
if motiv:
    fdf = fdf[fdf["Motivation_Level"].isin(motiv)]

avg_score = fdf["Exam_Score"].mean() if len(fdf) else 0
attendance = fdf["Attendance"].mean() if len(fdf) else 0
study = fdf["Hours_Studied"].mean() if len(fdf) else 0
high_support = int((fdf["Exam_Score"] < 60).sum()) if len(fdf) else 0

k1,k2,k3,k4 = st.columns(4)
k1.metric("Students", f"{len(fdf):,}")
k2.metric("Average Exam Score", f"{avg_score:.1f}")
k3.metric("Average Attendance", f"{attendance:.1f}%")
k4.metric("High-Support Students", f"{high_support:,}")

left, right = st.columns(2)
with left:
    fig = px.histogram(
        fdf, x="Exam_Score", nbins=20,
        title="Exam Score Distribution", marginal="box"
    )
    st.plotly_chart(fig, use_container_width=True)
with right:
    grp = fdf.groupby("Motivation_Level", dropna=False)["Exam_Score"].mean().reset_index()
    fig = px.bar(grp, x="Motivation_Level", y="Exam_Score",
                 title="Average Score by Motivation Level", text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("2. Sales/Product-style Analysis → Academic Drivers")
d1, d2 = st.columns(2)
with d1:
    fig = px.scatter(
        fdf, x="Hours_Studied", y="Exam_Score",
        size="Attendance", color="Motivation_Level",
        hover_data=["Previous_Scores","Tutoring_Sessions"],
        title="Study Hours vs Exam Score"
    )
    st.plotly_chart(fig, use_container_width=True)
with d2:
    grp = fdf.groupby("Access_to_Resources")["Exam_Score"].mean().reset_index()
    fig = px.bar(grp, x="Access_to_Resources", y="Exam_Score",
                 title="Average Score by Resource Access", text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

# Model / drivers
st.subheader("3. Drivers, Risk & Recommended Actions")
try:
    model, mae, r2, numeric, categorical = train_model(df)
    st.caption(f"Random Forest validation — MAE: {mae:.2f} points | R²: {r2:.3f}")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp["Feature"] = imp["Feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    top = imp.sort_values("Importance", ascending=False).head(10)

    fig = px.bar(top.sort_values("Importance"), x="Importance", y="Feature",
                 orientation="h", title="Top Predictive Drivers")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Model could not be trained: {e}")

r1, r2c = st.columns(2)
with r1:
    threshold = st.slider("Support-risk threshold", 50, 75, 60)
    risk_count = int((fdf["Exam_Score"] < threshold).sum())
    risk_pct = (risk_count / len(fdf) * 100) if len(fdf) else 0
    st.metric("Students below threshold", f"{risk_count:,}", f"{risk_pct:.1f}% of filtered group")
    if risk_count:
        st.info(
            "Risk action: identify students below the threshold and prioritize "
            "attendance checks, study-plan support, tutoring and resource access."
        )
    else:
        st.success("No students are below the selected threshold in the current filter.")
with r2c:
    st.markdown("**Action playbook**")
    actions = [
        ("Attendance", "Monitor students with low attendance and create early follow-up."),
        ("Study time", "Offer structured study plans where study hours are low."),
        ("Previous score", "Use previous performance to prioritize mentoring."),
        ("Resources", "Check access to learning resources for lower-performing groups."),
        ("Motivation", "Use engagement activities and regular progress feedback."),
    ]
    for area, action in actions:
        st.write(f"**{area}:** {action}")

st.subheader("4. Individual Prediction")
with st.form("prediction_form"):
    a,b,c,d = st.columns(4)
    with a:
        hours = st.number_input("Hours Studied", 0, 50, 20)
        attendance_in = st.number_input("Attendance %", 0, 100, 80)
        prev = st.number_input("Previous Score", 0, 100, 70)
        sleep = st.number_input("Sleep Hours", 0, 14, 7)
    with b:
        tutoring = st.number_input("Tutoring Sessions", 0, 10, 2)
        motivation_in = st.selectbox("Motivation", ["Low","Medium","High"])
        resources_in = st.selectbox("Resources", ["Low","Medium","High"])
        parental = st.selectbox("Parental Involvement", ["Low","Medium","High"])
    with c:
        extra = st.selectbox("Extracurricular", ["Yes","No"])
        internet = st.selectbox("Internet Access", ["Yes","No"])
        income = st.selectbox("Family Income", ["Low","Medium","High"])
        teacher = st.selectbox("Teacher Quality", ["Low","Medium","High"])
    with d:
        school_in = st.selectbox("School Type", ["Public","Private"])
        peer = st.selectbox("Peer Influence", ["Negative","Neutral","Positive"])
        activity = st.number_input("Physical Activity", 0, 15, 3)
        disability = st.selectbox("Learning Disability", ["Yes","No"])
    predict = st.form_submit_button("Predict Exam Score")

if predict:
    row = pd.DataFrame([{
        "Hours_Studied": hours, "Attendance": attendance_in,
        "Parental_Involvement": parental, "Access_to_Resources": resources_in,
        "Extracurricular_Activities": extra, "Sleep_Hours": sleep,
        "Previous_Scores": prev, "Motivation_Level": motivation_in,
        "Internet_Access": internet, "Tutoring_Sessions": tutoring,
        "Family_Income": income, "Teacher_Quality": teacher,
        "School_Type": school_in, "Peer_Influence": peer,
        "Physical_Activity": activity, "Learning_Disabilities": disability,
        "Parental_Education_Level": "College", "Distance_from_Home": "Near",
        "Gender": "Male"
    }])
    try:
        prediction = float(model.predict(row)[0])
        band, action = risk_band(prediction)
        st.success(f"Predicted exam score: **{prediction:.1f} / 100**")
        st.write(f"**Risk band:** {band}")
        st.write(f"**Recommended action:** {action}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "EduInsight is an educational analytics prototype. Predictions are decision-support "
    "outputs, not definitive judgments about individual students."
)
