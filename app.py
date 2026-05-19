import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Retention and Academic Risk Analytics",
    layout="wide"
)

# ---------------- SESSION STATE ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_type" not in st.session_state:
    st.session_state.user_type = None

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    data = pd.read_csv("StudentsPerformance.csv")
    return data

data = load_data()

# ---------------- CREATE RISK LEVEL ----------------

data["average_score"] = (
    data["math score"] +
    data["reading score"] +
    data["writing score"]
) / 3

def risk_category(score):

    if score >= 75:
        return "Low Risk"

    elif score >= 50:
        return "Medium Risk"

    else:
        return "High Risk"

data["risk_level"] = data["average_score"].apply(risk_category)

# ---------------- ENCODING ----------------

ml_data = data.copy()

label_cols = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

encoders = {}

for col in label_cols:

    le = LabelEncoder()

    ml_data[col] = le.fit_transform(ml_data[col])

    encoders[col] = le

risk_encoder = LabelEncoder()

ml_data["risk_level_encoded"] = risk_encoder.fit_transform(
    ml_data["risk_level"]
)

# ---------------- FEATURES ----------------

X = ml_data[
    [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "math score",
        "reading score",
        "writing score"
    ]
]

y = ml_data["risk_level_encoded"]

# ---------------- TRAIN MODEL ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------- LOGIN DETAILS ----------------

STUDENT_USERNAME = "student"
STUDENT_PASSWORD = "1234"

TEACHER_USERNAME = "teacher"
TEACHER_PASSWORD = "admin123"

# ---------------- SIDEBAR ----------------

st.sidebar.title("Login Panel")

login_type = st.sidebar.selectbox(
    "Select User Type",
    ["Student", "Teacher"]
)

username = st.sidebar.text_input("Username")

password = st.sidebar.text_input(
    "Password",
    type="password"
)

login_button = st.sidebar.button("Login")

logout_button = st.sidebar.button("Logout")

# ---------------- LOGOUT ----------------

if logout_button:

    st.session_state.logged_in = False
    st.session_state.user_type = None

# ---------------- LOGIN VALIDATION ----------------

if login_button:

    if (
        login_type == "Student"
        and username == STUDENT_USERNAME
        and password == STUDENT_PASSWORD
    ):

        st.session_state.logged_in = True
        st.session_state.user_type = "Student"

    elif (
        login_type == "Teacher"
        and username == TEACHER_USERNAME
        and password == TEACHER_PASSWORD
    ):

        st.session_state.logged_in = True
        st.session_state.user_type = "Teacher"

    else:

        st.error("Invalid Username or Password")

# ---------------- TITLE ----------------

st.title(
    "A Data-Driven Framework for Student Retention and Academic Risk Analytics"
)

st.write(
    "This system uses Machine Learning, Dashboard Analytics, "
    "and Real Kaggle Dataset for student risk prediction."
)

# ---------------- STUDENT PAGE ----------------

if (
    st.session_state.logged_in
    and st.session_state.user_type == "Student"
):

    st.success("Student Login Successful")

    st.header("Student Prediction System")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            data["gender"].unique()
        )

        race = st.selectbox(
            "Race / Ethnicity",
            data["race/ethnicity"].unique()
        )

        parent_edu = st.selectbox(
            "Parental Level of Education",
            data["parental level of education"].unique()
        )

        lunch = st.selectbox(
            "Lunch Type",
            data["lunch"].unique()
        )

        attendance = st.slider(
            "Attendance Percentage",
            0,
            100,
            70
        )

    with col2:

        test_prep = st.selectbox(
            "Test Preparation Course",
            data["test preparation course"].unique()
        )

        math_score = st.slider(
            "Math Score",
            0,
            100,
            70
        )

        reading_score = st.slider(
            "Reading Score",
            0,
            100,
            70
        )

        writing_score = st.slider(
            "Writing Score",
            0,
            100,
            70
        )

    if st.button("Predict Academic Risk"):

        input_data = pd.DataFrame({

            "gender": [
                encoders["gender"].transform([gender])[0]
            ],

            "race/ethnicity": [
                encoders["race/ethnicity"].transform([race])[0]
            ],

            "parental level of education": [
                encoders["parental level of education"].transform(
                    [parent_edu]
                )[0]
            ],

            "lunch": [
                encoders["lunch"].transform([lunch])[0]
            ],

            "test preparation course": [
                encoders["test preparation course"].transform(
                    [test_prep]
                )[0]
            ],

            "math score": [math_score],
            "reading score": [reading_score],
            "writing score": [writing_score]
        })

        prediction = model.predict(input_data)

        risk = risk_encoder.inverse_transform(prediction)[0]

        average = (
            math_score +
            reading_score +
            writing_score
        ) / 3

        st.subheader("Prediction Result")

        st.write("Average Score:", round(average, 2))

        # Weakest Subject

        scores = {
            "Math": math_score,
            "Reading": reading_score,
            "Writing": writing_score
        }

        weakest_subject = min(scores, key=scores.get)

        # Attendance Analysis

        required_attendance = 75

        if attendance < required_attendance:

            needed = required_attendance - attendance

        else:

            needed = 0

        # Risk Display

        if risk == "High Risk":

            st.error(f"Academic Risk Level: {risk}")

        elif risk == "Medium Risk":

            st.warning(f"Academic Risk Level: {risk}")

        else:

            st.success(f"Academic Risk Level: {risk}")

        # Suggestions

        st.info(
            f"You should concentrate more on {weakest_subject} subject."
        )

        if needed > 0:

            st.warning(
                f"You need approximately {needed}% more attendance to reach safe academic level."
            )

        else:

            st.success(
                "Your attendance level is safe."
            )

# ---------------- TEACHER PAGE ----------------

elif (
    st.session_state.logged_in
    and st.session_state.user_type == "Teacher"
):

    st.success("Teacher Login Successful")

    st.header("Teacher Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Students",
            len(data)
        )

    with col2:

        st.metric(
            "Average Math Score",
            round(data["math score"].mean(), 2)
        )

    with col3:

        st.metric(
            "Average Reading Score",
            round(data["reading score"].mean(), 2)
        )

    with col4:

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    st.subheader("Student Dataset")

    st.dataframe(data)

    st.subheader("Risk Distribution")

    risk_counts = data["risk_level"].value_counts()

    st.bar_chart(risk_counts)

    st.subheader("Subject Score Analytics")

    subject_scores = data[
        [
            "math score",
            "reading score",
            "writing score"
        ]
    ].mean()

    st.bar_chart(subject_scores)

    st.subheader("Gender-wise Performance")

    gender_avg = data.groupby("gender")[
        [
            "math score",
            "reading score",
            "writing score"
        ]
    ].mean()

    st.dataframe(gender_avg)

    st.bar_chart(gender_avg)

    st.subheader("High Risk Students")

    high_risk = data[
        data["risk_level"] == "High Risk"
    ]

    st.dataframe(high_risk)

# ---------------- DEFAULT ----------------

else:

    st.info("Please Login from Sidebar")

# ---------------- FOOTER ----------------

st.markdown("---")

st.write(
    "Student Retention and Academic Risk Analytics System"
)