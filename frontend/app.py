import io
import pandas as pd
import streamlit as st

from api_client import predict
from api_client import predict_csv
from api_client import health


st.set_page_config(
    page_title="Student Health Prediction",
    layout="wide"
)


st.sidebar.title(" Student Health Prediction")

status = health()

if status:

    st.sidebar.success("Backend Connected")

else:

    st.sidebar.error("Backend Not Running")


st.sidebar.markdown("---")

st.sidebar.write("### Model")

st.sidebar.info(
"""
**Algorithm**

XGBoost

**Validation Accuracy**

96.77%

**Backend**

FastAPI

**Frontend**

Streamlit
"""
)




st.title("Student Health Risk Prediction System")

st.write(
"""
Predict whether a student belongs to the **Fit**, **At-risk** or
**Unhealthy** category using lifestyle and health measurements.
"""
)

st.markdown("---")

tab1, tab2 = st.tabs(
[
"🧍 Single Prediction",
"📄 Batch Prediction"
]
)


with tab1:

    left, right = st.columns([3, 2])

    with left:

        st.markdown("### Input Features")
        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        with c1:
            bmi = st.slider("BMI", 10.0, 50.0, 23.5)
        with c2:
            heart_rate = st.slider("Resting heart rate (bpm)", 30, 220, 72)
        with c3:
            exercise_duration = st.slider("Exercise (min)", 0, 300, 30)

        c1, c2 = st.columns(2)
        with c1:
            sleep_duration = st.slider(
                "Sleep duration (hours)",
                0.0,
                24.0,
                7.0,
                help="Recommended: 6–9 hours"
            )
        with c2:
            water_intake = st.slider("Water (L/day)", 0.0, 10.0, 2.0)

        c1, c2 = st.columns(2)
        with c1:
            step_count = st.number_input("Daily steps", 0, 50000, 8000)
        with c2:
            calorie_expenditure = st.number_input("Calories (kcal)", 500, 6000, 2200)

        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox(
                "Gender",
                ["male", "female", "other", "missing"]
            )
        with c2:
            physical_activity_level = st.selectbox(
                "Activity level",
                ["sedentary", "moderate", "active", "missing"]
            )
        with c3:
            sleep_quality = st.selectbox(
                "Sleep quality",
                ["poor", "average", "good", "missing"]
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            diet_type = st.selectbox(
                "Diet type",
                ["balanced", "veg", "non-veg", "missing"]
            )
        with c2:
            smoking_alcohol = st.selectbox(
                "Smoking / alcohol use",
                ["no", "occasional", "yes", "missing"]
            )
        with c3:
            stress_level = st.selectbox(
                "Stress level",
                ["low", "medium", "high", "missing"]
            )

        st.markdown("---")

        predict_clicked = st.button(
            " Run Prediction",
            use_container_width=True
        )

    with right:

        st.markdown("### Prediction")
        st.markdown("---")

        if predict_clicked:

            payload = {

                "sleep_duration": sleep_duration,
                "heart_rate": heart_rate,
                "bmi": bmi,
                "calorie_expenditure": calorie_expenditure,
                "step_count": step_count,
                "exercise_duration": exercise_duration,
                "water_intake": water_intake,
                "diet_type": diet_type,
                "stress_level": stress_level,
                "sleep_quality": sleep_quality,
                "physical_activity_level": physical_activity_level,
                "smoking_alcohol": smoking_alcohol,
                "gender": gender

            }

            with st.spinner("Generating Prediction..."):

                result = predict(payload)

            if "error" in result:

                st.error(result["error"])

            else:

                prediction = result["predicted_health_condition"]

                if prediction == "fit":
                    condition_color = "green"
                    explanation = (
                        "Your indicators suggest you're maintaining a healthy "
                        "lifestyle. Keep up the good habits around sleep, "
                        "activity, and diet."
                    )

                elif prediction == "at-risk":
                    condition_color = "orange"
                    explanation = (
                        "You may be at risk of developing health issues. "
                        "Consider improving your lifestyle by increasing "
                        "physical activity, maintaining a balanced diet, "
                        "managing stress, and getting adequate sleep."
                    )

                else:
                    condition_color = "red"
                    explanation = (
                        "Your indicators suggest significant health risks. "
                        "Consider consulting a healthcare professional and "
                        "making changes across activity, diet, sleep, and "
                        "stress management."
                    )

                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div style="text-align:center;">
                            <div style="font-size:16px; font-weight:600;">Predicted Condition</div>
                            <div style="font-size:28px; font-weight:700; color:{condition_color}; margin-top:8px;">
                                {prediction.upper()}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("**Class Probabilities**")

                probs = result["probabilities"]

                sorted_probs = sorted(
                    probs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for label, value in sorted_probs:

                    pct = value * 100

                    st.write(f"{label}")

                    bc1, bc2 = st.columns([5, 1])

                    with bc1:
                        st.progress(value)

                    with bc2:
                        st.write(f"{pct:.1f}%")

                st.info("Probabilities represent the model's confidence for each class.")

                with st.container(border=True):
                    st.markdown("**What does this mean?**")
                    st.write(explanation)

        else:

            st.info(
                "Fill in the form and click **Run Prediction** to see results here."
            )


with tab2:

    st.subheader("Batch Prediction")

    st.write(
"""
Upload the Kaggle **test.csv** file to generate predictions for
multiple students.
"""
)

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type="csv"
    )

    if uploaded_file is not None:

        preview = pd.read_csv(uploaded_file)

        st.write("Preview")

        st.dataframe(
            preview.head(),
            use_container_width=True
        )

        if st.button(
            "🚀 Run Batch Prediction",
            use_container_width=True
        ):

            uploaded_file.seek(0)

            with st.spinner("Predicting..."):

                csv_data = predict_csv(
                    uploaded_file
                )

            result_df = pd.read_csv(
                io.BytesIO(csv_data)
            )

            st.success(
                "Prediction Completed"
            )

            st.dataframe(
                result_df.head(20),
                use_container_width=True
            )

            st.download_button(
                "⬇ Download predictions.csv",
                csv_data,
                "predictions.csv",
                "text/csv"
            )

            st.markdown("### Prediction Distribution")

            st.bar_chart(
                result_df[
                    "predicted_health_condition"
                ].value_counts()
            )

