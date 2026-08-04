# Student Health Risk Prediction System

## Overview

The **Student Health Risk Prediction System** is a machine learning-based web application developed to predict a student's health condition using lifestyle and health-related information.

The system classifies students into one of three health conditions:

- Fit
- At-risk
- Unhealthy

The application uses an XGBoost Classification Model trained on the Kaggle Student Health Risk Prediction Dataset. The trained model is deployed using FastAPI, while Streamlit provides an interactive web interface for users to obtain real-time predictions.

---

# Technologies Used

- Python
- XGBoost
- Scikit-learn
- FastAPI
- Streamlit
- Pandas
- NumPy
- Joblib
- Requests
- Pydantic
- Git & GitHub

---

# Project Structure

```
Student_Health_Prediction_Model/
│
├── Backend/
│   ├── api.py
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── schemas.py
│   ├── config.py
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   └── api_client.py
│
├── Models/
│   ├── xgboost_model.pkl
│   ├── feature_encoders.pkl
│   ├── preprocessing_info.pkl
│   ├── feature_names.pkl
│   └── target_encoder.pkl
│
├── Dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Random_Forest_Model.ipynb
│   ├── MLP_Classifier.ipynb
│   ├── LGB_Model.ipynb
│   ├── XGBoost_Model.ipynb
│   └── Kaggle_Submission.ipynb
│
├── Submissions/
│
└── README.md
```

---

# Software Requirements

The project was developed using the following software.

| Software | Version |
|----------|---------|
| Python | 3.13 |
| Visual Studio Code | Latest |
| Git | Latest |
| Google Chrome / Edge | Latest |

---

# Python Library Requirements

Install the required Python libraries.

```
fastapi
uvicorn
streamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
joblib
requests
pydantic
jupyter
notebook
```

or simply install all required packages using:

```bash
pip install -r Backend/requirements.txt
```

---

# Installation

## Step 1

Clone the repository.

```bash
git clone https://github.com/udananirmal12-cell/Student-Health-Risk-Prediction.git
```

or download the project as a ZIP file.

---

## Step 2

Navigate to the project directory.

```bash
cd Student_Health_Prediction_Model
```

---

## Step 3

Install the required libraries.

```bash
pip install -r Backend/requirements.txt
```

If Streamlit is not installed:

```bash
pip install streamlit
```

---

# Training the Machine Learning Model (Optional)

The repository already contains a pre-trained XGBoost model.

Retraining is optional and only required if you wish to reproduce the training process or experiment with different parameters.

## Dataset

Download the Kaggle Student Health Risk Prediction dataset and place the following files inside the **Dataset** folder.

```
train.csv
test.csv
sample_submission.csv
```

---

## Train the Model

Open the notebook:

```
Notebooks/XGBoost_Model.ipynb
```

Run all notebook cells sequentially.

The notebook performs:

- Data loading
- Data preprocessing
- Missing value handling
- Feature encoding
- Model training
- Model evaluation
- Model saving

After training, the following files will be generated inside the **Models** folder.

```
xgboost_model.pkl
feature_encoders.pkl
feature_names.pkl
preprocessing_info.pkl
target_encoder.pkl
```

These files are required by the FastAPI backend.

---

# Running the Application

The application consists of two components.

- FastAPI Backend
- Streamlit Frontend

Start the backend first.

---

# Running the Backend

Open a terminal.

Navigate to the Backend folder.

```bash
cd Backend
```

Run the API server.

```bash
python -m uvicorn api:app --reload
```

The backend will start at:

```
http://127.0.0.1:8000
```

Swagger API Documentation:

```
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

Open another terminal.

Navigate to the frontend folder.

```bash
cd frontend
```

Run Streamlit.

```bash
streamlit run app.py
```

The application will automatically open in your web browser.

---

# Application Workflow

1. The user enters health and lifestyle information through the Streamlit interface.

2. Streamlit sends the input data to the FastAPI backend using an HTTP POST request.

3. FastAPI validates the request using Pydantic schemas.

4. The preprocessing module applies the same preprocessing pipeline used during model training.

5. The trained XGBoost model generates the prediction.

6. The prediction probabilities are calculated.

7. The prediction results are returned to Streamlit.

8. Streamlit displays the predicted health condition and class probabilities.

---

# Machine Learning Model

| Item | Description |
|------|-------------|
| Algorithm | XGBoost Classifier |
| Validation Accuracy | 96.77% |
| Output Classes | Fit, At-risk, Unhealthy |

---

# Input Features

### Numerical Features

- Sleep Duration
- Heart Rate
- BMI
- Calorie Expenditure
- Step Count
- Exercise Duration
- Water Intake

### Categorical Features

- Diet Type
- Stress Level
- Sleep Quality
- Physical Activity Level
- Smoking / Alcohol
- Gender

---

# Output

The application returns:

- Predicted Health Condition
- Probability of each health condition

Example:

```
Predicted Health Condition

AT-RISK

Prediction Probabilities

Fit          12%
At-risk      82%
Unhealthy     6%
```

---

# Features

The application supports:

- Single student prediction
- Batch CSV prediction
- Download prediction results
- Probability estimation
- REST API
- Interactive Streamlit interface
- Automatic input validation

---

# Notes

- Start the FastAPI backend before launching the Streamlit frontend.
- The trained model and preprocessing files must remain inside the **Models** folder.
- Retraining the model is optional because a pre-trained model is already included.
- The backend uses the same preprocessing pipeline that was applied during model training to ensure consistent predictions.

---
