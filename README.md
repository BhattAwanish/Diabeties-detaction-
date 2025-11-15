🩺 Diabetes Detection AI — Machine Learning Based Disease Prediction

This project is a machine learning–powered diabetes detection system, built using Random Forest Classifier, Python, and Streamlit.
It allows users to upload patient health data (CSV format) and instantly receive a diabetes risk prediction along with probability scores.

🚀 Features

✅ Upload CSV file containing patient data
✅ Automatic preprocessing (missing value handling, zero correction, scaling)
✅ Machine Learning predictions using Random Forest
✅ Displays uploaded patient data in a clean UI table
✅ Shows prediction + probability for each patient
✅ Allows downloading prediction results as CSV
✅ Beautiful pastel hospital-themed UI
✅ Fully interactive Streamlit web app

📁 Supported Input Columns

Your CSV file should contain the following optional/required columns:

Glucose

BloodPressure

SkinThickness

Insulin

BMI

Age

Sex

Outcome (optional)

The app will automatically fill missing numeric fields using column mean.

🧠 Machine Learning Model

The app uses a Random Forest Classifier, trained on the PIMA Diabetes Dataset (or your custom dataset).
Input features are preprocessed using:

Standard Scaling

Missing value imputation

Zero-value normalization

🖥️ How It Works

Upload your patient CSV file

App processes and cleans the data

Random Forest model predicts diabetes outcome

Shows results in a styled table

Download results as a CSV

Simple, fast, and intuitive.

🎨 UI & Design

The interface uses a custom-designed pastel hospital theme, including:

Soft yellow background

Blue healthcare accent colors

Custom success banners

Styled HTML data tables

Clean card-based layout

Designed for clarity and ease of use.

📦 Technologies Used

Python

Streamlit

Pandas

NumPy

Scikit-learn

Joblib

HTML + Custom CSS

🧪 Demo

You can run the app using:

streamlit run app.py

📷 Project Screenshot

(Place your uploaded screenshot or app preview image here)

👨‍💻 Author

Awanish Bhatt
Machine Learning & AI Developer
