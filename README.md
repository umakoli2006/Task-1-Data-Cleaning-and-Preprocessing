Task 1: Data Cleaning and Preprocessing

📌 Project Overview

This project focuses on cleaning and preprocessing a Healthcare Patient Records dataset. The main goal is to convert raw data into a clean, consistent, and analysis-ready dataset.

🎯 Objectives
Identify and handle missing values.
Remove duplicate records.
Fix incorrect data types.
Standardize column names.
Clean and standardize text values.
Convert date columns into datetime format.
Extract useful date features.
Prepare the dataset for EDA and Machine Learning.
🏥 Dataset

Domain: Healthcare / Hospital Patient Records

The dataset contains patient information such as:

Patient ID
Age
Gender
Blood Group
City
Department
Diagnosis
Admission and Discharge Dates
Vital Signs
BMI
Diabetes
Hypertension
Medication
Treatment Cost
Insurance Type
Outcome
🛠️ Data Cleaning Steps
1. Data Inspection

Used head(), info(), shape, and describe() to understand the dataset.

2. Data Type Correction

Converted numerical columns to numeric data types and admission/discharge dates to datetime.

3. Date Preprocessing

Created:

Admission Year
Admission Month
Admission Day
Admission Day Name
Discharge Year
Discharge Month
Discharge Day
Discharge Day Name
4. Column Name Cleaning

Standardized column names using lowercase and underscores.

Example:

Patient_ID → patient_id
Admission_Date → admission_date
Treatment_Cost_INR → treatment_cost_inr
5. Text Standardization

Standardized categorical values such as gender, blood group, city, and department.

6. Duplicate Removal

Identified and removed duplicate records using:

df.drop_duplicates()
7. Missing Value Handling

Missing values were identified using:

df.isnull().sum()

Numerical values were handled using the median, categorical values using the mode, and missing medication values were marked as unknown.

💻 Technologies Used
Python
Pandas
NumPy
Jupyter Notebook
Matplotlib
Seaborn
📊 Result

The raw healthcare dataset was successfully cleaned and transformed into an analysis-ready dataset suitable for Exploratory Data Analysis, visualization, and Machine Learning.

👩‍💻 Author

Uma Koli

Task: Data Analyst Internship – Task 1
Topic: Data Cleaning and Preprocessing
