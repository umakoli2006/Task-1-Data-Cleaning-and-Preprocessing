# Task 1: Data Cleaning and Preprocessing

## Hospital Patient Records

###  improt the required library


```python
# import numpy 
# this library help us to perform numerical operation on diffrent data structure

import numpy as np

# import pandas 
# this library help us to convert diffrent data entry in pandas data frame 

import pandas as pd
# import matplotlib and seaborn
# this library help us visulization diffrent data structure

import matplotlib.pyplot as plt
import seaborn as sns
```

### Load The  Raw Dataset


```python
# read the data from the local 

df_raw_dataset_Healthcare_Patient_Records =pd.read_csv('raw_dataset_Healthcare_Patient_Records.csv')
```


```python
# show the first five file observations of given data frame 
df_raw_dataset_Healthcare_Patient_Records.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patient_ID</th>
      <th>Patient_Name</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Blood_Group</th>
      <th>City</th>
      <th>Department</th>
      <th>Diagnosis</th>
      <th>Admission_Date</th>
      <th>Discharge_Date</th>
      <th>...</th>
      <th>BMI</th>
      <th>Smoking_Status</th>
      <th>Diabetes</th>
      <th>Hypertension</th>
      <th>Medication</th>
      <th>Treatment_Cost_INR</th>
      <th>Insurance_Type</th>
      <th>Doctor_ID</th>
      <th>Outcome</th>
      <th>Admission_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P106252</td>
      <td>Sneha Singh</td>
      <td>31.0</td>
      <td>Male</td>
      <td>AB+</td>
      <td>Ahmedabad</td>
      <td>Oncology</td>
      <td>Gastritis</td>
      <td>2025-04-18</td>
      <td>2025-04-20</td>
      <td>...</td>
      <td>15.7</td>
      <td>Never</td>
      <td>Yes</td>
      <td>No</td>
      <td>Insulin</td>
      <td>13238.96</td>
      <td>Government</td>
      <td>D696</td>
      <td>Recovered</td>
      <td>Elective</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P104684</td>
      <td>Vivaan Reddy</td>
      <td>38.0</td>
      <td>Male</td>
      <td>B-</td>
      <td>Kolkata</td>
      <td>Orthopedics</td>
      <td>Arthritis</td>
      <td>2022-05-09</td>
      <td>2022-05-12</td>
      <td>...</td>
      <td>34.5</td>
      <td>Never</td>
      <td>No</td>
      <td>No</td>
      <td>Metformin</td>
      <td>8232.69</td>
      <td>Employer</td>
      <td>D738</td>
      <td>Improved</td>
      <td>Emergency</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P101731</td>
      <td>Isha Mehta</td>
      <td>67.0</td>
      <td>Male</td>
      <td>AB-</td>
      <td>Rajkot</td>
      <td>Neurology</td>
      <td>Diabetes</td>
      <td>2024-09-12</td>
      <td>2024-09-23</td>
      <td>...</td>
      <td>29.1</td>
      <td>Never</td>
      <td>No</td>
      <td>No</td>
      <td>Amoxicillin</td>
      <td>23295.42</td>
      <td>Government</td>
      <td>D350</td>
      <td>Referred</td>
      <td>Emergency</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P104742</td>
      <td>Rahul Koli</td>
      <td>53.0</td>
      <td>M</td>
      <td>O+</td>
      <td>Ahmedabad</td>
      <td>Orthopedics</td>
      <td>Gastritis</td>
      <td>2023-04-07</td>
      <td>2023-04-13</td>
      <td>...</td>
      <td>23.3</td>
      <td>Never</td>
      <td>No</td>
      <td>No</td>
      <td>Metformin</td>
      <td>25632.82</td>
      <td>Private</td>
      <td>D634</td>
      <td>Referred</td>
      <td>Urgent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P104521</td>
      <td>Vikram Desai</td>
      <td>39.0</td>
      <td>Male</td>
      <td>B+</td>
      <td>Vadodara</td>
      <td>Oncology</td>
      <td>Diabetes</td>
      <td>2022-10-05</td>
      <td>2022-10-08</td>
      <td>...</td>
      <td>35.5</td>
      <td>Current</td>
      <td>No</td>
      <td>Yes</td>
      <td>Paracetamol</td>
      <td>20111.41</td>
      <td>Government</td>
      <td>D737</td>
      <td>Recovered</td>
      <td>Elective</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
# show the last five file observations of given data frame 
df_raw_dataset_Healthcare_Patient_Records.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patient_ID</th>
      <th>Patient_Name</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Blood_Group</th>
      <th>City</th>
      <th>Department</th>
      <th>Diagnosis</th>
      <th>Admission_Date</th>
      <th>Discharge_Date</th>
      <th>...</th>
      <th>BMI</th>
      <th>Smoking_Status</th>
      <th>Diabetes</th>
      <th>Hypertension</th>
      <th>Medication</th>
      <th>Treatment_Cost_INR</th>
      <th>Insurance_Type</th>
      <th>Doctor_ID</th>
      <th>Outcome</th>
      <th>Admission_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9995</th>
      <td>P105734</td>
      <td>Vikram Reddy</td>
      <td>37.0</td>
      <td>Male</td>
      <td>A+</td>
      <td>Rajkot</td>
      <td>Oncology</td>
      <td>Fracture</td>
      <td>2023-12-22</td>
      <td>2024-01-03</td>
      <td>...</td>
      <td>29.5</td>
      <td>Never</td>
      <td>Yes</td>
      <td>No</td>
      <td>Metformin</td>
      <td>12065.84</td>
      <td>Self-Pay</td>
      <td>D415</td>
      <td>Recovered</td>
      <td>ELECTIVE</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>P105191</td>
      <td>Pooja Singh</td>
      <td>56.0</td>
      <td>Female</td>
      <td>AB-</td>
      <td>Vadodara</td>
      <td>Pediatrics</td>
      <td>Hypertension</td>
      <td>2025-10-24</td>
      <td>2025-10-27</td>
      <td>...</td>
      <td>16.1</td>
      <td>Never</td>
      <td>No</td>
      <td>No</td>
      <td>Atorvastatin</td>
      <td>15477.88</td>
      <td>Private</td>
      <td>D781</td>
      <td>Recovered</td>
      <td>Elective</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>P105390</td>
      <td>Rahul Koli</td>
      <td>31.0</td>
      <td>Female</td>
      <td>B+</td>
      <td>Pune</td>
      <td>Pediatrics</td>
      <td>Fracture</td>
      <td>2022-03-12</td>
      <td>2022-03-23</td>
      <td>...</td>
      <td>32.4</td>
      <td>Never</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Amoxicillin</td>
      <td>59541.18</td>
      <td>Employer</td>
      <td>D131</td>
      <td>Recovered</td>
      <td>emergency</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>P100860</td>
      <td>Neha Patel</td>
      <td>49.0</td>
      <td>Female</td>
      <td>B+</td>
      <td>Hyderabad</td>
      <td>Dermatology</td>
      <td>Migraine</td>
      <td>2023-01-09</td>
      <td>2023-01-18</td>
      <td>...</td>
      <td>15.7</td>
      <td>Former</td>
      <td>No</td>
      <td>No</td>
      <td>Amlodipine</td>
      <td>30732.58</td>
      <td>Employer</td>
      <td>D155</td>
      <td>Recovered</td>
      <td>Elective</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>P107270</td>
      <td>Isha Reddy</td>
      <td>23.0</td>
      <td>Male</td>
      <td>AB-</td>
      <td>Hyderabad</td>
      <td>ENT</td>
      <td>Fracture</td>
      <td>2022-05-17</td>
      <td>2022-05-30</td>
      <td>...</td>
      <td>28.9</td>
      <td>Current</td>
      <td>No</td>
      <td>No</td>
      <td>Insulin</td>
      <td>23623.18</td>
      <td>Government</td>
      <td>D767</td>
      <td>Recovered</td>
      <td>Emergency</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



### Data Overview


```python
# check the shap of the data
df_raw_dataset_Healthcare_Patient_Records.shape
```




    (10000, 30)



**Interpretation**
- we have 10000 observations and 30 attributes.


```python
# check the columns present in the data
df_raw_dataset_Healthcare_Patient_Records.columns
```




    Index(['Patient_ID', 'Patient_Name', 'Age', 'Gender', 'Blood_Group', 'City',
           'Department', 'Diagnosis', 'Admission_Date', 'Discharge_Date',
           'Length_of_Stay', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP',
           'Temperature_C', 'SpO2', 'Respiratory_Rate', 'Glucose_mg_dL',
           'Cholesterol_mg_dL', 'Hemoglobin_g_dL', 'BMI', 'Smoking_Status',
           'Diabetes', 'Hypertension', 'Medication', 'Treatment_Cost_INR',
           'Insurance_Type', 'Doctor_ID', 'Outcome', 'Admission_Type'],
          dtype='object')


# information about columns 
'Patient_ID': Unique identifier,
'Patient_Name': Name/identifier,
'Age': Numeric,
'Gender': Categorical,
'Blood_Group': Categorical,
'City': Categorical,
'Department': Categorical,
'Diagnosis': Categorical,
'Admission_Date': Date,
'Discharge_Date': Date,
'Length_of_Stay': Numeric,
'Heart_Rate': Numeric,
'Systolic_BP': Numeric,
'Diastolic_BP': Numeric,
'Temperature_C': Numeric,
'SpO2': Numeric,
'Respiratory_Rate':Numeric,
'Glucose_mg_dL': Numeric,
'Cholesterol_mg_dL': Numeric,
'Hemoglobin_g_dL': Numeric,
'BMI': Numeric,
'Smoking_Status': Categorical,
'Diabetes': Boolean/Categorical,
'Hypertension': Boolean/Categorical,
'Medication': Categorical,
'Treatment_Cost_INR': Numeric,
'Insurance_Type': Categorical,
'Doctor_ID': Identifier,
'Outcome': Categorical,
'Admission_Type': Categorical

```python
# check the size and data type of the columns 
df_raw_dataset_Healthcare_Patient_Records.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 30 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   Patient_ID          10000 non-null  object 
     1   Patient_Name        10000 non-null  object 
     2   Age                 9800 non-null   float64
     3   Gender              9808 non-null   object 
     4   Blood_Group         9801 non-null   object 
     5   City                9801 non-null   object 
     6   Department          9800 non-null   object 
     7   Diagnosis           9798 non-null   object 
     8   Admission_Date      10000 non-null  object 
     9   Discharge_Date      10000 non-null  object 
     10  Length_of_Stay      10000 non-null  int64  
     11  Heart_Rate          9799 non-null   float64
     12  Systolic_BP         10000 non-null  int64  
     13  Diastolic_BP        10000 non-null  int64  
     14  Temperature_C       10000 non-null  float64
     15  SpO2                10000 non-null  int64  
     16  Respiratory_Rate    10000 non-null  int64  
     17  Glucose_mg_dL       9801 non-null   float64
     18  Cholesterol_mg_dL   10000 non-null  int64  
     19  Hemoglobin_g_dL     10000 non-null  float64
     20  BMI                 9802 non-null   float64
     21  Smoking_Status      9941 non-null   object 
     22  Diabetes            10000 non-null  object 
     23  Hypertension        10000 non-null  object 
     24  Medication          8374 non-null   object 
     25  Treatment_Cost_INR  10000 non-null  object 
     26  Insurance_Type      9799 non-null   object 
     27  Doctor_ID           10000 non-null  object 
     28  Outcome             10000 non-null  object 
     29  Admission_Type      10000 non-null  object 
    dtypes: float64(6), int64(6), object(18)
    memory usage: 2.3+ MB
    

**Interpretation** 
- we have 12 numerical columns and 18 categorical column
- the data size is 2.3+ MB
- we have missing data present in data frame

### Convert date formats to a consistent type (e.g., dd-mm-yyyy).


```python
# To see the data in Admission Date and Discharge Date
df_raw_dataset_Healthcare_Patient_Records[['Admission_Date','Discharge_Date']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Admission_Date</th>
      <th>Discharge_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-04-18</td>
      <td>2025-04-20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-09</td>
      <td>2022-05-12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-09-12</td>
      <td>2024-09-23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-07</td>
      <td>2023-04-13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-10-05</td>
      <td>2022-10-08</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert the column to datetime
df_raw_dataset_Healthcare_Patient_Records['Admission_Date'] = pd.to_datetime(df_raw_dataset_Healthcare_Patient_Records['Admission_Date'],errors='coerce')
```


```python
# Create separate Year, Month, Day columns
df_raw_dataset_Healthcare_Patient_Records['Admission_Year'] = (df_raw_dataset_Healthcare_Patient_Records['Admission_Date'].dt.year)

df_raw_dataset_Healthcare_Patient_Records['Admission_Month'] = (df_raw_dataset_Healthcare_Patient_Records['Admission_Date'].dt.month)

df_raw_dataset_Healthcare_Patient_Records['Admission_Day'] = (df_raw_dataset_Healthcare_Patient_Records['Admission_Date'].dt.day)
```


```python
# Do the same for Discharge Date 
df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'] = pd.to_datetime(df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'],errors='coerce')

df_raw_dataset_Healthcare_Patient_Records['Discharge_Year'] = (df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'].dt.year)

df_raw_dataset_Healthcare_Patient_Records['Discharge_Month'] = (df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'].dt.month)

df_raw_dataset_Healthcare_Patient_Records['Discharge_Day'] = (df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'].dt.day)
```


```python
# Chaek the result
df_raw_dataset_Healthcare_Patient_Records[
['Admission_Date','Admission_Year','Admission_Month','Admission_Day','Discharge_Date','Discharge_Year','Discharge_Month','Discharge_Day']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Admission_Date</th>
      <th>Admission_Year</th>
      <th>Admission_Month</th>
      <th>Admission_Day</th>
      <th>Discharge_Date</th>
      <th>Discharge_Year</th>
      <th>Discharge_Month</th>
      <th>Discharge_Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-04-18</td>
      <td>2025</td>
      <td>4</td>
      <td>18</td>
      <td>2025-04-20</td>
      <td>2025</td>
      <td>4</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-09</td>
      <td>2022</td>
      <td>5</td>
      <td>9</td>
      <td>2022-05-12</td>
      <td>2022</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-09-12</td>
      <td>2024</td>
      <td>9</td>
      <td>12</td>
      <td>2024-09-23</td>
      <td>2024</td>
      <td>9</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-07</td>
      <td>2023</td>
      <td>4</td>
      <td>7</td>
      <td>2023-04-13</td>
      <td>2023</td>
      <td>4</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-10-05</td>
      <td>2022</td>
      <td>10</td>
      <td>5</td>
      <td>2022-10-08</td>
      <td>2022</td>
      <td>10</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# if i want the day name, like Monday, Tuesday, etc. so
# Use .dt.day_name()
## for Admission Date
df_raw_dataset_Healthcare_Patient_Records['Admission_Date'] = pd.to_datetime(df_raw_dataset_Healthcare_Patient_Records['Admission_Date'],errors='coerce')

df_raw_dataset_Healthcare_Patient_Records['Admission_Day_Name'] = (df_raw_dataset_Healthcare_Patient_Records['Admission_Date'].dt.day_name())
## for  Discharge Date
df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'] = pd.to_datetime(df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'],errors='coerce')

df_raw_dataset_Healthcare_Patient_Records['Discharge_Day_Name'] = (df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'].dt.day_name())
```


```python
# chek the result
df_raw_dataset_Healthcare_Patient_Records[['Admission_Date', 'Admission_Day_Name','Discharge_Date', 'Discharge_Day_Name']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Admission_Date</th>
      <th>Admission_Day_Name</th>
      <th>Discharge_Date</th>
      <th>Discharge_Day_Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-04-18</td>
      <td>Friday</td>
      <td>2025-04-20</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-09</td>
      <td>Monday</td>
      <td>2022-05-12</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-09-12</td>
      <td>Thursday</td>
      <td>2024-09-23</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-07</td>
      <td>Friday</td>
      <td>2023-04-13</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-10-05</td>
      <td>Wednesday</td>
      <td>2022-10-08</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
</div>



### Check and fix data types (e.g., age should be int, date as datetime)


```python
# check current data type
# we also check with  info()
df_raw_dataset_Healthcare_Patient_Records.dtypes
```




    Patient_ID                    object
    Patient_Name                  object
    Age                          float64
    Gender                        object
    Blood_Group                   object
    City                          object
    Department                    object
    Diagnosis                     object
    Admission_Date        datetime64[ns]
    Discharge_Date        datetime64[ns]
    Length_of_Stay                 int64
    Heart_Rate                   float64
    Systolic_BP                    int64
    Diastolic_BP                   int64
    Temperature_C                float64
    SpO2                           int64
    Respiratory_Rate               int64
    Glucose_mg_dL                float64
    Cholesterol_mg_dL              int64
    Hemoglobin_g_dL              float64
    BMI                          float64
    Smoking_Status                object
    Diabetes                      object
    Hypertension                  object
    Medication                    object
    Treatment_Cost_INR            object
    Insurance_Type                object
    Doctor_ID                     object
    Outcome                       object
    Admission_Type                object
    Admission_Year                 int32
    Admission_Month                int32
    Admission_Day                  int32
    Discharge_Year                 int32
    Discharge_Month                int32
    Discharge_Day                  int32
    Admission_Day_Name            object
    Discharge_Day_Name            object
    dtype: object




```python
# Fix the important data types
# Age → numeric
df_raw_dataset_Healthcare_Patient_Records['Age'] = pd.to_numeric(df_raw_dataset_Healthcare_Patient_Records['Age'],errors='coerce').astype('Int64')
# Dates → datetime
df_raw_dataset_Healthcare_Patient_Records['Admission_Date'] = pd.to_datetime(df_raw_dataset_Healthcare_Patient_Records['Admission_Date'],errors='coerce')

df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'] = pd.to_datetime(
    df_raw_dataset_Healthcare_Patient_Records['Discharge_Date'],
    errors='coerce'
)


```


```python
# check the result
df_raw_dataset_Healthcare_Patient_Records.dtypes
```




    Patient_ID                    object
    Patient_Name                  object
    Age                            Int64
    Gender                        object
    Blood_Group                   object
    City                          object
    Department                    object
    Diagnosis                     object
    Admission_Date        datetime64[ns]
    Discharge_Date        datetime64[ns]
    Length_of_Stay                 int64
    Heart_Rate                   float64
    Systolic_BP                    int64
    Diastolic_BP                   int64
    Temperature_C                float64
    SpO2                           int64
    Respiratory_Rate               int64
    Glucose_mg_dL                float64
    Cholesterol_mg_dL              int64
    Hemoglobin_g_dL              float64
    BMI                          float64
    Smoking_Status                object
    Diabetes                      object
    Hypertension                  object
    Medication                    object
    Treatment_Cost_INR            object
    Insurance_Type                object
    Doctor_ID                     object
    Outcome                       object
    Admission_Type                object
    Admission_Year                 int32
    Admission_Month                int32
    Admission_Day                  int32
    Discharge_Year                 int32
    Discharge_Month                int32
    Discharge_Day                  int32
    Admission_Day_Name            object
    Discharge_Day_Name            object
    dtype: object




```python
# Length of Stay → numeric
df_raw_dataset_Healthcare_Patient_Records['Length_of_Stay'] = pd.to_numeric(
    df_raw_dataset_Healthcare_Patient_Records['Length_of_Stay'],
    errors='coerce'
).astype('Int64')


# Vital signs / medical measurements → numeric
numeric_columns = [
    'Heart_Rate',
    'Systolic_BP',
    'Diastolic_BP',
    'Temperature_C',
    'SpO2',
    'Respiratory_Rate',
    'Glucose_mg_dL',
    'Cholesterol_mg_dL'
]

for col in numeric_columns:
    df_raw_dataset_Healthcare_Patient_Records[col] = pd.to_numeric(
        df_raw_dataset_Healthcare_Patient_Records[col],
        errors='coerce'
    )
```


```python
# check the result
df_raw_dataset_Healthcare_Patient_Records.dtypes
```




    Patient_ID                    object
    Patient_Name                  object
    Age                            Int64
    Gender                        object
    Blood_Group                   object
    City                          object
    Department                    object
    Diagnosis                     object
    Admission_Date        datetime64[ns]
    Discharge_Date        datetime64[ns]
    Length_of_Stay                 Int64
    Heart_Rate                   float64
    Systolic_BP                    int64
    Diastolic_BP                   int64
    Temperature_C                float64
    SpO2                           int64
    Respiratory_Rate               int64
    Glucose_mg_dL                float64
    Cholesterol_mg_dL              int64
    Hemoglobin_g_dL              float64
    BMI                          float64
    Smoking_Status                object
    Diabetes                      object
    Hypertension                  object
    Medication                    object
    Treatment_Cost_INR            object
    Insurance_Type                object
    Doctor_ID                     object
    Outcome                       object
    Admission_Type                object
    Admission_Year                 int32
    Admission_Month                int32
    Admission_Day                  int32
    Discharge_Year                 int32
    Discharge_Month                int32
    Discharge_Day                  int32
    Admission_Day_Name            object
    Discharge_Day_Name            object
    dtype: object



### Rename column headers to be clean and uniform (e.g., lowercase, no spaces).


```python
# check the columns
df_raw_dataset_Healthcare_Patient_Records.columns
```




    Index(['Patient_ID', 'Patient_Name', 'Age', 'Gender', 'Blood_Group', 'City',
           'Department', 'Diagnosis', 'Admission_Date', 'Discharge_Date',
           'Length_of_Stay', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP',
           'Temperature_C', 'SpO2', 'Respiratory_Rate', 'Glucose_mg_dL',
           'Cholesterol_mg_dL', 'Hemoglobin_g_dL', 'BMI', 'Smoking_Status',
           'Diabetes', 'Hypertension', 'Medication', 'Treatment_Cost_INR',
           'Insurance_Type', 'Doctor_ID', 'Outcome', 'Admission_Type',
           'Admission_Year', 'Admission_Month', 'Admission_Day', 'Discharge_Year',
           'Discharge_Month', 'Discharge_Day', 'Admission_Day_Name',
           'Discharge_Day_Name'],
          dtype='object')




```python
# renaming or repalce in columns 
df_raw_dataset_Healthcare_Patient_Records.columns = (
    df_raw_dataset_Healthcare_Patient_Records.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace(r'[^a-z0-9_]', '', regex=True)
)

print(df_raw_dataset_Healthcare_Patient_Records.columns)
```

    Index(['patient_id', 'patient_name', 'age', 'gender', 'blood_group', 'city',
           'department', 'diagnosis', 'admission_date', 'discharge_date',
           'length_of_stay', 'heart_rate', 'systolic_bp', 'diastolic_bp',
           'temperature_c', 'spo2', 'respiratory_rate', 'glucose_mg_dl',
           'cholesterol_mg_dl', 'hemoglobin_g_dl', 'bmi', 'smoking_status',
           'diabetes', 'hypertension', 'medication', 'treatment_cost_inr',
           'insurance_type', 'doctor_id', 'outcome', 'admission_type',
           'admission_year', 'admission_month', 'admission_day', 'discharge_year',
           'discharge_month', 'discharge_day', 'admission_day_name',
           'discharge_day_name'],
          dtype='object')
    


```python
# check the first five record
df_raw_dataset_Healthcare_Patient_Records.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>patient_name</th>
      <th>age</th>
      <th>gender</th>
      <th>blood_group</th>
      <th>city</th>
      <th>department</th>
      <th>diagnosis</th>
      <th>admission_date</th>
      <th>discharge_date</th>
      <th>...</th>
      <th>outcome</th>
      <th>admission_type</th>
      <th>admission_year</th>
      <th>admission_month</th>
      <th>admission_day</th>
      <th>discharge_year</th>
      <th>discharge_month</th>
      <th>discharge_day</th>
      <th>admission_day_name</th>
      <th>discharge_day_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P106252</td>
      <td>Sneha Singh</td>
      <td>31</td>
      <td>Male</td>
      <td>AB+</td>
      <td>Ahmedabad</td>
      <td>Oncology</td>
      <td>Gastritis</td>
      <td>2025-04-18</td>
      <td>2025-04-20</td>
      <td>...</td>
      <td>Recovered</td>
      <td>Elective</td>
      <td>2025</td>
      <td>4</td>
      <td>18</td>
      <td>2025</td>
      <td>4</td>
      <td>20</td>
      <td>Friday</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P104684</td>
      <td>Vivaan Reddy</td>
      <td>38</td>
      <td>Male</td>
      <td>B-</td>
      <td>Kolkata</td>
      <td>Orthopedics</td>
      <td>Arthritis</td>
      <td>2022-05-09</td>
      <td>2022-05-12</td>
      <td>...</td>
      <td>Improved</td>
      <td>Emergency</td>
      <td>2022</td>
      <td>5</td>
      <td>9</td>
      <td>2022</td>
      <td>5</td>
      <td>12</td>
      <td>Monday</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P101731</td>
      <td>Isha Mehta</td>
      <td>67</td>
      <td>Male</td>
      <td>AB-</td>
      <td>Rajkot</td>
      <td>Neurology</td>
      <td>Diabetes</td>
      <td>2024-09-12</td>
      <td>2024-09-23</td>
      <td>...</td>
      <td>Referred</td>
      <td>Emergency</td>
      <td>2024</td>
      <td>9</td>
      <td>12</td>
      <td>2024</td>
      <td>9</td>
      <td>23</td>
      <td>Thursday</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P104742</td>
      <td>Rahul Koli</td>
      <td>53</td>
      <td>M</td>
      <td>O+</td>
      <td>Ahmedabad</td>
      <td>Orthopedics</td>
      <td>Gastritis</td>
      <td>2023-04-07</td>
      <td>2023-04-13</td>
      <td>...</td>
      <td>Referred</td>
      <td>Urgent</td>
      <td>2023</td>
      <td>4</td>
      <td>7</td>
      <td>2023</td>
      <td>4</td>
      <td>13</td>
      <td>Friday</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P104521</td>
      <td>Vikram Desai</td>
      <td>39</td>
      <td>Male</td>
      <td>B+</td>
      <td>Vadodara</td>
      <td>Oncology</td>
      <td>Diabetes</td>
      <td>2022-10-05</td>
      <td>2022-10-08</td>
      <td>...</td>
      <td>Recovered</td>
      <td>Elective</td>
      <td>2022</td>
      <td>10</td>
      <td>5</td>
      <td>2022</td>
      <td>10</td>
      <td>8</td>
      <td>Wednesday</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



### Standardize text values like gender, country names, etc.


```python
# First check the unique text values
text_columns = [
    'gender',
    'blood_group',
    'city',
    'department',
    'diagnosis',
    'smoking_status',
    'diabetes',
    'hypertension',
    'medication',
    'insurance_type',
    'outcome',
    'admission_type'
]

for col in text_columns:
    print(f"\n{col}:")
    print(df_raw_dataset_Healthcare_Patient_Records[col].unique())
```

    
    gender:
    ['Male' 'M' 'Female' nan 'Other' 'F' 'female']
    
    blood_group:
    ['AB+' 'B-' 'AB-' 'O+' 'B+' 'O-' nan 'A-' 'A+']
    
    city:
    ['Ahmedabad' 'Kolkata' 'Rajkot' 'Vadodara' 'Bengaluru' 'Hyderabad' 'Delhi'
     'Chennai' 'vadodara' 'Pune' 'Mumbai' 'Jaipur' 'Surat' nan 'Vadodara '
     'AHMEDABAD']
    
    department:
    ['Oncology' 'Orthopedics' 'Neurology' 'Cardiology' 'Emergency'
     'Pediatrics' 'ENT' 'General Medicine' nan 'Dermatology']
    
    diagnosis:
    ['Gastritis' 'Arthritis' 'Diabetes' 'Viral Infection' 'Heart Disease'
     'Pneumonia' 'Fracture' 'Hypertension' 'Asthma' 'Migraine' nan]
    
    smoking_status:
    ['Never' 'Current' 'Former' 'current' nan]
    
    diabetes:
    ['Yes' 'No']
    
    hypertension:
    ['No' 'Yes']
    
    medication:
    ['Insulin' 'Metformin' 'Amoxicillin' 'Paracetamol' nan 'Atorvastatin'
     'Amlodipine']
    
    insurance_type:
    ['Government' 'Employer' 'Private' 'Self-Pay' nan]
    
    outcome:
    ['Recovered' 'Improved' 'Referred' 'Critical']
    
    admission_type:
    ['Elective' 'Emergency' 'Urgent' 'emergency' 'ELECTIVE']
    


```python
# General text cleaning
# i want only  change in general
for col in text_columns:
    df_raw_dataset_Healthcare_Patient_Records[col] = (
        df_raw_dataset_Healthcare_Patient_Records[col]
        .astype('string')
        .str.strip()
        .str.lower()
    )
# Standardize Gender

    df_raw_dataset_Healthcare_Patient_Records['gender'] = (
    df_raw_dataset_Healthcare_Patient_Records['gender']
    .replace({
        'm': 'male',
        'man': 'male',
        'male': 'male',
        'f': 'female',
        'woman': 'female',
        'female': 'female'
    })
)
```


```python
# check the result 
df_raw_dataset_Healthcare_Patient_Records['gender'].value_counts(dropna=False)
```




    gender
    male      4807
    female    4797
    other      204
    <NA>       192
    Name: count, dtype: Int64




```python
# check the first five record
df_raw_dataset_Healthcare_Patient_Records.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>patient_name</th>
      <th>age</th>
      <th>gender</th>
      <th>blood_group</th>
      <th>city</th>
      <th>department</th>
      <th>diagnosis</th>
      <th>admission_date</th>
      <th>discharge_date</th>
      <th>...</th>
      <th>outcome</th>
      <th>admission_type</th>
      <th>admission_year</th>
      <th>admission_month</th>
      <th>admission_day</th>
      <th>discharge_year</th>
      <th>discharge_month</th>
      <th>discharge_day</th>
      <th>admission_day_name</th>
      <th>discharge_day_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P106252</td>
      <td>Sneha Singh</td>
      <td>31</td>
      <td>male</td>
      <td>ab+</td>
      <td>ahmedabad</td>
      <td>oncology</td>
      <td>gastritis</td>
      <td>2025-04-18</td>
      <td>2025-04-20</td>
      <td>...</td>
      <td>recovered</td>
      <td>elective</td>
      <td>2025</td>
      <td>4</td>
      <td>18</td>
      <td>2025</td>
      <td>4</td>
      <td>20</td>
      <td>Friday</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P104684</td>
      <td>Vivaan Reddy</td>
      <td>38</td>
      <td>male</td>
      <td>b-</td>
      <td>kolkata</td>
      <td>orthopedics</td>
      <td>arthritis</td>
      <td>2022-05-09</td>
      <td>2022-05-12</td>
      <td>...</td>
      <td>improved</td>
      <td>emergency</td>
      <td>2022</td>
      <td>5</td>
      <td>9</td>
      <td>2022</td>
      <td>5</td>
      <td>12</td>
      <td>Monday</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P101731</td>
      <td>Isha Mehta</td>
      <td>67</td>
      <td>male</td>
      <td>ab-</td>
      <td>rajkot</td>
      <td>neurology</td>
      <td>diabetes</td>
      <td>2024-09-12</td>
      <td>2024-09-23</td>
      <td>...</td>
      <td>referred</td>
      <td>emergency</td>
      <td>2024</td>
      <td>9</td>
      <td>12</td>
      <td>2024</td>
      <td>9</td>
      <td>23</td>
      <td>Thursday</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P104742</td>
      <td>Rahul Koli</td>
      <td>53</td>
      <td>male</td>
      <td>o+</td>
      <td>ahmedabad</td>
      <td>orthopedics</td>
      <td>gastritis</td>
      <td>2023-04-07</td>
      <td>2023-04-13</td>
      <td>...</td>
      <td>referred</td>
      <td>urgent</td>
      <td>2023</td>
      <td>4</td>
      <td>7</td>
      <td>2023</td>
      <td>4</td>
      <td>13</td>
      <td>Friday</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P104521</td>
      <td>Vikram Desai</td>
      <td>39</td>
      <td>male</td>
      <td>b+</td>
      <td>vadodara</td>
      <td>oncology</td>
      <td>diabetes</td>
      <td>2022-10-05</td>
      <td>2022-10-08</td>
      <td>...</td>
      <td>recovered</td>
      <td>elective</td>
      <td>2022</td>
      <td>10</td>
      <td>5</td>
      <td>2022</td>
      <td>10</td>
      <td>8</td>
      <td>Wednesday</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



### Remove duplicate rows using .drop_duplicates() or Excel’s “Remove Duplicates”.


```python
# Check how many duplicate rows exist
duplicates = df_raw_dataset_Healthcare_Patient_Records.duplicated().sum()

print("Number of duplicate rows:", duplicates)
```

    Number of duplicate rows: 100
    


```python
# Remove duplicate rows
df_raw_dataset_Healthcare_Patient_Records = (
    df_raw_dataset_Healthcare_Patient_Records.drop_duplicates()
)
```


```python
# check the duplicate again
print(
    "Duplicate rows after cleaning:",
    df_raw_dataset_Healthcare_Patient_Records.duplicated().sum()
)
```

    Duplicate rows after cleaning: 0
    


```python
# Check the number of rows before and after
print("Rows after removing duplicates:",
      len(df_raw_dataset_Healthcare_Patient_Records))
```

    Rows after removing duplicates: 9900
    


```python
# Check final shape
print("Final dataset shape:",
      df_raw_dataset_Healthcare_Patient_Records.shape)
```

    Final dataset shape: (9900, 38)
    

### Identify and handle missing values using .isnull() in Python or filters in Excel


```python
# Check total missing values in each column
missing_values = df_raw_dataset_Healthcare_Patient_Records.isnull().sum()

print(missing_values)
```

    patient_id               0
    patient_name             0
    age                    197
    gender                 192
    blood_group            197
    city                   195
    department             199
    diagnosis              200
    admission_date           0
    discharge_date           0
    length_of_stay           0
    heart_rate             200
    systolic_bp              0
    diastolic_bp             0
    temperature_c            0
    spo2                     0
    respiratory_rate         0
    glucose_mg_dl          197
    cholesterol_mg_dl        0
    hemoglobin_g_dl          0
    bmi                    198
    smoking_status          59
    diabetes                 0
    hypertension             0
    medication            1610
    treatment_cost_inr       0
    insurance_type         199
    doctor_id                0
    outcome                  0
    admission_type           0
    admission_year           0
    admission_month          0
    admission_day            0
    discharge_year           0
    discharge_month          0
    discharge_day            0
    admission_day_name       0
    discharge_day_name       0
    dtype: int64
    

**Interpretation**
- we have  column of that have medication its have 1610 missing null value


```python
missing_percentage = (
    df_raw_dataset_Healthcare_Patient_Records.isnull().mean() * 100
).round(2)

print(missing_percentage[missing_percentage > 0])
```

    age                1.99
    gender             1.94
    blood_group        1.99
    city               1.97
    department         2.01
    diagnosis          2.02
    heart_rate         2.02
    glucose_mg_dl      1.99
    bmi                2.00
    smoking_status     0.60
    medication        16.26
    insurance_type     2.01
    dtype: float64
    


```python
# Handle numerical missing values
numeric_columns = [
    'age',
    'heart_rate',
    'glucose_mg_dl',
    'bmi'
]

for col in numeric_columns:
    df_raw_dataset_Healthcare_Patient_Records[col] = (
        df_raw_dataset_Healthcare_Patient_Records[col]
        .fillna(df_raw_dataset_Healthcare_Patient_Records[col].median())
    )
```


```python
# Handle categorical missing values
categorical_columns = [
    'gender',
    'blood_group',
    'city',
    'department',
    'diagnosis',
    'smoking_status',
    'insurance_type'
]

for col in categorical_columns:
    df_raw_dataset_Healthcare_Patient_Records[col] = (
        df_raw_dataset_Healthcare_Patient_Records[col]
        .fillna(df_raw_dataset_Healthcare_Patient_Records[col].mode()[0])
    )
```


```python
# check the result 
missing_after = df_raw_dataset_Healthcare_Patient_Records.isnull().sum()

print(missing_after[missing_after > 0])
```

    medication    1610
    dtype: int64
    


```python
# for the  medication
df_raw_dataset_Healthcare_Patient_Records['medication'] = (
    df_raw_dataset_Healthcare_Patient_Records['medication']
    .fillna('unknown')
)
```


```python
# check the result
missing_after = df_raw_dataset_Healthcare_Patient_Records.isnull().sum()

print(missing_after[missing_after > 0])

```

    Series([], dtype: int64)
    


```python
# Check total missing values in each column
missing_values = df_raw_dataset_Healthcare_Patient_Records.isnull().sum()

print(missing_values)
```

    patient_id            0
    patient_name          0
    age                   0
    gender                0
    blood_group           0
    city                  0
    department            0
    diagnosis             0
    admission_date        0
    discharge_date        0
    length_of_stay        0
    heart_rate            0
    systolic_bp           0
    diastolic_bp          0
    temperature_c         0
    spo2                  0
    respiratory_rate      0
    glucose_mg_dl         0
    cholesterol_mg_dl     0
    hemoglobin_g_dl       0
    bmi                   0
    smoking_status        0
    diabetes              0
    hypertension          0
    medication            0
    treatment_cost_inr    0
    insurance_type        0
    doctor_id             0
    outcome               0
    admission_type        0
    admission_year        0
    admission_month       0
    admission_day         0
    discharge_year        0
    discharge_month       0
    discharge_day         0
    admission_day_name    0
    discharge_day_name    0
    dtype: int64
    


```python
# show the first five file observations of given data frame 
df_raw_dataset_Healthcare_Patient_Records.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>patient_name</th>
      <th>age</th>
      <th>gender</th>
      <th>blood_group</th>
      <th>city</th>
      <th>department</th>
      <th>diagnosis</th>
      <th>admission_date</th>
      <th>discharge_date</th>
      <th>...</th>
      <th>outcome</th>
      <th>admission_type</th>
      <th>admission_year</th>
      <th>admission_month</th>
      <th>admission_day</th>
      <th>discharge_year</th>
      <th>discharge_month</th>
      <th>discharge_day</th>
      <th>admission_day_name</th>
      <th>discharge_day_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P106252</td>
      <td>Sneha Singh</td>
      <td>31</td>
      <td>male</td>
      <td>ab+</td>
      <td>ahmedabad</td>
      <td>oncology</td>
      <td>gastritis</td>
      <td>2025-04-18</td>
      <td>2025-04-20</td>
      <td>...</td>
      <td>recovered</td>
      <td>elective</td>
      <td>2025</td>
      <td>4</td>
      <td>18</td>
      <td>2025</td>
      <td>4</td>
      <td>20</td>
      <td>Friday</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>P104684</td>
      <td>Vivaan Reddy</td>
      <td>38</td>
      <td>male</td>
      <td>b-</td>
      <td>kolkata</td>
      <td>orthopedics</td>
      <td>arthritis</td>
      <td>2022-05-09</td>
      <td>2022-05-12</td>
      <td>...</td>
      <td>improved</td>
      <td>emergency</td>
      <td>2022</td>
      <td>5</td>
      <td>9</td>
      <td>2022</td>
      <td>5</td>
      <td>12</td>
      <td>Monday</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>P101731</td>
      <td>Isha Mehta</td>
      <td>67</td>
      <td>male</td>
      <td>ab-</td>
      <td>rajkot</td>
      <td>neurology</td>
      <td>diabetes</td>
      <td>2024-09-12</td>
      <td>2024-09-23</td>
      <td>...</td>
      <td>referred</td>
      <td>emergency</td>
      <td>2024</td>
      <td>9</td>
      <td>12</td>
      <td>2024</td>
      <td>9</td>
      <td>23</td>
      <td>Thursday</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>P104742</td>
      <td>Rahul Koli</td>
      <td>53</td>
      <td>male</td>
      <td>o+</td>
      <td>ahmedabad</td>
      <td>orthopedics</td>
      <td>gastritis</td>
      <td>2023-04-07</td>
      <td>2023-04-13</td>
      <td>...</td>
      <td>referred</td>
      <td>urgent</td>
      <td>2023</td>
      <td>4</td>
      <td>7</td>
      <td>2023</td>
      <td>4</td>
      <td>13</td>
      <td>Friday</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>P104521</td>
      <td>Vikram Desai</td>
      <td>39</td>
      <td>male</td>
      <td>b+</td>
      <td>vadodara</td>
      <td>oncology</td>
      <td>diabetes</td>
      <td>2022-10-05</td>
      <td>2022-10-08</td>
      <td>...</td>
      <td>recovered</td>
      <td>elective</td>
      <td>2022</td>
      <td>10</td>
      <td>5</td>
      <td>2022</td>
      <td>10</td>
      <td>8</td>
      <td>Wednesday</td>
      <td>Saturday</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
print(type(df_raw_dataset_Healthcare_Patient_Records))
print(df_raw_dataset_Healthcare_Patient_Records.shape)
```

    <class 'pandas.core.frame.DataFrame'>
    (9900, 38)
    


```python
# Save cleaned dataset
df_raw_dataset_Healthcare_Patient_Records.to_csv(
    "cleaned_healthcare_patient_records.csv",
    index=False
)

print("Cleaned dataset saved successfully!")
```

    Cleaned dataset saved successfully!
    


```python

```
