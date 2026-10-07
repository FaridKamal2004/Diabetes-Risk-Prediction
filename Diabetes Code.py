# -*- coding: utf-8 -*-

"""
Section01A_Diabetes (Coding)
"""

# Required Libraries

!pip install -q kagglehub scikit-fuzzy

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt


# Load Dataset

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "akshaydattatraykhare/diabetes-dataset",
    "diabetes.csv"
)


# Define Fuzzy Input Variables

pregnancies = ctrl.Antecedent(np.arange(0, 18, 1), 'Pregnancies')
glucose = ctrl.Antecedent(np.arange(0, 210, 1), 'Glucose')
blood_pressure = ctrl.Antecedent(np.arange(0, 140, 1), 'BloodPressure')
skin_thickness = ctrl.Antecedent(np.arange(0, 110, 1), 'SkinThickness')
insulin = ctrl.Antecedent(np.arange(0, 900, 1), 'Insulin')
bmi = ctrl.Antecedent(np.arange(0, 75, 1), 'BMI')
dpf = ctrl.Antecedent(np.arange(0, 2.6, 0.01), 'DPF')
age = ctrl.Antecedent(np.arange(20, 101, 1), 'Age')


# Define Fuzzy Output Variables

risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Risk')


# Pregnancies Membership Functions & Graph

pregnancies['None'] = fuzz.trapmf(
    pregnancies.universe, [0, 0, 1, 2]
)

pregnancies['Few'] = fuzz.trimf(
    pregnancies.universe, [1, 4, 7]
)

pregnancies['Many'] = fuzz.trapmf(
    pregnancies.universe, [5, 8, 17, 17]
)

plt.figure(figsize=(8, 4))

for term in pregnancies.terms:
    plt.plot(
        pregnancies.universe,
        pregnancies[term].mf,
        label=term
    )

plt.title("Pregnancies Membership Functions")
plt.xlabel("Pregnancies")
plt.ylabel("Membership Degree")
plt.ylim(0, 1.05)
plt.xlim(
    min(pregnancies.universe),
    max(pregnancies.universe)
)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# Glucose Membership Function

glucose['Low'] = fuzz.trapmf(
    glucose.universe, [0, 0, 70, 100]
)

glucose['Medium'] = fuzz.trimf(
    glucose.universe, [90, 125, 160]
)

glucose['High'] = fuzz.trapmf(
    glucose.universe, [140, 170, 200, 210]
)

plt.figure(figsize=(8, 4))

for term in glucose.terms:
    plt.plot(
        glucose.universe,
        glucose[term].mf,
        label=term
    )

plt.title("Glucose Membership Functions")
plt.xlabel("Glucose (mg/dL)")
plt.ylabel("Membership Degree")
plt.ylim(0, 1.05)
plt.xlim(
    min(glucose.universe),
    max(glucose.universe)
)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# Blood Pressure Membership Function

blood_pressure['Low'] = fuzz.trapmf(
    blood_pressure.universe, [0, 0, 50, 65]
)

blood_pressure['Normal'] = fuzz.trimf(
    blood_pressure.universe, [60, 75, 90]
)

blood_pressure['High'] = fuzz.trapmf(
    blood_pressure.universe, [85, 100, 130, 140]
)

plt.figure(figsize=(8, 4))

for term in blood_pressure.terms:
    plt.plot(
        blood_pressure.universe,
        blood_pressure[term].mf,
        label=term
    )

plt.title("Blood Pressure Membership Functions")
plt.xlabel("Blood Pressure (mm Hg)")
plt.ylabel("Membership Degree")
plt.ylim(0, 1.05)
plt.xlim(
    min(blood_pressure.universe),
    max(blood_pressure.universe)
)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# Skin Thickness Membership Function

skin_thickness['Thin'] = fuzz.trapmf(
    skin_thickness.universe, [0, 0, 10, 20]
)

skin_thickness['Normal'] = fuzz.trimf(
    skin_thickness.universe, [15, 30, 45]
)

skin_thickness['Thick'] = fuzz.trapmf(
    skin_thickness.universe, [40, 55, 100, 110]
)

plt.figure(figsize=(8, 4))

for term in skin_thickness.terms:
    plt.plot(
        skin_thickness.universe,
        skin_thickness[term].mf,
        label=term
    )

plt.title("Skin Thickness Membership Functions")
plt.xlabel("Skin Thickness (mm)")
plt.ylabel("Membership Degree")
plt.ylim(0, 1.05)
plt.xlim(
    min(skin_thickness.universe),
    max(skin_thickness.universe)
)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# Insulin Membership Function

insulin['Low'] = fuzz.trapmf(
    insulin.universe, [0, 0, 50, 100]
)

insulin['Medium'] = fuzz.trimf(
    insulin.universe, [80, 200, 320]
)

insulin['High'] = fuzz.trapmf(
    insulin.universe, [300, 500, 850, 900]
)

plt.figure(figsize=(8, 4))

for term in insulin.terms:
    plt.plot(
        insulin.universe,
        insulin[term].mf,
        label=term
    )

plt.title("Insulin Membership Functions")
plt.xlabel("Insulin (μU/mL)")
plt.ylabel("Membership Degree")
plt.ylim(0, 1.05)
plt.xlim(
    min(insulin.universe),
    max(insulin.universe)
)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# BMI Membership Function

bmi['Low'] = fuzz.trapmf(
    bmi.universe, [0, 0, 18, 22]
)

bmi['Medium'] = fuzz.trimf(
    bmi.universe, [20, 27, 34]
)

bmi['High'] = fuzz.trapmf(
    bmi.universe, [30, 35, 70, 75]
)

plt.figure(figsize=(8, 4))

for term in bmi.terms:
    plt.plot(
        bmi.universe,
        bmi[term].mf,
        label=term
    )

plt.title("BMI Membership Functions")
plt.xlabel("BMI (kg/m²)")
plt.ylabel("Membership Degree")
plt.ylim(0, 1.05)
plt.xlim(
    min(bmi.universe),
    max(bmi.universe)
)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# DPF Membership Function

dpf['Low'] = fuzz.trapmf(
    dpf.universe, [0.0, 0.0, 0.3, 0.5]
)

dpf['Medium'] = fuzz.trimf(
    dpf.universe, [0.4, 0.8, 1.2]
)

dpf['High'] = fuzz.trapmf(
    dpf.universe, [1.0, 1.5, 2.5, 2.6]
)

plt.figure(figsize=(8, 4))

for term in dpf.terms:
    plt.plot(
        dpf.universe,
        dpf[term].mf,
        label=term
    )

plt.title("DPF Membership Functions")
plt.xlabel("Diabetes Pedigree Function")
plt.ylabel("Membership Degree")
plt.ylim(0, 1.05)
plt.xlim(
    min(dpf.universe),
    max(dpf.universe)
)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# Age Membership Function

age['Young'] = fuzz.trapmf(
    age.universe, [20, 20, 30, 40]
)

age['Middle'] = fuzz.trimf(
    age.universe, [35, 50, 65]
)

age['Old'] = fuzz.trapmf(
    age.universe, [60, 70, 100, 100]
)

plt.figure(figsize=(8, 4))

for term in age.terms:
    plt.plot(
        age.universe,
        age[term].mf,
        label=term
    )

plt.title("Age Membership Functions")
plt.xlabel("Age (years)")
plt.ylabel("Membership Degree")
plt.ylim(0, 1.05)
plt.xlim(
    min(age.universe),
    max(age.universe)
)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# Risk Output

risk['Low'] = fuzz.trapmf(
    risk.universe, [0.0, 0.0, 0.2, 0.4]
)

risk['Medium'] = fuzz.trimf(
    risk.universe, [0.3, 0.5, 0.7]
)

risk['High'] = fuzz.trapmf(
    risk.universe, [0.6, 0.8, 1.0, 1.0]
)

for term in risk.terms:
    plt.plot(
        risk.universe,
        risk[term].mf,
        label=term
    )

    plt.fill_between(
        risk.universe,
        0,
        risk[term].mf,
        alpha=0.2
    )

plt.title("Output Risk Membership Functions")
plt.xlabel("Risk Score")
plt.ylabel("Membership Degree")
plt.ylim(0, 1.05)
plt.xlim(
    min(risk.universe),
    max(risk.universe)
)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# Define High Risk Rules

rule1 = ctrl.Rule(
    glucose['High'] & bmi['High'],
    risk['High']
)

rule2 = ctrl.Rule(
    glucose['High'] & age['Old'],
    risk['High']
)

rule3 = ctrl.Rule(
    glucose['High'] & insulin['High'],
    risk['High']
)

rule4 = ctrl.Rule(
    glucose['High'] & dpf['High'],
    risk['High']
)

rule5 = ctrl.Rule(
    pregnancies['Many'] & glucose['High'],
    risk['High']
)

rule6 = ctrl.Rule(
    bmi['High'] & dpf['High'],
    risk['High']
)

rule7 = ctrl.Rule(
    glucose['High'] &
    age['Middle'] &
    bmi['High'],
    risk['High']
)

rule8 = ctrl.Rule(
    blood_pressure['High'] & bmi['High'],
    risk['High']
)

rule9 = ctrl.Rule(
    insulin['High'] & age['Old'],
    risk['High']
)

rule10 = ctrl.Rule(
    glucose['High'] &
    skin_thickness['Thick'] &
    insulin['High'],
    risk['High']
)

rule11 = ctrl.Rule(
    glucose['High'] &
    bmi['High'] &
    pregnancies['Many'],
    risk['High']
)

rule12 = ctrl.Rule(
    dpf['High'] & age['Old'],
    risk['High']
)

rule13 = ctrl.Rule(
    bmi['High'] & skin_thickness['Thick'],
    risk['High']
)


# Define Medium Risk Rules

rule14 = ctrl.Rule(
    glucose['Medium'] & bmi['Medium'],
    risk['Medium']
)

rule15 = ctrl.Rule(
    glucose['Medium'] & age['Middle'],
    risk['Medium']
)

rule16 = ctrl.Rule(
    bmi['Medium'] & dpf['Medium'],
    risk['Medium']
)

rule17 = ctrl.Rule(
    pregnancies['Few'] & glucose['Medium'],
    risk['Medium']
)

rule18 = ctrl.Rule(
    insulin['Medium'] & glucose['Medium'],
    risk['Medium']
)

rule19 = ctrl.Rule(
    blood_pressure['Normal'] & bmi['Medium'],
    risk['Medium']
)

rule20 = ctrl.Rule(
    age['Middle'] & dpf['Medium'],
    risk['Medium']
)

rule21 = ctrl.Rule(
    skin_thickness['Normal'] & bmi['Medium'],
    risk['Medium']
)

rule22 = ctrl.Rule(
    blood_pressure['High'] & glucose['Medium'],
    risk['Medium']
)

rule23 = ctrl.Rule(
    pregnancies['Many'] & age['Middle'],
    risk['Medium']
)

rule24 = ctrl.Rule(
    glucose['Medium'] & dpf['High'],
    risk['Medium']
)

rule25 = ctrl.Rule(
    insulin['Medium'] & bmi['Medium'],
    risk['Medium']
)

rule26 = ctrl.Rule(
    glucose['High'] & dpf['Medium'],
    risk['Medium']
)

rule27 = ctrl.Rule(
    skin_thickness['Thick'] & bmi['Medium'],
    risk['Medium']
)


# Define Low Risk Rules

rule28 = ctrl.Rule(
    glucose['Low'] & bmi['Low'],
    risk['Low']
)

rule29 = ctrl.Rule(
    glucose['Low'] & age['Young'],
    risk['Low']
)

rule30 = ctrl.Rule(
    glucose['Low'] & dpf['Low'],
    risk['Low']
)

rule31 = ctrl.Rule(
    pregnancies['None'] & glucose['Low'],
    risk['Low']
)

rule32 = ctrl.Rule(
    blood_pressure['Normal'] & glucose['Low'],
    risk['Low']
)

rule33 = ctrl.Rule(
    bmi['Low'] & skin_thickness['Thin'],
    risk['Low']
)

rule34 = ctrl.Rule(
    insulin['Low'] & glucose['Low'],
    risk['Low']
)

rule35 = ctrl.Rule(
    dpf['Low'] & age['Young'],
    risk['Low']
)

rule36 = ctrl.Rule(
    insulin['Low'] & bmi['Low'],
    risk['Low']
)

rule37 = ctrl.Rule(
    pregnancies['None'] &
    dpf['Low'] &
    glucose['Low'],
    risk['Low']
)

rule38 = ctrl.Rule(
    age['Young'] & bmi['Low'],
    risk['Low']
)

rule39 = ctrl.Rule(
    glucose['Low'] & skin_thickness['Thin'],
    risk['Low']
)

rule40 = ctrl.Rule(
    blood_pressure['Low'] & glucose['Low'],
    risk['Low']
)


# Combine All Rules Into a List

rules = [
    rule1, rule2, rule3, rule4, rule5,
    rule6, rule7, rule8, rule9, rule10,
    rule11, rule12, rule13, rule14, rule15,
    rule16, rule17, rule18, rule19, rule20,
    rule21, rule22, rule23, rule24, rule25,
    rule26, rule27,
    rule28, rule29, rule30, rule31, rule32,
    rule33, rule34, rule35, rule36, rule37,
    rule38, rule39, rule40
]


# Create Control System and Simulation Engine

risk_ctrl = ctrl.ControlSystem(rules)

risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)


# Run Simulation on all Rows

risk_scores = []
fuzzy_results = []

for i, row in df.iterrows():

    try:

        # Feed inputs into the fuzzy system

        risk_simulation.input['Pregnancies'] = row['Pregnancies']
        risk_simulation.input['Glucose'] = row['Glucose']
        risk_simulation.input['BloodPressure'] = row['BloodPressure']
        risk_simulation.input['SkinThickness'] = row['SkinThickness']
        risk_simulation.input['Insulin'] = row['Insulin']
        risk_simulation.input['BMI'] = row['BMI']
        risk_simulation.input['DPF'] = row['DiabetesPedigreeFunction']
        risk_simulation.input['Age'] = row['Age']


        # Compute output

        risk_simulation.compute()

        score = risk_simulation.output['Risk']

        risk_scores.append(score)


        # Apply threshold for binary class

        fuzzy_results.append(
            1 if score >= 0.5 else 0
        )

    except:

        # Handle fuzzy system errors
        # (e.g. values outside MF range)

        risk_scores.append(None)
        fuzzy_results.append(None)


# Store results in dataframe

df['Risk_Score'] = risk_scores
df['Fuzzy_Result'] = fuzzy_results

df[['Risk_Score', 'Fuzzy_Result']].head()


# Optional: Create labeled result
# (for clarity in the dataset)

df_cleaned = df.dropna(
    subset=['Fuzzy_Result']
).copy()

df_cleaned['Fuzzy_Classification'] = (
    df_cleaned['Fuzzy_Result']
    .replace({
        1: 'Diabetic',
        0: 'Non-Diabetic'
    })
)


pd.set_option(
    'display.max_columns',
    None
)

pd.set_option(
    'display.width',
    None
)

df_cleaned


# Save final dataset

df_cleaned.to_csv(
    'final_fuzzy_dataset.csv',
    index=False
)


# Evaluate Classification Accuracy

total_predictions = len(df_cleaned)

correct_predictions = (
    df_cleaned['Fuzzy_Result'] ==
    df_cleaned['Outcome']
).sum()

accuracy = (
    correct_predictions /
    total_predictions
) * 100


# Output Summary

print(
    f"Total Predictions: {total_predictions}"
)

print(
    f"Correct Predictions: {correct_predictions}"
)

print(
    f"Accuracy: {accuracy:.2f}%"
)