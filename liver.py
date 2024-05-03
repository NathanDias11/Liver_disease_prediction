import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular

# Title
st.title("Liver Disease Prediction")

# Load data
df = pd.read_csv('live.csv')

# Features (symptoms)
features = df.columns[:-1]

# Sidebar for symptom selection
st.sidebar.markdown("# Select Symptoms")
symptom_selection = {}
for symptom in features:
    symptom_selection[symptom] = st.sidebar.checkbox(symptom)

# Function to predict disease
def rf_classifier(selected_symptoms):
    # Train the model
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # LimeTabularExplainer
    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
                                                       feature_names=features,
                                                       class_names=['No Disease', 'Disease'],
                                                       discretize_continuous=True)

    # Predict disease based on selected symptoms
    input_data = [selected_symptoms]
    predict = clf.predict(input_data)
    predicted_disease = predict[0]
    
    # Calculate accuracy using cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    accuracy = np.mean(cv_scores)
    
    return predicted_disease, accuracy, clf, explainer

# Submit button
if st.sidebar.button("Submit"):
    # Prepare input data for prediction
    selected_symptoms = [1 if symptom_selection[symptom] else 0 for symptom in features]

    # Call the rf_classifier function to predict disease and calculate accuracy
    predicted_disease, accuracy, clf, explainer = rf_classifier(selected_symptoms)

    # Display predicted disease
    st.subheader("Predicted Disease")
    st.write(predicted_disease)
    

    # LIME explanation
    exp = explainer.explain_instance(np.array(selected_symptoms), clf.predict_proba)
    st.subheader("LIME Explanation")
    for explanation in exp.as_list():
        st.write(explanation[0], explanation[1])
