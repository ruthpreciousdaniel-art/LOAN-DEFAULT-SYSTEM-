
import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="wide"
)

# LOAD MODEL

model = joblib.load("loan_default_model.pkl")

# CUSTOM CSS

st.markdown('''

 ''', unsafe_allow_html=True)

# SIDEBAR

with st.sidebar:

    st.image(
            "https://cdn-icon-png.flaticon.com/512/3135/3135715.png",
            width=150
    )

    st.title("📊 Project Overview")

    st.markdown('''
    ### Model is used

    - Logistic Regression
    - Random Forest
    - XGBoost

    ### Objective

    Predict whether a borrower is likely to default on a loan.

     ### Features

    -Real-time prediction
    -Risk probability
    -ML-powered decision support
    ''')

    st.divider()

    st.info(
        "Built with streamlit, Scikit-Learn and XGBooost."
    )

# HEADER

st.markdown(
    "💳 Loan Default Prediction System",
     unsafe_allow_html=True
)

st.markdown(
    "Machine Learning Powered Credit Risk Assessment Dashboard"
)

st.divider()

# INPUT SECTION

col1, col2, col3 = st.columns(3)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    Income = st.number_input(
        "Income",
        min_value=1000,
        value=5000
    )

    Loan_Amount = st.number_input(
          "Loan Amount",
          min_value=1000,
          value=10000
    )

with col2:

    CreditScore = st.number_input(
          "Credit Score",
          min_value=300,
          max_value=850,
          value=650
    )

    InterestRate = st.number_input(
          "Interest Rate",
          min_value=1.0,
          max_value=40.0,
          value=10.0
    )

    LoanTerm = st.number_input(
          "Loan Term (Months)",
          min_value=6,
          max_value=360,
          value=36
    )

with col3:

    MonthsEmployed = st.number_input(
          "Months Employed",
          min_value=0,
          value=24
    )

    NumCreditLines = st.number_input(
          "Number of Credit Lines",
          min_value=0,
          value=5
    )

    DTIRatio = st.number_input(
          "Debt-to-Income Ratio",
          min_value=0.0,
          max_value=1.0,
          value=0.30
    )

# PREDICTION

if st.button("🔍  Predict Loan Risk"):

    # Create a Dictionary for the input features
    input_dict = {
        'Age': Age,
        'Income': Income,
        'LoanAmount': Loan_Amount,
        'CreditScore': CreditScore,
        'InterestRate': InterestRate,
        'LoanTerm': LoanTerm,
        'MonthsEmployed': MonthsEmployed,
        'NumCreditLines': NumCreditLines,
        'DTIRatio': DTIRatio
    }

    # Create a DataFrame from the input dictionary
    input_data_df = pd.DataFrame([input_dict])

    # Get the column names from the training data X
    # This list is obtained from the kernel state 's 'X' variable.
    expected_column_names = [
         'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
         'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
         'Education_High School',"Education_Master's", 'Education_PhD',
         'EmploymentType_Part-time', 'EmploymentType_Self-employed', 'EmploymentType_Unemployed',
         'MaritalStatus_Married', 'MaritalStatus_Single',
         'HasMortgage_Yes', 'HasDependents_Yes', 'LoanPurpose_Business',
         'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other',
         'HasCoSigner_Yes'
    ]

    # Create an empty DataFrame with the expected column names
    final_input_df = pd.DataFrame(columns=expected_column_names)


    # Populate the known numeric features
    for col in input_data_df.columns:
        if col in final_input_df.columns:
            final_input_df[col] = input_data_df[col]

    # Fill all other columns with (categorical one-hot encoded) with 0
    for col in final_input_df.columns:
        if col not in input_data_df.columns:
            final_input_df[col] = 0


    # Ensure the order of columns matches the training data 'X'
    final_input_df = final_input_df[expected_column_names]


    prediction = model.predict(final_input_df)[0]


    probability = model.predict_proba(
       final_input_df
    )[0][1]

    st.divider()

    st.subheader("Prediction Result")

    colA, colB = st.columns(2)

    with colA:

       if prediction == 1:

           st.error(
               f'''
               ⚠️HIGH DEFAULT RISK

               Probability of Default:
               {probability:.2%}
               '''
           )

       else:

           st.success(
               f'''
               ✅ LOW DEFAULT RISK

                Probability of Default:
                {probability:.2%}
                '''
           )

    with colB:

           st.metric(
                "Default Probability",
                f"{probability:.2%}"
           )

           st.progress(float(probability))
           st.metric(
                "Risk Level",
                "High Risk" if prediction == 1 else "Low Risk"
           )

# FOOTER

st.divider()

st.caption(
     "Loan Default Prediction Dashboard | Machine Learning Project"
)
