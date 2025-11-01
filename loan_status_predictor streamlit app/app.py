import pickle
import streamlit as st
from typing import Optional, List


def try_load_model(base_names: List[str]) -> Optional[object]:
    """Try loading a pickle model from several filename variants.

    Returns the loaded object or None if none of the files exist / can be loaded.
    """
    for name in base_names:
        for ext in ('', '.sav', '.pkl', '.bin'):
            path = f"{name}{ext}"
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                continue
            except Exception as e:
                # If file exists but can't be unpickled, raise so the user sees the error
                st.error(f"Found {path} but failed to load it: {e}")
                return None
    return None


st.title('Loan Status Prediction')

# Desired input column order requested by the user
FEATURE_COLUMNS = ['Married', 'Dependents', 'LoanAmount', 'Credit_History']

# Try to load model named 'loan_staus_model' (try a few filename variants)
model = try_load_model(['loan_staus_model', 'loan_status_model'])

if model is None:
    st.warning("Model 'loan_staus_model' not found in the app directory. Place the pickled model file (e.g. 'loan_staus_model.sav') next to this script.")


def encode_married(value: str) -> int:
    return 1 if value.lower() in ('yes', 'y', 'true', '1') else 0


def encode_dependents(value: str) -> int:
    # common representation: '0','1','2','3+' -> convert '3+' to 3
    if value.endswith('+'):
        try:
            return int(value.rstrip('+'))
        except ValueError:
            return 3
    try:
        return int(value)
    except Exception:
        return 0


with st.form('loan_form'):
    st.subheader('Enter applicant details')

    married = st.selectbox('Married', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    loan_amount = st.number_input('LoanAmount', min_value=0.0, value=100.0, step=0.5)
    credit_history = st.selectbox('Credit_History', ['0', '1'])

    submitted = st.form_submit_button('Predict Loan Status')

    if submitted:
        if model is None:
            st.error("Cannot make prediction because the model wasn't loaded. See warning above.")
        else:
            # Build feature vector in the requested order
            x_married = encode_married(married)
            x_dependents = encode_dependents(dependents)
            x_loan = float(loan_amount)
            try:
                x_credit = int(credit_history)
            except Exception:
                x_credit = 0

            input_vector = [x_married, x_dependents, x_loan, x_credit]

            try:
                prediction = model.predict([input_vector])
                # If model provides predict_proba, show probability
                proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba([input_vector])[0].tolist()
                    except Exception:
                        proba = None

                label = prediction[0] 
                if label == 1:
                    label = 'ELIGIBLE'              
                    st.success(f'Prediction: {label}')
                else:
                    label = 'NOT ELIGIBLE'
                    st.error(f'Prediction: {label}')
                    
                if proba is not None:
                    st.info(f'Probabilities: {proba}')
            except Exception as e:
                st.error(f'Prediction failed: {e}')

st.markdown('---')

