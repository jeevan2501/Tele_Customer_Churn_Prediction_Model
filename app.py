import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# PascalCase for class names, snake_case for variables and functions

class ChurnPredictionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def preprocess_data(self, df):
        # Drop customerID as it's not useful
        df = df.drop('customerID', axis=1)
        
        # Handle TotalCharges: convert to float, replace spaces with 0
        df['TotalCharges'] = df['TotalCharges'].replace(' ', 0).astype(float)
        
        # Convert SeniorCitizen to object for consistency
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.drop('Churn')
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Encode target variable
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        return df

    def split_data(self, df):
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def evaluate_model(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        cm = confusion_matrix(self.y_test, self.y_pred)
        return accuracy, report, cm

    def plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        return fig

    def predict_churn(self, input_data):
        # Preprocess input data using stored label encoders
        input_df = pd.DataFrame([input_data])
        for col, le in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        prediction = self.model.predict(input_df)
        return 'Yes' if prediction[0] == 1 else 'No'

# Streamlit App
def main():
    st.set_page_config(page_title="Churn Prediction Model", page_icon="📊", layout="wide")

    # Use an attractive template with custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize session state for the model
    if 'model' not in st.session_state:
        st.session_state.model = ChurnPredictionModel(r"C:\Users\JEEVAN\OneDrive\Desktop\Bia_Mini_Project\Model\Telco_Customer_Churn.csv")

    st.title("📈 Telco Customer Churn Prediction Model")
    st.markdown("This app uses a Random Forest Classifier to predict customer churn based on the Telco dataset.")

    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Choose a section", ["Model Training & Evaluation", "Predict Churn"])

    if page == "Model Training & Evaluation":
        st.header("Model Training & Evaluation")

        if st.button("Train Model"):
            with st.spinner("Loading data..."):
                df = st.session_state.model.load_data()
            with st.spinner("Preprocessing data..."):
                df = st.session_state.model.preprocess_data(df)
            with st.spinner("Splitting data..."):
                st.session_state.model.split_data(df)
            with st.spinner("Training model..."):
                st.session_state.model.train_model()
            with st.spinner("Evaluating model..."):
                accuracy, report, cm = st.session_state.model.evaluate_model()

            st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")

            st.subheader("Classification Report")
            st.text(report)

            st.subheader("Confusion Matrix")
            fig = st.session_state.model.plot_confusion_matrix(cm)
            st.pyplot(fig)

    elif page == "Predict Churn":
        st.header("Predict Customer Churn")

        if st.session_state.model.model is None:
            st.warning("Please train the model first in the 'Model Training & Evaluation' section.")
        else:
            st.subheader("Enter Customer Details")

            # Input fields based on dataset columns (excluding customerID and Churn)
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=100.0)

            input_data = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }

            if st.button("Predict"):
                prediction = st.session_state.model.predict_churn(input_data)
                st.success(f"Predicted Churn: {prediction}")

if __name__ == "__main__":
    main()