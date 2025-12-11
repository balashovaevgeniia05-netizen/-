import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================
st.set_page_config(
    page_title="💰 Income Predictor | ML Model",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== СТИЛИ CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #5D5D5D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(45deg, #2E86AB, #A23B72);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(46, 134, 171, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ==================== ЗАГРОВОК ====================
st.markdown('<h1 class="main-header">💰 Income Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict if annual income exceeds $50,000</p>', unsafe_allow_html=True)

# ==================== ЗАГРУЗКА МОДЕЛИ ====================
@st.cache_resource
def load_model():
    """Load trained model and preprocessing objects"""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        st.success("✅ Model loaded successfully!")
        return model, scaler, encoder
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("Please ensure model.pkl, scaler.pkl, and encoder.pkl are in the same directory.")
        return None, None, None

model, scaler, encoder = load_model()

# ==================== БОКОВАЯ ПАНЕЛЬ ====================
with st.sidebar:
    st.header("👤 Personal Information")
    
    # Личная информация
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=17, max_value=90, value=35, step=1, 
                             help="Age of the individual")
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
    
    st.divider()
    
    # Образование и работа
    st.subheader("🎓 Education & Work")
    education = st.selectbox(
        "Highest Education",
        ["Bachelors", "Some-college", "HS-grad", "Masters", "Doctorate", 
         "Assoc-voc", "Assoc-acdm", "Prof-school", "11th", "10th", 
         "9th", "12th", "7th-8th", "5th-6th", "1st-4th", "Preschool"],
        help="Highest level of education achieved"
    )
    
    workclass = st.selectbox(
        "Employment Sector",
        ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
         "Local-gov", "State-gov", "Without-pay", "Never-worked"],
        help="Type of employment"
    )
    
    occupation = st.selectbox(
        "Occupation",
        ["Tech-support", "Craft-repair", "Other-service", "Sales",
         "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
         "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
         "Transport-moving", "Priv-house-serv", "Protective-serv",
         "Armed-Forces"],
        help="Current occupation"
    )
    
    hours_per_week = st.slider("Hours per week", 1, 99, 40,
                              help="Average hours worked per week")
    
    st.divider()
    
    # Семейное положение и финансы
    st.subheader("🏠 Family & Finance")
    marital_status = st.selectbox(
        "Marital Status",
        ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
         "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
        help="Current marital status"
    )
    
    relationship = st.selectbox(
        "Family Role",
        ["Wife", "Own-child", "Husband", "Not-in-family",
         "Other-relative", "Unmarried"],
        help="Role in the family"
    )
    
    col3, col4 = st.columns(2)
    with col3:
        capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0,
                                      help="Income from investments")
    with col4:
        capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=5000, value=0,
                                      help="Losses from investments")
    
    # Остальные поля
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", 
                                "Amer-Indian-Eskimo", "Other"])
    fnlwgt = st.number_input("Population Weight", min_value=10000, max_value=2000000, 
                            value=189000, help="Statistical weight in population data")
    education_num = st.slider("Years of Education", 1, 16, 10,
                             help="Number of years of formal education")

# ==================== ФУНКЦИЯ ПРЕДСКАЗАНИЯ ====================
def make_prediction():
    """Prepare data and make prediction"""
    # Create input dataframe
    input_data = {
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'education-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week]
    }
    
    df = pd.DataFrame(input_data)
    
    # Prepare features
    numeric_features = ['age', 'fnlwgt', 'education-num', 
                       'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status',
                           'occupation', 'relationship', 'race', 'sex']
    
    # Preprocessing
    try:
        X_num = scaler.transform(df[numeric_features])
        X_cat = encoder.transform(df[categorical_features])
        X_final = np.hstack([X_num, X_cat])
        
        # Prediction
        prediction = model.predict(X_final)[0]
        probability = model.predict_proba(X_final)[0][1]
        
        return prediction, probability, df
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, df

# ==================== ОСНОВНОЕ ОКНО ====================
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Info", "📈 Data Analysis"])

with tab1:
    st.header("Income Prediction")
    
    if model is None:
        st.warning("Model not loaded. Please check if model files are uploaded.")
    else:
        # Кнопка для предсказания
        predict_btn = st.button("🚀 Predict Income", type="primary", use_container_width=True)
        
        if predict_btn:
            with st.spinner("Analyzing demographic data..."):
                prediction, probability, input_df = make_prediction()
            
            if prediction is not None:
                # Display results in cards
                st.markdown("### 📋 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Predicted Income",
                        value="> $50K" if prediction == 1 else "≤ $50K",
                        delta="High Income" if prediction == 1 else "Low Income",
                        delta_color="normal"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Probability",
                        value=f"{probability:.1%}",
                        delta=f"Confidence: {'High' if probability > 0.7 or probability < 0.3 else 'Medium'}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    confidence_level = "High" if probability > 0.7 else "Medium" if probability > 0.6 else "Low"
                    st.metric(
                        label="Confidence Level",
                        value=confidence_level
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Progress bar
                st.markdown("#### Confidence Indicator")
                st.progress(float(probability))
                
                # Visualization
                st.markdown("#### 📊 Probability Distribution")
                chart_data = pd.DataFrame({
                    'Income Category': ['≤ $50K', '> $50K'],
                    'Probability': [1 - probability, probability]
                })
                st.bar_chart(chart_data.set_index('Income Category'), use_container_width=True)
                
                # Interpretation
                st.markdown("#### 💡 Interpretation & Insights")
                if probability > 0.7:
                    st.success("""
                    **✅ High probability of earning more than $50K annually**
                    
                    **Likely contributing factors:**
                    - Higher education level
                    - Professional/managerial occupation  
                    - Full-time employment (40+ hours/week)
                    - Married status
                    - Age in prime earning years
                    """)
                elif probability > 0.5:
                    st.info("""
                    **🟡 Moderate probability of earning more than $50K**
                    
                    **Consider for improvement:**
                    - Additional education or certifications
                    - Career advancement opportunities
                    - Gaining more work experience
                    - Developing specialized skills
                    """)
                else:
                    st.warning("""
                    **🟠 Lower probability of exceeding $50K income**
                    
                    **Potential areas for development:**
                    - Pursue higher education
                    - Skills training programs
                    - Explore career change opportunities
                    - Seek full-time employment
                    """)

with tab2:
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Gradient Boosting Classifier
        This machine learning model predicts annual income based on demographic and employment factors.
        
        #### Model Performance
        - **AUC-ROC Score**: 0.9194
        - **Accuracy**: ~87%
        - **Precision**: 0.75
        - **Recall**: 0.65
        - **Cross-validation**: 5-Fold
        
        #### Dataset
        - **Source**: UCI Adult Income Dataset
        - **Samples**: 32,561 individuals
        - **Features**: 14 attributes
        """)
    
    with col2:
        st.markdown("""
        #### Features Used
        
        **Numerical Features (6):**
        1. Age
        2. Final weight (fnlwgt)
        3. Education years
        4. Capital gain
        5. Capital loss
        6. Hours per week
        
        **Categorical Features (7):**
        1. Workclass
        2. Education
        3. Marital status
        4. Occupation
        5. Relationship
        6. Race
        7. Gender
        
        #### Model Parameters
        ```python
        GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            criterion='friedman_mse',
            max_features=None,
            random_state=42
        )
        ```
        """)

with tab3:
    st.header("📈 Data Analysis & Insights")
    
    st.markdown("""
    ### Dataset Overview
    The model was trained on the **Adult Income Dataset** from the UCI Machine Learning Repository.
    This dataset contains demographic information from the 1994 US Census.
    
    #### Key Statistics
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", "32,561")
    
    with col2:
        st.metric("Features", "14")
    
    with col3:
        st.metric("Target Classes", "2")
    
    st.markdown("""
    #### Class Distribution
    - **≤ $50K**: 76% of individuals
    - **> $50K**: 24% of individuals
    
    #### Most Important Features
    1. **Age** - Prime earning years are 35-55
    2. **Education Level** - Higher education correlates with higher income
    3. **Occupation** - Professional/managerial roles pay more
    4. **Hours per Week** - Full-time work increases earning potential
    5. **Marital Status** - Married individuals tend to earn more
    
    #### Model Development
    This model was developed as part of a machine learning course project. 
    It demonstrates the application of ensemble methods for classification tasks.
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>💡 <em>This predictive model uses Gradient Boosting algorithms trained on demographic data from the 1994 US Census.</em></p>
    <p>⚠️ <em>Predictions are statistical estimates based on historical patterns and should not be used for financial or career decisions.</em></p>
    <p>📚 <em>Educational Project | Machine Learning | Data Science | 2024</em></p>
    <p>🔗 <em>GitHub Repository: github.com/your-username/income-predictor-app</em></p>
</div>
""", unsafe_allow_html=True)

# ==================== СКРЫТЫЕ ПРОВЕРКИ ====================
def check_files():
    """Check if required files exist"""
    import os
    required_files = ['model.pkl', 'scaler.pkl', 'encoder.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files and model is None:
        st.sidebar.warning(f"Missing files: {', '.join(missing_files)}")

check_files()