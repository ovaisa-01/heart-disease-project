# ❤️ Heart Disease Prediction Project  

### A Machine Learning Approach Using the UCI Heart Disease Dataset  

---

## 1. Introduction  

Cardiovascular diseases remain one of the leading causes of death worldwide. Early detection of heart disease can significantly improve patient outcomes through timely intervention.  
This project applies **machine learning techniques** to predict the presence of heart disease based on clinical and physiological patient data.  

The dataset used in this study is the **Heart Disease Dataset from Kaggle (UCI Source)**, containing attributes related to age, blood pressure, cholesterol, and other vital indicators.  

---

## 2. Objectives  

The objectives of this project are to:  
1. Preprocess and explore the heart disease dataset to identify patterns and relationships between variables.  
2. Apply a **Logistic Regression model** for binary classification (disease vs. no disease).  
3. Evaluate model performance using accuracy, precision, recall, and f1-score metrics.  
4. Build foundational understanding of the **machine learning pipeline** using Python and Scikit-learn.

---

## 3. Dataset Description  

- **Source:** [Kaggle – Heart Disease UCI Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
- **Number of Samples:** 1,025  
- **Target Variable:** `target`  
  - 1 → Presence of heart disease  
  - 0 → Absence of heart disease  
- **Missing Values:** None detected  

### Key Features  

| Feature | Description |
|----------|-------------|
| age | Age of the patient |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels (0–3) colored by fluoroscopy |
| thal | Thalassemia |
| target | Presence (1) or absence (0) of heart disease |

---

## 4. Methodology  

### Step 1: Data Preprocessing  
- Imported the dataset using `pandas`.  
- Checked for missing values (none found).  
- Normalized numerical features and encoded categorical data.  
- Split data into **80% training** and **20% testing** sets.

### Step 2: Model Development  
- Applied **Logistic Regression** for binary classification.  
- Utilized Scikit-learn’s `train_test_split`, `LogisticRegression`, and `metrics` libraries.  

### Step 3: Model Evaluation  
Performance was evaluated using accuracy and a classification report:

| Metric | Score |
|---------|-------|
| **Accuracy** | 79.5% |
| **Precision (No Disease)** | 0.85 |
| **Precision (Disease)** | 0.76 |
| **Recall (No Disease)** | 0.72 |
| **Recall (Disease)** | 0.87 |

---

## 5. Tools and Libraries  

| Tool | Purpose |
|------|----------|
| **Python** | Programming language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical computation |
| **Scikit-learn** | Machine learning modeling |
| **Matplotlib & Seaborn** | Data visualization |

---

## 6. Implementation  

### Running the Project  

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ovaisa-01/heart-disease-project.git
   cd heart-disease-project
   
2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # For Windows


3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt


4. **Run the Script**
   ```bash
   python heart_disease.py


### The program will display the model’s accuracy and classification report in the terminal.

---
   
## 7. Results and Discussion

The Logistic Regression model achieved an accuracy of approximately 79.5%, demonstrating promising predictive performance for a simple classification model.
Further experiments with Random Forest or Support Vector Machines (SVM) could potentially enhance accuracy and robustness.

The results validate the dataset’s potential for machine learning–based predictive modeling in healthcare analytics.

---

## 8. Future Work

 - Explore additional classification algorithms (SVM, Random Forest, Gradient Boosting).
 - Apply cross-validation for improved generalization.
 - Deploy the model using Streamlit or Flask for web-based prediction.
 - Conduct feature importance analysis to interpret model behavior.

---

## 9. Conclusion

This project successfully demonstrates the use of **machine learning** in healthcare prediction tasks using accessible tools.  
Through systematic **data preprocessing**, **model training**, and **performance evaluation**, it establishes a reproducible baseline for future exploration in **predictive health analytics**.

---

