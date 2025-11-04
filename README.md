🫀 Heart Disease Prediction Using Machine Learning
Abstract

This project investigates the use of machine learning algorithms to predict the presence of heart disease in patients based on key clinical parameters.
Using the Kaggle Heart Disease dataset, the project applies Logistic Regression for binary classification, achieving an accuracy of approximately 79.5%.
The aim is to explore how simple models can assist in early medical diagnosis and decision support systems.

1. Introduction

Heart disease remains one of the leading causes of death globally.
Early detection through data-driven methods can significantly improve patient outcomes.
This project represents a beginner-level implementation of predictive modeling for healthcare analytics.

2. Methodology

The following workflow was applied:

Data Preprocessing:

Cleaning data, normalizing features, and splitting into training/testing sets.

Feature Selection:

Variables include age, sex, cholesterol, chest pain type, blood pressure, and ECG results.

Model Training:

Logistic Regression classifier using scikit-learn.

Evaluation Metrics:

Accuracy, precision, recall, and F1-score.

3. Results

The trained model achieved the following performance:

Accuracy: 79.51%
Precision, Recall, and F1-score show balanced performance across both classes.


These results demonstrate that even a simple logistic regression model can provide useful baseline predictions for clinical data.

4. Tools and Libraries

Programming Language: Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

Install all dependencies using:

pip install -r requirements.txt

5. How to Run

Clone this repository:

git clone https://github.com/ovaisa-01/heart-disease-project.git
cd heart-disease-project


Run the script:

python heart_disease.py

6. Dataset

This project uses the Heart Disease Dataset (Kaggle)
,
containing patient data such as:

Age, sex, cholesterol, blood pressure

ECG and exercise-related data

Binary target variable: 0 (no disease) and 1 (disease present)

7. Conclusion

This study demonstrates how basic machine learning techniques can assist in medical prediction tasks.
Future improvements may include ensemble methods, hyperparameter tuning, and neural network architectures to enhance model accuracy and generalization.

8. Author

Ovaisa KT
Beginner AI/ML Learner
📧 Email: ovaisabloom@gmail.com

🌐 GitHub: @ovaisa-01