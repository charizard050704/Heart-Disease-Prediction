# Heart Disease Prediction using ANN

## Keywords
Artificial Neural Network, Deep Learning, Binary Classification, Medical Analytics, TensorFlow/Keras, Streamlit, ROC-AUC

## Abstract
This project implements an Artificial Neural Network (ANN) to predict the likelihood of heart disease based on patient data. Using clinical attributes such as age, cholesterol levels, and chest pain type, the ANN classifies whether a patient is at risk of heart disease. The trained model is deployed as a web application using Streamlit, providing an interactive interface for healthcare practitioners and users.

## Methodology
1. **Dataset Acquisition**: UCI/Kaggle Heart Disease dataset.
2. **Preprocessing**: Handling missing values, scaling numeric features, encoding categorical variables.
3. **Model**: ANN with two hidden layers (32 and 16 neurons, ReLU activation), dropout for regularization, and sigmoid output.
4. **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC. Visualizations include loss/accuracy curves, confusion matrix, ROC, PR curve.
5. **Deployment**: Model and preprocessor serialized. Streamlit app provides input form and prediction results with a gauge chart.

## Results
- Training Accuracy: ~85–90%
- Test Accuracy: ~82–87%
- ROC-AUC: ~0.90

## Limitations
- Dataset size is limited (~300 samples), may not generalize to all populations.
- Model interpretability is limited compared to decision trees or logistic regression.

## Future Work
- Integrate SHAP for explainability.
- Deploy on Streamlit Cloud or Flask API for broader access.
