# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using logistic regression with feature engineering and comprehensive data analysis.

## Overview

This notebook analyzes the famous Titanic dataset to build a predictive model for passenger survival. The project includes exploratory data analysis (EDA), feature engineering, data preprocessing, and model training with accuracy evaluation.

## Dataset

The dataset contains information about Titanic passengers with the following features:

- **PassengerId**: Unique identifier for each passenger
- **Survived**: Survival indicator (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

**Dataset Size**: 891 passengers

## Project Structure

### 1. Data Exploration
- Load and inspect the dataset
- Check data types and missing values
- Generate statistical summaries
- Analyze data distributions

### 2. Data Cleaning
- Handle missing values in Age, Cabin, and Embarked columns
- Remove unnecessary columns (Name, Ticket, Cabin)
- Standardize column names to lowercase

### 3. Feature Engineering
- **Age Groups**: Categorize ages into bins (Child, Teen, Young Adult, Adult, Senior)
- **Fare Groups**: Categorize fares into bins (Low, Medium, High, Very High)
- Create meaningful categorical variables for better model performance

### 4. Exploratory Data Analysis
- Correlation heatmap to identify feature relationships
- Survival rate analysis by:
  - Passenger class
  - Gender
  - Age group
  - Fare group
  - Embarkation port
  - Family size (SibSp, Parch)
- Multiple subplots showing survival rates across different features

### 5. Model Building

#### Model 1: Basic Logistic Regression
- **Features**: age, sibsp, parch, fare, pclass, sex, embarked
- **Preprocessing**: 
  - Numerical features: Median imputation + StandardScaler
  - Categorical features: Most frequent imputation + OneHotEncoder
- **Accuracy**: 77.5%

#### Model 2: Enhanced Logistic Regression (Final Model)
- **Features**: sibsp, parch, pclass, sex, embarked, age_group, fare_group
- **Preprocessing**:
  - Numerical features: Median imputation + StandardScaler
  - Categorical features: Most frequent imputation + OneHotEncoder
- **Accuracy**: 80.3%
- **Performance Metrics**:
  - Precision: 85% (non-survivors), 74% (survivors)
  - Recall: 83% (non-survivors), 77% (survivors)
  - F1-Score: 84% (non-survivors), 75% (survivors)

## Key Insights

From the exploratory data analysis:
- **Gender**: Females had significantly higher survival rates than males
- **Class**: First-class passengers had better survival rates than lower classes
- **Age**: Children had higher survival rates than adults
- **Fare**: Higher fare passengers (likely in better cabins) survived more
- **Family Size**: Passengers with moderate family size had better survival rates

## Requirements

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Ensure you have the `train.csv` file in the same directory as the notebook
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook titanic.ipynb
   ```
3. Run all cells sequentially to:
   - Load and explore the data
   - Perform feature engineering
   - Train the model
   - Evaluate performance

## Model Pipeline

The final model uses a scikit-learn Pipeline with:
1. **ColumnTransformer** for separate preprocessing of numerical and categorical features
2. **SimpleImputer** for handling missing values
3. **StandardScaler** for numerical feature scaling
4. **OneHotEncoder** for categorical feature encoding
5. **LogisticRegression** as the classification model

## Results

The enhanced model with engineered features (age_group and fare_group) achieved **80.3% accuracy** on the test set, outperforming the basic model by approximately 3%.

## Future Improvements

- Experiment with other algorithms (Random Forest, Gradient Boosting, XGBoost)
- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Create additional feature interactions
- Implement cross-validation for more robust evaluation
- Try ensemble methods combining multiple models

## Author


[Emad Anwer Naguib]
- Email: [emadanwer888@gmial.com]
Created as part of a data science learning project.

## License

This project is available for educational and research purposes.
