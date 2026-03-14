# Airbnb Price Prediction ML

Machine learning project for predicting Airbnb listing prices from listing features such as location, room type, amenities, host information, and property characteristics.

## Project Overview

This project explores an Airbnb dataset and builds several machine learning models to predict **log-transformed listing prices**.

The notebook includes:
- data exploration
- missing value treatment
- feature engineering
- categorical encoding
- amenities extraction
- model training
- model comparison
- prediction error analysis

## Objectives

The main goal of this project is to understand which listing features influence Airbnb prices and to build a regression model capable of predicting prices accurately.

## Dataset

The project uses a training dataset named:

- `airbnb_train.csv`

Target variable:
- `log_price`

## Workflow

### 1. Data Exploration
The notebook starts with exploratory data analysis to better understand:
- dataset shape
- feature types
- missing values
- summary statistics
- price distribution
- correlations between numerical variables
- geographical distribution of listings

### 2. Data Cleaning
Several preprocessing steps are applied:
- median imputation for numerical variables such as `bathrooms`, `beds`, and `bedrooms`
- cleaning and conversion of `host_response_rate`
- handling missing review dates
- datetime conversion for date columns
- creation of `host_age_in_days`
- treatment of missing boolean values

### 3. Feature Engineering
The project includes:
- boolean feature normalization
- one-hot encoding for categorical variables such as:
  - `property_type`
  - `city`
  - `room_type`
  - `cancellation_policy`
  - `bed_type`
- target encoding for `neighbourhood`
- amenities parsing and extraction
- creation of binary columns for the top 50 most frequent amenities

### 4. Model Training
Several regression models are trained and compared:
- Linear Regression
- Gradient Boosting Regressor
- XGBoost Regressor
- Linear SVR
- Random Forest Regressor

Grid search is used for hyperparameter tuning on some models.

### 5. Evaluation
The models are evaluated using:
- **R²**
- **MAE**
- **RMSE**

According to the notebook, the best results were obtained with ensemble models such as **XGBoost** and **Gradient Boosting**, with performance around **R² = 0.72** in early experiments.

## Tech Stack

- Python
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- xgboost


