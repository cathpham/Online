{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "d0d80412",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: xgboost in /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages (3.0.0)\n",
      "Requirement already satisfied: numpy in /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages (from xgboost) (1.25.0)\n",
      "Requirement already satisfied: scipy in /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages (from xgboost) (1.10.1)\n",
      "\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m23.1.2\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m25.1.1\u001b[0m\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip3.11 install --upgrade pip\u001b[0m\n",
      "Requirement already satisfied: lightgbm in /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages (4.6.0)\n",
      "Requirement already satisfied: numpy>=1.17.0 in /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages (from lightgbm) (1.25.0)\n",
      "Requirement already satisfied: scipy in /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages (from lightgbm) (1.10.1)\n",
      "\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m23.1.2\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m25.1.1\u001b[0m\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip3.11 install --upgrade pip\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "!{sys.executable} -m pip install xgboost\n",
    "\n",
    "!{sys.executable} -m pip install lightgbm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "4a769bd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "# Convert data to machine learning\n",
    "from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder #normalize numeric & convert categorical -> numerical\n",
    "from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split # split data into training (80%) and testing (%20)\n",
    "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
    "from sklearn.linear_model import LinearRegression, Ridge\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
    "import xgboost as xgb\n",
    "import lightgbm as lgb\n",
    "\n",
    "# Load data (adjust path as needed)\n",
    "df = pd.read_csv('insurance.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "16068c29",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert categorical variables into numerical format\n",
    "df['sex'] = df['sex'].map({'male': 1, 'female': 0})\n",
    "df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})\n",
    "\n",
    "# One-hot encode 'region' -> category to number\n",
    "df = pd.get_dummies(df, columns=['region'], drop_first=True) #convert region into 3 new categories (the other left can be 4)\n",
    "\n",
    "# Define features (X) and target variable (y)\n",
    "X = df.drop(columns=['charges'])  # Independent variables\n",
    "y = df['charges']  # Target variable\n",
    "\n",
    "# Scale numerical features\n",
    "scaler = StandardScaler() #compute means + stdev\n",
    "X_scaled = scaler.fit_transform(X) #scaling parameters and applying transform \n",
    "\n",
    "# Split dataset into training (80%) and testing (20%)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Log-transform the target\n",
    "y_log = np.log(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "9c9c911c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to calculate metrics\n",
    "def calculate_metrics(y_true, y_pred):\n",
    "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
    "    mae = mean_absolute_error(y_true, y_pred)\n",
    "    r2 = r2_score(y_true, y_pred)\n",
    "    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100\n",
    "    return rmse, mae, r2, mape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "0a126dbb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results for Gradient Boosting:\n",
      "    RMSE=4690.20, MAE=2173.81, R²=0.8499, MAPE=17.05%\n"
     ]
    }
   ],
   "source": [
    "# Gradient Boosting\n",
    "# Models to evaluate\n",
    "gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=55)\n",
    "\n",
    "# 5-Fold CV\n",
    "y_true_gb = y  # Actual charges in original scale\n",
    "log_pred_gb = cross_val_predict(gb, X, y_log, cv=5)\n",
    "    \n",
    "y_pred_gb = np.exp(log_pred_gb)\n",
    "rmse, mae, r2, mape = calculate_metrics(y_true_gb, y_pred_gb)\n",
    "\n",
    "r2_gb = r2\n",
    "\n",
    "# Print results\n",
    "print(\"Results for Gradient Boosting:\")\n",
    "print(f\"    RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "539b1bbe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results for XGBoost:\n",
      "    RMSE=4596.56, MAE=2138.65, R²=0.8558, MAPE=16.85%\n"
     ]
    }
   ],
   "source": [
    "# XGBoost\n",
    "# Models to evaluate\n",
    "xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=55)\n",
    "\n",
    "# 5-Fold CV\n",
    "y_true_xgb = y  # Actual charges in original scale\n",
    "log_pred_xgb = cross_val_predict(xgb, X, y_log, cv=5)\n",
    "    \n",
    "y_pred_xgb = np.exp(log_pred_xgb)\n",
    "rmse, mae, r2, mape = calculate_metrics(y_true_xgb, y_pred_xgb)\n",
    "\n",
    "r2_xgb = r2\n",
    "\n",
    "# Print results\n",
    "print(\"Results for XGBoost:\")\n",
    "print(f\"    RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "97dbee7f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results for LightGBM:\n",
      "    RMSE=4631.93, MAE=2165.67, R²=0.8536, MAPE=16.68%\n"
     ]
    }
   ],
   "source": [
    "# LightGBM\n",
    "# Models to evaluate\n",
    "lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=55)\n",
    "\n",
    "# 5-Fold CV\n",
    "y_true_lgb = y  # Actual charges in original scale\n",
    "log_pred_lgb = cross_val_predict(lgb, X, y_log, cv=5)\n",
    "    \n",
    "y_pred_lgb = np.exp(log_pred_lgb)\n",
    "rmse, mae, r2, mape = calculate_metrics(y_true_lgb, y_pred_lgb)\n",
    "\n",
    "r2_lgb = r2\n",
    "\n",
    "# Print results\n",
    "print(\"Results for LightGBM:\")\n",
    "print(f\"    RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "52912dc7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results for Ensemble:\n",
      "    RMSE=4604.84, MAE=2145.24, R²=0.8553, MAPE=16.77%\n"
     ]
    }
   ],
   "source": [
    "# Ensemble (Weighting average)\n",
    "# Calculate weights\n",
    "w_gb = r2_gb/(r2_gb + r2_xgb + r2_lgb)\n",
    "w_xgb = r2_xgb/(r2_gb + r2_xgb + r2_lgb)\n",
    "w_lgb = r2_lgb/(r2_gb + r2_xgb + r2_lgb)\n",
    "\n",
    "# Ensemble\n",
    "y_pred_ensemble = w_gb*y_pred_gb + w_xgb*y_pred_xgb + w_lgb*y_pred_lgb\n",
    "rmse, mae, r2, mape = calculate_metrics(y, y_pred_ensemble)\n",
    "# Print results\n",
    "print(\"Results for Ensemble:\")\n",
    "print(f\"    RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "ea404821",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'age': 45, 'bmi': 30.36, 'children': 0, 'smoker': 1, 'sex': 1, 'region_northwest': 0, 'region_southeast': 1, 'region_southwest': 0}\n",
      "Estimated value (Ensemble): $39416.33\n"
     ]
    }
   ],
   "source": [
    "# Example: new person data as dictionary\n",
    "new_person = {\n",
    "    'age': 45,\n",
    "    'bmi': 30.36,\n",
    "    'children': 0,\n",
    "    'smoker': 1,\n",
    "    'sex': 1,\n",
    "    'region_northwest': 0,\n",
    "    'region_southeast': 1,\n",
    "    'region_southwest': 0\n",
    "    # Add all necessary features as in your X\n",
    "}\n",
    "\n",
    "# Convert to DataFrame and reorder columns to match training data\n",
    "new_X = pd.DataFrame([new_person])\n",
    "new_X = new_X[X.columns]  # Reorder columns to match X\n",
    "\n",
    "# Predict log-transformed output for each model\n",
    "log_gb = gb.predict(new_X)\n",
    "log_xgb = xgb.predict(new_X)\n",
    "log_lgb = lgb.predict(new_X)\n",
    "\n",
    "# Combine with weights (already calculated)\n",
    "log_ensemble = w_gb * log_gb + w_xgb * log_xgb + w_lgb * log_lgb\n",
    "\n",
    "# Convert back to original scale\n",
    "ensemble_prediction = np.exp(log_ensemble)\n",
    "print(new_person)\n",
    "print(f\"Estimated value (Ensemble): ${ensemble_prediction[0]:.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "72f46aa8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
