# Crop Recommendation System

## 1\. Project Overview

This project develops a machine learning model to recommend the most suitable crop for a given piece of land based on its soil and environmental characteristics. By analyzing a dataset of various agricultural parameters, the model learns to predict the optimal crop type, aiming to assist farmers in making informed decisions for maximizing yield and sustainability.

The primary objective is to build a robust classification model that accurately predicts one of 22 possible crop types. The project follows a comprehensive data science workflow:

  - **Data Loading and Exploration**: Understanding the dataset's structure, features, and target variable.
  - **Data Preprocessing**: Handling missing values and encoding categorical data for model training.
  - **Model Building and Evaluation**: Training and evaluating multiple classification algorithms, including Decision Tree and Random Forest.
  - **Hyperparameter Tuning**: Optimizing the chosen model using `GridSearchCV` to enhance its predictive accuracy and generalization capabilities.
  - **Final Prediction**: Using the fine-tuned model to make predictions on a new, unseen dataset.

The target variable for this classification task is `label`, which represents the recommended crop.

## 2\. Dataset Description

The project utilizes a dataset containing various soil and environmental metrics that are crucial for crop selection. The key features include:

| Column Name | Description | Data Type |
| :--- | :--- | :--- |
| `N` | The ratio of Nitrogen content in the soil. | Numerical |
| `P` | The ratio of Phosphorous content in the soil. | Numerical |
| `K` | The ratio of Potassium content in the soil. | Numerical |
| `temperature` | The temperature in degrees Celsius. | Numerical |
| `humidity` | The relative humidity in percentage. | Numerical |
| `ph` | The pH value of the soil. | Numerical |
| `rainfall` | The rainfall in mm. | Numerical |
| **`label`** | **(Target Variable)** The type of crop recommended. | Categorical |


## 3\. Project Pipeline

The project was executed through the following structured stages:

### 3.1. Data Loading and Initial Analysis

  - The training data (`farm_data.csv`) was loaded into a pandas DataFrame.
  - An initial check for missing values was performed using `dados.isnull().sum()`, which confirmed that the dataset was clean with no missing entries.
  - The target variable `label` was identified, and its categorical nature was confirmed.

### 3.2. Data Preprocessing

  - **Feature Separation**: The dataset was split into features (X) and the target variable (y).
  - **Categorical Encoding**: The `label` column, being categorical, was encoded into numerical format using `LabelEncoder` from Scikit-learn. This step is essential as machine learning models require numerical input.
  - **Data Splitting**: The dataset was divided into training (80%) and testing (20%) sets using `train_test_split` to ensure that the model could be evaluated on unseen data.

### 3.3. Model Building and Evaluation

#### a) Decision Tree Classifier (Base Model)

A `DecisionTreeClassifier` was trained as a baseline model to quickly assess the problem's feasibility.

  - **Result**: The model achieved an impressive F1-score of **0.985** on the test set, indicating that the features are highly predictive of the crop type.

#### b) Random Forest Classifier

To improve upon the baseline, a `RandomForestClassifier`, an ensemble of Decision Trees, was implemented.

  - **Result**: The Random Forest model showed even better performance, achieving an F1-score of **0.995** on the test set. This improvement is expected as ensemble methods tend to be more robust and accurate.

### 3.4. Hyperparameter Tuning with GridSearchCV

Although the Random Forest model performed exceptionally well, `GridSearchCV` was employed to ensure the model was optimized and to demonstrate a best-practice approach.

  - **Hyperparameter Grid**: A parameter grid was defined to search for the best combination of:
      - `n_estimators`: The number of trees in the forest.
      - `max_depth`: The maximum depth of each tree.
      - `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
      - `min_samples_split`: The minimum number of samples required to split an internal node.
  - **Training and Selection**: `GridSearchCV` was trained on the data using 5-fold cross-validation. The best parameters were identified based on the F1-score metric.
  - **Result**: The optimized Random Forest classifier was selected as the final model for prediction.

### 3.5. Final Prediction

  - The best performing model—the optimized `RandomForestClassifier`—was used to make predictions on the new, unseen test dataset (`farm_test.csv`).
  - The numerical predictions were transformed back to their original crop labels using `inverse_transform`.
  - The predictions were appended as a new `Predicted_Crop` column to the test DataFrame.
  - The final result, including the predictions, was saved to a new CSV file named `Prediction_Farm.csv`.


## 4\. Technologies and Libraries Used

  - **Programming Language**: Python 3
  - **Core Libraries**:
      - **Pandas**: For data manipulation, loading CSV files, and creating DataFrames.
      - **NumPy**: For numerical operations.
      - **Scikit-learn**: The primary library for machine learning, utilized for:
          - `train_test_split` for splitting data into training and testing sets.
          - `LabelEncoder` for encoding the categorical target variable.
          - `DecisionTreeClassifier` and `RandomForestClassifier` as the classification algorithms.
          - `GridSearchCV` for hyperparameter tuning.
          - `f1_score` and `accuracy_score` for model evaluation.
  - **Tools**:
      - **Jupyter Notebook**: As the interactive development environment for coding, analysis, and documentation.


## 5\. Model Performance Summary

The performance of the models was evaluated on the test set, with the Random Forest Classifier showing near-perfect results.

| Model | Accuracy (Test) | F1-Score (Test) |
| :--- | :--- | :--- |
| Decision Tree Classifier | 0.986 | 0.985 |
| **Random Forest Classifier** | **0.995** | **0.995** |

The optimized Random Forest was chosen as the final model due to its superior performance and robustness.


## 6\. How to Run the Project

1.  **Clone the repository:**

    ```sh
    git clone [URL_OF_YOUR_REPOSITORY]
    ```

2.  **Install the necessary dependencies.** It is recommended to use a virtual environment.

    ```sh
    pip install pandas numpy scikit-learn
    ```

3.  **Place the data files** (`farm_data.csv` and `farm_test.csv`) in the root directory of the project.

4.  **Run the Jupyter Notebook:**
    Open and execute the `Farm_Project.ipynb` file to run the entire workflow, from data preprocessing to model training and final prediction.

## 7\. Final Output

The project generates a file named **`Prediction_Farm.csv`**, which contains the original features from the test dataset along with a new `Predicted_Crop` column that holds the model's recommendations.
