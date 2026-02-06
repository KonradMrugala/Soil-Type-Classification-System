# Soil Type Classification System

A Machine Learning application that classifies soil types based on geographical and physical features using the Random Forest algorithm.

## Project Overview

This Python application connects to the UCI Machine Learning Repository to fetch the **Covertype** dataset and performs the following operations:
- **Data Preprocessing**: Converts raw cartographic data and transforms target variables (from One-Hot Encoding to Label Encoding).
- **Model Training**: Trains a **Random Forest Classifier** to predict one of 40 distinct soil types based on 54 input features.
- **Evaluation**: Assesses model performance using accuracy metrics and classification reports.

## Technologies Used

- **Python 3.x**
- **pandas** (Data Manipulation)
- **scikit-learn** (Machine Learning: Random Forest, Metrics)
- **UCI Machine Learning Repository** (Data Source)

## Features

- **Data Acquisition**: Automated fetching of the dataset from a remote URL.
- **Feature Engineering**: Dynamic column naming and transformation of 40 binary soil columns into a single target variable.
- **Classification**: Uses an ensemble learning method (100 Decision Trees) for high accuracy.
- **Evaluation**: Outputs global accuracy and a detailed classification report (Precision, Recall, F1-Score).
- **Practical Verification**: Sample prediction for a single test instance.

## Sample Output

The application runs as a console script and outputs the training progress and evaluation metrics.

**Console Execution:**

```bash
python soil_classification.py
```

**Sample Response:**

```text
Downloading data from UCI Repository...
Loaded 581012 rows and 55 columns.

Processing soil types...

Splitting data into training and test sets...
Training set size: 464809 samples
Test set size: 116203 samples

Training Random Forest Classifier...

Predicting on test set...
Model accuracy: 89.55%

Making a sample prediction...
Predicted Soil Type: 38, True Soil Type: 38
```

## Model Performance

- **Accuracy**: ~90%
- **Algorithm**: Random Forest Classifier
- **Parameters**: `n_estimators=100`, `n_jobs=-1` (Parallel processing)
- **Dataset**: Imbalanced (some soil types are rare), which is handled by the robustness of the Random Forest algorithm.

## How to Run

1. **Prerequisites:** Make sure **Python 3.x** is installed.

2. **Clone the repository:**
   ```bash
   git clone https://github.com/KonradMrugala/Soil-Type-Classification.git
   ```
   
3. **Navigate to the project directory**.
   ```bash
   cd Soil-Type-Classification
   ```

4. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn
   ```

5. **Run the application:**
   ```bash
   python soil_classification.py
   ```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
