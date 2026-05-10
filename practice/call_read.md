# Call Center Sentiment Analysis & Feature Engineering Pipeline

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📋 Overview
This repository contains a comprehensive data science workflow designed to process and prepare call center performance data for predictive modeling. The notebook focuses on transforming raw, unstructured Excel data into a refined feature set, specifically targeting **Customer Sentiment Analysis**.

The pipeline handles complex data cleaning, categorical consolidation, and high-dimensional feature engineering using industry-standard Scikit-Learn preprocessing techniques.

## 🚀 Key Features
*   **Data Wrangling & Normalization**: Automated cleaning of non-standard Excel headers, handling null values, and string normalization (lowercasing, stripping whitespace, and snake_case formatting).
*   **Target Label Consolidation**: Strategy implemented to merge granular sentiment classes (e.g., merging "Very Positive" into "Positive") to improve model stability and reduce class sparsity.
*   **Advanced Preprocessing Pipeline**:
    *   **One-Hot Encoding**: Applied to categorical variables including `reason`, `channel`, `response_time`, and `call_center`.
    *   **Feature Scaling**: Standardized numerical metrics like `csat_score` and `call_duration` to ensure uniform variance.
    *   **ColumnTransformer Integration**: A modular approach to applying different transformations to different subsets of features simultaneously.

## 🛠 Technology Stack
*   **Language**: Python 3.x
*   **Data Manipulation**: `pandas`
*   **Machine Learning Preprocessing**: `scikit-learn`
    *   `LabelEncoder`
    *   `OneHotEncoder`
    *   `StandardScaler`
    *   `ColumnTransformer`

## 📊 Data Transformation Architecture
The model utilizes a multi-input pipeline to prepare the following feature matrix:

| Feature Type | Variables | Transformation |
| :--- | :--- | :--- |
| **Categorical** | Reason, Channel, Response Time, Call Center | One-Hot Encoding |
| **Numerical** | CSAT Score, Call Duration | StandardScaler |
| **Target** | Sentiment | Label Encoding |

## 🔍 Insights & Findings
*   **Class Distribution**: The initial analysis utilized `value_counts()` to identify the frequency of customer sentiments, leading to a strategic decision to consolidate sentiment labels for a more balanced target variable.
*   **Dimensionality**: Through One-Hot Encoding, the feature space is expanded to capture the nuances of different communication channels and call centers, allowing a model to learn specific patterns associated with each.
*   **Data Quality**: Extensive cleaning was required to handle header offsets and index resets inherent in the raw sample data.

## ⚙️ Usage
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/call-center-sentiment-analysis.git
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas scikit-learn openpyxl
    ```
3.  **Data Placement**:
    Ensure the `Call-Center-Sentiment-Sample-Data.xlsx` file is located in the appropriate directory or update the path in the second cell:
    ```python
    df = pd.read_excel('path/to/your/data.xlsx')
    ```
4.  **Run the Notebook**:
    Execute the cells sequentially to perform the data cleaning and generate the transformed `X_trans` and `y_trans` arrays.

---
**Author**: [Your Name/GitHub Profile]
**Project Category**: Customer Analytics / Machine Learning Preprocessing
