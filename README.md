# **Credit Card Fraud Detection**

This repository contains a Jupyter Notebook (`CreditCardFraud.ipynb`) that implements machine learning models to detect fraudulent credit card transactions. The project addresses the critical business problem of minimizing financial losses and maintaining customer trust by identifying and flagging suspicious transactions.

## **Table of Contents**

* [Business Problem](#business-problem)

* [Solution](#solution)

* [Dataset](#dataset)

* [Methodology](#methodology)

* [How to Run](#how-to-run)

* [Key Steps in the Code](#key-steps-in-the-code)

* [Results](#results)

* [Visualizations](#visualizations)

* [Future Improvements](#future-improvements)


## **Business Problem**

Credit card fraud is a significant challenge for financial institutions and businesses, leading to substantial financial losses, damage to reputation, and erosion of customer trust. Manually reviewing every transaction for potential fraud is impractical and inefficient due to the sheer volume of transactions. There is a critical need for an automated, accurate, and efficient system to identify fraudulent transactions in real-time or near real-time. The highly imbalanced nature of transaction data (where fraudulent transactions are a tiny fraction of the total) further complicates this problem, making traditional classification models less effective.

## **Solution**

This project provides a data-driven solution to the credit card fraud detection problem. It leverages machine learning algorithms to build models capable of distinguishing between legitimate and fraudulent transactions. The approach involves:

1. **Data Preprocessing:** Handling potential issues in the dataset, including scaling features to ensure models are not biased by the magnitude of values.

2. **Addressing Data Imbalance:** Implementing techniques like Undersampling (RandomUnderSampler) and Oversampling (SMOTE) to mitigate the challenge posed by the small number of fraudulent transactions compared to legitimate ones.

3. **Model Selection and Training:** Exploring different classification models, including Isolation Forest (an anomaly detection algorithm), XGBoost, and Random Forest, which are well-suited for this type of problem.

4. **Evaluation:** Assessing the performance of the trained models using appropriate metrics like classification reports (Precision, Recall, F1-score) and ROC curves, which are crucial for evaluating performance on imbalanced datasets.

By implementing these steps, the project aims to develop a model that can effectively identify fraudulent transactions, thereby helping to reduce financial losses and improve the security of transactions.

## **Dataset**

The project uses a dataset named `creditcard.csv`. This dataset contains credit card transactions, where each row represents a transaction. The features `V1` through `V28` are the result of a PCA transformation for anonymity. The dataset also includes 'Time' (seconds elapsed between this transaction and the first transaction in the dataset) and 'Amount' (the transaction amount). The target variable is 'Class', which is `1` for fraudulent transactions and `0` otherwise.

The dataset is highly imbalanced, with a very small percentage of transactions belonging to the 'fraudulent' class (Class = 1).

## **Methodology**

The analysis follows these main steps:

1. **Load Libraries:** Import necessary Python libraries for data manipulation, visualization, preprocessing, and machine learning.

2. **Load Data:** Read the `creditcard.csv` file into a pandas DataFrame.

3. **Data Inspection:** Perform initial data inspection, including viewing the head and tail of the data, checking data types and non-null counts (`.info()`), getting descriptive statistics (`.describe().T`), and checking unique values (`.nunique()`).

4. **Data Preprocessing:**

   * Handle any potential missing values (though the notebook indicates no missing values initially, a fillna step is included).

   * Scale the features using `RobustScaler` to handle potential outliers effectively.

   * Separate the features (X) and the target variable (y).

5. **Data Splitting:** Split the data into training and testing sets (80/20 split). Stratification is used when addressing imbalance to ensure the test set has a representative proportion of fraudulent transactions.

6. **Model Training and Evaluation (Initial):**

   * Train Isolation Forest, XGBoost Classifier, and Random Forest Classifier on the initial (imbalanced) training data.

   * Evaluate the models using accuracy score and ROC curves.

7. **Addressing Data Imbalance (Undersampling):**

   * Create a pipeline using `RandomUnderSampler` and `XGBClassifier` or `RandomForestClassifier`.

   * Train the pipeline on the training data.

   * Evaluate the performance using a classification report and confusion matrix, and ROC curves.

8. **Addressing Data Imbalance (Oversampling):**

   * Apply SMOTE (`SMOTE`) to the training data to oversample the minority class.

   * Train `RandomForestClassifier` and `XGBClassifier` on the SMOTE-resampled training data.

   * Evaluate the performance using a classification report, confusion matrix, and ROC curves.
  
 ## **How to Run**

1. Download the `creditcard.csv` dataset from kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and the Jupyter Notebook (`CreditCardFraud.ipynb`).

2. Ensure both files are in the same directory, or update the file path in the notebook's data loading cell (`pd.read_csv("/content/creditcard.csv")`) to the correct location of `creditcard.csv`.

3. Open the notebook in a Jupyter environment (like Jupyter Notebook, JupyterLab, or Google Colab).

4. Run the cells sequentially. The notebook will execute the data loading, preprocessing, model training, and evaluation steps for different models and data balancing techniques.

## **Key Steps in the Code**

This section outlines the main logical steps performed in the Python script without including the code itself.

* Importing necessary libraries.

* Loading the credit card transactions dataset.

* Performing initial data exploration and inspection (info, describe, nunique).

* Handling potential missing values.

* Scaling features using `RobustScaler`.

* Separating features (X) and target variable (y).

* Splitting data into training and testing sets.

* Training and evaluating Isolation Forest, XGBoost, and Random Forest on imbalanced data.

* Implementing Undersampling using `RandomUnderSampler` with pipelines and evaluating models.

* Implementing Oversampling using SMOTE and evaluating models (Random Forest and XGBoost).

* Generating classification reports, confusion matrices, and ROC curves for model evaluation.

## **Results**

The notebook will output various results, including:

1. The head and tail of the loaded dataset.

2. Data types, non-null counts, descriptive statistics, and unique value counts.

3. A correlation heatmap visualization.

4. Accuracy scores and ROC curves for the initial models on imbalanced data.

5. Classification reports, confusion matrices, and ROC curves for models trained with Undersampling.

6. Classification reports, confusion matrices, and ROC curves for models trained with Oversampling (SMOTE).

These outputs demonstrate the performance of different models and the impact of data balancing techniques on the ability to detect fraudulent transactions. Pay close attention to metrics like Precision, Recall, and F1-score in the classification reports, especially for the minority class (fraudulent transactions), as accuracy alone can be misleading on imbalanced datasets.

## **Visualizations**

The script generates the following visualizations:

* **Correlation Heatmap:** Shows the pairwise correlation between the features.

* **ROC Curves:** Generated for each trained model to visualize the trade-off between the true positive rate and false positive rate at various threshold settings.

These visualizations aid in understanding the data and comparing the performance of the different models.

## **Future Improvements**

* Experiment with other data balancing techniques (e.g., ADASYN, NearMiss).

* Explore different classification algorithms (e.g., Support Vector Machines, Neural Networks).

* Perform hyperparameter tuning for the selected models to optimize performance.

* Implement cross-validation for more robust model evaluation.

* Investigate feature importance to understand which features are most predictive of fraud.

* Consider time-series aspects of the data for potential sequential patterns in fraud.

* Explore anomaly detection techniques further.


