# Credit-Card-Fraud-detection
This project uses Apache Spark and machine learning to detect fraudulent credit card transactions in imbalanced datasets. It handles class imbalance via oversampling and undersampling, trains models like Logistic Regression, Random Forest, and Gradient Boosting, and evaluates them on metrics like accuracy, precision, recall, and F1 score.

Project Overview
The script performs the following tasks:

Data Loading: Reads the credit card transaction data from a CSV file.
Data Inspection: Analyzes the schema and handles missing values.
Class Distribution Analysis: Examines the distribution of fraudulent vs. non-fraudulent transactions.
Class Imbalance Handling:
Oversampling: Increases the number of fraudulent transactions to balance the dataset.
Undersampling: Reduces the number of non-fraudulent transactions to balance the dataset.
Feature Engineering: Transforms features into a suitable format for machine learning models.
Model Training and Evaluation:
Trains and evaluates models using Logistic Regression, Random Forest, Naive Bayes, and Gradient Boosting.
Assesses model performance based on accuracy, precision, recall, and F1 score.
Results Visualization: Presents model performance metrics in tables and plots.
Getting Started
Prerequisites
Apache Spark (with PySpark)
Python 3.x
Required Python libraries: pyspark, matplotlib, tabulate

Installation
Clone the repository:
bash
Copy code
'''bash
git clone https://github.com/your-username/credit-card-fraud-detection.git

Navigate to the project directory:
bash
Copy code
cd credit-card-fraud-detection
Install the required Python packages:
bash
Copy code
pip install pyspark matplotlib tabulate
Running the Script
Ensure your data file (creditcard.csv) is located at C:/Users/hingn/Videos/CFD/creditcard.csv.
Run the script:
bash
Copy code
spark-submit credit_card_fraud_detection.py
Results
The script will output the performance metrics for each model to the console.
Visualizations of model performance will be generated and saved in the project directory.
File Structure
credit_card_fraud_detection.py: The main script for credit card fraud detection.
README.md: This README file.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Apache Spark
PySpark Documentation
Matplotlib Documentation
