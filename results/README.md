# Thyroid Disease Classification

## 1. Project Overview

This project develops a machine learning system to classify thyroid-related health conditions using clinical and laboratory attributes. It is a multiclass classification problem in which the model predicts one of several thyroid-related condition labels.

## 2. Objectives

- Analyze clinical and laboratory data related to thyroid conditions.
- Preprocess missing values and categorical features.
- Train and evaluate machine learning classification models.
- Compare model performance using appropriate evaluation metrics.
- Examine the challenges caused by imbalanced class distributions.

## 3. Dataset

The dataset is stored in `dataset/thyroid.csv`.

- Total records used in the evaluation: 22,629
- Number of classes evaluated: 13
- Features include age, sex, thyroid laboratory measurements, clinical indicators, and referral information.
- The target variable is `class`.

The dataset contains missing values and a highly imbalanced distribution of target classes. Classes with fewer than 10 records were excluded from the final benchmark.

## 4. Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## 5. Methodology

1. Load and inspect the dataset.
2. Remove identifier columns that are not useful for prediction.
3. Separate the input features from the target class.
4. Handle missing numerical and categorical values.
5. Encode categorical features and scale numerical features.
6. Split the data into training, validation, and test sets.
7. Train Logistic Regression models with different class-weighting strengths.
8. Select the model using validation Macro F1.
9. Evaluate the selected model on the held-out test set.
10. Save the model, evaluation metrics, and confusion matrix.

## 6. Final Model and Results

The selected model is Logistic Regression with a class-weighting parameter of alpha = 0.75.

| Metric | Result |
|---|---:|
| Test accuracy | 73.24% |
| Balanced accuracy | 54.34% |
| Macro F1-score | 17.76% |

The results show that performance varies considerably across classes. The dataset is highly imbalanced, so accuracy alone is not sufficient to assess model quality. The Macro F1-score highlights the difficulty of achieving good performance across both common and rare classes.

## 7. Project Structure

```text
Thyroid_Disease_Classification/
├── dataset/
│   └── thyroid.csv
├── results/
│   ├── confusion_matrix_final.png
│   ├── final_model_metrics.csv
│   ├── validation_weight_results.csv
│   └── thyroid_best_model.joblib
├── experiments/
├── thyroid_classification.py
├── thyroid_final_evaluation.py
├── requirements.txt
└── README.md
```

Additional result files may also be present in the `results` folder.

## 8. How to Run the Project

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

Run the final evaluation script from the project root directory:

```bash
python thyroid_final_evaluation.py
```

The script evaluates the model and saves the final metrics, trained model, and confusion matrix in the `results` directory.

## 9. Limitations

- The dataset has a highly imbalanced class distribution.
- Some conditions have very few examples, making reliable evaluation difficult.
- The final benchmark includes 13 classes and excludes two classes with fewer than 10 records.
- Performance on the test set is lower than performance on the validation set.
- The model has not been clinically validated and must not be used as a substitute for professional medical diagnosis.

## 10. Conclusion

This project demonstrates the use of machine learning for multiclass thyroid disease classification using clinical and laboratory attributes. The evaluation highlights the importance of class imbalance, suitable evaluation metrics, and careful model selection. Further work is needed to improve performance across minority classes and assess generalization on independent datasets.
