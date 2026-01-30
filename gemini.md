# Sales Prediction Project Summary

This document summarizes the sales prediction project completed by the Gemini agent.

## Project Objective
The main objective of this project was to build a machine learning model to predict sales based on advertising expenditure across different channels (TV, Radio, Newspaper) using the provided `advertising.csv` dataset.

## Project Steps and Outcomes

1.  **Dataset Loading and Initial Exploration**:
    *   The `advertising.csv` dataset was successfully loaded into a pandas DataFrame.
    *   Initial exploration confirmed the dataset contains 200 entries and 4 columns (`TV`, `Radio`, `Newspaper`, `Sales`), all of float64 type, with no missing values.
    *   Descriptive statistics were generated to understand data distribution.

2.  **Data Preparation**:
    *   Features (X) were identified as `TV`, `Radio`, and `Newspaper`.
    *   The target variable (y) was identified as `Sales`.
    *   The data was successfully split into training (80%) and testing (20%) sets for model development and evaluation, ensuring reproducibility with `random_state=42`.

3.  **Model Training**:
    *   A Linear Regression model from `scikit-learn` was chosen and trained on the training data.
    *   The trained model's coefficients and intercept were obtained:
        *   Coefficients: `[0.05450927 (TV), 0.10094536 (Radio), 0.00433665 (Newspaper)]`
        *   Intercept: `4.714126402214127`

4.  **Model Evaluation**:
    *   The trained model made predictions on the test set.
    *   Performance metrics were calculated:
        *   **Mean Squared Error (MSE)**: `2.91`
        *   **R-squared (R2)**: `0.91`
    *   The high R-squared value indicates that approximately 91% of the variance in sales can be explained by the advertising expenditures.

5.  **Visualization**:
    *   A scatter plot comparing actual sales (`y_test`) against predicted sales (`y_pred`) was generated.
    *   A red dashed line representing perfect predictions was added for reference.
    *   The visualization was saved as `actual_vs_predicted_sales.png`, demonstrating a strong correlation between actual and predicted values.

## Conclusion
The project successfully developed a Linear Regression model capable of predicting sales based on advertising spend with a high degree of accuracy. The model provides insights into the impact of different advertising channels on sales.
