# rna_modeling
CMSE_381 Project

# Predicting Protein Counts from RNA Copies

## Introduction
Statistical learning algorithms play a crucial role in various applications, from image classification to driving vehicles. One notable area of development is in cell biology, where understanding cellular processes has wide-reaching implications. This project focuses on predicting protein counts based on the number of RNA copies within a cell, offering a more cost-effective and less complex alternative to direct protein measurement.

## Methodology and Results
The project begins with data exploration and manipulation of .h5ad files. The data is structured with various tables, including RNA and protein attributes. The focus is on predicting protein counts, making RNA copies a key variable. Data visualization reveals challenges in predicting protein counts solely based on RNA copies due to complex cellular processes.

### Baseline Models
- **Baseline PCA Mean:** Initial PCA analysis resulted in an RMSE of 0.58.
- **Baseline PCA Linear:** Improved upon the mean model with an RMSE of 0.3827.

### Linear Models with Regularization
- **Lasso with PCA:** Utilized Lasso regression to address overfitting, yielding an RMSE of 0.48.
- **Ridge with PCA:** Applied Ridge regression for redundancy, achieving an RMSE of 0.3815.

### Multi-Layer Perceptron (MLP)
- **Basic MLP:** Introduced a 2-layer MLP but achieved an RMSE of over 0.4.
- **MLP with Batch Normalization (BN):** Added BN to reduce testing error, yielding marginal improvement.
- **MLP with BN and PCA:** Incorporated PCA, modified hyperparameters, and activation functions, resulting in an RMSE of 0.3680.

### Magic Imputation + MLP + PCA
- **Magic + MLP + PCA:** Experimented with the magic-impute library to fill in missing RNA copy values. Despite efforts, the RMSE increased to 0.4.

## Conclusion
The project successfully surpassed baseline models, achieving an RMSE of 0.3680. However, the journey highlighted the challenges and nuances of statistical modeling, emphasizing the need for creativity, time, and a deep understanding of model behavior. Frustrations emerged in competing with basic linear models and the time-consuming nature of model evaluation. Despite the challenges, the project provided valuable insights into statistical learning and the importance of intuition in model development.
