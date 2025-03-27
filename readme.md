# PAD Experiment Series

This repository contains a series of Jupyter Notebooks designed to explore various aspects of predictive analytics and decision-making. The experiments cover hypothesis testing, model selection, performance evaluation, and a wide array of techniques for regression, classification, clustering, and anomaly detection. In addition, several notebooks focus on pattern and anomaly detection methods using both classical and modern machine learning techniques.

## Overview

The notebooks in this repository serve as a comprehensive guide to:
- Validating assumptions and statistical hypotheses
- Selecting and evaluating predictive models
- Visualizing and interpreting classification performance through confusion matrices
- Applying advanced methods for regression, classification, clustering, and outlier detection
- Detecting patterns and anomalies in data, including intrusion detection using logs

Each notebook includes:
- Theoretical background and practical examples
- Step-by-step code implementations using Python and popular data science libraries
- Discussions on performance metrics and visualizations to provide clear insights

## Notebooks Overview

1. **PAD_Exp1_HypothesisTesting.ipynb**  
   Explores hypothesis testing techniques for validating assumptions in data analysis. Topics include:
   - Null and alternative hypotheses
   - Statistical significance testing
   - Implementation of t-tests, chi-square tests, and other key tests
   - Practical examples using synthetic or real-world datasets

2. **PAD_Exp2_ModelSelection.ipynb**  
   Focuses on methodologies for selecting the best-performing predictive models. It covers:
   - Comparison of multiple machine learning algorithms
   - Cross-validation techniques
   - Hyperparameter tuning (grid search, random search)
   - Performance evaluation metrics such as accuracy, precision, and recall

3. **PAD_Exp3_ConfusionMatrixAnalysis.ipynb**  
   Provides a detailed analysis of confusion matrices to assess classification models. Key topics include:
   - Construction and interpretation of confusion matrices
   - Calculation of performance metrics (F1 score, specificity, sensitivity)
   - Visualization techniques for better model insights

4. **PAD_Exp4_PolynomialCurveFitting.ipynb**  
   Demonstrates polynomial curve fitting to model non-linear relationships in regression tasks. This notebook explains:
   - Polynomial feature transformation
   - Fitting and evaluating polynomial regression models
   - Visual comparison of linear vs. polynomial fits

5. **PAD_Exp5_LinearRegression.ipynb**  
   Focuses on linear regression techniques for predictive modeling. Topics include:
   - Model formulation and estimation
   - Evaluation metrics like RMSE and R²
   - Case studies using real-world datasets

6. **PAD_Exp6_LinearClassification.ipynb**  
   Explores linear classification methods and decision boundaries. This notebook covers:
   - Logistic regression and linear discriminant analysis
   - Implementation details and performance evaluation
   - Visualization of classification boundaries

7. **PAD_Exp7_NeuralNetworks.ipynb**  
   Introduces neural networks as a powerful tool for capturing complex data patterns. Key points include:
   - Network architectures and activation functions
   - Backpropagation and training techniques
   - Applications in classification and regression problems

8. **PAD_Exp8_GaussianMixtureModel.ipynb**  
   Applies Gaussian Mixture Models (GMM) for clustering and density estimation. Topics include:
   - Model formulation and parameter estimation using the Expectation-Maximization algorithm
   - Comparison with other clustering methods
   - Visualization of cluster assignments

9. **PAD_Exp9_ZScoreOutlierDetection.ipynb**  
   Demonstrates how to detect outliers using Z-score analysis. This notebook explains:
   - Calculation and interpretation of Z-scores
   - Setting thresholds for outlier detection
   - Application examples on different datasets

10. **PAD_Exp10_DBScan.ipynb**  
    Focuses on the DBScan clustering algorithm for detecting clusters and anomalies. It covers:
    - Density-based clustering fundamentals
    - Parameter selection and evaluation
    - Case studies on real-world datasets

11. **PAD_Exp11_KLEOutlier.ipynb**  
    Explores the KLEOutlier method for detecting anomalous data points. Topics include:
    - Overview of the KLEOutlier algorithm
    - Implementation details and performance metrics
    - Comparative analysis with other outlier detection techniques

12. **PAD_Exp12_IsolationForest.ipynb**  
    Applies the Isolation Forest algorithm for anomaly detection. This notebook provides:
    - An introduction to tree-based anomaly detection
    - Model training and evaluation
    - Visualizations to identify anomalies in high-dimensional data

13. **PAD_Exp13_IntrusionDetectionUsingLogs.ipynb**  
    Focuses on using log data for intrusion detection by identifying unusual patterns and anomalies. Key aspects include:
    - Preprocessing and feature extraction from logs
    - Pattern recognition techniques
    - Implementation of various anomaly detection methods to flag potential intrusions

## Requirements

To run the notebooks, ensure you have the following installed:
- **Python 3.8+**
- **Jupyter Notebook** or **JupyterLab**

### Required Python Libraries

- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**
- **scikit-learn**
- *(Additional libraries may be required for neural networks, e.g., TensorFlow or PyTorch)*

You can install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
