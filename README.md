# Breast-Cancer-Detection

#### Context and Goal

Breast Cancer prediction has been a widely studied problem in the medical community. We want to predict breast cancer in patients using a small data set of ~600 samples. The dataset has been downloaded from this source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html.

In this project, we will build a Support Vector Machine (SVM) Classification model from scratch to reliably classify data samples corresponding to Malignant and Benign tumor. 

#### Training

There are two training algorithms we will use:

- **Stochastic Gradient Descent**: Uses a single training sample for each iteration
- **Mini-Batch Atochastic Gradient Descent**: Uses a batch of training samples for each iteration

  
#### Early Stopping

Develop an early stopping algorithm to prevent overfitting using L2 regularization. 

#### Sampling Strategy

We will employ an active sampling strategy to minimize the cost of training. This technique used was first proposed in 2000 at ICML, a premier venue for state-of-the-art ML research.

Thus, the main steps of the algorithm are as follows:
1. Train an initial classifier with random samples.
2. Perform prediction on all remaining samples.
3. Select the next probable sample, using the above sampling strategy.
4. Train the classifier further.
5. Repeat steps 2 – 4, till you get a satisfactory performance.
