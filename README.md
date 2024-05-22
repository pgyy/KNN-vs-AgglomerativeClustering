# Gene Expression Classification with KNN and Agglomerative Clustering

This repository contains a Python program that classifies gene expression data derived from RNA samples. Two machine learning approaches are used for classification: a manually implemented k-nearest neighbors (KNN) classifier and a clustering-based approach using scikit-learn's AgglomerativeClustering.

## Data

The dataset used in this program is derived from the NCBI GEO repository (accession number GSE994). It consists of gene expression data measured using Affymetrix microarrays from airway samples of subjects who either smoke or have never smoked. The expression values are log-transformed and stored in two files:

- `GSE994-train.txt`: Training set with 15 current smokers and 15 people who have never smoked.
- `GSE994-test.txt`: Testing set with 25 patient samples where the class labels are unknown.

Both files are tab-delimited with rows labeled by gene name and columns labeled by patient sample name. The last row of each file contains the class labels indicating whether the patient is a current smoker or a "never smoker."

## KNN Classifier

### Implementation

The KNN classifier is implemented from scratch. The main components include:
- **`KNNClassifier` class**: Handles the training and prediction processes.
- **`get_majority_vote` method**: Determines the most common label among the nearest neighbors.
- **`get_Euclidean_dist` method**: Computes the Euclidean distance between two vectors.
- **`predict` method**: Predicts labels for query samples by finding k-nearest neighbors.

### Cross-Validation

A 5-fold cross-validation is performed to evaluate the KNN classifier with various values of k. The program records and plots the accuracy for each k value.

### Usage

To use the KNN classifier:
1. Load the training data using `load_data("GSE994-train.txt")`.
2. Initialize the classifier with desired k value: `knn = KNNClassifier(k, X_train, y_train)`.
3. Make predictions for test data: `y_pred = knn.predict(X_test)`.

## Agglomerative Clustering

The program uses scikit-learn's `AgglomerativeClustering` to perform hierarchical clustering on the training data. The clustering model is evaluated by comparing the predicted cluster labels with the actual class labels.

### Usage

To use the clustering-based approach:
1. Initialize and fit the model: `cluster_model = AgglomerativeClustering(n_clusters=2, linkage='average').fit(X_train)`.
2. Evaluate the clustering results by checking the accuracy and calculating TP, TN, FP, and FN metrics.

## Evaluation Metrics

The program computes the following evaluation metrics for both classifiers:
- Accuracy
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

## Output

The program generates the following output files:
- `Prob5-1NNoutput.txt`: Predictions for test data using k=1 KNN classifier.
- `Prob5-3NNoutput.txt`: Predictions for test data using k=3 KNN classifier.
- `knn_accuracies.png`: Plot of k values versus accuracy for the KNN classifier.

## How to Run

1. Ensure you have the required libraries installed:
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```
2. Place the data files (`GSE994-train.txt` and `GSE994-test.txt`) in the same directory as the script.
3. Run the script:
    ```bash
    python classify.py
    ```

## Conclusion

This program demonstrates the application of two different machine learning approaches for classifying gene expression data. The KNN classifier, with its simplicity and effectiveness, and the clustering-based approach, offering an alternative perspective, both provide valuable insights into the data. 

For further improvements, consider exploring other machine learning algorithms and feature selection techniques to enhance classification performance.
