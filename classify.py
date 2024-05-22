import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

class KNNClassifier:
    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train

    def get_majority_vote(self, neighbors):
        # Determines the most common label among the nearest neighbors
        return np.bincount(neighbors).argmax()

    def get_Euclidean_dist(self, a, b):
        # Computes the Euclidean distance between two vectors a and b
        return np.linalg.norm(a - b)

    def predict(self, x_query):
        # Predicts labels for query samples by finding k-nearest neighbors
        predictions = []

        for query in x_query:
            # Calculate distances from query sample to all training samples
            distances = [self.get_Euclidean_dist(query, curr_x_train) for curr_x_train in self.X_train]
            
            # Get indices of k nearest neighbors
            nearest_idx = np.argsort(distances)[:self.k]
            # Retrieve labels of these neighbors
            nearest_labels = self.y_train[nearest_idx]

            # Determine the majority label and store it
            predictions.append(self.get_majority_vote(nearest_labels))

        return np.array(predictions)
    
def load_data(filename):
    """
    Loads the data from a tab-separated file and converts it into training data.
    
    :param filename: Path to the file.
    :return: X_data, y_data - 2D numpy array of features and 1D numpy array of labels.
    """
    data = pd.read_csv(filename, sep="\t", index_col=0)

    # Convert all gene expression values to floats
    data = data.applymap(lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)

    # Extract class labels (last row)
    y_data = data.iloc[-1, :].apply(lambda x: 1 if x == "CurrentSmoker" else 0).values

    # Extract features (all rows except the last)
    X_data = data.iloc[:-1, :].T.values

    return X_data, y_data

def accuracy(y_true, y_pred):
    # Computes the fraction of correctly predicted labels
    return np.mean(y_true == y_pred)

def cross_validation(X_train, y_train, k_vals):
    """
    Performs 5-fold cross-validation and records results.
    """
    folds = [
        (range(0, 6), list(range(6, 30))),
        (range(6, 12), list(range(0, 6)) + list(range(12, 30))),
        (range(12, 18), list(range(0, 12)) + list(range(18, 30))),
        (range(18, 24), list(range(0, 18)) + list(range(24, 30))),
        (range(24, 30), list(range(0, 24))),
    ]

    acc_by_k = {}
    y_pred_by_k = {}

    for k in k_vals:
        fold_accuracies = []

        for val_idx, train_idx in folds:
            # Create training and validation sets for this fold
            x_train_fold, y_train_fold = X_train[list(train_idx)], y_train[list(train_idx)]
            x_val_fold, y_val_fold = X_train[list(val_idx)], y_train[list(val_idx)]

            # Initialize and use KNNClassifier for this fold
            knn = KNNClassifier(k, x_train_fold, y_train_fold)
            y_pred = knn.predict(x_val_fold)

            # Record fold accuracy
            fold_accuracies.append(accuracy(y_val_fold, y_pred))

        acc_by_k[k] = np.mean(fold_accuracies) # Record average accuracy for k
        y_pred_by_k[k] = knn.predict(X_train) # Record predictions for entire training set

    return acc_by_k, y_pred_by_k

def calc_TP_TN_FP_FN(y_true, y_pred):
    # Computes counts for TP, TN, FP, and FN based on true and predicted labels
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def main():
    # Load training and test data
    X_train, y_train = load_data("GSE994-train.txt")
    X_test, _ = load_data("GSE994-test.txt")

    # Make predictions for test data with k=1 and k=3
    knn1 = KNNClassifier(1, X_train, y_train)
    y_pred1 = knn1.predict(X_test)
    
    knn3 = KNNClassifier(3, X_train, y_train)
    y_pred3 = knn3.predict(X_test)

    # Write predictions to output files
    with open("Prob5-1NNoutput.txt", "w") as f1, open("Prob5-3NNoutput.txt", "w") as f3:
        for i, (label1, label3) in enumerate(zip(y_pred1, y_pred3)):
            f1.write(f"PATIENT{i + 31}\t{'CurrentSmoker' if label1 == 1 else 'NeverSmoker'}\n")
            f3.write(f"PATIENT{i + 31}\t{'CurrentSmoker' if label3 == 1 else 'NeverSmoker'}\n")

    # Perform cross-validation for different k values
    k_vals = [1, 3, 5, 7, 11, 21, 23]
    acc_by_k, y_pred_by_k = cross_validation(X_train, y_train, k_vals)

    # Plot k vs. accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, list(acc_by_k.values()), marker="o", linestyle="-", color="b")
    plt.title("kNN Classifier Accuracy vs. k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig("knn_accuracies.png")

    # Perform clustering and evaluate its accuracy
    cluster_model = AgglomerativeClustering(n_clusters=2, linkage='average')
    cluster_model.fit(X_train)
    cluster_labels = cluster_model.labels_
    # Check for correct mapping to classes
    if accuracy(y_train, cluster_labels) < accuracy(y_train, 1 - cluster_labels):
        cluster_labels = 1 - cluster_labels

    # Calculate accuracy and other metrics for clustering
    acc_cluster = accuracy(y_train, cluster_labels)
    TP, TN, FP, FN = calc_TP_TN_FP_FN(y_train, cluster_labels)

    print(f"Accuracy (Cluster): {acc_cluster:.2f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    # Metrics for k=5 KNN classifier
    acc_classifier = acc_by_k[5]
    y_pred5 = y_pred_by_k[5]

    TP, TN, FP, FN = calc_TP_TN_FP_FN(y_train, y_pred5)

    print(f"Accuracy (k=5 Classifier): {acc_classifier:.2f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

if __name__ == "__main__":
    main()
