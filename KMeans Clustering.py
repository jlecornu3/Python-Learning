# Example of KMeans Clustering
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load Dataset
bc = load_breast_cancer()
print(bc)

X = scale(bc.data)
y = bc.target

# Split train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

# Initialise model
KM = KMeans(n_clusters=2, random_state=11)

# Fit model to X
KM.fit(X_train)

# Predictions
predictions = KM.predict(X_test)
labels = KM.labels_

# View results
print('Labels: ', labels)
print("Predictions: ", predictions)
print("Actual Values: ", y_test)
print("KM Model Accuracy: ", accuracy_score(y_test,predictions))

# Note: Some very low accuracy scores came up, this is as the clustering algorithm randomly assigned 0 label to cluster 1, 1 label to cluster 0
# Print Cross Tab test
print(pd.crosstab(y_train, labels))

# Define and Use some KMeans Benchmarks
from sklearn import metrics
def bench_k_means(kmeans, name, data, labels):
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(data, estimator[-1].labels_,
                                 metric="euclidean", sample_size=300,)
    ]

    # Show the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                        "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))

# Evaluate Model
bench_k_means(KM, '1', X, y)