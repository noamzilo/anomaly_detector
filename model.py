import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pyod.models.knn import KNN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.
from pyod.utils.data import generate_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from Utils.warning_utils.warning_utils import silence_warnings
from image_paths import create_links


def main():
    plt.close('all')
    matplotlib.use('Qt5Agg')  # override PyCharm pro's scientific view

    create_links()

    warnings.showwarning = silence_warnings
    contamination = 0.1  # percentage of outliers
    n_train = 500  # number of training points
    n_test = 500  # number of testing points
    n_features = 25  # Number of features

    X_test, y_test, X_train, y_train = _generate_random_data(contamination, n_features, n_test, n_train)
    X_test, y_test, X_train, y_train = ?

    _plot_using_pca(X_train, y_train)

    hidden_neurons = [25, 2, 2, 25]
    clf1 = AutoEncoder(hidden_neurons=hidden_neurons)
    clf1.fit(X_train)
    y_train_scores = clf1.decision_scores_

    # Predict the anomaly scores
    y_test_scores = clf1.decision_function(X_test)  # outlier scores
    y_test_scores = pd.Series(y_test_scores)

    # Plot anomaly scores
    plt.hist(y_test_scores, bins='auto')
    plt.title("Histogram for Model Clf1 Anomaly Scores")
    plt.show()

    manual_score_thres = 4
    df_test = X_test.copy()
    df_test['score'] = y_test_scores
    # assign cluster=0 to samples with low anomaly score, and cluster=1 to samples with high anomaly score.
    df_test['cluster'] = np.where(df_test['score'] < manual_score_thres, 0, 1)
    df_test['cluster'].value_counts()

    df_test.groupby('cluster').mean()
    print(df_test)


def _generate_random_data(contamination, n_features, n_test, n_train):
    X_train, y_train, X_test, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        contamination=contamination,
        random_state=1234,
        behaviour="old")
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_train = StandardScaler().fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    X_test = pd.DataFrame(X_test)
    return X_test, y_test, X_train, y_train


def _create_data(contamination, n_features, n_test, n_train):
    X_train, y_train, X_test, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        contamination=contamination,
        random_state=1234,
        behaviour="old")
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_train = StandardScaler().fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    X_test = pd.DataFrame(X_test)
    return X_train, y_train, X_test, y_test


def _plot_using_pca(X_train, y_train):
    # see the data distribution using PCA
    pca = PCA(2)
    x_pca = pca.fit_transform(X_train)
    x_pca = pd.DataFrame(x_pca)
    x_pca.columns = ['PC1', 'PC2']
    plt.scatter(X_train[0], X_train[1], c=y_train, alpha=0.8)
    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    main()