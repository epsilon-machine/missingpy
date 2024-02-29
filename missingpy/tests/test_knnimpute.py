import numpy as np

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_raise_message
# from sklearn.utils._testing import assert_equal
from numpy.testing import assert_equal

from missingpy import KNNImputer
from missingpy.pairwise_external import masked_euclidean_distances
from missingpy.pairwise_external import pairwise_distances


def test_knn_imputation_shape():
    # Verify the shapes of the imputed matrix for different weights and
    # number of neighbors.
    n_rows = 10
    n_cols = 2
    X = np.random.rand(n_rows, n_cols)
    X[0, 0] = np.nan

    for weights in ['uniform', 'distance']:
        for n_neighbors in range(1, 6):
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            X_imputed = imputer.fit_transform(X)
            assert_equal(X_imputed.shape, (n_rows, n_cols))


def test_knn_imputation_zero():
    # Test imputation when missing_values == 0
    missing_values = 0
    n_neighbors = 2
    imputer = KNNImputer(missing_values=missing_values,
                         n_neighbors=n_neighbors,
                         weights="uniform")

    # Test with missing_values=0 when NaN present
    X = np.array([
        [np.nan, 0, 0, 0, 5],
        [np.nan, 1, 0, np.nan, 3],
        [np.nan, 2, 0, 0, 0],
        [np.nan, 6, 0, 5, 13],
    ])
    msg = f"Input contains NaN."
    assert_raise_message(ValueError, msg, imputer.fit, X)

    # Test with % zeros in column > col_max_missing
    X = np.array([
        [1, 0, 0, 0, 5],
        [2, 1, 0, 2, 3],
        [3, 2, 0, 0, 0],
        [4, 6, 0, 5, 13],
    ])
    msg = "Some column(s) have more than {}% missing values".format(
        imputer.col_max_missing * 100)
    assert_raise_message(ValueError, msg, imputer.fit, X)


def test_knn_imputation_zero_p2():
    # Test with an imputable matrix and also compare with missing_values="NaN"
    X_zero = np.array([
        [1, 0, 1, 1, 1.],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 0],
        [6, 6, 0, 6, 6],
    ])

    X_nan = np.array([
        [1, np.nan, 1,      1,      1.],
        [2, 2,      2,      2,      2],
        [3, 3,      3,      3,      np.nan],
        [6, 6,      np.nan, 6,      6],
    ])
    statistics_mean = np.nanmean(X_nan, axis=0)

    X_imputed = np.array([
        [1, 2.5,    1,   1, 1.],
        [2, 2,      2,   2, 2],
        [3, 3,      3,   3, 1.5],
        [6, 6,      2.5, 6, 6],
    ])

    imputer_zero = KNNImputer(missing_values=0, n_neighbors=2,
                              weights="uniform")

    imputer_nan = KNNImputer(missing_values="NaN",
                             n_neighbors=2,
                             weights="uniform")

    assert_array_equal(imputer_zero.fit_transform(X_zero), X_imputed)
    assert_array_equal(imputer_zero.statistics_, statistics_mean)
    assert_array_equal(imputer_zero.fit_transform(X_zero),
                       imputer_nan.fit_transform(X_nan))


def test_knn_imputation_default():
    # Test imputation with default parameter values

    # Test with an imputable matrix
    X = np.array([
        [1,      0,      0,      1],
        [2,      1,      2,      np.nan],
        [3,      2,      3,      np.nan],
        [np.nan, 4,      5,      5],
        [6,      np.nan, 6,      7],
        [8,      8,      8,      8],
        [16,     15,     18,    19],
    ])
    statistics_mean = np.nanmean(X, axis=0)

    X_imputed = np.array([
        [1,      0,      0,      1],
        [2,      1,      2,      8],
        [3,      2,      3,      8],
        [4,      4,      5,      5],
        [6,      3,      6,      7],
        [8,      8,      8,      8],
        [16,     15,     18,    19],
    ])

    imputer = KNNImputer()
    assert_array_equal(imputer.fit_transform(X), X_imputed)
    assert_array_equal(imputer.statistics_, statistics_mean)

    # Test with % missing in row > row_max_missing
    X = np.array([
        [1,      0,      0,      1],
        [2,      1,      2,      np.nan],
        [3,      2,      3,      np.nan],
        [np.nan, 4,      5,      5],
        [6,      np.nan, 6,      7],
        [8,      8,      8,      8],
        [19,     19,     19,     19],
        [np.nan, np.nan, np.nan, 19],
    ])
    statistics_mean = np.nanmean(X, axis=0)
    r7c0, r7c1, r7c2, _ = statistics_mean

    X_imputed = np.array([
        [1,      0,      0,      1],
        [2,      1,      2,      8],
        [3,      2,      3,      8],
        [4,      4,      5,      5],
        [6,      3,      6,      7],
        [8,      8,      8,      8],
        [19,     19,     19,     19],
        [r7c0,   r7c1,   r7c2,   19],
    ])

    imputer = KNNImputer()
    assert_array_almost_equal(imputer.fit_transform(X), X_imputed, decimal=6)
    assert_array_almost_equal(imputer.statistics_, statistics_mean, decimal=6)

    # Test with all neighboring donors also having missing feature values
    X = np.array([
        [1, 0, 0, np.nan],
        [2, 1, 2, np.nan],
        [3, 2, 3, np.nan],
        [4, 4, 5, np.nan],
        [6, 7, 6, np.nan],
        [8, 8, 8, np.nan],
        [20, 20, 20, 20],
        [22, 22, 22, 22]
    ])
    statistics_mean = np.nanmean(X, axis=0)

    X_imputed = np.array([
        [1, 0, 0, 21],
        [2, 1, 2, 21],
        [3, 2, 3, 21],
        [4, 4, 5, 21],
        [6, 7, 6, 21],
        [8, 8, 8, 21],
        [20, 20, 20, 20],
        [22, 22, 22, 22]
    ])

    imputer = KNNImputer()
    assert_array_equal(imputer.fit_transform(X), X_imputed)
    assert_array_equal(imputer.statistics_, statistics_mean)

    # Test when data in fit() and transform() are different
    X = np.array([
        [0,      0],
        [np.nan, 2],
        [4,      3],
        [5,      6],
        [7,      7],
        [9,      8],
        [11,     16]
    ])
    statistics_mean = np.nanmean(X, axis=0)

    Y = np.array([
        [1,      0],
        [3,      2],
        [4,      np.nan]
        ])

    Y_imputed = np.array([
        [1,      0],
        [3,      2],
        [4,      4.8]
        ])

    imputer = KNNImputer()
    assert_array_equal(imputer.fit(X).transform(Y), Y_imputed)
    assert_array_equal(imputer.statistics_, statistics_mean)


def test_default_with_invalid_input():
    # Test imputation with default values and invalid input

    # Test with % missing in a column > col_max_missing
    X = np.array([
        [np.nan, 0, 0, 0, 5],
        [np.nan, 1, 0, np.nan, 3],
        [np.nan, 2, 0, 0, 0],
        [np.nan, 6, 0, 5, 13],
        [np.nan, 7, 0, 7, 8],
        [np.nan, 8, 0, 8, 9],
    ])
    imputer = KNNImputer()
    msg = "Some column(s) have more than {}% missing values".format(
        imputer.col_max_missing * 100)
    assert_raise_message(ValueError, msg, imputer.fit, X)

    # Test with insufficient number of neighbors
    X = np.array([
        [1, 1, 1, 2, np.nan],
        [2, 1, 2, 2, 3],
        [3, 2, 3, 3, 8],
        [6, 6, 2, 5, 13],
    ])
    msg = "There are only %d samples, but n_neighbors=%d." % \
          (X.shape[0], imputer.n_neighbors)
    assert_raise_message(ValueError, msg, imputer.fit, X)

    # Test with inf present
    X = np.array([
        [np.inf, 1, 1, 2, np.nan],
        [2, 1, 2, 2, 3],
        [3, 2, 3, 3, 8],
        [np.nan, 6, 0, 5, 13],
        [np.nan, 7, 0, 7, 8],
        [6, 6, 2, 5, 7],
    ])
    msg = "+/- inf values are not allowed."
    assert_raise_message(ValueError, msg, KNNImputer().fit, X)

    # Test with inf present in matrix passed in transform()
    X = np.array([
        [np.inf, 1, 1, 2, np.nan],
        [2, 1, 2, 2, 3],
        [3, 2, 3, 3, 8],
        [np.nan, 6, 0, 5, 13],
        [np.nan, 7, 0, 7, 8],
        [6, 6, 2, 5, 7],
    ])

    X_fit = np.array([
        [0, 1, 1, 2, np.nan],
        [2, 1, 2, 2, 3],
        [3, 2, 3, 3, 8],
        [np.nan, 6, 0, 5, 13],
        [np.nan, 7, 0, 7, 8],
        [6, 6, 2, 5, 7],
    ])
    msg = "+/- inf values are not allowed in data to be transformed."
    assert_raise_message(ValueError, msg, KNNImputer().fit(X_fit).transform, X)


def test_knn_n_neighbors():

    X = np.array([
        [0,       0],
        [np.nan,  2],
        [4,       3],
        [5,       np.nan],
        [7,       7],
        [np.nan,  8],
        [14,      13]
    ])
    statistics_mean = np.nanmean(X, axis=0)

    # Test with 1 neighbor
    X_imputed_1NN = np.array([
        [0,      0],
        [4,      2],
        [4,      3],
        [5,      3],
        [7,      7],
        [7,      8],
        [14,     13]
    ])

    n_neighbors = 1
    imputer = KNNImputer(n_neighbors=n_neighbors)

    assert_array_equal(imputer.fit_transform(X), X_imputed_1NN)
    assert_array_equal(imputer.statistics_, statistics_mean)

    # Test with 6 neighbors
    X = np.array([
        [0,      0],
        [np.nan, 2],
        [4,      3],
        [5,      np.nan],
        [7,      7],
        [np.nan, 8],
        [14,      13]
    ])

    X_imputed_6NN = np.array([
        [0,      0],
        [6,      2],
        [4,      3],
        [5,      5.5],
        [7,      7],
        [6,      8],
        [14,     13]
    ])

    n_neighbors = 6
    imputer = KNNImputer(n_neighbors=6)
    imputer_plus1 = KNNImputer(n_neighbors=n_neighbors + 1)

    assert_array_equal(imputer.fit_transform(X), X_imputed_6NN)
    assert_array_equal(imputer.statistics_, statistics_mean)
    assert_array_equal(imputer.fit_transform(X), imputer_plus1.fit(
        X).transform(X))


def test_weight_uniform():
    X = np.array([
        [0,      0],
        [np.nan, 2],
        [4,      3],
        [5,      6],
        [7,      7],
        [9,      8],
        [11,     10]
    ])

    # Test with "uniform" weight (or unweighted)
    X_imputed_uniform = np.array([
        [0,      0],
        [5,      2],
        [4,      3],
        [5,      6],
        [7,      7],
        [9,      8],
        [11,     10]
    ])

    imputer = KNNImputer(weights="uniform")
    assert_array_equal(imputer.fit_transform(X), X_imputed_uniform)

    # Test with "callable" weight
    def no_weight(dist=None):
        return None

    imputer = KNNImputer(weights=no_weight)
    assert_array_equal(imputer.fit_transform(X), X_imputed_uniform)


def test_weight_distance():
    X = np.array([
        [0,      0],
        [np.nan, 2],
        [4,      3],
        [5,      6],
        [7,      7],
        [9,      8],
        [11,    10]
    ])

    # Test with "distance" weight

    # Get distance of "n_neighbors" neighbors of row 1
    dist_matrix = pairwise_distances(X, metric="masked_euclidean")

    index = np.argsort(dist_matrix)[1, 1:6]
    dist = dist_matrix[1, index]
    weights = 1 / dist
    values = X[index, 0]
    imputed = np.dot(values, weights) / np.sum(weights)

    # Manual calculation
    X_imputed_distance1 = np.array([
        [0,                 0],
        [3.850394,          2],
        [4,                 3],
        [5,                 6],
        [7,                 7],
        [9,                 8],
        [11,                10]
    ])

    # NearestNeighbor calculation
    X_imputed_distance2 = np.array([
        [0,                 0],
        [imputed,           2],
        [4,                 3],
        [5,                 6],
        [7,                 7],
        [9,                 8],
        [11,                10]
    ])

    imputer = KNNImputer(weights="distance")
    assert_array_almost_equal(imputer.fit_transform(X), X_imputed_distance1,
                              decimal=6)
    assert_array_almost_equal(imputer.fit_transform(X), X_imputed_distance2,
                              decimal=6)

    # Test with weights = "distance" and n_neighbors=2
    X = np.array([
        [np.nan, 0,      0],
        [2,      1,      2],
        [3,      2,      3],
        [4,      5,      5],
    ])
    statistics_mean = np.nanmean(X, axis=0)

    X_imputed = np.array([
        [2.3828, 0,     0],
        [2,      1,     2],
        [3,      2,     3],
        [4,      5,     5],
    ])

    imputer = KNNImputer(n_neighbors=2, weights="distance")
    assert_array_almost_equal(imputer.fit_transform(X), X_imputed,
                              decimal=4)
    assert_array_equal(imputer.statistics_, statistics_mean)

    # Test with varying missingness patterns
    X = np.array([
        [1,         0,          0,  1],
        [0,         np.nan,     1,  np.nan],
        [1,         1,          1,  np.nan],
        [0,         1,          0,  0],
        [0,         0,          0,  0],
        [1,         0,          1,  1],
        [10,        10,         10, 10],
    ])
    statistics_mean = np.nanmean(X, axis=0)

    # Get weights of donor neighbors
    dist = masked_euclidean_distances(X)
    r1c1_nbor_dists = dist[1, [0, 2, 3, 4, 5]]
    r1c3_nbor_dists = dist[1, [0, 3, 4, 5, 6]]
    r1c1_nbor_wt = (1/r1c1_nbor_dists)
    r1c3_nbor_wt = (1 / r1c3_nbor_dists)

    r2c3_nbor_dists = dist[2, [0, 3, 4, 5, 6]]
    r2c3_nbor_wt = 1/r2c3_nbor_dists

    # Collect donor values
    col1_donor_values = np.ma.masked_invalid(X[[0, 2, 3, 4, 5], 1]).copy()
    col3_donor_values = np.ma.masked_invalid(X[[0, 3, 4, 5, 6], 3]).copy()

    # Final imputed values
    r1c1_imp = np.ma.average(col1_donor_values, weights=r1c1_nbor_wt)
    r1c3_imp = np.ma.average(col3_donor_values, weights=r1c3_nbor_wt)
    r2c3_imp = np.ma.average(col3_donor_values, weights=r2c3_nbor_wt)

    print(r1c1_imp, r1c3_imp, r2c3_imp)
    X_imputed = np.array([
        [1,         0,          0,  1],
        [0,         r1c1_imp,   1,  r1c3_imp],
        [1,         1,          1,  r2c3_imp],
        [0,         1,          0,  0],
        [0,         0,          0,  0],
        [1,         0,          1,  1],
        [10,        10,         10, 10],
    ])

    imputer = KNNImputer(weights="distance")
    assert_array_almost_equal(imputer.fit_transform(X), X_imputed, decimal=6)
    assert_array_equal(imputer.statistics_, statistics_mean)


def test_metric_type():
    X = np.array([
        [0,      0],
        [np.nan, 2],
        [4,      3],
        [5,      6],
        [7,      7],
        [9,      8],
        [11,     10]
    ])

    # Test with a metric type without NaN support
    imputer = KNNImputer(metric="euclidean")
    bad_metric_msg = "The selected metric does not support NaN values."
    assert_raise_message(ValueError, bad_metric_msg, imputer.fit, X)


def test_callable_metric():

    # Define callable metric that returns the l1 norm:
    def custom_callable(x, y, missing_values="NaN", squared=False):
        x = np.ma.array(x, mask=np.isnan(x))
        y = np.ma.array(y, mask=np.isnan(y))
        dist = np.nansum(np.abs(x-y))
        return dist

    X = np.array([
        [4, 3, 3, np.nan],
        [6, 9, 6, 9],
        [4, 8, 6, 9],
        [np.nan, 9, 11, 10.]
    ])

    X_imputed = np.array([
        [4, 3, 3, 9],
        [6, 9, 6, 9],
        [4, 8, 6, 9],
        [5, 9, 11, 10.]
    ])

    imputer = KNNImputer(n_neighbors=2, metric=custom_callable)
    assert_array_equal(imputer.fit_transform(X), X_imputed)


def test_complete_features():

    # Test with use_complete=True
    X = np.array([
        [0,      np.nan,    0,       np.nan],
        [1,      1,         1,       np.nan],
        [2,      2,         np.nan,  2],
        [3,      3,         3,       3],
        [4,      4,         4,       4],
        [5,      5,         5,       5],
        [6,      6,         6,       6],
        [np.nan, 7,         7,       7]
    ])

    r0c1 = np.mean(X[1:6, 1])
    r0c3 = np.mean(X[2:-1, -1])
    r1c3 = np.mean(X[2:-1, -1])
    r2c2 = np.nanmean(X[:6, 2])
    r7c0 = np.mean(X[2:-1, 0])

    X_imputed = np.array([
        [0,     r0c1,   0,    r0c3],
        [1,     1,      1,    r1c3],
        [2,     2,      r2c2, 2],
        [3,     3,      3,    3],
        [4,     4,      4,    4],
        [5,     5,      5,    5],
        [6,     6,      6,    6],
        [r7c0,  7,      7,    7]
    ])

    imputer_comp = KNNImputer()
    assert_array_almost_equal(imputer_comp.fit_transform(X), X_imputed)


def test_complete_features_weighted():

    # Test with use_complete=True
    X = np.array([
        [0,      0,     0,       np.nan],
        [1,      1,     1,       np.nan],
        [2,      2,     np.nan,  2],
        [3,      3,     3,       3],
        [4,      4,     4,       4],
        [5,      5,     5,       5],
        [6,      6,     6,       6],
        [np.nan, 7,     7,       7]
    ])

    dist = pairwise_distances(X,
                              metric="masked_euclidean",
                              squared=False)

    # Calculate weights
    r0c3_w = 1.0 / dist[0, 2:-1]
    r1c3_w = 1.0 / dist[1, 2:-1]
    r2c2_w = 1.0 / dist[2, (0, 1, 3, 4, 5)]
    r7c0_w = 1.0 / dist[7, 2:7]

    # Calculate weighted averages
    r0c3 = np.average(X[2:-1, -1], weights=r0c3_w)
    r1c3 = np.average(X[2:-1, -1], weights=r1c3_w)
    r2c2 = np.average(X[(0, 1, 3, 4, 5), 2], weights=r2c2_w)
    r7c0 = np.average(X[2:7, 0], weights=r7c0_w)

    X_imputed = np.array([
        [0,     0,  0,    r0c3],
        [1,     1,  1,    r1c3],
        [2,     2,  r2c2, 2],
        [3,     3,  3,    3],
        [4,     4,  4,    4],
        [5,     5,  5,    5],
        [6,     6,  6,    6],
        [r7c0,  7,  7,    7]
    ])

    imputer_comp_wt = KNNImputer(weights="distance")
    assert_array_almost_equal(imputer_comp_wt.fit_transform(X), X_imputed)
