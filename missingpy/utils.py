"""Utility Functions"""
# Author: Ashim Bhattarai
# License: BSD 3 clause

import numpy as np


def masked_euclidean_distances(X, Y=None, squared=False,
                               missing_values="NaN", copy=True):
    """Calculates euclidean distances in the presence of missing values

    Computes the euclidean distance between each pair of samples (rows) in X
    and Y, where Y=X is assumed if Y=None.
    When calculating the distance between a pair of samples, this formulation
    essentially zero-weights feature coordinates with a missing value in either
    sample and scales up the weight of the remaining coordinates:

        dist(x,y) = sqrt(weight * sq. distance from non-missing coordinates)
        where,
        weight = Total # of coordinates / # of non-missing coordinates

    Note that if all the coordinates are missing or if there are no common
    non-missing coordinates then NaN is returned for that pair.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)

    squared : boolean, optional
        Return squared Euclidean distances.

    missing_values : "NaN" or integer, optional
        Representation of missing value

    copy : boolean, optional
        Make and use a deep copy of X and Y (if Y exists)

    Returns
    -------
    distances : {array}, shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from missingpy.utils import masked_euclidean_distances
    >>> nan = float("NaN")
    >>> X = [[0, 1], [1, nan]]
    >>> # distance between rows of X
    >>> masked_euclidean_distances(X, X)
    array([[0.        , 1.41421356],
           [1.41421356, 0.        ]])

    >>> # get distance to origin
    >>> masked_euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])

    References
    ----------
    * John K. Dixon, "Pattern Recognition with Partly Missing Data",
      IEEE Transactions on Systems, Man, and Cybernetics, Volume: 9, Issue:
      10, pp. 617 - 621, Oct. 1979.
      http://ieeexplore.ieee.org/abstract/document/4310090/

    See also
    --------
    paired_distances : distances betweens pairs of elements of X and Y.
    """
    # Import here to prevent circular import
    from .pairwise_external import _get_mask, check_pairwise_arrays

    # NOTE: force_all_finite=False allows not only NaN but also +/- inf
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False,
                                 force_all_finite=False, copy=copy)
    if (np.any(np.isinf(X)) or
            (Y is not X and np.any(np.isinf(Y)))):
        raise ValueError(
            "+/- Infinite values are not allowed.")

    # Get missing mask for X and Y.T
    mask_X = _get_mask(X, missing_values)

    YT = Y.T
    mask_YT = mask_X.T if Y is X else _get_mask(YT, missing_values)

    # Check if any rows have only missing value
    if np.any(mask_X.sum(axis=1) == X.shape[1])\
            or (Y is not X and np.any(mask_YT.sum(axis=0) == Y.shape[1])):
        raise ValueError("One or more rows only contain missing values.")

    # else:
    if missing_values not in ["NaN", np.nan] and (
            np.any(np.isnan(X)) or (Y is not X and np.any(np.isnan(Y)))):
        raise ValueError(
            "NaN values present but missing_value = {0}".format(
                missing_values))

    # Get mask of non-missing values set Y.T's missing to zero.
    # Further, casting the mask to int to be used in formula later.
    not_YT = (~mask_YT).astype(np.int32)
    YT[mask_YT] = 0

    # Get X's mask of non-missing values and set X's missing to zero
    not_X = (~mask_X).astype(np.int32)
    X[mask_X] = 0

    # Calculate distances
    # The following formula derived by:
    # Shreya Bhattarai <shreya.bhattarai@gmail.com>

    distances = (
            (X.shape[1] / (np.dot(not_X, not_YT))) *
            (np.dot(X * X, not_YT) - 2 * (np.dot(X, YT)) +
             np.dot(not_X, YT * YT)))

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)
