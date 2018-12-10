## missingpy

`missingpy` is a library for missing data imputation in Python. It has an 
API consistent with [scikit-learn](http://scikit-learn.org/stable/), so users 
already comfortable with that interface will find themselves in familiar 
terrain. Currently, the library supports k-Nearest Neighbors based 
imputation and Random Forest based imputation (MissForest) but we plan to add 
other imputation tools in the future so please stay tuned!

## Installation

`pip install missingpy`

## k-Nearest Neighbors (kNN) Imputation

### Example
```
# Let X be an array containing missing values
from missingpy import KNNImputer
imputer = KNNImputer()
X_imputed = imputer.fit_transform(X)
```
 
### Description
The `KNNImputer` class provides imputation for completing missing
values using the k-Nearest Neighbors approach. Each sample's missing values
are imputed using values from `n_neighbors` nearest neighbors found in the
training set. Note that if a sample has more than one feature missing, then
the sample can potentially have multiple sets of `n_neighbors`
donors depending on the particular feature being imputed.

Each missing feature is then imputed as the average, either weighted or
unweighted, of these neighbors. Where the number of donor neighbors is less
than `n_neighbors`, the training set average for that feature is used
for imputation. The total number of samples in the training set is, of course,
always greater than or equal to the number of nearest neighbors available for
imputation, depending on both the overall sample size as well as the number of
samples excluded from nearest neighbor calculation because of too many missing
features (as controlled by `row_max_missing`).
For more information on the methodology, see [1].

The following snippet demonstrates how to replace missing values,
encoded as `np.nan`, using the mean feature value of the two nearest
neighbors of the rows that contain the missing values::

    >>> import numpy as np
    >>> from missingpy import KNNImputer
    >>> nan = np.nan
    >>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
    >>> imputer = KNNImputer(n_neighbors=2, weights="uniform")
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
           [8. , 8. , 7. ]])

### API
    KNNImputer(missing_values="NaN", n_neighbors=5, weights="uniform", 
                     metric="masked_euclidean", row_max_missing=0.5, 
                     col_max_missing=0.8, copy=True)
                 
    Parameters
    ----------
    missing_values : integer or "NaN", optional (default = "NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as
        ``np.nan``, use the string value "NaN".

    n_neighbors : int, optional (default = 5)
        Number of neighboring samples to use for imputation.

    weights : str or callable, optional (default = "uniform")
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    metric : str or callable, optional (default = "masked_euclidean")
        Distance metric for searching neighbors. Possible values:
        - 'masked_euclidean'
        - [callable] : a user-defined function which conforms to the
        definition of _pairwise_callable(X, Y, metric, **kwds). In other
        words, the function accepts two arrays, X and Y, and a
        ``missing_values`` keyword in **kwds and returns a scalar distance
        value.

    row_max_missing : float, optional (default = 0.5)
        The maximum fraction of columns (i.e. features) that can be missing
        before the sample is excluded from nearest neighbor imputation. It
        means that such rows will not be considered a potential donor in
        ``fit()``, and in ``transform()`` their missing feature values will be
        imputed to be the column mean for the entire dataset.

    col_max_missing : float, optional (default = 0.8)
        The maximum fraction of rows (or samples) that can be missing
        for any feature beyond which an error is raised.

    copy : boolean, optional (default = True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, if metric is
        "masked_euclidean" and copy=False then missing_values in the
        input matrix X will be overwritten with zeros.

    Attributes
    ----------
    statistics_ : 1-D array of length {n_features}
        The 1-D array contains the mean of each feature calculated using
        observed (i.e. non-missing) values. This is used for imputing
        missing values in samples that are either excluded from nearest
        neighbors search because they have too many ( > row_max_missing)
        missing features or because all of the sample's k-nearest neighbors
        (i.e., the potential donors) also have the relevant feature value
        missing.

    Methods
    -------
    fit(X, y=None):
        Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
            
            
    transform(X):
        Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            The imputed dataset.
    
    
    fit_transform(X, y=None, **fit_params):
        Fit KNNImputer and impute all missing values in X.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
    
        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.       

### References
1. Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor
    Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing value
    estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17 no. 6, 2001
    Pages 520-525.

## Random Forest Imputation (MissForest)

### Example
```
# Let X be an array containing missing values
from missingpy import MissForest
imputer = MissForest()
X_imputed = imputer.fit_transform(X)
```

### Description
MissForest imputes missing values using Random Forests in an iterative
fashion [1]. By default, the imputer begins imputing missing values of the
column (which is expected to be a variable) with the smallest number of
missing values -- let's call this the candidate column.
The first step involves filling any missing values of the remaining,
non-candidate, columns with an initial guess, which is the column mean for
columns representing numerical variables and the column mode for columns
representing categorical variables. Note that the categorical variables 
need to be explicitly identified during the imputer's `fit()` method call 
(see API for more information). After that, the imputer fits a random
forest model with the candidate column as the outcome variable and the
remaining columns as the predictors over all rows where the candidate
column values are not missing.
After the fit, the missing rows of the candidate column are
imputed using the prediction from the fitted Random Forest. The
rows of the non-candidate columns act as the input data for the fitted
model.
Following this, the imputer moves on to the next candidate column with the
second smallest number of missing values from among the non-candidate
columns in the first round. The process repeats itself for each column
with a missing value, possibly over multiple iterations or epochs for
each column, until the stopping criterion is met.
The stopping criterion is governed by the "difference" between the imputed
arrays over successive iterations. For numerical variables (`num_vars_`),
the difference is defined as follows:

     sum((X_new[:, num_vars_] - X_old[:, num_vars_]) ** 2) /
     sum((X_new[:, num_vars_]) ** 2)

For categorical variables(`cat_vars_`), the difference is defined as follows:

    sum(X_new[:, cat_vars_] != X_old[:, cat_vars_])) / n_cat_missing

where `X_new` is the newly imputed array, `X_old` is the array imputed in the
previous round, `n_cat_missing` is the total number of categorical
values that are missing, and the `sum()` is performed both across rows
and columns. Following [1], the stopping criterion is considered to have
been met when difference between `X_new` and `X_old` increases for the first
time for both types of variables (if available).


    >>> from missingpy import MissForest
    >>> nan = float("NaN")
    >>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
    >>> imputer = MissForest()
    >>> imputer.fit_transform(X)
    Iteration: 0
    Iteration: 1
    Iteration: 2
    Iteration: 3
    array([[1.  , 2. , 4. ],
           [3.  , 4. , 3. ],
           [3.16, 6. , 5. ],
           [8.  , 8. , 7. ]])

### API
    MissForest(max_iter=10, decreasing=False, missing_values=np.nan,
                 copy=True, n_estimators=100, criterion=('mse', 'gini'),
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='auto',
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
                 verbose=0, warm_start=False, class_weight=None)
                 
    Parameters
    ----------
    NOTE: Most parameter definitions below are taken verbatim from the
    Scikit-Learn documentation at [2] and [3].

    max_iter : int, optional (default = 10)
        The maximum iterations of the imputation process. Each column with a
        missing value is imputed exactly once in a given iteration.

    decreasing : boolean, optional (default = False)
        If set to True, columns are sorted according to decreasing number of
        missing values. In other words, imputation will move from imputing
        columns with the largest number of missing values to columns with
        fewest number of missing values.

    missing_values : np.nan, integer, optional (default = np.nan)
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.

    copy : boolean, optional (default = True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.

    criterion : tuple, optional (default = ('mse', 'gini'))
        The function to measure the quality of a split.The first element of
        the tuple is for the Random Forest Regressor (for imputing numerical
        variables) while the second element is for the Random Forest
        Classifier (for imputing categorical variables).

    n_estimators : integer, optional (default=100)
        The number of trees in the forest.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        NOTE: This parameter is only applicable for Random Forest Classifier
        objects (i.e., for categorical variables).

    Attributes
    ----------
    statistics_ : Dictionary of length two
        The first element is an array with the mean of each numerical feature
        being imputed while the second element is an array of modes of
        categorical features being imputed (if available, otherwise it
        will be None).
    
    Methods
    -------
    fit(self, X, y=None, cat_vars=None):
        Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        cat_vars : int or array of ints, optional (default = None)
            An int or an array containing column indices of categorical
            variable(s)/feature(s) present in the dataset X.
            ``None`` if there are no categorical variables in the dataset.

        Returns
        -------
        self : object
            Returns self.
        
            
    transform(X):
        Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            The imputed dataset.
        
    
    fit_transform(X, y=None, **fit_params):
        Fit MissForest and impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.

### References

* [1] Stekhoven, Daniel J., and Peter Bühlmann. "MissForest—non-parametric
  missing value imputation for mixed-type data." Bioinformatics 28.1
  (2011): 112-118.
* [2] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
* [3] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier