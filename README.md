## missingpy

`missingpy` is a library for missing data imputation in Python. It has an 
API consistent with [scikit-learn](http://scikit-learn.org/stable/), so users 
already comfortable with that interface will find themselves in familiar 
terrain. Currently, the library only supports k-Nearest Neighbors based 
imputation but we plan to add other imputation tools in the future so 
please stay tuned!

## Installation

`pip install missingpy`

## Example

```
from missingpy import KNNImputer
imputer = KNNImputer()
X_imputed = imputer.fit_transform(X)
```
Note: Please check out the imputer's docstring for more information.
 
## k-Nearest Neighbors (kNN) Imputation

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
