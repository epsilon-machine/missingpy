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

## References
1. Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor
    Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing value
    estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17 no. 6, 2001
    Pages 520-525.
