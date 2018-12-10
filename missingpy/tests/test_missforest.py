import numpy as np
from scipy.stats import mode

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from missingpy import MissForest

def gen_array(n_rows=20, n_cols=5, missingness=0.2, min_val=0, max_val=10,
              missing_values=np.nan, rand_seed=1337):
    """Generate an array with NaNs"""

    rand_gen = np.random.RandomState(seed=rand_seed)
    X = rand_gen.randint(
        min_val, max_val, n_rows * n_cols).reshape(n_rows, n_cols).astype(
        np.float)

    # Introduce NaNs if missingness > 0
    if missingness > 0:
        # If missingness >= 1 then use it as approximate (see below) count
        if missingness >= 1:
            n_missing = missingness
        else:
            # If missingness is between (0, 1] then use it as approximate %
            # of total cells that are NaNs
            n_missing = int(np.ceil(missingness * n_rows * n_cols))

        # Generate row, col index pairs and introduce NaNs
        # NOTE: Below does not account for repeated index pairs so NaN
        # count/percentage might be less than specified in function call
        nan_row_idx = rand_gen.randint(0, n_rows, n_missing)
        nan_col_idx = rand_gen.randint(0, n_cols, n_missing)
        X[nan_row_idx, nan_col_idx] = missing_values

    return X


def test_missforest_imputation_shape():
    # Verify the shapes of the imputed matrix
    n_rows = 10
    n_cols = 2
    X = gen_array(n_rows, n_cols)
    imputer = MissForest()
    X_imputed = imputer.fit_transform(X)
    assert_equal(X_imputed.shape, (n_rows, n_cols))


def test_missforest_zero():
    # Test imputation when missing_values == 0
    missing_values = 0
    imputer = MissForest(missing_values=missing_values,
                         random_state=0)

    # Test with missing_values=0 when NaN present
    X = gen_array(min_val=0)
    msg = "Input contains NaN, infinity or a value too large for %r." % X.dtype
    assert_raise_message(ValueError, msg, imputer.fit, X)

    # Test with all zeroes in a column
    X = np.array([
        [1, 0, 0, 0, 5],
        [2, 1, 0, 2, 3],
        [3, 2, 0, 0, 0],
        [4, 6, 0, 5, 13],
    ])
    msg = "One or more columns have all rows missing."
    assert_raise_message(ValueError, msg, imputer.fit, X)


def test_missforest_zero_part2():
    # Test with an imputable matrix and compare with missing_values="NaN"
    X_zero = gen_array(min_val=1, missing_values=0)
    X_nan = gen_array(min_val=1, missing_values=np.nan)
    statistics_mean = np.nanmean(X_nan, axis=0)

    imputer_zero = MissForest(missing_values=0, random_state=1337)
    imputer_nan = MissForest(missing_values=np.nan, random_state=1337)

    assert_array_equal(imputer_zero.fit_transform(X_zero),
                       imputer_nan.fit_transform(X_nan))
    assert_array_equal(imputer_zero.statistics_.get("col_means"),
                       statistics_mean)


def test_missforest_numerical_single():
    # Test imputation with default parameter values

    # Test with a single missing value
    df = np.array([
        [1,      0,      0,      1],
        [2,      1,      2,      2],
        [3,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [6,      7,      6,      7],
        [8,      8,      8,      8],
        [16,     15,     18,    19],
    ])
    statistics_mean = np.nanmean(df, axis=0)

    y = df[:, 0]
    X = df[:, 1:]
    good_rows = np.where(~np.isnan(y))[0]
    bad_rows = np.where(np.isnan(y))[0]

    rf = RandomForestRegressor(n_estimators=10, random_state=1337)
    rf.fit(X=X[good_rows], y=y[good_rows])
    pred_val = rf.predict(X[bad_rows])

    df_imputed = np.array([
        [1,         0,      0,      1],
        [2,         1,      2,      2],
        [3,         2,      3,      2],
        [pred_val,  4,      5,      5],
        [6,         7,      6,      7],
        [8,         8,      8,      8],
        [16,        15,     18,    19],
    ])

    imputer = MissForest(n_estimators=10, random_state=1337)
    assert_array_equal(imputer.fit_transform(df), df_imputed)
    assert_array_equal(imputer.statistics_.get('col_means'), statistics_mean)


def test_missforest_numerical_multiple():
    # Test with two missing values for multiple iterations
    df = np.array([
        [1,      0,      np.nan, 1],
        [2,      1,      2,      2],
        [3,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [6,      7,      6,      7],
        [8,      8,      8,      8],
        [16,     15,     18,    19],
    ])
    statistics_mean = np.nanmean(df, axis=0)
    n_rows, n_cols = df.shape

    # Fit missforest and transform
    imputer = MissForest(random_state=1337)
    df_imp1 = imputer.fit_transform(df)

    # Get iterations used by missforest above
    max_iter = imputer.iter_count_

    # Get NaN mask
    nan_mask = np.isnan(df)
    nan_rows, nan_cols = np.where(nan_mask)

    # Make initial guess for missing values
    df_imp2 = df.copy()
    df_imp2[nan_rows, nan_cols] = np.take(statistics_mean, nan_cols)

    # Loop for max_iter count over the columns with NaNs
    for _ in range(max_iter):
        for c in nan_cols:
            # Identify all other columns (i.e. predictors)
            not_c = np.setdiff1d(np.arange(n_cols), c)
            # Identify rows with NaN and those without in 'c'
            y = df_imp2[:, c]
            X = df_imp2[:, not_c]
            good_rows = np.where(~nan_mask[:, c])[0]
            bad_rows = np.where(nan_mask[:, c])[0]

            # Fit model and predict
            rf = RandomForestRegressor(n_estimators=100, random_state=1337)
            rf.fit(X=X[good_rows], y=y[good_rows])
            pred_val = rf.predict(X[bad_rows])

            # Fill in values
            df_imp2[bad_rows, c] = pred_val

    assert_array_equal(df_imp1, df_imp2)
    assert_array_equal(imputer.statistics_.get('col_means'), statistics_mean)


def test_missforest_categorical_single():
    # Test imputation with default parameter values

    # Test with a single missing value
    df = np.array([
        [0,      0,      0,      1],
        [0,      1,      2,      2],
        [0,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [1,      7,      6,      7],
        [1,      8,      8,      8],
        [1,     15,     18,     19],
    ])

    y = df[:, 0]
    X = df[:, 1:]
    good_rows = np.where(~np.isnan(y))[0]
    bad_rows = np.where(np.isnan(y))[0]

    rf = RandomForestClassifier(n_estimators=10, random_state=1337)
    rf.fit(X=X[good_rows], y=y[good_rows])
    pred_val = rf.predict(X[bad_rows])

    df_imputed = np.array([
        [0,         0,      0,      1],
        [0,         1,      2,      2],
        [0,         2,      3,      2],
        [pred_val,  4,      5,      5],
        [1,         7,      6,      7],
        [1,         8,      8,      8],
        [1,         15,     18,     19],
    ])

    imputer = MissForest(n_estimators=10, random_state=1337)
    assert_array_equal(imputer.fit_transform(df, cat_vars=0), df_imputed)
    assert_array_equal(imputer.fit_transform(df, cat_vars=[0]), df_imputed)


def test_missforest_categorical_multiple():
    # Test with two missing values for multiple iterations
    df = np.array([
        [0,      0,      np.nan, 1],
        [0,      1,      1,      2],
        [0,      2,      1,      2],
        [np.nan, 4,      1,      5],
        [1,      7,      0,      7],
        [1,      8,      0,      8],
        [1,     15,      0,     19],
        [1,     18,      0,     17],
    ])
    cat_vars = [0, 2]
    statistics_mode = mode(df, axis=0, nan_policy='omit').mode[0]
    n_rows, n_cols = df.shape

    # Fit missforest and transform
    imputer = MissForest(random_state=1337)
    df_imp1 = imputer.fit_transform(df, cat_vars=cat_vars)

    # Get iterations used by missforest above
    max_iter = imputer.iter_count_

    # Get NaN mask
    nan_mask = np.isnan(df)
    nan_rows, nan_cols = np.where(nan_mask)

    # Make initial guess for missing values
    df_imp2 = df.copy()
    df_imp2[nan_rows, nan_cols] = np.take(statistics_mode, nan_cols)

    # Loop for max_iter count over the columns with NaNs
    for _ in range(max_iter):
        for c in nan_cols:
            # Identify all other columns (i.e. predictors)
            not_c = np.setdiff1d(np.arange(n_cols), c)
            # Identify rows with NaN and those without in 'c'
            y = df_imp2[:, c]
            X = df_imp2[:, not_c]
            good_rows = np.where(~nan_mask[:, c])[0]
            bad_rows = np.where(nan_mask[:, c])[0]

            # Fit model and predict
            rf = RandomForestClassifier(n_estimators=100, random_state=1337)
            rf.fit(X=X[good_rows], y=y[good_rows])
            pred_val = rf.predict(X[bad_rows])

            # Fill in values
            df_imp2[bad_rows, c] = pred_val

    assert_array_equal(df_imp1, df_imp2)
    assert_array_equal(imputer.statistics_.get('col_modes')[0],
                       statistics_mode[cat_vars])


def test_missforest_mixed_multiple():
    # Test with mixed data type
    df = np.array([
        [np.nan, 0,      0,      1],
        [0,      1,      2,      2],
        [0,      2,      3,      2],
        [1,      4,      5,      5],
        [1,      7,      6,      7],
        [1,      8,      8,      8],
        [1,     15,     18,      np.nan],
    ])

    n_rows, n_cols = df.shape
    cat_vars = [0]
    num_vars = np.setdiff1d(range(n_cols), cat_vars)
    statistics_mode = mode(df, axis=0, nan_policy='omit').mode[0]
    statistics_mean = np.nanmean(df, axis=0)

    # Fit missforest and transform
    imputer = MissForest(random_state=1337)
    df_imp1 = imputer.fit_transform(df, cat_vars=cat_vars)

    # Get iterations used by missforest above
    max_iter = imputer.iter_count_

    # Get NaN mask
    nan_mask = np.isnan(df)
    nan_rows, nan_cols = np.where(nan_mask)

    # Make initial guess for missing values
    df_imp2 = df.copy()
    df_imp2[0, 0] = statistics_mode[0]
    df_imp2[6, 3] = statistics_mean[3]

    # Loop for max_iter count over the columns with NaNs
    for _ in range(max_iter):
        for c in nan_cols:
            # Identify all other columns (i.e. predictors)
            not_c = np.setdiff1d(np.arange(n_cols), c)
            # Identify rows with NaN and those without in 'c'
            y = df_imp2[:, c]
            X = df_imp2[:, not_c]
            good_rows = np.where(~nan_mask[:, c])[0]
            bad_rows = np.where(nan_mask[:, c])[0]

            # Fit model and predict
            if c in cat_vars:
                rf = RandomForestClassifier(n_estimators=100,
                                            random_state=1337)
            else:
                rf = RandomForestRegressor(n_estimators=100,
                                           random_state=1337)
            rf.fit(X=X[good_rows], y=y[good_rows])
            pred_val = rf.predict(X[bad_rows])

            # Fill in values
            df_imp2[bad_rows, c] = pred_val

    assert_array_equal(df_imp1, df_imp2)
    assert_array_equal(imputer.statistics_.get('col_means'),
                       statistics_mean[num_vars])
    assert_array_equal(imputer.statistics_.get('col_modes')[0],
                       statistics_mode[cat_vars])


def test_statstics_fit_transform():
    # Test statistics_ when data in fit() and transform() are different
    X = np.array([
        [1,      0,      0,      1],
        [2,      1,      2,      2],
        [3,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [6,      7,      6,      7],
        [8,      8,      8,      8],
        [16,     15,     18,    19],
    ])
    statistics_mean = np.nanmean(X, axis=0)

    Y = np.array([
        [0,      0,      0,      0],
        [2,      2,      2,      1],
        [3,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [6,      7,      6,      7],
        [9,      9,      8,      8],
        [16,     15,     18,    19],
    ])

    imputer = MissForest()
    imputer.fit(X).transform(Y)
    assert_array_equal(imputer.statistics_.get('col_means'), statistics_mean)


def test_default_with_invalid_input():
    # Test imputation with default values and invalid input

    # Test with all rows missing in a column
    X = np.array([
        [np.nan,    0,      0,      1],
        [np.nan,    1,      2,      np.nan],
        [np.nan,    2,      3,      np.nan],
        [np.nan,    4,      5,      5],
    ])
    imputer = MissForest(random_state=1337)
    msg = "One or more columns have all rows missing."
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
    msg = "+/- inf values are not supported."
    assert_raise_message(ValueError, msg, MissForest().fit, X)

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
    msg = "+/- inf values are not supported."
    assert_raise_message(ValueError, msg, MissForest().fit(X_fit).transform, X)
