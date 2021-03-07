# Selected Bugs for Assignment 2

## Table of Content

- [#18963](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#sklearnfeature_extractiongrid_to_graph-may-reorder-vertices-18963)
- [#19352](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#interactive-imputer-cannot-accept-plsregression-as-an-estimator-due-to-shape-mismatch-19352)
- [#16710](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#pipeline-requires-both-fit-and-transform-method-to-be-available-instead-of-only-fit_transform-16710)
- [#16924](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#matthews-correlation-coefficient-metric-throws-misleading-division-by-zero-runtimewarning-16924)
- [#19071](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#simpleimputer-missing_values-and-none-19071)

## sklearn.feature_extraction.grid_to_graph may reorder vertices #18963

[Link](https://github.com/scikit-learn/scikit-learn/issues/18963) to the issue page.

- Location

    /sklearn/feature_extraction/image.py: grid_to_graph

- Description

    In grid\_to\_graph, you expect the vertices to correspond to the implicit order defined by the mask. This is not always the case, due to the occurrence of isolated vertices that are dismissed in the reindexing of the vertices.

- Reproduce

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/18963-1.png "File to reproduce")

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/18963-2.png "Reproduce output")

## Interactive Imputer cannot accept PLSRegression() as an estimator due to "shape mismatch" #19352

[Link](https://github.com/scikit-learn/scikit-learn/issues/19352) to the issue page.

- Location

  /sklearns/impute/_iterative.py

- Description

  As the issues mention in github, when user setting the estimator as PLSRegression(), a ValueError is triggered by module '\_iteractive.py' located in impute package in line 348, caused by "shape mismatch"

- Reproduce

  - Expected Results

    ```python
    [[   8.3252       41.            6.98412698 ...    2.55555556
    37.88       -122.25930206]
    [   8.3014       21.            6.23813708 ...    2.10984183
    37.86       -122.22      ]
    [   7.2574       52.            8.28813559 ...    2.80225989
    37.85       -122.24      ]
    ...
    [   3.60438721   50.            5.33480176 ...    2.30396476
    37.88       -122.29      ]
    [   5.1675       52.            6.39869281 ...    2.44444444
    37.89       -122.29      ]
    [   5.1696       52.            6.11590296 ...    2.70619946
    37.8709526  -122.29      ]]
    ```

    - Actual Results

    ```python
    ValueError: shape mismatch: value array of shape (27,1) could not be broadcast to indexing result of shape (27,)
    ```

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-1.png "File to reproduce")

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-2.png "Reproduce output")

- Fix

  - Source Code

    - Original

      ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-original.png "Original Source Code")

    - Fixed

      ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-fixed.png "Fixed Source Code")

  - Fixed Result 1

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-fixed-result-1.png "Fixed Result 1")

  - Fixed Result 2

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-fixed-result-2.png "Fixed Result 2")

  - Explaination

    In \_iterative.py, the dimension of the array is not correct in line 346. A one-dimensional array is needed, resulting in a missing two-dimensional array.The solution for this would be Convert the two-dimensional array to a one-dimensional array.

  - Test File

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-test-file-1.png "Test File 1")

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-test-file-2.png "Test File 2")

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-test-file-3.png "Test File 3")

  - Test Result

    - 1D array

      ```python
      pytest ./sklearn/impute/tests/test_impute.py -v -k imputation_1D
      ```

      ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-test-result-1d.png "Test Result of 1D array")

    - 2D array

      ```python
      pytest ./sklearn/impute/tests/test_impute.py -v -k imputation_example
      ```

      ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19352-test-result-2d.png "Test Result of 2D array")

## Pipeline requires both fit and transform method to be available instead of only fit_transform #16710

[Link](https://github.com/scikit-learn/scikit-learn/issues/16710) to the issue page.

- Location

  /sklearn/pipeline.py

- Description

  As the issues mention in github, when a user calling a pipeline with a nonparametric function causes an error in pipeline.py. a regular transform() method does not exist since there is no projection or mapping that is learned.

- Reproduce

  - Expected Results

    No warning is thrown.

- Actual Results

    ```python
    TypeError: All intermediate steps should be transformers and implement fit and transform or be the string 'passthrough' 'TSNE(angle=0.5,...
    ```

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16710-1.png "File to reproduce")

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16710-2.png "Reproduce output")

- Fix

  - Source Code

    - Original

      ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16710-original.png "Original Source Code")

    - Fixed

      ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16710-fixed.png "Fixed Source Code")

  - Fixed Result 1

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16710-fixed-result-1.png "Fixed Result 1")

  - Fixed Result 2

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16710-fixed-result-2.png "Fixed Result 2")

  - Explaination

    In pipeline.py Pipeline requires both fit and transform methods to be available instead of only fit_transform however, TSNE has "fit", has "fit\_transform", but it does not have "transform". PCA has "fit", has "fit\_transform", but it does not have "transform". The solution would be we change the first if statement to check whether the class have “transform” function or not then raise the error.

  - Test File

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16710-test-file.png "Test File")

  - Test Result

    ```python
    pytest ./sklearn/tests/test_pipeline.py -v -k make_pipeline
    ```

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16710-test-result.png "Test Result")

## Matthews correlation coefficient metric throws misleading division by zero RuntimeWarning #16924

[Link](https://github.com/scikit-learn/scikit-learn/issues/16924) to the issue page.

- Location

  /sklearn/metrics/\_classification.py: matthews\_corrcoef

- Description

    While using **_sklearn.metrics.matthews\_corrcoef_** with the steps below, program throws a **_RuntimeWarning_**, reporting a division by zero.

- Reproduce

  - Expected Results

    No warning is thrown.

  - Actual Results

    ```python
    RuntimeWarning: invalid value encountered in double_scalars
        mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
    ```

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16924-1.png "File to reproduce")

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/16924-2.png "Reproduce output")

- Fix

  - Source Code

    - Original

    ```python
    def matthews_corrcoef(y_true, y_pred, *, sample_weight=None):
        ...
        t_sum = C.sum(axis=1, dtype=np.float64)
        p_sum = C.sum(axis=0, dtype=np.float64)
        ...
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
        mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

        if np.isnan(mcc):
            return 0.
        else:
            return mcc
    ```

    - Fixed

    ```python
    def matthews_corrcoef(y_true, y_pred, *, sample_weight=None):
        ...
        t_sum = C.sum(axis=1, dtype=np.float64)
        p_sum = C.sum(axis=0, dtype=np.float64)
        ...
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
        t_nonzero = np.nonzero(t_sum)[0].size
        p_nonzero = np.nonzero(p_sum)[0].size

        if t_nonzero == 1 or p_nonzero == 1:
            return 0.
        else:
            mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
            return mcc
    ```

  - Explaination

    The original MCC function included the case of confusion matrix has all zeros but for one column. However, the original solution did not expect the warning messeage would pop up, even the return value is correct.

    According to [Jurman, Riccadonna, Furlanello, (2012). A Comparison of MCC and CEN Error Measures in MultiClass Prediction](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041882)
    > MCC is equal to 0 when C is all zeros but for one column (all samples have been classified to be of a class k)

    and [Wikipedia of Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient),

    > If any of the four sums in the denominator is zero, the denominator can be arbitrarily set to one

    the fix is checking whenever there is only a column or a row in confusion matrix is non-zero, the MCC automatically set to 0.

  - Test Cases

    Location: /sklearn/metrics/tests/test\_classification.py

    - Original

    ```python
    @ignore_warnings
    def test_matthews_corrcoef_nan():
        assert matthews_corrcoef([0], [1]) == 0.0
        assert matthews_corrcoef([0, 0], [0, 1]) == 0.0
    
    def test_matthews_corrcoef():
        ...
        # For the zero vector case, the corrcoef cannot be calculated and should
        # result in a RuntimeWarning
        mcc = assert_warns_div0(matthews_corrcoef, [0, 0, 0, 0], [0, 0, 0, 0])

        # But will output 0
        assert_almost_equal(mcc, 0.)

        # And also for any other vector with 0 variance
        mcc = assert_warns_div0(matthews_corrcoef, y_true, ['a'] * len(y_true))

        # But will output 0
        assert_almost_equal(mcc, 0.)
        ...
    
    def test_matthews_corrcoef_multiclass():
        ...
        # Zero variance will result in an mcc of zero and a Runtime Warning
        y_true = [0, 1, 2]
        y_pred = [3, 3, 3]
        mcc = assert_warns_message(RuntimeWarning, 'invalid value encountered',
                                matthews_corrcoef, y_true, y_pred)
        assert_almost_equal(mcc, 0.0)
        ...
        # For the zero vector case, the corrcoef cannot be calculated and should
        # result in a RuntimeWarning
        y_true = [0, 0, 1, 2]
        y_pred = [0, 0, 1, 2]
        sample_weight = [1, 1, 0, 0]
        mcc = assert_warns_message(RuntimeWarning, 'invalid value encountered',
                                matthews_corrcoef, y_true, y_pred,
                                sample_weight=sample_weight)

        # But will output 0
        assert_almost_equal(mcc, 0.)
    ```

    - Modified

    ```python
    def test_matthews_corrcoef_nan():
        assert matthews_corrcoef([0], [1]) == 0.0
        assert matthews_corrcoef([0, 0], [0, 1]) == 0.0
    
    def test_matthews_corrcoef():
        ...
        # For the zero vector case, the corrcoef cannot be calculated and should
        # output 0
        assert_almost_equal(matthews_corrcoef([0, 0, 0, 0], [0, 0, 0, 0]), 0.)

        # And also for any other vector with 0 variance
        assert_almost_equal(matthews_corrcoef(y_true, ['a'] * len(y_true)), 0.)
        ...
    
    def test_matthews_corrcoef_multiclass():
        ...
        # Zero variance will result in an mcc of zero
        y_true = [0, 1, 2]
        y_pred = [3, 3, 3]
        mcc = matthews_corrcoef(y_true, y_pred)
        assert_almost_equal(mcc, 0.0)

        # Also for ground truth with zero variance
        y_true = [3, 3, 3]
        y_pred = [0, 1, 2]
        mcc = matthews_corrcoef(y_true, y_pred)
        assert_almost_equal(mcc, 0.0)
        ...
        # For the zero vector case, the corrcoef cannot be calculated and should
        # output 0
        y_true = [0, 0, 1, 2]
        y_pred = [0, 0, 1, 2]
        sample_weight = [1, 1, 0, 0]
        assert_almost_equal(matthews_corrcoef(y_true, y_pred,
                                sample_weight=sample_weight), 0.)
    ```

## SimpleImputer, missing_values and None #19071

[Link](https://github.com/scikit-learn/scikit-learn/issues/19071) to the issue page.

- Location:

    /sklearn/impute/\_base.py: class SimpleImputer

- Description:

    As the documentation in scikit learn mentions, the simpleImputer can take None as missing value, but it throws a value error when there is None in the input array.

- Reproduce
  - Expected Results:

    No error throws.

  - Actual Results:

    ```python
    ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    ```

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19071-1.png "File to reproduce")

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19071-2.png "Reproduce output")

- Fix

  - Source Code

    - Original

      ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19071-original.png "Original Source Code")

    - Modified

      ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19071-fixed.png "Fiexed Source Code")

  - Fixed Result 1

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19071-fixed-result-1.png "Fixed Result 1")

  - Fixed Result 2

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19071-fixed-result-2.png "Fixed Result 2")

  - Explaination

    When we called _SimpleImputer(missing\_values=None).fit\_transform(a)_, the original error message was just the python error message _“ValueError: Input contains NaN, infinity or a value too large for dtype('float64')”_. The case which the missing value is of None type was not handled from line 256 to line 262 in the **original src code**. Thus, the users would be confused if they did not know None is not numeric data and can not be used with either strategy mean or strategy median.

    In the **fixed src code**, we have handled this error from line 261 to line 265. Now the new error message is shown as **Fixed result 1**, which tells you why this is error happened (Cannot use mean strategy with non-numeric data, and None is an non-numeric data) and provides an alternative missing value parameter np.NaN _(SimpleImputer(missing\_values=np.nan).fit\_transform(a))_ to deal with the None case. The result of calling the alternative function is shown as **Fixed result2**, no error is thrown as expected.

  - PyTest test case

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/19071-test.png "PyTest test case")
