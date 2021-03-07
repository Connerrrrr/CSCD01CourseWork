# Selected Bugs for Assignment 2

## Table of Content

- [#18963](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#sklearnfeature_extractiongrid_to_graph-may-reorder-vertices-18963)
- [#19352](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#interactive-imputer-cannot-accept-plsregression-as-an-estimator-due-to-shape-mismatch-19352)
- [#16710](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#pipeline-requires-both-fit-and-transform-method-to-be-available-instead-of-only-fit_transform-16710)
- [#16924](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#pipeline-requires-both-fit-and-transform-method-to-be-available-instead-of-only-fit_transform-16710)
- [#19071](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#simpleimputer-missing_values-and-none-19071)
- [#18940](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a2#diabetes-data-set-description-possibly-inaccurate-18940)

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
