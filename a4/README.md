# Selected Bugs for Assignment 4

## Table of Content

- [#15336](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a4#add-sparse-matrix-support-for-histgradientboostingclassifier)
  - Investigation
    - New Feature
    - Reproduce
    - Analysis
  - Design
  - Interactions
- [#19679](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a4#pass-the-confusion-matrix-as-a-parameter)
  - Investigation
    - Reproduce
    - Analysis
  - Design
  - Interactions
  - Implementation
  - User Guide
  - Testing
    - Acceptance Test
    - Regression Test
    - Unit Test
- [Work Log](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a4#work-log)

In Assignment 4, we have one new feature #19679 and one hard enhencement #15336. Fixed solution and investigation can be found below each issue section.

## Add Sparse Matrix Support For HistGradientBoostingClassifier

[Link](https://github.com/scikit-learn/scikit-learn/issues/15336) to the issue page.

- Investigation

  Currently, when run **fit()** in _**HistGradientBoostingClassifier class**_, it requires to have a **dense matrix** as input, however some time the size of the dense data is huge (may run out of memory) and when we have a **sparse matrix** on hand, we may want to transfer it to a **dense one**, then extra time for that transformation is required.

  - New Feature

    Provide an option to directly take a sparse matrix as an input when run _**fit()**_ in _**HistGradientBoostingClassifier**_ class.

  - Reproduce

    ![alt text]( "Reproduce")

    - Expected Result:

        No error raised.

    - Actual Result:

      ```python
      TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
      ```

    Note: in line 18, vecs is a sparse matrix. If we convert it into a dense matrix by using toarray() method, then it will be fine.

    ![alt text]( "Reproduce with alternative")

  - Analysis

    ![alt text]( "UML")

    HistGradientBoostingClassifier inherits the fit() method from BaseHistGradientBoosting.

    From: **sklearn/ensemble/\_hist\_gradient\_boosting/gradient\_boosting.py**

    In line 403, the fit() method has called _raw_predict() method.

    ![alt text]( "_raw_predict")

    From: **sklearn/ensemble/\_hist\_gradient\_boosting/gradient\_boosting.py**

    In line 736, the _raw_predict() has then called the check_array() method.

    ![alt text]( "check_array")

    From: **sklearn/utils/validation.py**

    In line 593, the check\_array() method has further called \_ensure\_sparse\_format() method, with the parameter accept_sparse set to false.

    ![alt text]( "_ensure_sparse_format")

    From: **sklearn/utils/validation.py**

    In line 360, the TypeError is finally raised in the \_ensure\_sparse\_format() method.

    ![alt text]( "_ensure_sparse_format")

- Design and Interactions

## Pass the confusion matrix as a parameter

[Link](https://github.com/scikit-learn/scikit-learn/issues/19679) to the issue page.

- Investigation

  - Reproduce

  - Analysis

- Design

- Interactions

- Implementation

- User Guide

- Testing

  - Acceptance Test

  - Regression Test

  - Unit Test

## Work Log
