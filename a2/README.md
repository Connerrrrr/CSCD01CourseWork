# Selected Bugs for Assignment 2

## Table of Content

- [#18940]()
- [#18963]()
- [#19352]()
- [#16710]()
- [#16924]()
- [#19071]()

## Diabetes data set description. Possibly inaccurate? #18940

[Link](https://github.com/scikit-learn/scikit-learn/issues/18940) to the issue page.

## sklearn.feature_extraction.grid_to_graph may reorder vertices #18963

[Link](https://github.com/scikit-learn/scikit-learn/issues/18963) to the issue page.

- Location

    /sklearn/feature_extraction/image.py: grid_to_graph

- Description

    In grid\_to\_graph, you expect the vertices to correspond to the implicit order defined by the mask. This is not always the case, due to the occurrence of isolated vertices that are dismissed in the reindexing of the vertices.

- Reproduce
    - version
        - Python 3.9.1
        - Numpy 1.21.0.dev0+577.g48808e1a6

![alt text][https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/18963-1.png "File to reproduce"]

![alt text][https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a2/Images/18963-2.png "Reproduce output"]

## Interactive Imputer cannot accept PLSRegression() as an estimator due to "shape mismatch" #19352

[Link](https://github.com/scikit-learn/scikit-learn/issues/19352) to the issue page.

## Pipeline requires both fit and transform method to be available instead of only fit_transform #16710

[Link](https://github.com/scikit-learn/scikit-learn/issues/16710) to the issue page.

## Matthews correlation coefficient metric throws misleading division by zero RuntimeWarning #16924

[Link](https://github.com/scikit-learn/scikit-learn/issues/16924) to the issue page.

- Description

    While using **_sklearn.metrics.matthews\_corrcoef_** with the steps below, program throws a **_RuntimeWarning_**, reporting a division by zero.

- Reproduce

```python
import sklearn.metrics                         
trues = [1,0,1,1,0]                            
preds = [0,0,0,0,0]                            
sklearn.metrics.matthews_corrcoef(trues, preds)
```

- Expected Results

    No warning is thrown.

- Actual Results

```python
C:\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:870: RuntimeWarning: invalid value encountered in double_scalars
  mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
```

## SimpleImputer, missing_values and None #19071

[Link](https://github.com/scikit-learn/scikit-learn/issues/19071) to the issue page.
