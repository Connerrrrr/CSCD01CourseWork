# Selected Bugs for Assignment 4

## Table of Content

- [#15336](https://github.com/UTSCCSCD01/course-project-apple_team/tree/master/a4#add-sparse-matrix-support-for-histgradientboostingclassifier)
  - Investigation
    - New Feature
    - Reproduce
    - Analysis
  - Design and Interactions
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

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a4/Images/15336-Reproduce.png "Reproduce")

    - Expected Result:

        No error raised.

    - Actual Result:

      ```python
      TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
      ```

    Note: in line 18, vecs is a sparse matrix. If we convert it into a dense matrix by using toarray() method, then it will be fine.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a4/Images/15336-Reproduce_with_alternative.png "Reproduce with alternative")

  - Analysis

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a4/Images/15336-UML.png "UML")

    HistGradientBoostingClassifier inherits the fit() method from BaseHistGradientBoosting.

    From: **sklearn/ensemble/\_hist\_gradient\_boosting/gradient\_boosting.py**

    In line 403, the fit() method has called _raw_predict() method.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a4/Images/15336-_raw_predict.png "_raw_predict")

    From: **sklearn/ensemble/\_hist\_gradient\_boosting/gradient\_boosting.py**

    In line 736, the _raw_predict() has then called the check_array() method.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a4/Images/15336-check_array.png "check_array")

    From: **sklearn/utils/validation.py**

    In line 593, the check\_array() method has further called \_ensure\_sparse\_format() method, with the parameter accept_sparse set to false.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a4/Images/15336-_ensure_sparse_format-1.png "_ensure_sparse_format")

    From: **sklearn/utils/validation.py**

    In line 360, the TypeError is finally raised in the \_ensure\_sparse\_format() method.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a4/Images/15336-_ensure_sparse_format-2.png "_ensure_sparse_format")

- Design and Interactions

## Pass the confusion matrix as a parameter

[Link](https://github.com/scikit-learn/scikit-learn/issues/19679) to the issue page.

- Investigation

  - Reproduce

  - Analysis

- Design

- Interactions

- Implementation

  - Original Code

    - matthews_corrcoef

      ```python
      @_deprecate_positional_args
      def matthews_corrcoef(y_true, y_pred, *, sample_weight=None):

          ...

          C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
          t_sum = C.sum(axis=1, dtype=np.float64)
          p_sum = C.sum(axis=0, dtype=np.float64)
          n_correct = np.trace(C, dtype=np.float64)
          n_samples = p_sum.sum()
          cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
          cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
          cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
          mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

          if np.isnan(mcc):
              return 0.
          else:
              return mcc
      ```

    - jaccard_score

      ```python
      @_deprecate_positional_args
      def jaccard_score(y_true, y_pred, *, labels=None, pos_label=1,
                        average='binary', sample_weight=None, zero_division="warn"):

          ...

          labels = _check_set_wise_labels(y_true, y_pred, average, labels,
                                          pos_label)
          samplewise = average == 'samples'
          MCM = multilabel_confusion_matrix(y_true, y_pred,
                                            sample_weight=sample_weight,
                                            labels=labels, samplewise=samplewise)

          numerator = MCM[:, 1, 1]
          denominator = MCM[:, 1, 1] + MCM[:, 0, 1] + MCM[:, 1, 0]

          if average == 'micro':
              numerator = np.array([numerator.sum()])
              denominator = np.array([denominator.sum()])

          jaccard = _prf_divide(numerator, denominator, 'jaccard',
                                'true or predicted', average, ('jaccard',),
                                zero_division=zero_division)
          if average is None:
              return jaccard
          if average == 'weighted':
              weights = MCM[:, 1, 0] + MCM[:, 1, 1]
              if not np.any(weights):
                  # numerator is 0, and warning should have already been issued
                  weights = None
          elif average == 'samples' and sample_weight is not None:
              weights = sample_weight
          else:
              weights = None
          return np.average(jaccard, weights=weights)
      ```

    - precision_recall_fscore_support

      ```python
      @_deprecate_positional_args
      def precision_recall_fscore_support(y_true, y_pred, *, beta=1.0, labels=None,
                                          pos_label=1, average=None,
                                          warn_for=('precision', 'recall',
                                                    'f-score'),
                                          sample_weight=None,
                                          zero_division="warn"):

          ...

          _check_zero_division(zero_division)
          if beta < 0:
              raise ValueError("beta should be >=0 in the F-beta score")
          labels = _check_set_wise_labels(y_true, y_pred, average, labels,
                                          pos_label)

          # Calculate tp_sum, pred_sum, true_sum ###
          samplewise = average == 'samples'
          MCM = multilabel_confusion_matrix(y_true, y_pred,
                                            sample_weight=sample_weight,
                                            labels=labels, samplewise=samplewise)
          tp_sum = MCM[:, 1, 1]
          pred_sum = tp_sum + MCM[:, 0, 1]
          true_sum = tp_sum + MCM[:, 1, 0]

          if average == 'micro':
              tp_sum = np.array([tp_sum.sum()])
              pred_sum = np.array([pred_sum.sum()])
              true_sum = np.array([true_sum.sum()])

          # Finally, we have all our sufficient statistics. Divide! #
          beta2 = beta ** 2

          # Divide, and on zero-division, set scores and/or warn according to
          # zero_division:
          precision = _prf_divide(tp_sum, pred_sum, 'precision',
                                  'predicted', average, warn_for, zero_division)
          recall = _prf_divide(tp_sum, true_sum, 'recall',
                              'true', average, warn_for, zero_division)

          # warn for f-score only if zero_division is warn, it is in warn_for
          # and BOTH prec and rec are ill-defined
          if zero_division == "warn" and ("f-score",) == warn_for:
              if (pred_sum[true_sum == 0] == 0).any():
                  _warn_prf(
                      average, "true nor predicted", 'F-score is', len(true_sum)
                  )

          # if tp == 0 F will be 1 only if all predictions are zero, all labels are
          # zero, and zero_division=1. In all other case, 0
          if np.isposinf(beta):
              f_score = recall
          else:
              denom = beta2 * precision + recall

              denom[denom == 0.] = 1  # avoid division by 0
              f_score = (1 + beta2) * precision * recall / denom

          # Average the results
          if average == 'weighted':
              weights = true_sum
              if weights.sum() == 0:
                  zero_division_value = np.float64(1.0)
                  if zero_division in ["warn", 0]:
                      zero_division_value = np.float64(0.0)
                  # precision is zero_division if there are no positive predictions
                  # recall is zero_division if there are no positive labels
                  # fscore is zero_division if all labels AND predictions are
                  # negative
                  if pred_sum.sum() == 0:
                      return (zero_division_value,
                              zero_division_value,
                              zero_division_value,
                              None)
                  else:
                      return (np.float64(0.0),
                              zero_division_value,
                              np.float64(0.0),
                              None)

          elif average == 'samples':
              weights = sample_weight
          else:
              weights = None

          if average is not None:
              assert average != 'binary' or len(precision) == 1
              precision = np.average(precision, weights=weights)
              recall = np.average(recall, weights=weights)
              f_score = np.average(f_score, weights=weights)
              true_sum = None  # return no support

          return precision, recall, f_score, true_sum 
      ```

      - fbeta_score

        ```python
        @_deprecate_positional_args
        def fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1,
                        average='binary', sample_weight=None, zero_division="warn"):

            ...

            _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                        beta=beta,
                                                        labels=labels,
                                                        pos_label=pos_label,
                                                        average=average,
                                                        warn_for=('f-score',),
                                                        sample_weight=sample_weight,
                                                        zero_division=zero_division)
            return f
        ```

      - f1_score

        ```python
        @_deprecate_positional_args
        def f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
                     sample_weight=None, zero_division="warn"):

            ...

            return fbeta_score(y_true, y_pred, beta=1, labels=labels,
                              pos_label=pos_label, average=average,
                              sample_weight=sample_weight,
                              zero_division=zero_division)
        ```

      - precision_score

        ```python
        @_deprecate_positional_args
        def precision_score(y_true, y_pred, *, labels=None, pos_label=1,
                            average='binary', sample_weight=None,
                            zero_division="warn"):

            ...

            p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                        labels=labels,
                                                        pos_label=pos_label,
                                                        average=average,
                                                        warn_for=('precision',),
                                                        sample_weight=sample_weight,
                                                        zero_division=zero_division)
            return p
        ```

      - recall_score

        ```python
        @_deprecate_positional_args
        def recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
                         sample_weight=None, zero_division="warn"):

            ...

            _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                        labels=labels,
                                                        pos_label=pos_label,
                                                        average=average,
                                                        warn_for=('recall',),
                                                        sample_weight=sample_weight,
                                                        zero_division=zero_division)
            return r
        ```

    - balanced_accuracy_score

      ```python
      @_deprecate_positional_args
      def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None,
                                  adjusted=False):

          ...

          C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
          with np.errstate(divide='ignore', invalid='ignore'):
              per_class = np.diag(C) / C.sum(axis=1)
          if np.any(np.isnan(per_class)):
              warnings.warn('y_pred contains classes not in y_true')
              per_class = per_class[~np.isnan(per_class)]
          score = np.mean(per_class)
          if adjusted:
              n_classes = len(per_class)
              chance = 1 / n_classes
              score -= chance
              score /= 1 - chance
          return score
      ```

  - Changed Code

    - matthews_corrcoef

      ```python
      @_deprecate_positional_args
      def matthews_corrcoef(y_true, y_pred, *, sample_weight=None):

          ...

          C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)

          return matthews_corrcoef_from_confusion(C)
      ```

    - jaccard_score

      ```python
      @_deprecate_positional_args
      def jaccard_score(y_true, y_pred, *, labels=None, pos_label=1,
                        average='binary', sample_weight=None, zero_division="warn"):

          ... 

          MCM = multilabel_confusion_matrix(y_true, y_pred,
                                            sample_weight=sample_weight,
                                            labels=labels, samplewise=samplewise)

          return jaccard_score_from_confusion(MCM, average=average, sample_weight=sample_weight,
                                              zero_division=zero_division)
      ```

    - precision_recall_fscore_support

      ```python
      @_deprecate_positional_args
      def precision_recall_fscore_support(y_true, y_pred, *, beta=1.0, labels=None,
                                          pos_label=1, average=None,
                                          warn_for=('precision', 'recall',
                                                    'f-score'),
                                          sample_weight=None,
                                          zero_division="warn"):

          ...

          MCM = multilabel_confusion_matrix(y_true, y_pred,
                                            sample_weight=sample_weight,
                                            labels=labels, samplewise=samplewise)

          return precision_recall_fscore_support_from_confusion(MCM, beta=beta, average=average,
                                                                warn_for=warn_for,
                                                                sample_weight=sample_weight,
                                                                zero_division=zero_division)
      ```

    - balanced_accuracy_score

      ```python
      @_deprecate_positional_args
      def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None,
                                  adjusted=False):

          C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
          return balanced_accuracy_score_from_confusion(C, adjusted=adjusted)
      ```

  - New Code Blocks

    - matthews_corrcoef_from_confusion

      ```python
      @_deprecate_positional_args
      def matthews_corrcoef_from_confusion(cm):

          t_sum = cm.sum(axis=1, dtype=np.float64)
          p_sum = cm.sum(axis=0, dtype=np.float64)
          n_correct = np.trace(cm, dtype=np.float64)
          n_samples = p_sum.sum()
          cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
          cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
          cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
          mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

          if np.isnan(mcc):
              return 0.
          else:
              return mcc
      ```

    - jaccard_score_from_confusion

      ```python
      @_deprecate_positional_args
      def jaccard_score_from_confusion(MCM, *, average='binary', sample_weight=None,
                                       zero_division="warn"):

          numerator = MCM[:, 1, 1]
          denominator = MCM[:, 1, 1] + MCM[:, 0, 1] + MCM[:, 1, 0]

          if average == 'micro':
              numerator = np.array([numerator.sum()])
              denominator = np.array([denominator.sum()])

          jaccard = _prf_divide(numerator, denominator, 'jaccard',
                                'true or predicted', average, ('jaccard',),
                                zero_division=zero_division)
          if average is None:
              return jaccard
          if average == 'weighted':
              weights = MCM[:, 1, 0] + MCM[:, 1, 1]
              if not np.any(weights):
                  # numerator is 0, and warning should have already been issued
                  weights = None
          elif average == 'samples' and sample_weight is not None:
              weights = sample_weight
          else:
              weights = None
          return np.average(jaccard, weights=weights)


      ```

    - precision_recall_fscore_support_from_confusion

      ```python
      @_deprecate_positional_args
      def precision_recall_fscore_support_from_confusion(MCM, *, beta=1.0, average=None,
                                                         warn_for=('precision', 'recall',
                                                                   'f-score'),
                                                         sample_weight=None,
                                                         zero_division="warn"):

          tp_sum = MCM[:, 1, 1]
          pred_sum = tp_sum + MCM[:, 0, 1]
          true_sum = tp_sum + MCM[:, 1, 0]

          if average == 'micro':
              tp_sum = np.array([tp_sum.sum()])
              pred_sum = np.array([pred_sum.sum()])
              true_sum = np.array([true_sum.sum()])

          # Finally, we have all our sufficient statistics. Divide! #
          beta2 = beta ** 2

          # Divide, and on zero-division, set scores and/or warn according to
          # zero_division:
          precision = _prf_divide(tp_sum, pred_sum, 'precision',
                                  'predicted', average, warn_for, zero_division)
          recall = _prf_divide(tp_sum, true_sum, 'recall',
                              'true', average, warn_for, zero_division)

          # warn for f-score only if zero_division is warn, it is in warn_for
          # and BOTH prec and rec are ill-defined
          if zero_division == "warn" and ("f-score",) == warn_for:
              if (pred_sum[true_sum == 0] == 0).any():
                  _warn_prf(
                      average, "true nor predicted", 'F-score is', len(true_sum)
                  )

          # if tp == 0 F will be 1 only if all predictions are zero, all labels are
          # zero, and zero_division=1. In all other case, 0
          if np.isposinf(beta):
              f_score = recall
          else:
              denom = beta2 * precision + recall

              denom[denom == 0.] = 1  # avoid division by 0
              f_score = (1 + beta2) * precision * recall / denom

          # Average the results
          if average == 'weighted':
              weights = true_sum
              if weights.sum() == 0:
                  zero_division_value = np.float64(1.0)
                  if zero_division in ["warn", 0]:
                      zero_division_value = np.float64(0.0)
                  # precision is zero_division if there are no positive predictions
                  # recall is zero_division if there are no positive labels
                  # fscore is zero_division if all labels AND predictions are
                  # negative
                  if pred_sum.sum() == 0:
                      return (zero_division_value,
                              zero_division_value,
                              zero_division_value,
                              None)
                  else:
                      return (np.float64(0.0),
                              zero_division_value,
                              np.float64(0.0),
                              None)

          elif average == 'samples':
              weights = sample_weight
          else:
              weights = None

          if average is not None:
              assert average != 'binary' or len(precision) == 1
              precision = np.average(precision, weights=weights)
              recall = np.average(recall, weights=weights)
              f_score = np.average(f_score, weights=weights)
              true_sum = None  # return no support

          return precision, recall, f_score, true_sum
      ```

      - fbeta_score_from_confusion

        ```python
        @_deprecate_positional_args
        def fbeta_score_from_confusion(MCM, *, beta, average='binary',
                                       sample_weight=None, zero_division="warn"):

            _, _, f, _ = precision_recall_fscore_support_from_confusion(MCM,
                                                                        beta=beta,
                                                                        average=average,
                                                                        warn_for=('f-score',),
                                                                        sample_weight=sample_weight,
                                                                        zero_division=zero_division)
            return f
        ```

      - f1_score_from_confusion

        ```python
        @_deprecate_positional_args
        def f1_score_from_confusion(MCM, *, average='binary', sample_weight=None,
                               zero_division="warn"):

        return fbeta_score_from_confusion(MCM, beta=1, average=average,
                                          sample_weight=sample_weight,
                                          zero_division=zero_division)
        ```

      - precision_score_from_confusion

        ```python
        @_deprecate_positional_args
        def precision_score_from_confusion(MCM, average='binary', sample_weight=None,
                                          zero_division="warn"):

            p, _, _, _ = precision_recall_fscore_support_from_confusion(MCM,
                                                                        average=average,
                                                                        warn_for=('precision',),
                                                                        sample_weight=sample_weight,
                                                                        zero_division=zero_division)
            return p
        ```

      - recall_score_from_confusion

        ```python
        @_deprecate_positional_args
        def recall_score_from_confusion(MCM, average='binary', sample_weight=None,
                                        zero_division="warn"):

            _, r, _, _ = precision_recall_fscore_support_from_confusion(MCM,
                                                                        average=average,
                                                                        warn_for=('recall',),
                                                                        sample_weight=sample_weight,
                                                                        zero_division=zero_division)
            return r
        ```

    - balanced_accuracy_score_from_confusion

      ```python
      @_deprecate_positional_args
      def balanced_accuracy_score_from_confusion(cm, *, adjusted=False):

          with np.errstate(divide='ignore', invalid='ignore'):
              per_class = np.diag(cm) / cm.sum(axis=1)
          if np.any(np.isnan(per_class)):
              warnings.warn('y_pred contains classes not in y_true')
              per_class = per_class[~np.isnan(per_class)]
          score = np.mean(per_class)
          if adjusted:
              n_classes = len(per_class)
              chance = 1 / n_classes
              score -= chance
              score /= 1 - chance
          return score
      ```

- User Guide

  - matthews_corrcoef_from_confusion

    ```python
    def matthews_corrcoef_from_confusion(cm):
    """Compute the Matthews correlation coefficient (MCC) from given confusion matrix.

    Read more in matthews_corrcoef() below.

    Parameters
    ----------

    cm : ndarray of shape (n_classes, n_classes)
         Confusion matrix whose i-th row and j-th
         column entry indicates the number of
         samples with true label being i-th class
         and predicted label being j-th class.

    Returns
    -------
    mcc : float
        The Matthews correlation coefficient (+1 represents a perfect
        prediction, 0 an average random prediction and -1 and inverse
        prediction).

    Examples
    --------
    >>> from sklearn.metrics import matthews_corrcoef_from_confusion
    >>> cm = [[0 1]
              [1 2]]
    >>> matthews_corrcoef_from_confusion(cm)
    -0.33...
    """
    ```

  - jaccard_score_from_confusion

    ```python
    def jaccard_score_from_confusion(MCM, *, average='binary', sample_weight=None,
                                     zero_division="warn"):
    """Jaccard similarity coefficient score from given multilabel confusion matrix.

    Read more in jaccard_score() below.

    Note:
        1. average='sample' can only be used when multilabel confusion matrix is calculated with
           'samplewise' param is set to True.
        2. For binary case, the provided confusion matrix is supposed to be corresponding
           to the right class or sample, otherwise the score would be wrong.

    Parameters
    ----------
    multi_confusion : ndarray of shape (n_outputs, 2, 2)
        A 2x2 confusion matrix corresponding to each output in the input.
        When calculating class-wise multi_confusion (default), then
        n_outputs = n_labels; when calculating sample-wise multi_confusion
        (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
        the results will be returned in the order specified in ``labels``,
        otherwise the results will be returned in sorted order by default.

    average : {None, 'micro', 'macro', 'samples', 'weighted', \
            'binary'}, default='binary'
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification).
        Note: 'sample' can only be used when multilabel confusion matrix is calculated with
              'samplewise' param is set to True

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", {0.0, 1.0}, default="warn"
        Sets the value to return when there is a zero division, i.e. when there
        there are no negative values in predictions and labels. If set to
        "warn", this acts like 0, but a warning is also raised.

    Returns
    -------
    score : float (if average is not None) or array of floats, shape =\
            [n_unique_labels]

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import jaccard_score_from_confusion
    >>> y_true = np.array([[0, 1, 1],
    ...                    [1, 1, 0]])
    >>> y_pred = np.array([[1, 1, 1],
    ...                    [1, 0, 0]])

    In the binary case:

    >>> MCM = multilabel_confusion_matrix(y_true[0], y_pred[0], labels=[1])
    >>> jaccard_score_from_confusion(MCM)
    0.6666...

    In the multilabel case:

    >>> MCM = multilabel_confusion_matrix(y_true, y_pred, samplewise=True)
    >>> jaccard_score_from_confusion(MCM, average='samples')
    0.5833...
    >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
    >>> jaccard_score_from_confusion(MCM, average='macro')
    0.6666...
    >>> jaccard_score_from_confusion(MCM, average=None)
    array([0.5, 0.5, 1. ])

    In the multiclass case:

    >>> y_pred = [0, 2, 1, 2]
    >>> y_true = [0, 1, 2, 2]
    >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
    >>> jaccard_score(MCM, average=None)
    array([1. , 0. , 0.33...])
    """

    ```

  - precision_recall_fscore_support_from_confusion

    ```python
    def precision_recall_fscore_support_from_confusion(MCM, *, beta=1.0, average=None,
                                                       warn_for=('precision', 'recall',
                                                                 'f-score'),
                                                       sample_weight=None,
                                                       zero_division="warn"):
    """Compute precision, recall, F-measure and support for each class from given multilabel confusion matrix.

    Read more in precision_recall_fscore_support() below.

    Note:
        1. sample_weight='sample' can only be used when multilabel confusion matrix is calculated with
           'samplewise' param is set to True.
        2. For binary case, the provided confusion matrix is supposed to be corresponding
           to the right class or sample, otherwise the score would be wrong.

    Parameters
    ----------
    multi_confusion : ndarray of shape (n_outputs, 2, 2)
        A 2x2 confusion matrix corresponding to each output in the input.
        When calculating class-wise multi_confusion (default), then
        n_outputs = n_labels; when calculating sample-wise multi_confusion
        (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
        the results will be returned in the order specified in ``labels``,
        otherwise the results will be returned in sorted order by default.

    beta : float, default=1.0
        The strength of recall versus precision in the F-score.

    average : {'binary', 'micro', 'macro', 'samples','weighted'}, \
            default=None
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
        Note: 'sample' can only be used when multilabel confusion matrix is calculated with
              'samplewise' param is set to True.

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division:
           - recall: when there are no positive labels
           - precision: when there are no positive predictions
           - f-score: both

        If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    recall : float (if average is not None) or array of float, , shape =\
        [n_unique_labels]

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support : None (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_fscore_support_from_confusion
    >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
    >>> precision_recall_fscore_support_from_confusion(MCM, average='macro')
    (0.22..., 0.33..., 0.26..., None)
    >>> precision_recall_fscore_support_from_confusion(MCM, average='micro')
    (0.33..., 0.33..., 0.33..., None)
    >>> precision_recall_fscore_support_from_confusion(MCM, average='weighted')
    (0.22..., 0.33..., 0.26..., None)

    It is possible to compute per-label precisions, recalls, F1-scores and
    supports instead of averaging:

    # Labels' order are declared when contructing multilabel confusion matrix
    >>> MCM = multilabel_confusion_matrix(y_true, y_pred, labels=['pig', 'dog', 'cat'])
    >>> precision_recall_fscore_support_from_confusion(y_true, y_pred, average=None)
    (array([0.        , 0.        , 0.66...]),
     array([0., 0., 1.]), array([0. , 0. , 0.8]),
     array([2, 2, 2]))
    """

    ```

    - fbeta_score_from_confusion

      ```python
      def fbeta_score_from_confusion(MCM, *, beta, average='binary',
                                     sample_weight=None, zero_division="warn"):
      """Compute the F-beta score from the given multilabel confusion matrix.

      Read more in fbeta_score() below.

      Note:
          1. average='sample' can only be used when multilabel confusion matrix is calculated with
            'samplewise' param is set to True.
          2. For binary case, the provided confusion matrix is supposed to be corresponding
            to the right class or sample, otherwise the score would be wrong.

      Parameters
      ----------
      multi_confusion : ndarray of shape (n_outputs, 2, 2)
          A 2x2 confusion matrix corresponding to each output in the input.
          When calculating class-wise multi_confusion (default), then
          n_outputs = n_labels; when calculating sample-wise multi_confusion
          (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
          the results will be returned in the order specified in ``labels``,
          otherwise the results will be returned in sorted order by default.

      beta : float
          Determines the weight of recall in the combined score.

      average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None \
              default='binary'
          This parameter is required for multiclass/multilabel targets.
          If ``None``, the scores for each class are returned. Otherwise, this
          determines the type of averaging performed on the data:

          ``'binary'``:
              Only report results for the class specified by ``pos_label``.
              This is applicable only if targets (``y_{true,pred}``) are binary.
          ``'micro'``:
              Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
          ``'macro'``:
              Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
          ``'weighted'``:
              Calculate metrics for each label, and find their average weighted
              by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an
              F-score that is not between precision and recall.
          ``'samples'``:
              Calculate metrics for each instance, and find their average (only
              meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).
          Note: 'sample' can only be used when multilabel confusion matrix is calculated with
                'samplewise' param is set to True


      sample_weight : array-like of shape (n_samples,), default=None
          Sample weights.

      zero_division : "warn", 0 or 1, default="warn"
          Sets the value to return when there is a zero division, i.e. when all
          predictions and labels are negative. If set to "warn", this acts as 0,
          but warnings are also raised.

      Returns
      -------
      fbeta_score : float (if average is not None) or array of float, shape =\
          [n_unique_labels]
          F-beta score of the positive class in binary classification or weighted
          average of the F-beta score of each class for the multiclass task.

      Examples
      --------
      >>> from sklearn.metrics import fbeta_score_from_confusion
      >>> y_true = [0, 1, 2, 0, 1, 2]
      >>> y_pred = [0, 2, 1, 0, 0, 1]
      >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
      >>> fbeta_score_from_confusion(MCM, average='macro', beta=0.5)
      0.23...
      >>> fbeta_score_from_confusion(MCM, average='micro', beta=0.5)
      0.33...
      >>> fbeta_score_from_confusion(MCM, average='weighted', beta=0.5)
      0.23...
      >>> fbeta_score_from_confusion(MCM, average=None, beta=0.5)
      array([0.71..., 0.        , 0.        ])
      """
      ```

    - f1_score_from_confusion

      ```python
      def f1_score_from_confusion(MCM, *, average='binary', sample_weight=None,
                                  zero_division="warn"):
      """Compute the F1 score from the given multilabel confusion matrix, also known as balanced F-score or F-measure.

      Read more in f1_score() below.

      Note:
          1. average='sample' can only be used when multilabel confusion matrix is calculated with
            'samplewise' param is set to True.
          2. For binary case, the provided confusion matrix is supposed to be corresponding
            to the right class or sample, otherwise the score would be wrong.

      Parameters
      ----------
      multi_confusion : ndarray of shape (n_outputs, 2, 2)
          A 2x2 confusion matrix corresponding to each output in the input.
          When calculating class-wise multi_confusion (default), then
          n_outputs = n_labels; when calculating sample-wise multi_confusion
          (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
          the results will be returned in the order specified in ``labels``,
          otherwise the results will be returned in sorted order by default.

      average : {'micro', 'macro', 'samples','weighted', 'binary'} or None, \
              default='binary'
          This parameter is required for multiclass/multilabel targets.
          If ``None``, the scores for each class are returned. Otherwise, this
          determines the type of averaging performed on the data:

          ``'binary'``:
              Only report results for the class specified by ``pos_label``.
              This is applicable only if targets (``y_{true,pred}``) are binary.
          ``'micro'``:
              Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
          ``'macro'``:
              Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
          ``'weighted'``:
              Calculate metrics for each label, and find their average weighted
              by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an
              F-score that is not between precision and recall.
          ``'samples'``:
              Calculate metrics for each instance, and find their average (only
              meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).
          Note: 'sample' can only be used when multilabel confusion matrix is calculated with
                'samplewise' param is set to True

      sample_weight : array-like of shape (n_samples,), default=None
          Sample weights.

      zero_division : "warn", 0 or 1, default="warn"
          Sets the value to return when there is a zero division, i.e. when all
          predictions and labels are negative. If set to "warn", this acts as 0,
          but warnings are also raised.

      Returns
      -------
      f1_score : float or array of float, shape = [n_unique_labels]
          F1 score of the positive class in binary classification or weighted
          average of the F1 scores of each class for the multiclass task.

      Examples
      --------
      >>> from sklearn.metrics import f1_score_from_confusion
      >>> y_true = [0, 1, 2, 0, 1, 2]
      >>> y_pred = [0, 2, 1, 0, 0, 1]
      >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
      >>> f1_score_from_confusion(MCM, average='macro')
      0.26...
      >>> f1_score_from_confusion(MCM, average='micro')
      0.33...
      >>> f1_score_from_confusion(MCM, average='weighted')
      0.26...
      >>> f1_score_from_confusion(MCM, average=None)
      array([0.8, 0. , 0. ])
      >>> y_true = [0, 0, 0, 0, 0, 0]
      >>> y_pred = [0, 0, 0, 0, 0, 0]
      >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
      >>> f1_score_from_confusion(MCM, zero_division=1)
      1.0...
      """

      ```

    - precision_score_from_confusion

      ```python
      def precision_score_from_confusion(MCM, average='binary', sample_weight=None,
                                         zero_division="warn"):
      """Compute the precision from the given multilabel confusion matrix.

      Read more in precision_score() below.

      Note:
          1. sample_weight='sample' can only be used when multilabel confusion matrix is calculated with
            'samplewise' param is set to True.
          2. For binary case, the provided confusion matrix is supposed to be corresponding
            to the right class or sample, otherwise the score would be wrong.

      Parameters
      ----------
      multi_confusion : ndarray of shape (n_outputs, 2, 2)
          A 2x2 confusion matrix corresponding to each output in the input.
          When calculating class-wise multi_confusion (default), then
          n_outputs = n_labels; when calculating sample-wise multi_confusion
          (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
          the results will be returned in the order specified in ``labels``,
          otherwise the results will be returned in sorted order by default.

      average : {'micro', 'macro', 'samples', 'weighted', 'binary'} \
              default='binary'
          This parameter is required for multiclass/multilabel targets.
          If ``None``, the scores for each class are returned. Otherwise, this
          determines the type of averaging performed on the data:

          ``'binary'``:
              Only report results for the class specified by ``pos_label``.
              This is applicable only if targets (``y_{true,pred}``) are binary.
          ``'micro'``:
              Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
          ``'macro'``:
              Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
          ``'weighted'``:
              Calculate metrics for each label, and find their average weighted
              by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an
              F-score that is not between precision and recall.
          ``'samples'``:
              Calculate metrics for each instance, and find their average (only
              meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).
          Note: 'sample' can only be used when multilabel confusion matrix is calculated with
                'samplewise' param is set to True.

      sample_weight : array-like of shape (n_samples,), default=None
          Sample weights.

      zero_division : "warn", 0 or 1, default="warn"
          Sets the value to return when there is a zero division. If set to
          "warn", this acts as 0, but warnings are also raised.

      Returns
      -------
      precision : float (if average is not None) or array of float of shape
          (n_unique_labels,)
          Precision of the positive class in binary classification or weighted
          average of the precision of each class for the multiclass task.

      Examples
      --------
      >>> from sklearn.metrics import precision_score_from_confusion
      >>> y_true = [0, 1, 2, 0, 1, 2]
      >>> y_pred = [0, 2, 1, 0, 0, 1]
      >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
      >>> precision_score_from_confusion(MCM, average='macro')
      0.22...
      >>> precision_score_from_confusion(MCM, average='micro')
      0.33...
      >>> precision_score_from_confusion(MCM, average='weighted')
      0.22...
      >>> precision_score_from_confusion(MCM, average=None)
      array([0.66..., 0.        , 0.        ])
      >>> y_pred = [0, 0, 0, 0, 0, 0]
      >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
      >>> precision_score_from_confusion(MCM, average=None)
      array([0.33..., 0.        , 0.        ])
      >>> precision_score_from_confusion(MCM, average=None, zero_division=1)
      array([0.33..., 1.        , 1.        ])

      """
      ```

    - recall_score_from_confusion

      ```python
      def recall_score_from_confusion(MCM, average='binary', sample_weight=None,
                                      zero_division="warn"):
      """Compute the recall from the given multilabel confusion matrix.

      Read more in recall_score() below.

      Note:
          1. sample_weight='sample' can only be used when multilabel confusion matrix is calculated with
            'samplewise' param is set to True.
          2. For binary case, the provided confusion matrix is supposed to be corresponding
            to the right class or sample, otherwise the score would be wrong.

      Parameters
      ----------
      multi_confusion : ndarray of shape (n_outputs, 2, 2)
          A 2x2 confusion matrix corresponding to each output in the input.
          When calculating class-wise multi_confusion (default), then
          n_outputs = n_labels; when calculating sample-wise multi_confusion
          (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
          the results will be returned in the order specified in ``labels``,
          otherwise the results will be returned in sorted order by default.

      average : {'micro', 'macro', 'samples', 'weighted', 'binary'} \
              default='binary'
          This parameter is required for multiclass/multilabel targets.
          If ``None``, the scores for each class are returned. Otherwise, this
          determines the type of averaging performed on the data:

          ``'binary'``:
              Only report results for the class specified by ``pos_label``.
              This is applicable only if targets (``y_{true,pred}``) are binary.
          ``'micro'``:
              Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
          ``'macro'``:
              Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
          ``'weighted'``:
              Calculate metrics for each label, and find their average weighted
              by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an
              F-score that is not between precision and recall.
          ``'samples'``:
              Calculate metrics for each instance, and find their average (only
              meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).
          Note: 'sample' can only be used when multilabel confusion matrix is calculated with
                'samplewise' param is set to True.

      sample_weight : array-like of shape (n_samples,), default=None
          Sample weights.

      zero_division : "warn", 0 or 1, default="warn"
          Sets the value to return when there is a zero division. If set to
          "warn", this acts as 0, but warnings are also raised.

      Returns
      -------
      Returns
      -------
      recall : float (if average is not None) or array of float of shape
          (n_unique_labels,)
          Recall of the positive class in binary classification or weighted
          average of the recall of each class for the multiclass task.

      Examples
      --------
      >>> from sklearn.metrics import recall_score_from_confusion
      >>> y_true = [0, 1, 2, 0, 1, 2]
      >>> y_pred = [0, 2, 1, 0, 0, 1]
      >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
      >>> recall_score_from_confusion(MCM, average='macro')
      0.33...
      >>> recall_score_from_confusion(MCM, average='micro')
      0.33...
      >>> recall_score_from_confusion(MCM, average='weighted')
      0.33...
      >>> recall_score_from_confusion(MCM, average=None)
      array([1., 0., 0.])
      >>> y_true = [0, 0, 0, 0, 0, 0]
      >>> MCM = multilabel_confusion_matrix(y_true, y_pred)
      >>> recall_score_from_confusion(MCM, average=None)
      array([0.5, 0. , 0. ])
      >>> recall_score_from_confusion(MCM, average=None, zero_division=1)
      array([0.5, 1. , 1. ])
      """
      ```

  - balanced_accuracy_score_from_confusion

    ```python
    def balanced_accuracy_score_from_confusion(cm, *, adjusted=False):
    """Compute the balanced accuracy from given confusion matrix.

    Read more in balanced_accuracy_score() below.

    Parameters
    ----------
    cm : ndarray of shape (n_classes, n_classes)
         Confusion matrix whose i-th row and j-th
         column entry indicates the number of
         samples with true label being i-th class
         and predicted label being j-th class.

    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, while keeping perfect performance at a score
        of 1.

    Returns
    -------
    balanced_accuracy : float

    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score_from_confusion
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> cm = confusion_matrix(y_true, y_pred)
    >>> balanced_accuracy_score_from_confusion(y_true, y_pred)
    0.625

    """
    ```

- Testing

  - Acceptance Test

  - Regression Test

    Regression Test passed 100% with the original test cases defined in:

    /sklearn/metrics/tests/test\_classification.py

    Thus, the system still meets the requirement specifications.

    ![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a4/Images/19679-RegressionTest.png "Regression Test")

  - Unit Test

    - matthews_corrcoef_from_confusion

      ```python
      @ignore_warnings
      def test_matthews_corrcoef_from_confusion_nan():
          C1 = confusion_matrix([0], [1], sample_weight=None)
          C2 = confusion_matrix([0, 0], [0, 1], sample_weight=None)
          assert matthews_corrcoef_from_confusion(C1) == 0.0
          assert matthews_corrcoef_from_confusion(C2) == 0.0
      

      def test_matthews_corrcoef_from_confusion():
          rng = np.random.RandomState(0)
          y_true = ["a" if i == 0 else "b" for i in rng.randint(0, 2, size=20)]

          # corrcoef of same vectors must be 1
          assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_true, y_true)), 1.0)

          # corrcoef, when the two vectors are opposites of each other, should be -1
          y_true_inv = ["b" if i == "a" else "a" for i in y_true]
          assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_true, y_true_inv)), -1)

          y_true_inv2 = label_binarize(y_true, classes=["a", "b"])
          y_true_inv2 = np.where(y_true_inv2, 'a', 'b')
          assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_true, y_true_inv2)), -1)

          # For the zero vector case, the corrcoef cannot be calculated and should
          # result in a RuntimeWarning
          mcc = assert_warns_div0(matthews_corrcoef_from_confusion, confusion_matrix([0,0,0,0], [0,0,0,0]))

          # But will output 0
          assert_almost_equal(mcc, 0.)

          # And also for any other vector with 0 variance
          mcc = assert_warns_div0(matthews_corrcoef_from_confusion, confusion_matrix(y_true, ['a'] * len(y_true)))

          # But will output 0
          assert_almost_equal(mcc, 0.)

          # These two vectors have 0 correlation and hence mcc should be 0
          y_1 = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
          y_2 = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
          assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_1, y_2)), 0.)

          # Check that sample weight is able to selectively exclude
          mask = [1] * 10 + [0] * 10
          # Now the first half of the vector elements are alone given a weight of 1
          # and hence the mcc will not be a perfect 0 as in the previous case
          with pytest.raises(AssertionError):
              assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_1, y_2,
                                                    sample_weight=mask)), 0.)


      def test_matthews_corrcoef_from_confusion_multiclass():
          rng = np.random.RandomState(0)
          ord_a = ord('a')
          n_classes = 4
          y_true = [chr(ord_a + i) for i in rng.randint(0, n_classes, size=20)]

          # corrcoef of same vectors must be 1
          assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_true, y_true)), 1.0)

          # with multiclass > 2 it is not possible to achieve -1
          y_true = [0, 0, 1, 1, 2, 2]
          y_pred_bad = [2, 2, 0, 0, 1, 1]
          assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_true, y_pred_bad)), -.5)

          # Maximizing false positives and negatives minimizes the MCC
          # The minimum will be different for depending on the input
          y_true = [0, 0, 1, 1, 2, 2]
          y_pred_min = [1, 1, 0, 0, 0, 0]
          assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_true, y_pred_min)),
                              -12 / np.sqrt(24 * 16))

          # Zero variance will result in an mcc of zero and a Runtime Warning
          y_true = [0, 1, 2]
          y_pred = [3, 3, 3]
          mcc = assert_warns_message(RuntimeWarning, 'invalid value encountered',
                                    matthews_corrcoef_from_confusion, confusion_matrix(y_true, y_pred))
          assert_almost_equal(mcc, 0.0)

          # These two vectors have 0 correlation and hence mcc should be 0
          y_1 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
          y_2 = [1, 1, 1, 2, 2, 2, 0, 0, 0]
          assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_1, y_2)), 0.)

          # We can test that binary assumptions hold using the multiclass computation
          # by masking the weight of samples not in the first two classes

          # Masking the last label should let us get an MCC of -1
          y_true = [0, 0, 1, 1, 2]
          y_pred = [1, 1, 0, 0, 2]
          sample_weight = [1, 1, 1, 1, 0]
          assert_almost_equal(matthews_corrcoef_from_confusion(confusion_matrix(y_true, y_pred,
                                                sample_weight=sample_weight)), -1)

          # For the zero vector case, the corrcoef cannot be calculated and should
          # result in a RuntimeWarning
          y_true = [0, 0, 1, 2]
          y_pred = [0, 0, 1, 2]
          sample_weight = [1, 1, 0, 0]
          mcc = assert_warns_message(RuntimeWarning, 'invalid value encountered',
                                    matthews_corrcoef_from_confusion, confusion_matrix(y_true, y_pred,
                                              sample_weight=sample_weight))

          # But will output 0
          assert_almost_equal(mcc, 0.)

      ```

    - jaccard_score_from_confusion

      ```python
      def test_multilabel_jaccard_score_from_confusion(recwarn):
          # multilabel case
          y1 = np.array([[0, 1, 1], [1, 0, 1]])
          y2 = np.array([[0, 0, 1], [1, 0, 1]])
          MCM1 = multilabel_confusion_matrix(y1, y2, samplewise=True)
          MCM2 = multilabel_confusion_matrix(y1, y1, samplewise=True)
          MCM3 = multilabel_confusion_matrix(y2, y2, samplewise=True)
          assert jaccard_score_from_confusion(MCM1, average='samples') == 0.75
          assert jaccard_score_from_confusion(MCM2, average='samples') == 1
          assert jaccard_score_from_confusion(MCM3, average='samples') == 1

          y_true = np.array([[0, 1, 1], [1, 0, 0]])
          y_pred = np.array([[1, 1, 1], [1, 0, 1]])

          # average='samples'
          MCM = multilabel_confusion_matrix(y_true, y_pred, samplewise=True)
          assert_almost_equal(jaccard_score_from_confusion(MCM, average='samples'),
                              7. / 12)
      

      def test_average_binary_jaccard_score_from_confusion(recwarn):
          # tp=0, fp=0, fn=1, tn=0
          MCM = multilabel_confusion_matrix([1], [0])
          assert jaccard_score_from_confusion(MCM, average='binary') == 0.
          y_true = np.array([1, 0, 1, 1, 0])
          y_pred = np.array([1, 0, 1, 1, 1])
          MCM = multilabel_confusion_matrix(y_true, y_pred, labels=[1])
          assert_almost_equal(jaccard_score_from_confusion(MCM,
                                            average='binary'), 3. / 4)

          assert not list(recwarn)
      

      def test_jaccard_score_from_division_zero_division_warning():
          # check that we raised a warning with default behavior if a zero division
          # happens
          y_true = np.array([[1, 0, 1], [0, 0, 0]])
          y_pred = np.array([[0, 0, 0], [0, 0, 0]])
          msg = ('Jaccard is ill-defined and being set to 0.0 in '
                'samples with no true or predicted labels.'
                ' Use `zero_division` parameter to control this behavior.')
          with pytest.warns(UndefinedMetricWarning, match=msg):
              score = jaccard_score_from_confusion(
                  multilabel_confusion_matrix(y_true, y_pred), average='samples', zero_division='warn'
              )
              assert score == pytest.approx(0.0)
      

      @pytest.mark.parametrize(
          "zero_division, expected_score", [(0, 0), (1, 0.5)]
      )
      def test_jaccard_score_from_confusion_zero_division_set_value(zero_division, expected_score):
          # check that we don't issue warning by passing the zero_division parameter
          y_true = np.array([[1, 0, 1], [0, 0, 0]])
          y_pred = np.array([[0, 0, 0], [0, 0, 0]])
          with pytest.warns(None) as record:
              score = jaccard_score_from_confusion(
                  multilabel_confusion_matrix(y_true, y_pred, samplewise=True), average="samples", zero_division=zero_division
              )
          assert score == pytest.approx(expected_score)
          assert len(record) == 0
      ```

    - precision_recall_fscore_support_from_confusion

      ```python
      def test_precision_recall_f1_score_from_confusion_binary():
          # Test Precision Recall and F1 Score for binary classification task
          y_true, y_pred, _ = make_prediction(binary=True)

          # detailed measures for each class
          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred), average=None)
          assert_array_almost_equal(p, [0.73, 0.85], 2)
          assert_array_almost_equal(r, [0.88, 0.68], 2)
          assert_array_almost_equal(f, [0.80, 0.76], 2)
          assert_array_equal(s, [25, 25])


      def test_precision_recall_f1_score_from_confusion_multiclass():
          # Test Precision Recall and F1 Score for multiclass classification task
          y_true, y_pred, _ = make_prediction(binary=False)

          # compute scores with default labels introspection
          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred), average=None)
          assert_array_almost_equal(p, [0.83, 0.33, 0.42], 2)
          assert_array_almost_equal(r, [0.79, 0.09, 0.90], 2)
          assert_array_almost_equal(f, [0.81, 0.15, 0.57], 2)
          assert_array_equal(s, [24, 31, 20])

          # same prediction but with and explicit label ordering
          p, r, f, s = precision_recall_fscore_support_from_confusion(
              multilabel_confusion_matrix(y_true, y_pred, labels=[0, 2, 1]), average=None)
          assert_array_almost_equal(p, [0.83, 0.41, 0.33], 2)
          assert_array_almost_equal(r, [0.79, 0.90, 0.10], 2)
          assert_array_almost_equal(f, [0.81, 0.57, 0.15], 2)
          assert_array_equal(s, [24, 20, 31])


      @pytest.mark.parametrize('average',
                               ['samples', 'micro', 'macro', 'weighted', None])
      def test_precision_refcall_f1_score_from_confusion_multilabel_unordered_labels(average):
          # test that labels need not be sorted in the multilabel case
          y_true = np.array([[1, 1, 0, 0]])
          y_pred = np.array([[0, 0, 1, 1]])
          p, r, f, s = precision_recall_fscore_support_from_confusion(
              multilabel_confusion_matrix(y_true, y_pred, labels=[3, 0, 1, 2]), warn_for=[], average=average)
          assert_array_equal(p, 0)
          assert_array_equal(r, 0)
          assert_array_equal(f, 0)
          if average is None:
              assert_array_equal(s, [0, 1, 1, 0])
      

      def test_precision_recall_f1_score_from_confusion_binary_averaged():
          y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
          y_pred = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1])

          # compute scores with default labels introspection
          ps, rs, fs, _ = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                          average=None)
          p, r, f, _ = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                      average='macro')
          assert p == np.mean(ps)
          assert r == np.mean(rs)
          assert f == np.mean(fs)
          p, r, f, _ = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                      average='weighted')
          support = np.bincount(y_true)
          assert p == np.average(ps, weights=support)
          assert r == np.average(rs, weights=support)
          assert f == np.average(fs, weights=support)


      @ignore_warnings
      def test_precision_recall_f1_score_from_confusion_multilabel_1():
          # Test precision_recall_f1_score on a crafted multilabel example
          # First crafted example

          y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
          y_pred = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]])

          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred), average=None)

          # Check per class

          assert_array_almost_equal(p, [0.0, 0.5, 1.0, 0.0], 2)
          assert_array_almost_equal(r, [0.0, 1.0, 1.0, 0.0], 2)
          assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
          assert_array_almost_equal(s, [1, 1, 1, 1], 2)

          # Check macro
          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                      average="macro")
          assert_almost_equal(p, 1.5 / 4)
          assert_almost_equal(r, 0.5)
          assert_almost_equal(f, 2.5 / 1.5 * 0.25)
          assert s is None

          # Check micro
          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                      average="micro")
          assert_almost_equal(p, 0.5)
          assert_almost_equal(r, 0.5)
          assert_almost_equal(f, 0.5)
          assert s is None

          # Check weighted
          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                      average="weighted")
          assert_almost_equal(p, 1.5 / 4)
          assert_almost_equal(r, 0.5)
          assert_almost_equal(f, 2.5 / 1.5 * 0.25)
          assert s is None

          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred, samplewise=True),
                                                      average="samples")
          assert_almost_equal(p, 0.5)
          assert_almost_equal(r, 0.5)
          assert_almost_equal(f, 0.5)
          assert s is None


      @ignore_warnings
      @pytest.mark.parametrize('zero_division', ["warn", 0, 1])
      def test_precision_recall_f1_score_from_confusion_with_an_empty_prediction(zero_division):
          y_true = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])
          y_pred = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])

          zero_division = 1.0 if zero_division == 1.0 else 0.0
          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                      average=None,
                                                      zero_division=zero_division)
          assert_array_almost_equal(p, [zero_division, 1.0, 1.0, 0.0], 2)
          assert_array_almost_equal(r, [0.0, 0.5, 1.0, zero_division], 2)
          assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
          assert_array_almost_equal(s, [1, 2, 1, 0], 2)

          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                      average="macro",
                                                      zero_division=zero_division)
          assert_almost_equal(p, (2 + zero_division) / 4)
          assert_almost_equal(r, (1.5 + zero_division) / 4)
          assert_almost_equal(f, 2.5 / (4 * 1.5))
          assert s is None

          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                      average="micro",
                                                      zero_division=zero_division)
          assert_almost_equal(p, 2 / 3)
          assert_almost_equal(r, 0.5)
          assert_almost_equal(f, 2 / 3 / (2 / 3 + 0.5))
          assert s is None

          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred),
                                                      average="weighted",
                                                      zero_division=zero_division)
          assert_almost_equal(p, 3 / 4 if zero_division == 0 else 1.0)
          assert_almost_equal(r, 0.5)
          assert_almost_equal(f, (2 / 1.5 + 1) / 4)
          assert s is None

          p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred, samplewise=True),
                                                      average="samples")

          assert_almost_equal(p, 1 / 3)
          assert_almost_equal(r, 1 / 3)
          assert_almost_equal(f, 1 / 3)
          assert s is None


      @pytest.mark.parametrize('beta', [1])
      @pytest.mark.parametrize('average', ["macro", "micro", "weighted", "samples"])
      @pytest.mark.parametrize('zero_division', [0, 1])
      def test_precision_recall_f1_from_confusion_no_labels(beta, average, zero_division):
          y_true = np.zeros((20, 3))
          y_pred = np.zeros_like(y_true)

          p, r, f, s = assert_no_warnings(precision_recall_fscore_support_from_confusion, multilabel_confusion_matrix(y_true, y_pred), average=average, beta=beta,
                                          zero_division=zero_division)

          zero_division = float(zero_division)
          assert_almost_equal(p, zero_division)
          assert_almost_equal(r, zero_division)
          assert_almost_equal(f, zero_division)
          assert s is None
      

      @pytest.mark.parametrize('average', ["macro", "micro", "weighted", "samples"])
      def test_precision_recall_f1_from_confusion_no_labels_check_warnings(average):
          y_true = np.zeros((20, 3))
          y_pred = np.zeros_like(y_true)

          samplewise = average == 'samples'
          with pytest.warns(UndefinedMetricWarning):
              p, r, f, s = precision_recall_fscore_support_from_confusion(multilabel_confusion_matrix(y_true, y_pred, samplewise=samplewise), average=average, beta=1.0)

          assert_almost_equal(p, 0)
          assert_almost_equal(r, 0)
          assert_almost_equal(f, 0)
          assert s is None
      

      @pytest.mark.parametrize('zero_division', [0, 1])
      def test_precision_recall_f1_from_confusion_no_labels_average_none(zero_division):
          y_true = np.zeros((20, 3))
          y_pred = np.zeros_like(y_true)

          p, r, f, s = assert_no_warnings(precision_recall_fscore_support_from_confusion, multilabel_confusion_matrix(y_true, y_pred), average=None, beta=1.0,
                                          zero_division=zero_division)

          zero_division = float(zero_division)
          assert_array_almost_equal(
              p, [zero_division, zero_division, zero_division], 2
          )
          assert_array_almost_equal(
              r, [zero_division, zero_division, zero_division], 2
          )
          assert_array_almost_equal(
              f, [zero_division, zero_division, zero_division], 2
          )
          assert_array_almost_equal(s, [0, 0, 0], 2)
      

      def test_precision_recall_f1_from_confusion_no_labels_average_none_warn():
          y_true = np.zeros((20, 3))
          y_pred = np.zeros_like(y_true)

          with pytest.warns(UndefinedMetricWarning):
              p, r, f, s = precision_recall_fscore_support_from_confusion(
                  multilabel_confusion_matrix(y_true, y_pred), average=None, beta=1
              )

          assert_array_almost_equal(p, [0, 0, 0], 2)
          assert_array_almost_equal(r, [0, 0, 0], 2)
          assert_array_almost_equal(f, [0, 0, 0], 2)
          assert_array_almost_equal(s, [0, 0, 0], 2)
      ```

    - balanced_accuracy_score_from_confusion

      ```python
      def test_balanced_accuracy_score_from_confusion_unseen():
          assert_warns_message(UserWarning, 'y_pred contains classes not in y_true',
                               balanced_accuracy_score_from_confusion, confusion_matrix([0, 0, 0], [0, 0, 1]))
      
      
      @pytest.mark.parametrize('y_true,y_pred',
                              [
                                  (['a', 'b', 'a', 'b'], ['a', 'a', 'a', 'b']),
                                  (['a', 'b', 'c', 'b'], ['a', 'a', 'a', 'b']),
                                  (['a', 'a', 'a', 'b'], ['a', 'b', 'c', 'b']),
                              ])
      def test_balanced_accuracy_score_from_confusion(y_true, y_pred):
          macro_recall = recall_score(y_true, y_pred, average='macro',
                                      labels=np.unique(y_true))
          with ignore_warnings():
              # Warnings are tested in test_balanced_accuracy_score_unseen
              balanced = balanced_accuracy_score_from_confusion(confusion_matrix(y_true, y_pred))
          assert balanced == pytest.approx(macro_recall)
          adjusted = balanced_accuracy_score_from_confusion(confusion_matrix(y_true, y_pred), adjusted=True)
          chance = balanced_accuracy_score_from_confusion(confusion_matrix(y_true, np.full_like(y_true, y_true[0])))
          assert adjusted == (balanced - chance) / (1 - chance)
      ```

## Work Log
