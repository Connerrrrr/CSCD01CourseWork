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
    ```

  - jaccard_score_from_confusion

    ```python
    ```

  - precision_recall_fscore_support_from_confusion

    ```python
        
    ```

    - fbeta_score_from_confusion

      ```python
      ```

    - f1_score_from_confusion

      ```python
      ```

    - precision_score_from_confusion

      ```python
      ```

    - recall_score_from_confusion

      ```python
      ```

  - balanced_accuracy_score_from_confusion

    ```python
    ```

- Testing

  - Acceptance Test

  - Regression Test

  - Unit Test

## Work Log
