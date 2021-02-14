# CSCD01W21 Assignment 1

## Overall Architecture

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/sklearnFlowchart.svg "Overall Flow Diagram")

All modules in sklearn can be categorized as above. Users will first use some modules to analyze the input data sets and then do a preprocessing to encode, formalize and simplify input data sets. Then users will do some feature engineering to extract and select some useful features. Most importantly, sklearn provides various kinds of machine learning models over all three types of learnings (supervised, semi-supervised and unsupervised). It also provides some model selection modules to select a good model for classification or regression. Finally, users are able to review the selected models and then make a choice to reselect another model or output the result. Throughout the whole process, some other modules like base, utils, expectations and some external libraries such as numpy, scipy and cython are also frequently used.

--------

### Analysis

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/Analysis.svg "Analysis")

Scikit learn provides a few packages to perform data analysis such as covariance estimation, signals decomposition in components, cross decomposition, and imputation of missing values.

Because many statistical problems require the estimation of a population’s covariance matrix, and there are variants of covariance. The sklearn.covariance package provides a few types of covariance classes like Empirical Covariance, Shrunk Covariance, Sparse Inverse Covariance, and Robust Covariance. They have an inheritance relationship shown in the image above as using inheritance allows different covariance classes to share common attributes and methods.

In the real world, many datasets contain missing values, and those missing values may be valuable even so. In view of this problem, scikit learn comes up with a way to infer the missing values which is imputer. There are three imputers, Simple Imputer, Iterative Imputer, KNN Imputer. The imputers transform the datasets into being incompatible with scikit-learn estimators.

When it comes to a problem where a matrix contains too many zeros, scikit learn provides the decomposition package to reduce a matrix into constituent parts that make it easier to calculate more complex matrix operations. Same as covariance, there are a variety of decompositions like dictionary learning, PCA, ICA, etc. Some python classes can be categorized into these definitions. Cross decomposition is similar to decomposition in general.

--------

### Preprocessing

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/Preprocessing.svg "Preprocessing")

--------

### Feaure Engineering

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/FeatureEngineering.svg "Feature Engineering")

The **_feature extraction_** module can be used to extract features in a format supported by machine learning algorithms from datasets consisting of formats such as text and image. Where **_PatchExtractor_** can extract patches from a collection of images, **_Vectorizers_** and the **_Transformer_** are used to transform list of feature-values into vectors and **_FeatureHasher_** will turn sequences of feature names into matrices.

The **_feature selection_** module can be used for feature selection/dimensionality reduction on sample sets, either to improve estimators’ accuracy scores or to boost their performance on very high-dimensional datasets. Note that **_RFE_** and **_RFECV_** are for ranking features while other classes are for selecting features.

--------

### Modeling

The modeling module is the place to train the dataset, which is the most exciting part in machine learning, and consists of two main sections, i.e. supervised models and unsupervised & semi-supervised models. They both inherit from **_BaseEstimator_** class and include various **_Mixins_**, which can be viewed as interfaces with implemented methods. The Mixin pattern is an example of enforcing the **_dependency inversion principle_**, achieving the **_low coupling_**.

#### Supervised Models

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/SupervisedModel.svg "Supervised Learning Models")

Supervised models are the most common sub-branch of machine learning models today. They are designed to learn by labeled examples. The notion of “supervised” is from the idea that training the model is like having a teacher supervise the whole process. The major packages and classes that are commonly used are identified below:

1. naive_bayes

    * GaussianNB(Gaussian Naive Bayes)
2. svm
    * SVC(Support Vector Machine Classifier)
3. neural_network
    * MLPClassifier(Multilayer Perceptron Classifier)
4. linear_model
    * LogisticRegression
    * SGD(Stochastic Gradient Descent)
    * Perceptron
5. tree
    * DecisionTree
6. ensemble
    * Voting
    * RandomForest
    * Stacking
    * AdaBoost
    * GradientBoosting

#### Semi-supervised & Unsupervised Models

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/Unsupervised%26Semi-supervised%20Model.svg "Unsupervised and Semi-supervised learning Models")

In contrast to supervised models, datasets for unsupervised models do not contain labels, leaving models to identify the pattern on their own. Unsupervised models are able to learn on their own based on their algorithms, without being explicitly informed if what they are doing is correct. Meanwhile, indicated by its name, semi-supervised models lie between supervised and unsupervised models. Many machine learning researchers have found that using unlabeled data in combination with small amounts of labeled data can significantly improve learning accuracy.

1. neighbors
    * KNN(K-Nearest Neighbors)
2. neural_network
    * RBM(Restricted Boltzmann Machine)
3. cluster
    * KMeans
4. mixture
    * GaussianMixture

--------

### Model Selection

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/model_selection.svg "Model Selection")

For model selection, it focuses on two main purposes: evaluating estimator performance and tuning the hyper-parameters. This section also provided a learning curve for the validation and training score for the estimator.

First, in order to evaluate estimators, scikit-learn uses cross validation to approach this goal. It split the dataset into two parts, training set and test data. The class under **_BaseShuffleSplit_** are all different types of splits for different situations. The main function _train\_test\_split_ that was implemented inside **_ShuffleSplit_** can only perform strastratified splitting. That is why scikit-learn implemented a **_GroupShuffleSplit_** class to do the group split. The other main class would be KFold, which is also a classic cross validation technique.

Secondly would be tuning the hyper-parameters, which is also known as the parameters that is not learnt within estimators. In scikit-learn, it uses **_GridSearchCV_** for all combinations, and **_RandomizedSearchCV_** can sample a given value with a specific distribution. Both of the classes can cooperate with **_HalvingGridSearchCV_** and perform much faster at finding the combination.

--------

### Model Review

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/ModelReview.svg "Model Review")

Model Review’s main function would be to evaluate different estimator’s performance and give them a score. can be separated into two part metrics package and inspection package, where metrics folder, it records the score and uses the metrics function to predict possible error outcome. For the inspection package, it contains two main functions, this package did not contain any classes, however, these two functions are very important for model review.

First, the metrics package, it uses different metrics functions to plot graphs for the scores of the estimator.  The **_accuracy\_score_** function will compute the accuracy of either the fraction or predictions. The **_top\_k\_accuracy\_score_** function would be the generalization version of **_accuracy\_score_**. **_ConfusionMatrixDisplay_** class would evaluate classification accuracy of confusion matrix and **_plot\_confusion_matrix_** in this class would plot the confusion matrix visually.

Second the inspection package, it did not contain any classes but rather two useful functions for model review. The first one is **_partial\_dependence_**, which is a tool to dependence of features that belongs to the average estimator, the second one is **_permutation\_importance_** which would calculate the difference between baseline metric and metric that was created from permuting the feature column.

--------

### Others

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/Others.svg "Others")

Modules in this section including the three external libraries numpy, cython and scipy are commonly used everywhere in sklearn.

There is a base.py file in the home directory of sklearn package, it defines various **_Mixin_** classes that are used as base classes for this project.

The **_compose_** module contains meta-estimators for building composite models with transformers.

The **_utils_** package includes various utilities, where **_Buch_** is a subclass of python dictionary that exposing keys as attributes, **_MissingValues_** is a data class for missing data information, **_deprecated_** is a decorator to mark a function or class as deprecated and **_loguniform_** is class supporting log-uniform random variables which inherent from scipy.stats.reciprocal.

## Design Patterns

### Strategy Pattern

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/Strategy_Sequence_Diagram.svg "Sequence Diagram of Strategy")

The strategy pattern enables selecting an algorithm at runtime. Instead of implementing a single algorithm directly, code receives run-time instructions as to which in a family of algorithms to use. Since scikit-learn is a python machine learning library, which has a lot of different models with the same function signature, and the algorithms in those functions are different as well, therefore there are numerous cases where scikit-learn uses the strategy pattern. Like the image shown above, Class “LinearModel” is an abstract base class with an abstract method fit(). All its child classes has different implementations of fit().

#### Code Example

#### Class LinearModel: from sklearn/linear_model/_base.py

#### Class LinearRegression: from sklearn/linear_model/_base.py

#### Class BayesianRidge: from sklearn/linear_model/_baes.py

```python
class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""
```

```python
class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    def fit(self, X, y, sample_weight=None):
        _normalize = _deprecate_normalize(
            self.normalize, default=False,
            estimator_name=self.__class__.__name__
        )

        n_jobs_ = self.n_jobs

        accept_sparse = False if self.positive else ['csr', 'csc', 'coo']

        X, y = self._validate_data(X, y, accept_sparse=accept_sparse,
                                   y_numeric=True, multi_output=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=_normalize,
            copy=self.copy_X, sample_weight=sample_weight,
            return_mean=True)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if self.positive:
            if y.ndim < 2:
                self.coef_, self._residues = optimize.nnls(X, y)
            else:
                # scipy.optimize.nnls cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                        delayed(optimize.nnls)(X, y[:, j])
                        for j in range(y.shape[1]))
                self.coef_, self._residues = map(np.vstack, zip(*outs))
        elif sp.issparse(X):
            X_offset_scale = X_offset / X_scale

            def matvec(b):
                return X.dot(b) - b.dot(X_offset_scale)

            def rmatvec(b):
                return X.T.dot(b) - X_offset_scale * np.sum(b)

            X_centered = sparse.linalg.LinearOperator(shape=X.shape,
                                                      matvec=matvec,
                                                      rmatvec=rmatvec)

            if y.ndim < 2:
                out = sparse_lsqr(X_centered, y)
                self.coef_ = out[0]
                self._residues = out[3]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(sparse_lsqr)(X_centered, y[:, j].ravel())
                    for j in range(y.shape[1]))
                self.coef_ = np.vstack([out[0] for out in outs])
                self._residues = np.vstack([out[3] for out in outs])
        else:
            self.coef_, self._residues, self.rank_, self.singular_ = \
                linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self
```

```python
class BayesianRidge(RegressorMixin, LinearModel):
    def fit(self, X, y, sample_weight=None):
        if self.n_iter < 1:
            raise ValueError('n_iter should be greater than or equal to 1.'
                             ' Got {!r}.'.format(self.n_iter))

        X, y = self._validate_data(X, y, dtype=np.float64, y_numeric=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        X, y, X_offset_, y_offset_, X_scale_ = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_
        n_samples, n_features = X.shape

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero
        alpha_ = self.alpha_init
        lambda_ = self.lambda_init
        if alpha_ is None:
            alpha_ = 1. / (np.var(y) + eps)
        if lambda_ is None:
            lambda_ = 1.

        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        U, S, Vh = linalg.svd(X, full_matrices=False)
        eigen_vals_ = S ** 2

        # Convergence loop of the bayesian ridge regression
        for iter_ in range(self.n_iter):

            # update posterior mean coef_ based on alpha_ and lambda_ and
            # compute corresponding rmse
            coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
                                              XT_y, U, Vh, eigen_vals_,
                                              alpha_, lambda_)
            if self.compute_score:
                # compute the log marginal likelihood
                s = self._log_marginal_likelihood(n_samples, n_features,
                                                  eigen_vals_,
                                                  alpha_, lambda_,
                                                  coef_, rmse_)
                self.scores_.append(s)

            # Update alpha and lambda according to (MacKay, 1992)
            gamma_ = np.sum((alpha_ * eigen_vals_) /
                            (lambda_ + alpha_ * eigen_vals_))
            lambda_ = ((gamma_ + 2 * lambda_1) /
                       (np.sum(coef_ ** 2) + 2 * lambda_2))
            alpha_ = ((n_samples - gamma_ + 2 * alpha_1) /
                      (rmse_ + 2 * alpha_2))

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.n_iter_ = iter_ + 1

        # return regularization parameters and corresponding posterior mean,
        # log marginal likelihood and posterior covariance
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
                                               XT_y, U, Vh, eigen_vals_,
                                               alpha_, lambda_)
        if self.compute_score:
            # compute the log marginal likelihood
            s = self._log_marginal_likelihood(n_samples, n_features,
                                              eigen_vals_,
                                              alpha_, lambda_,
                                              coef_, rmse_)
            self.scores_.append(s)
            self.scores_ = np.array(self.scores_)

        # posterior covariance is given by 1/alpha_ * scaled_sigma_
        scaled_sigma_ = np.dot(Vh.T,
                               Vh / (eigen_vals_ +
                                     lambda_ / alpha_)[:, np.newaxis])
        self.sigma_ = (1. / alpha_) * scaled_sigma_

        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self
```

As you can see, both Class LinearRegression and BayesianRidge are the child classes of Class LinearModel, and their algorithm for function fit are completely different. This is a just small subset of large strategy design pattern usage.

### Dependency Injection Pattern

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/Dependency_injection_UML.svg "UML Diagram of Dependency Injection")

--------

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/Dependency_Injection_Sequence_Diagram.svg "Sequence Diagram of Dependency Injection")

Dependency injection plays a very important role when implementing IoC(Inversion of Control). The main purpose of dependency injection is to create the dependents objects outside the class and provide this object to a class through different ways. Dependency injection can broadly be separated into three different types, which are Constructor injection, Property Injection and Method Injection. Constructor injection will go through a constructor, Property injection will go through a property and method injection will go through a method. As our team read through the code of scikit-learn, we found out that there are lots of places that have been implemented using Dependency injection. Most of their codes are using Constructor injection.

#### Code Example

#### Class GirdSearchCV: from sklearn/model_selection/_search.py

#### Class RandomizedSearchCV: from sklearn/model_selection/_search.py

#### Class HavingGirdSearchCV: from sklearn/model_selection/_search_successive_halving.py

#### Class BaseEstimator: from sklearn/base.py

```python
class GridSearchCV(BaseSearchCV):
    _required_parameters = ["estimator", "param_grid"]

    @_deprecate_positional_args
    def __init__(self, estimator, param_grid, *, scoring=None,
                 n_jobs=None, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)
    
    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))
```

```python
class RandomizedSearchCV(BaseSearchCV):
    _required_parameters = ["estimator", "param_distributions"]
 
    @_deprecate_positional_args
    def __init__(self, estimator, param_distributions, *, n_iter=10,
                    scoring=None, n_jobs=None, refit=True,
                    cv=None, verbose=0, pre_dispatch='2*n_jobs',
                    random_state=None, error_score=np.nan,
                    return_train_score=False):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
    
    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        evaluate_candidates(ParameterSampler(
            self.param_distributions, self.n_iter,
            random_state=self.random_state))
```

```python
class HalvingGridSearchCV(BaseSuccessiveHalving):
    _required_parameters = ["estimator", "param_grid"]
 
    def __init__(self, estimator, param_grid, *,
                    factor=3, resource='n_samples', max_resources='auto',
                    min_resources='exhaust', aggressive_elimination=False,
                    cv=5, scoring=None, refit=True, error_score=np.nan,
                    return_train_score=True, random_state=None, n_jobs=None,
                    verbose=0):
        super().__init__(estimator, scoring=scoring,
                            n_jobs=n_jobs, refit=refit, verbose=verbose, cv=cv,
                            random_state=random_state, error_score=error_score,
                            return_train_score=return_train_score,
                            max_resources=max_resources, resource=resource,
                            factor=factor, min_resources=min_resources,
                            aggressive_elimination=aggressive_elimination)
        self.param_grid = param_grid
        _check_param_grid(self.param_grid)
    
    def _generate_candidate_params(self):
        return ParameterGrid(self.param_grid)
```

```python
class BaseEstimator:
    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
    
        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                    'Check the list of available parameters '
                                    'with `estimator.get_params().keys()`.' %
                                    (key, self))
    
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
    
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
    
        return self
```

As we can see from the above code example,  we can see that both GridSearchCV, and RandomizeSearchCV’s constructor in \_search.py takes an estimator like SVC in base.py as parameter. Also, HavingRandomSearchCV’s constructor in \_search\_successive_halving.py also takes an estimator as parameter. They all have imported it in the \_\_init\_\_ method.

### Iterator Pattern

![alt text](https://github.com/UTSCCSCD01/course-project-apple_team/blob/master/a1/Images/Iterator_Sequence_Diagram.svg "Sequence Diagram of Iterator")

Iterator is a behavioral design pattern that allows sequential traversal through a complex data structure without exposing its internal details. The containers must offer an *\_\_iter\_\_()* method that returns an iterator object. Supporting this method makes the containers **iterable**. Above are some examples of containers that are iterable from sklearn (BaseEnsemble, ParameterGrid, ParameterSampler).

#### Code example -- BaseEnsemble: /sklearn/ensemble/_base.py

```python
   def __iter__(self):
       """Return iterator over estimators in the ensemble."""
       return iter(self.estimators_)
```

The *iter()* function returns an iterator object.

#### Code example -- ParameterGrid: /sklearn/model_selection/_search.py

```python
def __iter__(self):
    """Iterate over the points in the grid.

    Returns
    -------
    params : iterator over dict of str to any
        Yields dictionaries mapping each estimator parameter to one of its
        allowed values.
    """
    for p in self.param_grid:
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(p.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params
```

The yield statement suspends function’s execution and sends a value back to the caller, but retains enough state to enable function to resume where it is left off. When resumed, the function continues execution immediately after the last yield run. This allows its code to produce a series of values over time, rather than computing them at once and sending them back like a list.

#### Code example -- ParameterSampler: /sklearn/model_selection/_search.py

```python
def __iter__(self):
    rng = check_random_state(self.random_state)

    # if all distributions are given as lists, we want to sample without
    # replacement
    if self._is_all_lists():
        # look up sampled parameter settings in parameter grid
        param_grid = ParameterGrid(self.param_distributions)
        grid_size = len(param_grid)
        n_iter = self.n_iter

        if grid_size < n_iter:
            warnings.warn(
                'The total space of parameters %d is smaller '
                'than n_iter=%d. Running %d iterations. For exhaustive '
                'searches, use GridSearchCV.'
                % (grid_size, self.n_iter, grid_size), UserWarning)
            n_iter = grid_size
        for i in sample_without_replacement(grid_size, n_iter,
                                            random_state=rng):
            yield param_grid[i]

    else:
        for _ in range(self.n_iter):
            dist = rng.choice(self.param_distributions)
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(dist.items())
            params = dict()
            for k, v in items:
                if hasattr(v, "rvs"):
                    params[k] = v.rvs(random_state=rng)
                else:
                    params[k] = v[rng.randint(len(v))]
            yield params
```

With an iterator design pattern, clients can go over elements of different collections in a similar fashion using a single **iterator** interface.

#### Code example -- _BaseHeterogeneousEnsemble: /sklearn/ensemble/_base.py

```python
for est in estimators:
    if est != 'drop' and not is_estimator_type(est):
        raise ValueError(
            "The estimator {} should be a {}.".format(
                est.__class__.__name__, is_estimator_type.__name__[3:]
            )
        )
```

Python’s for loop abstracts the Iterator Pattern so thoroughly that most Python programmers are never even aware of the object design pattern that it enacts beneath the surface. The for loop performs repeated assignment, running its indented block of code once for each item in the sequence it is iterating over.

Below is the sequence diagram for iterator design pattern. The implementation details of how to iterate elements in each collection are hidden. Clients only need to interact with a single iterator interface. Benefit from that, the level of coupling has decreased.

## Appendix

[https://refactoring.guru/design-patterns/iterator/python/example](https://refactoring.guru/design-patterns/iterator/python/example)

[https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/#:~:text=The%20yield%20statement%20suspends%20function's,after%20the%20last%20yield%20run](https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/#:~:text=The%20yield%20statement%20suspends%20function's,after%20the%20last%20yield%20run)

[https://reactiveprogramming.io/blog/en/design-patterns/strategy](https://reactiveprogramming.io/blog/en/design-patterns/strategy)

[https://www.tutorialsteacher.com/ioc/dependency-injection](https://www.tutorialsteacher.com/ioc/dependency-injection)
