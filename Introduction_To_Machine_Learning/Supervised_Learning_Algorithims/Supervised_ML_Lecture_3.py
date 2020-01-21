#!/usr/bin/env python
# coding: utf-8

# # Introduction to Machine Learning in python for invironmental problems 
# ## Lecture_3: Supervised Learning Algorithms 
#                                                      
# This module uses basic ML models – linear regression, logistic regression, decision trees, random forests, and gradient-boosted trees – to predict future rotation in numerically simulated thunderstorms. You will train different supervised machine learning models on the NCAR dataset introduced in the previous modules. 
# 
# **Please cite the notebook as follows:**
# 
# Kamangir, H., 2020: " Supervised Machine Learning: Python tutorial": https://github.com/alburke/ams-2020-ml-python-course/blob/supervised-machine-learning/Introduction_To_Machine_Learning/Lecture_2/Supervised_ML_Lecture_3.ipynb 
# 
# 
# **References** 
# 
# This notebook refers to a few publications, listed below.
# 
# Chisholm, D., J. Ball, K. Veigas, and P. Luty, 1968: "The diagnosis of upper-level humidity." Journal of Applied Meteorology, 7 (4), 613-619.
# 
# Hsu, W., and A. Murphy, 1986: "The attributes diagram: A geometrical framework for assessing the quality of probability forecasts." International Journal of Forecasting, 2, 285–293, https://doi.org/10.1016/0169-2070(86)90048-8.
# 
# Lagerquist, R., and D.J. Gagne II, 2019: "Basic machine learning for predicting thunderstorm rotation: Python tutorial". https://github.com/djgagne/ams-ml-python-course/blob/master/module_2/ML_Short_Course_Module_2_Basic.ipynb.
# 
# McGovern, A., D. Gagne II, J. Basara, T. Hamill, and D. Margolin, 2015: "Solar energy prediction: An international contest to initiate interdisciplinary research on compelling meteorological problems." Bulletin of the American Meteorological Society, 96 (8), 1388-1395.
# 
# Metz, C., 1978: "Basic principles of ROC analysis." Seminars in Nuclear Medicine, 8, 283–298, https://doi.org/10.1016/S0001-2998(78)80014-2.
# 
# Refaeilzadeh P., Tang L., Liu H. (2009) Cross-Validation. In: LIU L., ÖZSU M.T. (eds) Encyclopedia of Database Systems. Springer, Boston, MA
# 
# Roebber, P., 2009: "Visualizing multiple measures of forecast quality." Weather and Forecasting, 24, 601-608, https://doi.org/10.1175/2008WAF2222159.1.
# 
# Sammut, Claude; Webb, Geoffrey I. (Eds.) (2011-03-28). Encyclopedia of Machine Learning (1st ed.). Springer. p. 578. ISBN 978-0-387-30768-8.
# 
# Singh, A., Thakur, N. and Sharma, A., 2016, March. A review of supervised machine learning algorithms. In 2016 3rd International Conference on Computing for Sustainable Global Development (INDIACom) (pp. 1310-1315). IEEE. 
# 
# Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and Vanderplas, J., 2011. Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), pp.2825-2830.
# 
# https://towardsdatascience.com/support-vector-machine-simply-explained-fee28eba5496
# 
# Strobl, C., Boulesteix, A.L., Zeileis, A. and Hothorn, T., 2007. Bias in random forest variable importance measures: Illustrations, sources and a solution. BMC bioinformatics, 8(1), p.25.
# 
# 
# **Scikit-Learn**
# 
# Scikit-learn is an open-source Python library for machine learning. It's comprehensive, well-documented, and popular across disciplines. We'll use it later in this module and throughout future modules.
# 
# Find the user guide here: https://scikit-learn.org/stable/user_guide.html
# 
# And the API here: https://scikit-learn.org/stable/modules/classes.html

# # Setup
# 
# To use this notebook, you will need Python 3.6 and the following packages.
# 
# - scipy
# - TensorFlow
# - Keras
# - scikit-image
# - netCDF4
# - pyproj
# - scikit-learn
# - opencv-python
# - matplotlib
# - shapely
# - geopy
# - metpy
# - descartes
# 
# If you have Anaconda on a Linux or Mac, you can install these packages with the commands pip install scipy, pip install tensorflow, etc.

# ## Supervised Machine Learning 
# 
# 
# <img src="SML.png" alt="Decision-tree schematic" width="1000" /> 
# 
# 
# 
# 

# ## Supervised Machine Learning Categories:
# 
# <img src="SupervisedML.png" alt="Decision-tree schematic" width="1000" /> 

# ## Supervised Classification Machine Learning Models: 
# <img src="Models.png" alt="Decision-tree schematic" width="1000" /> 

# # Logistic Regression Classifier
# <img src="LR.png" alt="Decision-tree schematic" width="1000" /> 

# - Logistic regression is basically linear regression with targets y = (0, 1).i.e. binary classification.
# - Logistic regression is a linear regression with different decision rule, which is hyperplane. 
# - Logistic regression learns the conditional probability distribution $P(y|x)$ 
# - if the estimated probablity is greater than 50%, then the model predicts that instances belongs to htat class (label 1), if not class label 0. 

# ## Logistic regression is a discriminative classifiers 
# - For training X, and y as label, logistic regression learn $P(y|X)$. 
# - For $\beta$ as adjustable parameters, and two classes $y=o$ and $y=1$: 
# <br>
# $P_{1}(X; \beta) = \frac{1}{1+\exp(-\beta x)}$  and  $P_{0}(X; \beta) = 1 - \frac{1}{1+\exp(-\beta x)}$
# - This is equivalent to: 
# <br>
# $log \frac{P_{1}(X; \beta)}{P_{0}(X; \beta)} = \beta X$
# - Recall the equation for linear regression, the equation for logistic regression would be: 
# <br>
# $\textrm{ln}(\frac{P_{1}(X; \beta)}{1 - P_{1}(X; \beta)}) = \beta_0 + \sum\limits_{j = 1}^{M} \beta_j x_j$
# <br>
# $P(X; \beta) = \frac{\textrm{exp}(\beta_0 + \sum\limits_{j = 1}^{M} \beta_j x_j)}{1 + \textrm{exp}(\beta_0 + \sum\limits_{j = 1}^{M} \beta_j x_j)} $

# ## Constructing a learning algorithm for logistic regression
# - The coefficients ($\beta$) of the logistic regression algorithm must be estimated through training process.
# - The conditional data likelihood is the probability of the label (y) values in the training data, conditioned on thier corresponing X values. So, the log of the conditional likelihood: 
# <br>
# $\beta = arg max \sum_{j} \textrm{ln} P (y_{j} | X_{j}, \beta)$ 
# <br>

# - This conditional data log lokelihood, which can be rewritten as: 
# <br>
# $l(\beta) = -\frac{1}{N} \sum\limits_{i = 1}^{N} \left[ y_i\textrm{ln}P(y_i = 1 | X_i, \beta) + (1 - y_i)\textrm{ln}P(y_i = 0 | X_i, \beta) \right]$
# <br>
# <br>
# $\color{blue}{P(y_i =1 | X_i, \beta)}$ = forecast probability for $i^{\textrm{th}}$ example.  This is probability that the event occurs (so class = 1 or "yes").  In our case, this is probability that max future vorticity $\color{red}{\ge 3.850 \times 10^{-3}\textrm{s}^{-1}}$.
# <br>
# $\color{blue}{y_i}$ = true label for $i^{\textrm{th}}$ example (0 or 1); 
# <br>
# $\color{blue}{N}$ = number of training examples; 
# <br>
# $\color{blue}{l(\beta)}$ = cross-entropy

# # Imports
# 
# The next cell imports all libraries that will be used by this notebook.  If the notebook crashes anywhere, it will probably be here.

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import copy
import warnings
import numpy
import matplotlib.pyplot as pyplot
import utils
import roc_curves
import attr_diagrams  


warnings.filterwarnings('ignore')
DEFAULT_FEATURE_DIR_NAME = ('./data/track_data_ncar_ams_3km_csv_small')
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

MODULE2_DIR_NAME = '.'
SHORT_COURSE_DIR_NAME = '..'


# ### Prevent Auto-scrolling
# The next cell prevents output in the notebook from being nested in a scroll box (the scroll box is small and makes things hard to see).

# In[6]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# ### Find input data
# 
# Find the input data based on their directory, and categorized them based on training (2010-2014), validation (2015) and testing (2016-2017) periods. 

# In[10]:


training_file_names = utils.find_many_feature_files(
    first_date_string='20100101', last_date_string='20141231', feature_dir_name=DEFAULT_FEATURE_DIR_NAME)

validation_file_names = utils.find_many_feature_files(
    first_date_string='20150101', last_date_string='20151231', feature_dir_name=DEFAULT_FEATURE_DIR_NAME)

testing_file_names = utils.find_many_feature_files(
    first_date_string='20160101', last_date_string='20171231', feature_dir_name=DEFAULT_FEATURE_DIR_NAME)


# ### Read Data
# 
# Next step is reading the data, and explore the contents of the files. 

# In[11]:


(training_metadata_table, training_predictor_table_denorm,
 training_target_table
) = utils.read_many_feature_files(training_file_names)
print(MINOR_SEPARATOR_STRING)

(validation_metadata_table, validation_predictor_table_denorm,
 validation_target_table
) = utils.read_many_feature_files(validation_file_names)
print(MINOR_SEPARATOR_STRING)

(testing_metadata_table, testing_predictor_table_denorm,
 testing_target_table
) = utils.read_many_feature_files(testing_file_names)
print(MINOR_SEPARATOR_STRING)

print('Variables in metadata are as follows:\n{0:s}'.format(
    str(list(training_metadata_table))
))

print('\nPredictor variables are as follows:\n{0:s}'.format(
    str(list(training_predictor_table_denorm))
))

print('\nTarget variable is as follows:\n{0:s}'.format(
    str(list(training_target_table))
))

first_predictor_name = list(training_predictor_table_denorm)[0]
these_predictor_values = (
    training_predictor_table_denorm[first_predictor_name].values[:10]
)

message_string = (
    '\nValues of predictor variable "{0:s}" for the first training '
    'examples:\n{1:s}'
).format(first_predictor_name, str(these_predictor_values))
print(message_string)

target_name = list(training_target_table)[0]
these_target_values = training_target_table[target_name].values[:10]

message_string = (
    '\nValues of target variable for the first training examples:\n{0:s}'
).format(str(these_target_values))
print(message_string)


# In[12]:


training_predictor_table_denorm.head(5)


# In[13]:


training_target_table.head(5)


# In[14]:


training_predictor_table, validation_predictor_table, testing_predictor_table = utils.norm_predictors(
    training_predictor_table_denorm,validation_predictor_table_denorm, testing_predictor_table_denorm, "StandardScaler")

predictor_names = list(training_predictor_table_denorm)

original_values = (
    'Original values of "{0:s}" for the first training examples:\n{1:s}'
).format(predictor_names[0], str(training_predictor_table_denorm[predictor_names[0]].values[:10]))
print(original_values) 

normalized_values = (
    '\nNormalized values of "{0:s}" for the first training examples:\n{1:s}'
).format(predictor_names[0], str(training_predictor_table[predictor_names[0]].values[:10]))
print(normalized_values)


# # Binary Classification
# 
#  - **The rest of this module focuses on binary classification, rather than regression.**
#  - "Regression" is the prediction of a real number (*e.g.*, above, where we predicted max future vorticity).
#  - "Classification" is the prediction of a category (*e.g.*, low, medium, or high max future vorticity).
# <br><br>
#  - **In binary classification there are two categories.**
#  - Thus, prediction takes the form of answering a **yes-or-no question.**
#  - We will use the same target variable (max future vorticity), except we will binarize it.
#  - The problem will be predicting whether or not max future vorticity exceeds a threshold.

# # Binarization
# 
#  - The next cell "binarizes" the target variable (turns each value into a 0 or 1, yes or no).
#  - The threshold is the 90$^{\textrm{th}}$ percentile of max future vorticity over all training examples.
#  - The same threshold is used to binarize training, validation, and testing data.

# In[15]:


binarization_threshold = utils.get_binarization_threshold(
    csv_file_names=training_file_names, percentile_level=90.)
print(MINOR_SEPARATOR_STRING)

these_target_values = (
    training_target_table[utils.TARGET_NAME].values[:10]
)

message_string = (
    'Real-numbered target values for the first training examples:\n{0:s}'
).format(str(these_target_values))
print(message_string)

training_target_values = utils.binarize_target_values(
    target_values=training_target_table[utils.TARGET_NAME].values,
    binarization_threshold=binarization_threshold)

training_target_table = training_target_table.assign(
    **{utils.BINARIZED_TARGET_NAME: training_target_values}
)

print('\nBinarization threshold = {0:.3e} s^-1'.format(
    binarization_threshold
))

these_target_values = (
    training_target_table[utils.TARGET_NAME].values[:10]
)

message_string = (
    '\nBinarized target values for the first training examples:\n{0:s}'
).format(str(training_target_table))
print(message_string)

validation_target_values = utils.binarize_target_values(
    target_values=validation_target_table[utils.TARGET_NAME].values,
    binarization_threshold=binarization_threshold)

validation_target_table = validation_target_table.assign(
    **{utils.BINARIZED_TARGET_NAME: validation_target_values}
)

testing_target_values = utils.binarize_target_values(
    target_values=testing_target_table[utils.TARGET_NAME].values,
    binarization_threshold=binarization_threshold)

testing_target_table = testing_target_table.assign(
    **{utils.BINARIZED_TARGET_NAME: testing_target_values}
)


# # Supervised learning in Sckit-learn 
# 
# All machine learning models share a common set of functions, making it very easy to try different models on your dataset(s): 
#     There are three-four main functions: 
# <img src="SK.png" alt="Decision-tree schematic" width="700" />

# # setup a logistic regression model in python: 
# 
# - configuration of model: import the sckit learn library and linear model in python by: 
# <br>
# import sklearn.linear_model 
# <br>
# <br>
# - Setup a logistic regression function by: 
# <br>
# reg = sklearn.linear_model.SGDClassifier(
#             loss='log', penalty='none', fit_intercept=True, verbose=0,
#             random_state=RANDOM_SEED)
# <br>
# <br>
# - Train the model by using the "fit" function:
# <br>
# reg.fit(xtrain, ytrain)
# <br>
# <br>
# - Test on new data by using "predict" function:
# <br>
# reg.predict(xest)

# In[16]:


def setup_logistic_regression(lambda1=0., lambda2=0.):
    """Sets up (but does not train) logistic-regression model.
    :param lambda1: L1-regularization weight.
    :param lambda2: L2-regularization weight.
    :return: model_object: Instance of `sklearn.linear_model.SGDClassifier`.
    """

    assert lambda1 >= 0
    assert lambda2 >= 0

    if lambda1 < LAMBDA_TOLERANCE and lambda2 < LAMBDA_TOLERANCE:
        return sklearn.linear_model.SGDClassifier(
            loss='log', penalty='none', fit_intercept=True, verbose=0,
            random_state=RANDOM_SEED)

    if lambda1 < LAMBDA_TOLERANCE:
        return sklearn.linear_model.SGDClassifier(
            loss='log', penalty='l2', alpha=lambda2, fit_intercept=True,
            verbose=0, random_state=RANDOM_SEED)

    if lambda2 < LAMBDA_TOLERANCE:
        return sklearn.linear_model.SGDClassifier(
            loss='log', penalty='l1', alpha=lambda1, fit_intercept=True,
            verbose=0, random_state=RANDOM_SEED)

    alpha, l1_ratio = _lambdas_to_sklearn_inputs(
        lambda1=lambda1, lambda2=lambda2)

    return sklearn.linear_model.SGDClassifier(
        loss='log', penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio,
        fit_intercept=True, verbose=0, random_state=RANDOM_SEED)


# # Logistic Regression: Example
# 
#  - Trains a logistic-regression model (with default hyperparameters) to predict the label for each storm (whether or not max future vorticity $\ge 3.850 \times 10^{-3}\textrm{ s}^{-1}$).
#  - Evaluates the model on both training and validation data.
# 

# In[17]:


plain_log_model_object = utils.setup_logistic_regression(
    lambda1=0., lambda2=0.)

_ = utils.train_logistic_regression(
    model_object=plain_log_model_object,
    training_predictor_table=training_predictor_table,
    training_target_table=training_target_table)

training_probabilities = plain_log_model_object.predict_proba(
    training_predictor_table.as_matrix()
)[:, 1]
training_event_frequency = numpy.mean(
    training_target_table[utils.BINARIZED_TARGET_NAME].values
)


# In[18]:


_ = utils.eval_binary_classifn(
    observed_labels=training_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=training_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='training')


# In[19]:



validation_probabilities = plain_log_model_object.predict_proba(
    validation_predictor_table.as_matrix()
)[:, 1]

_ = utils.eval_binary_classifn(
    observed_labels=validation_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=validation_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='validation')


# # Logistic Regression: Coefficients
# 
#  - The next cell plots coefficients for the logistic-regression model.
#  - Positive (negative) coefficients mean that probability increases (decreases) with the predictor variable.
#  - Again, predictors have been normalized to the same scale ($z$-scores), so generally predictors with larger coefficients are more important.

# In[20]:


utils.plot_model_coefficients(
    model_object=plain_log_model_object,
    predictor_names=list(training_predictor_table)
)

pyplot.show()


# # Logistic Regression with $L_1$ and $L_2$
# 
#  - The next cell trains a logistic-regression model with both $L_1$ and $L_2$ regularization.
#  - As for linear regression, you could do a hyperparameter experiment to find the best $\lambda_1$ and $\lambda_2$ for logistic regression.
#  - Now you can see why just saying "lasso regression" or "ridge regression" or "elastic-net regression" is not descriptive enough.  This type of regularization can be applied to different base models.

# In[21]:


logistic_en_model_object = utils.setup_logistic_regression(
    lambda1=1e-3, lambda2=1e-3)

_ = utils.train_logistic_regression(
    model_object=logistic_en_model_object,
    training_predictor_table=training_predictor_table,
    training_target_table=training_target_table)

validation_probabilities = logistic_en_model_object.predict_proba(
    validation_predictor_table.as_matrix()
)[:, 1]
training_event_frequency = numpy.mean(
    training_target_table[utils.BINARIZED_TARGET_NAME].values
)

utils.eval_binary_classifn(
    observed_labels=validation_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=validation_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='validation')


# # Logistic Regression with $L_1$ and $L_2$: Coefficients
# 
# The next cell plots coefficients for the logistic-regression model with both penalties.
# <br><br>
# Many coefficients are zero, and the non-zero ones are about an order of magnitude smaller than for the original model.

# In[22]:


utils.plot_model_coefficients(
    model_object=logistic_en_model_object,
    predictor_names=list(training_predictor_table)
)

pyplot.show()


# # Decision Trees
#  - Decision trees have been used in meteorology since the 1960s (Chisholm 1968).
#  - They were built subjectively by human experts until the 1980s, when an objective algorithm (Quinlan 1986) was developed to "train" them (determine the best question at each branch node).
#  - A decision tree is a flow chart with **branch nodes** (ellipses) and **leaf nodes** (rectangles).  In the figure below, $f$ is the forecast probability of severe weather.
# <img src="tree_schematic.jpg" alt="Decision-tree schematic" width="500" />

# 
#  - The branch nodes are bifurcating, and the leaf nodes are terminal.
#  - In other words, each branch node has 2 children and each leaf node has 0 children.
#  - **Predictions are made at the leaf nodes, and questions are asked at the branch nodes.**
#  - Since the branch nodes are bifurcating, **questions asked at the branch nodes must be yes-or-no.**
# <br><br>
#  - **The prediction at leaf node $L$ is the average of all training examples that reached $L$.**
#  - For regression this is a real value (average max future vorticity of examples that reached $L$).
#  - For classification this is a probability (fraction of examples that reached $L$ with max future vorticity $\ge 3.850 \times 10^{-3}\textrm{ s}^{-1}$).
# <br><br>
#  

# - **The question chosen at each branch node is that which maximizes information gain.**
#  - **This is done by minimizing the "remainder,"** which is based on entropy of the child nodes.
#  - The entropy of one node is defined below.
# <br><br>
# $E = -\frac{1}{n} \left[ f\textrm{ log}_2(f) + f\textrm{ log}_2(f) \right]$
# <br><br>
#  - $n$ = number of examples at the node
#  - $f$ = fraction of these examples that are in the positive class
# <br>

# **The "remainder" is defined as follows.**
# <br><br>
# $R = \frac{n_{\textrm{left}} E_{\textrm{left}} + n_{\textrm{right}} E_{\textrm{right}}}{n_{\textrm{left}} + n_{\textrm{right}}}$
# <br><br>
#  - $n_{\textrm{left}}$ = number of examples sent to left child (for which the answer to the question is "no")
# <br>
#  - $n_{\textrm{left}}$ = number of examples sent to right child (for which answer is "yes")
# <br>
#  - $E_{\textrm{left}}$ = entropy of left child
# <br>
#  - $E_{\textrm{right}}$ = entropy of right child

# # Decision Tree: Example
# 
# The next cell trains a decision tree, with default hyperparameters, to forecast the probability that a storm will develop future vorticity $\ge 3.850 \times 10^{-3}\textrm{ s}^{-1}$.

# In[23]:


# Setup a decision tree learning model in python: 
def setup_classification_tree(min_examples_at_split=30,
                              min_examples_at_leaf=30):
    """Sets up (but does not train) decision tree for classification.
    :param min_examples_at_split: Minimum number of examples at split node.
    :param min_examples_at_leaf: Minimum number of examples at leaf node.
    :return: model_object: Instance of `sklearn.tree.DecisionTreeClassifier`.
    """

    return sklearn.tree.DecisionTreeClassifier(
        criterion='entropy', min_samples_split=min_examples_at_split,
        min_samples_leaf=min_examples_at_leaf, random_state=RANDOM_SEED)


def train_classification_tree(model_object, training_predictor_table,
                              training_target_table):
    """Trains decision tree for classification.
    :param model_object: Untrained model created by `setup_classification_tree`.
    :param training_predictor_table: See doc for `read_feature_file`.
    :param training_target_table: Same.
    :return: model_object: Trained version of input.
    """

    model_object.fit(
        X=training_predictor_table.as_matrix(),
        y=training_target_table[BINARIZED_TARGET_NAME].values
    )

    return model_object


# In[24]:


default_tree_model_object = utils.setup_classification_tree(
    min_examples_at_split=30, min_examples_at_leaf=30)

_ = utils.train_classification_tree(
    model_object=default_tree_model_object,
    training_predictor_table=training_predictor_table,
    training_target_table=training_target_table)

training_probabilities = default_tree_model_object.predict_proba(
    training_predictor_table.as_matrix()
)[:, 1]
training_event_frequency = numpy.mean(
    training_target_table[utils.BINARIZED_TARGET_NAME].values
)

utils.eval_binary_classifn(
    observed_labels=training_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=training_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='training')

validation_probabilities = default_tree_model_object.predict_proba(
    validation_predictor_table.as_matrix()
)[:, 1]

utils.eval_binary_classifn(
    observed_labels=validation_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=validation_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='validation')


# ## The bias-variance trade-off 
# - in supervised learning, we assume there is a real realtionship between features and labels and model estimate this unknown relationship. 
# - If we use a different training sets, we are likely to get different outputs for model. The amount by which model varies as we change the training sets is called **Variance**.  
# - For most real-life senarios, the true relationship between features and target is complicated and far from linear. Simplifying assumptions give **Bias** to a model. the more errorneous the assumption with respect to the true relationship, the higher the bias, and vice-versa. 
# - **Variance** and **Bias** are the sources of error for machine learning models. we need to keep both at their minimum. 

# - Inntuitively we want low bias to avoid building a model that is too simple. in most cases, a simple performs poorly on training data, and it is extremely likely to repeat the poor performance on the test data.
# - Similarly, we are looking for low variance to avoid building an overly complex model. More prone to falling overfitting problem. 
# - In practice, however, we need to accept a trade-off which we can not have both low values, so we want to have both variance and bias in middle. 
# - **Learning curves** provide the bias-variance trade-off. 
# - **learning Curves** show the relationship between training set size and your chosen evalutaion metrics (RMSE, Accuracy, etc.) on your training and validation sets.  

# <img src="LC.png" alt="Decision-tree schematic" width="1000" />

# In[25]:


classifier = default_tree_model_object 

_= utils.plot_learning_curves(classifier, training_predictor_table, training_target_values, validation_predictor_table, validation_target_table[
        utils.BINARIZED_TARGET_NAME].values, title='Learning Curve', xlabel='Training Set', ylabel='Performance(POD)')


# # Hyperparameter Experiment with Minimum Sample Size
# 
#  - **Two hyperparameters (among others) control the depth of a decision tree:** minimum examples per branch node ($N_b^{\textrm{min}}$) and per leaf node ($N_l^{\textrm{min}}$).
#  - If these values are set to 1, the tree can become very deep, which increases its ability to overfit.
#  - You can think of this another way: if there is one example at each leaf node, all predictions will be based on only one example and will probably not generalize well to new data.
#  - Conversely, if $N_b^{\textrm{min}}$ and $N_l^{\textrm{min}}$ are set too high, the tree will not become deep enough, causing it to underfit.
#  - For example, suppose that you have 1000 training examples and set $N_l^{\textrm{min}}$ to 1000.
#  - This will allow only one branch node (the root node); both children of the root node will have $<$ 1000 examples.
#  - Thus, predictions will be based on only one question.

# **Recall the four steps of any hyperparameter experiment:**
# 
#  1. Choose the values to be attempted.  We will try $N_b^{\textrm{min}} \in \lbrace 2, 5, 10, 20, 30, 40, 50, 100, 200, 500 \rbrace$ and $N_l^{\textrm{min}} \in \lbrace 1, 5, 10, 20, 30, 40, 50, 100, 200, 500 \rbrace$.  However, we will not try combinations where $N_l^{\textrm{min}} \ge N_b^{\textrm{min}}$, because this makes no sense (the child of a node with $N$ examples cannot have $\ge N$ examples).
#  2. Train a model with each combination.
#  3. Evaluate each model on the validation data.
#  4. Select the model that performs best on validation data.  Here we will define "best" as that with the highest Brier skill score.

# # Hyperparameter Experiment: Training
# 
# The next cell performs steps 1 and 2 of the hyperparameter experiment (defining the values to be attempted and training the models).
# 
# 

# In[26]:


min_per_split_values = numpy.array(
    [2, 5, 10, 20, 30, 40, 50, 100, 200, 500], dtype=int)
min_per_leaf_values = numpy.array(
    [1, 5, 10, 20, 30, 40, 50, 100, 200, 500], dtype=int)

num_min_per_split_values = len(min_per_split_values)
num_min_per_leaf_values = len(min_per_leaf_values)

validation_auc_matrix = numpy.full(
    (num_min_per_split_values, num_min_per_leaf_values), numpy.nan
)

validation_max_csi_matrix = validation_auc_matrix + 0.
validation_bs_matrix = validation_auc_matrix + 0.
validation_bss_matrix = validation_auc_matrix + 0.

training_event_frequency = numpy.mean(
    training_target_table[utils.BINARIZED_TARGET_NAME].values
)

for i in range(num_min_per_split_values):
    for j in range(num_min_per_leaf_values):
        if min_per_leaf_values[j] >= min_per_split_values[i]:
            continue

        this_message_string = (
            'Training model with minima of {0:d} examples per split node, '
            '{1:d} per leaf node...'
        ).format(min_per_split_values[i], min_per_leaf_values[j])

        print(this_message_string)

        this_model_object = utils.setup_classification_tree(
            min_examples_at_split=min_per_split_values[i],
            min_examples_at_leaf=min_per_leaf_values[j]
        )

        _ = utils.train_classification_tree(
            model_object=this_model_object,
            training_predictor_table=training_predictor_table,
            training_target_table=training_target_table)

        these_validation_predictions = this_model_object.predict_proba(
            validation_predictor_table.as_matrix()
        )[:, 1]

        this_evaluation_dict = utils.eval_binary_classifn(
            observed_labels=validation_target_table[
                utils.BINARIZED_TARGET_NAME].values,
            forecast_probabilities=these_validation_predictions,
            training_event_frequency=training_event_frequency,
            create_plots=False, verbose=False)

        validation_auc_matrix[i, j] = this_evaluation_dict[utils.AUC_KEY]
        validation_max_csi_matrix[i, j] = this_evaluation_dict[
            utils.MAX_CSI_KEY]
        validation_bs_matrix[i, j] = this_evaluation_dict[
            utils.BRIER_SCORE_KEY]
        validation_bss_matrix[i, j] = this_evaluation_dict[
            utils.BRIER_SKILL_SCORE_KEY]


# # Hyperparameter Experiment: Validation
# 
# The next cell performs step 3 of the hyperparameter experiment (evaluates each model on the validation data).

# In[27]:


utils.plot_scores_2d(
    score_matrix=validation_auc_matrix,
    min_colour_value=numpy.nanpercentile(validation_auc_matrix, 1.),
    max_colour_value=numpy.nanpercentile(validation_auc_matrix, 99.),
    x_tick_labels=min_per_leaf_values,
    y_tick_labels=min_per_split_values
)

pyplot.xlabel('Min num examples at leaf node')
pyplot.ylabel('Min num examples at split node')
pyplot.title('AUC (area under ROC curve) on validation data')

utils.plot_scores_2d(
    score_matrix=validation_max_csi_matrix,
    min_colour_value=numpy.nanpercentile(validation_max_csi_matrix, 1.),
    max_colour_value=numpy.nanpercentile(validation_max_csi_matrix, 99.),
    x_tick_labels=min_per_leaf_values,
    y_tick_labels=min_per_split_values
)

pyplot.xlabel('Min num examples at leaf node')
pyplot.ylabel('Min num examples at split node')
pyplot.title('Max CSI (critical success index) on validation data')

utils.plot_scores_2d(
    score_matrix=validation_bs_matrix,
    min_colour_value=numpy.nanpercentile(validation_bs_matrix, 1.),
    max_colour_value=numpy.nanpercentile(validation_bs_matrix, 99.),
    x_tick_labels=min_per_leaf_values,
    y_tick_labels=min_per_split_values
)

pyplot.xlabel('Min num examples at leaf node')
pyplot.ylabel('Min num examples at split node')
pyplot.title('Brier score on validation data')

utils.plot_scores_2d(
    score_matrix=validation_bss_matrix,
    min_colour_value=numpy.nanpercentile(validation_bss_matrix, 1.),
    max_colour_value=numpy.nanpercentile(validation_bss_matrix, 99.),
    x_tick_labels=min_per_leaf_values,
    y_tick_labels=min_per_split_values
)

pyplot.xlabel('Min num examples at leaf node')
pyplot.ylabel('Min num examples at split node')
pyplot.title('Brier skill score on validation data')


# # Hyperparameter Experiment: Selection
# 
# The next cell performs step 4 of the hyperparameter experiment (select model).

# In[28]:


best_linear_index = numpy.nanargmax(numpy.ravel(validation_bss_matrix))

best_split_index, best_leaf_index = numpy.unravel_index(
    best_linear_index, validation_bss_matrix.shape)

best_min_examples_per_split = min_per_split_values[best_split_index]
best_min_examples_per_leaf = min_per_leaf_values[best_leaf_index]
best_validation_bss = numpy.nanmax(validation_bss_matrix)

message_string = (
    'Best validation BSS = {0:.3f} ... corresponding min examples per split'
    ' node = {1:d} ... min examples per leaf node = {2:d}'
).format(
    best_validation_bss, best_min_examples_per_split,
    best_min_examples_per_leaf
)

print(message_string)

final_model_object = utils.setup_classification_tree(
    min_examples_at_split=best_min_examples_per_split,
    min_examples_at_leaf=best_min_examples_per_leaf
)

_ = utils.train_classification_tree(
    model_object=final_model_object,
    training_predictor_table=training_predictor_table,
    training_target_table=training_target_table)

testing_predictions = final_model_object.predict_proba(
    testing_predictor_table.as_matrix()
)[:, 1]
training_event_frequency = numpy.mean(
    training_target_table[utils.BINARIZED_TARGET_NAME].values
)

_ = utils.eval_binary_classifn(
    observed_labels=testing_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=testing_predictions,
    training_event_frequency=training_event_frequency,
    create_plots=True, verbose=True, dataset_name='testing')


# In[29]:


classifier = final_model_object

_= utils.plot_learning_curves(classifier, training_predictor_table, training_target_values, validation_predictor_table, validation_target_table[
        utils.BINARIZED_TARGET_NAME].values, title='Learning Curve', xlabel='Training Set', ylabel='Performance(POD)')


# # Random Forests
# 
#  - **A random forest is an ensemble of decision trees.**
#  - In the first example with decision trees, you may have noticed a lot of **overfitting.**
#  - **This is generally a problem with decision trees**, because they rely on exact thresholds, which introduce "jumps" into the decision function.
#  - For example, in the tree shown below, a difference of 0.0001 J kg$^{-1}$ in CAPE could lead to a difference of 55% in tornado probability.
# 
# <img src="tree_schematic.jpg" alt="Decision-tree schematic" width="500" />
# 

#  - One way to mitigate this overfitting is: train a bunch of decision trees.
#  - **If the decision trees are diverse enough, they will hopefully have offsetting biases** (overfit in different ways).
#  - **Random forests ensure diversity in two ways:**
#      - Example-bagging
#      - Predictor-bagging (or "feature-bagging")
# <br><br>
#  - **Example-bagging** is done by training each tree with a **bootstrapped replicate** of the training data.
#  - For a training set with $N$ examples, **a "bootstrapped replicate" is created by randomly sampling $N$ examples with replacement.**
#  - Sampling with replacement leads to duplicates.  On average, each bootstrapped replicate contains only 63.2% ($1 - e^{-1}$) of unique examples, with the other 37.8% being duplicates.
#  - This ensures that each tree is trained with a different set of unique examples.

#  - **Predictor-bagging** is done by looping over a random subset of predictors at each branch node.
#  - In other words, instead of trying all predictors to find the best question, try only a few predictors.
#  - If there are $M$ predictors, the general rule is to try $\sqrt{M}$ predictors at each branch node.
#  - In our case there are 41 predictors, so each branch node will loop over 6 randomly chosen predictors.

# # Random Forests: Example
# 
# The next cell trains a random forest with the following hyperparameters:
# 
#  - 100 trees
#  - 6 predictors attempted at each leaf node
#  - Minimum of 500 examples at a branch node
#  - Minimum of 200 examples at a leaf node

# In[30]:


# Setup a Random Forest learning model in python: 

def setup_classification_forest(
        max_predictors_per_split, num_trees=100, min_examples_at_split=30,
        min_examples_at_leaf=30):
    """Sets up (but does not train) random forest for classification.
    :param max_predictors_per_split: Max number of predictors to try at each
        split.
    :param num_trees: Number of trees.
    :param min_examples_at_split: Minimum number of examples at split node.
    :param min_examples_at_leaf: Minimum number of examples at leaf node.
    :return: model_object: Instance of
        `sklearn.ensemble.RandomForestClassifier`.
    """

    return sklearn.ensemble.RandomForestClassifier(
        n_estimators=num_trees, min_samples_split=min_examples_at_split,
        min_samples_leaf=min_examples_at_leaf,
        max_features=max_predictors_per_split, bootstrap=True,
        random_state=RANDOM_SEED, verbose=2)


def train_classification_forest(model_object, training_predictor_table,
                                training_target_table):
    """Trains random forest for classification.
    :param model_object: Untrained model created by
        `setup_classification_forest`.
    :param training_predictor_table: See doc for `read_feature_file`.
    :param training_target_table: Same.
    :return: model_object: Trained version of input.
    """

    model_object.fit(
        X=training_predictor_table.as_matrix(),
        y=training_target_table[BINARIZED_TARGET_NAME].values
    )

    return model_object


# In[31]:


num_predictors = len(list(training_predictor_table))
max_predictors_per_split = int(numpy.round(
    numpy.sqrt(num_predictors)
))

random_forest_model_object = utils.setup_classification_forest(
    max_predictors_per_split=max_predictors_per_split,
    num_trees=100, min_examples_at_split=500, min_examples_at_leaf=200)

_ = utils.train_classification_forest(
    model_object=random_forest_model_object,
    training_predictor_table=training_predictor_table,
    training_target_table=training_target_table)

training_probabilities = random_forest_model_object.predict_proba(
    training_predictor_table.as_matrix()
)[:, 1]
training_event_frequency = numpy.mean(
    training_target_table[utils.BINARIZED_TARGET_NAME].values
)

utils.eval_binary_classifn(
    observed_labels=training_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=training_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='training')

validation_probabilities = random_forest_model_object.predict_proba(
    validation_predictor_table.as_matrix()
)[:, 1]

utils.eval_binary_classifn(
    observed_labels=validation_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=validation_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='validation')


# # Practice 
# plot the learning curve for Random Forest and intrepret the results. 

# In[ ]:


classifier = random_forest_model_object

_= utils.plot_learning_curves(classifier, training_predictor_table, training_target_values, validation_predictor_table, validation_target_table[
        utils.BINARIZED_TARGET_NAME].values, title='Learning Curve', xlabel='Training Set', ylabel='Performance(POD)')


# # Hyperparameter Experiment with Minimum Sample Size
# 
#  - **Two hyperparameters (among others) control the depth of a decision tree:** minimum examples per branch node ($N_b^{\textrm{min}}$) and per leaf node ($N_l^{\textrm{min}}$).
#  - If these values are set to 1, the tree can become very deep, which increases its ability to overfit.
#  - You can think of this another way: if there is one example at each leaf node, all predictions will be based on only one example and will probably not generalize well to new data.
#  - Conversely, if $N_b^{\textrm{min}}$ and $N_l^{\textrm{min}}$ are set too high, the tree will not become deep enough, causing it to underfit.
#  - For example, suppose that you have 1000 training examples and set $N_l^{\textrm{min}}$ to 1000.
#  - This will allow only one branch node (the root node); both children of the root node will have $<$ 1000 examples.
#  - Thus, predictions will be based on only one question.

# # Practice at home:  
# use the gridsearch technique in sklearn library instead of suing the for loop to find the best configuration for randome forest. 
# 
# In the two next cells you can find the example for random forest hyperparameter tunning.  

# In[ ]:


import pprint
# Number of trees in random forest
n_estimators = [int(x) for x in numpy.linspace(start = 10, stop = 200, num = 4)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in numpy.linspace(1, 10, num = 4)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = numpy.array(
    [2, 5, 10, 20, 30, 40, 50, 100, 200, 500], dtype=int)
# Minimum number of samples required at each leaf node
min_samples_leaf = numpy.array(
    [1, 5, 10, 20, 30, 40, 50, 100, 200, 500], dtype=int)
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint.pprint(random_grid)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(RandomForestClassifier(), random_grid, cv=3, n_jobs=-1)
clf.fit(training_predictor_table, training_target_table[utils.BINARIZED_TARGET_NAME].values)
clf.best_params_


# # Random forest variable importance measures
# - A variable importance measure to use in tree-based ensemble methods is to merely count the number of times each variables is selected by all individual tree in the ensemble. 
# - In fact, the "Gini importance" availabel in random forest implementations, "gini" importance describes the improvement in the gini gain splitting criterion. 
# - Based on random forest, variable importanc measure is the "permutation accuracy importance" measure. Its rationale is: 
# 
# **by randomly permuting the predictor variable X, its original association with the response Y is broken. When the permuted variable x, together the remaining unpermuted predictor variables, is used to predict the response, the prediction accuracy decreases substantially, if the original variable x was associated with the response. Thus, a reasonable measure for variable importance is the difference in prediction accuracy before and after permuting x.**

# # Keep in mind!
# - The importance ranking using the impurity reduction is biased towards prefering variables with more categories. 
# - In case, when the dataset has two or more correlated features, any of these correlated features can be used as the predictors, with no concrete preference of one over the othres. But once one of them is used, the importance of others is significantly reduced since effectively the impurity they can remove is already removed by the first feature. As a consequence, they will have a lower reported importance. 
# - This is not an issue when we want to use the feature selection to reduce overfitting, not for interpreting the variables is a strong predictors. 

# In[32]:


classifier = random_forest_model_object 
importances = classifier.feature_importances_
std = numpy.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
indices = numpy.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(training_predictor_table.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, list(training_predictor_table_denorm)[f], importances[indices[f]]))

# Plot the feature importances of the forest
pyplot.figure(figsize=(20, 10))
pyplot.title("Feature importances")
pyplot.bar(range(training_predictor_table.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
pyplot.xticks(range(training_predictor_table.shape[1]), list(training_predictor_table_denorm), rotation=90)
pyplot.xlim([-1, training_predictor_table.shape[1]])
pyplot.show()


# # Gradient-boosted Trees
# 
#  - **Gradient-boosting is another way to ensemble decision trees.**
#  - In a random forest the trees are trained independently of each other.
#  - In a gradient-boosted ensemble (or "gradient-boosted forest"), the $k^{\textrm{th}}$ tree is trained to fit residuals from the first $k$ - 1 trees.
#  - The "residual" for the $i^{\textrm{th}}$ example is $y_i - p_i$.
# <br><br>
#  - Gradient-boosted trees can still use example-bagging and predictor-bagging.
#  - However, in most libraries the default is no example-bagging or predictor-bagging (train each tree with all examples and attempt all predictors at each branch node).

#  - In a random forest the trees can be trained in parallel (each tree is independent of the others), which makes random forests faster.
#  - In a gradient-boosted ensemble the trees must be trained in series, which makes them slower.
#  - However, **in practice gradient-boosting usually outperforms random forests.**
#  - In a recent contest for solar-energy prediction, the top 3 teams all used gradient-boosted trees (McGovern *et al*. 2015).

# # Gradient-boosted Trees: Example
# 
# The next cell trains a gradient-boosted ensemble with the following hyperparameters:
# 
#  - No example-bagging
#  - No predictor-bagging
#  - 100 trees
#  - Minimum of 500 examples at a branch node
#  - Minimum of 200 examples at a leaf node

# In[33]:


num_predictors = len(list(training_predictor_table))
# max_predictors_per_split = int(numpy.round(
#     numpy.sqrt(num_predictors)
# ))

gbt_model_object = utils.setup_classification_forest(
    max_predictors_per_split=num_predictors, num_trees=100,
    min_examples_at_split=500, min_examples_at_leaf=200)

_ = utils.train_classification_gbt(
    model_object=gbt_model_object,
    training_predictor_table=training_predictor_table,
    training_target_table=training_target_table)

training_probabilities = gbt_model_object.predict_proba(
    training_predictor_table.as_matrix()
)[:, 1]
training_event_frequency = numpy.mean(
    training_target_table[utils.BINARIZED_TARGET_NAME].values
)

utils.eval_binary_classifn(
    observed_labels=training_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=training_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='training')

validation_probabilities = gbt_model_object.predict_proba(
    validation_predictor_table.as_matrix()
)[:, 1]

utils.eval_binary_classifn(
    observed_labels=validation_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=validation_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='validation')


# # Multilayer Perceptron Neural Network
# - A multilayer perceptron (MLP) contains one or more hidden layers (apart from one input and one output). 
# - Single layer perceptron can only learn linear functions, a multi layer perceptron can also learn non-linear functins.
# <img src="NN.png" alt="Decision-tree schematic" width="700" />

# # Multilayer Perceptron Neural Network Components
# 
# **Key components:**
# 
# **Input layer:** The input is raw original input data and the output would be the input data plus bias which commonly is 1. 
# <br>
# **Neurons:** They are a computational units that have weights input and produce an output using an activation function.
# <br>
# **Neuron weights:** same as coefficients for linear regression.Each neurons has a bias too. Weights are often initialized to small random values, larger wieghts indicate increased complexity which need regularization techniques.  
# <br>
# **Activation Function:** The weights are summed and passed through an activation function (transfer function). It can a linear activation function or non-linear activation function such as sigmoid function. 

# # Training MLP: The Back-propagation Algorithm
# The process by which a MLP learns is called the Backpropagation algorithm. In fact, it is "learning from mistakes". The supervisor corrects the MLP whenever it makes mistakes. Generally there are two main steps:  
# **1. Forward propagation:**
# Initially all the edge weights are randomly assigned. For every input in training dataset, the ANN is activated (e.g., nodes is fired) by using the activation function and its output is generated. Assume the wights of the connections from the inputs to the outputs are formulated as follow: 
# <img src="FP.png" alt="Decision-tree schematic" width="650" />

# **2. Back propagation and weights updating**
# - The output is compared with the desired output (label) and the error is **propagated** back to the previous layer and generally all the network to calculate the (**gradient**). 
# - Using the optimization technique such as gradient descent to adjust the weights in all the network with an aim to reduce the error at the output layer. 
# - This process is repeated until the output error is below a predetermined threshold. 
# <img src="BP.png" alt="Decision-tree schematic" width="650" />

# In[34]:


from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
parameter_space = {
    'hidden_layer_sizes': [(20,20), (50,50,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

mlp = MLPClassifier(max_iter=100)
clf_MLP = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf_MLP.fit(training_predictor_table, training_target_table[utils.BINARIZED_TARGET_NAME].values)
print(clf_MLP.best_params_)


# In[47]:


from sklearn.neural_network import MLPClassifier
MLP_model_object = MLPClassifier(activation= 'tanh', solver='adam', alpha=0.05, hidden_layer_sizes=(20,20), random_state=1)
MLP_model_object.fit(training_predictor_table, training_target_table[utils.BINARIZED_TARGET_NAME].values)


# In[50]:


training_probabilities = MLP_model_object.predict(
    training_predictor_table.as_matrix())

training_event_frequency = numpy.mean(
    training_target_table[utils.BINARIZED_TARGET_NAME].values)
utils.eval_binary_classifn(
    observed_labels=training_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=training_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='training')

validation_probabilities = MLP_model_object.predict(
    validation_predictor_table.as_matrix()
)

utils.eval_binary_classifn(
    observed_labels=validation_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=validation_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='validation')


# # Support Vector Machines 
# - Instance-based model which the decision function is fully specified by a subset of training samples, the support vectors.  
# - The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points. 
# - To separate the two classes of data points, there are many possible hyperplanes but we are looking to find a plane with maximum margin (**called large margin classification**). 
# <img src="SVM.png" alt="Decision-tree schematic" width="650" />

# - SVM can seperate the classes linearly called linear SVM (**Hard margin clssification**). But in reality, dataset are probably not separatable linearly. SVM adress non-linearly separable cases by introducing two concepts: 
#  **soft margin classification** and **kernel method**. 
# - Applying **Soft Margin**, SVM tolerates a few dots to get misclassified and tries to balance the trade-off between finding a line that maximizes the margin and minimize the misclassification. 
# - Degree of tolerance is one of the most important hyperparametres for the SVM (both linear and non-linear- in Sklearn library is $\color{red}{C}$). If your SVM model is $\color{blue}{overfitting}$ , you can try regularizing it by reducing $\color{red}{C}$ value. 
# 

# **Kernel Trick**: kernel technique is using the existing features and applies some transformations, and creates new featuers to find the nonlinear decision boundary. It can be **Polynomial** and **Radial Basis Function(RBF)**. 
# <img src="Kernel.png" alt="Decision-tree schematic" width="850" />
# - RBF adds features computed using a similarity function that measure how much each instance resembles a particular landmark. The similarity function is the guassian radial basis function: 
# <br>
# $\phi \gamma (X, l) = exp(-\gamma||x-l||^{2})$
# 

# In[36]:


svm_model_object = utils.setup_classification_SVM(
    kernel = 'rbf', C = 10,
    gamma = 1e-2,probability=True)

_ = utils.train_classification_SVM(
    model_object=svm_model_object,
    training_predictor_table=training_predictor_table,
    training_target_table=training_target_table)


# In[53]:


training_probabilities = svm_model_object.predict(
    training_predictor_table.as_matrix())

training_event_frequency = numpy.mean(
    training_target_table[utils.BINARIZED_TARGET_NAME].values)
utils.eval_binary_classifn(
    observed_labels=training_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=training_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='training')

validation_probabilities = svm_model_object.predict(
    validation_predictor_table.as_matrix()
)

utils.eval_binary_classifn(
    observed_labels=validation_target_table[
        utils.BINARIZED_TARGET_NAME].values,
    forecast_probabilities=validation_probabilities,
    training_event_frequency=training_event_frequency,
    dataset_name='validation')


# # hyperparameter tuning for SVM model
#     
# In the next cell try to find the best configuration of SVM model based on Gridsearch strategy and then based on what you learned so far you are able to train the SVM model based on the best parameteres and evaluate it. 
# 
# - **C:** For soft margin SVM, the bigger the C, the more penalty SVM gets and narrower the margin and fewer support vecotrs. 
# - **Degree of polynomial:** low polynomial degree can not deal with very complex datasets, with high polynomial degree it creats a huge number of features making the model too slow. if your model is overfitting you might want to reduce the polynomiakl degree. 
# - **gamma**: increasing gamma makes the bell-shape (guassian similarity) cureve narrower and as a result each instance's range of influence is smaller, conversely, a small gamma value makes the bell-shaped curve wider and decision boundary ends up smoother. if your model is overfitting, you should reduce the gamma. 

# # Practice at home: 
# In next cell you can try different kernel for SVM model and tune the hyperparameters to find the best configuration for the SVM model. 

# In[ ]:


from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(SVC(), tuned_parameters, cv=3)
clf.fit(training_predictor_table, training_target_table[utils.BINARIZED_TARGET_NAME].values)

print("Best parameters set found on development set:")
print(clf.best_params_)

