
## Linear Model Selection and Regularization

OLS is used to build our Linear Regression model but it has some limitation. For e.g. OLS performs well when n>>p where n is the total no. of observation whereas p is the variables. But if that is not the case the Linear Regression build using OLS suffer from Variance and if the case is of p>n in that case there is no longer unique least square estimates and the variance is infinite. In this chapter we will see 3 ways to overcome this problem:

1. Subset Selection
2. Shrinkage
3. Dimensionality Reduction


### Subset Selection.
Consider selection the subset of best predictor.

1. Best subset selection - We take the combination of variables starting from one at a time to all the variable and fit the regression and compute the RSS for each selection category for e.g. RSS of all model with one predictors and so on. We then select the model with least RSS. At the end of this step we will (p+1) models one from each selection category and the null model which estimates the average value. Among this (p+1) we select the model using cross validation approach. This method suffers from the limitation of computationally expensive when p is large.


2. Stepwise selection -> This method have 3 subtypes:

a. Forward Selection -> In this method we start adding predictors based on lowest RSS and highest R-square value. This method can also work when n<p but we will only have submodels in that case.

b. Backward elimination -> In this method we start with a model containing all the predictors and start removing one predictor at a times based on lowest RSS and highest R-square value. However, unlike stepwise selection this method fails when n<p as we won't be able to fit a regression model using least square.

c. Hybrid -> It is the combination of above two methods.

Note -> The final model obtained from these different approaches might not have same predictors in their model.

The model assessment are done be looking at below stats. These stats take consider model complexity and variance and hence give a good assessment of models on unseen data:

1.Cp
2.AIC
3.BIC -> Almost same as above two but penalizes the model more for addtional predictors
4.Adjusted R-square

Given above metrics 'Cross-Validation' still is the most reliable way as it doesn't make assumption.

### Shrinkage Methods

There are two type of shrinkage method

1. Ridge Regression - Same as OLS but add a penalty term which updates the coefficient ->0. But unlike OLS it doesn't take care of scale of variables and therefore, before applying this method all variable should be scaled i.e. standardized to with the Standard deviation of 1. also, term as l2-loss.

2. Lasso Regression -> In this shrinkage mechanism the coefficient are forced to be zero. In a way Lasso act as subset selection method.

The main difference between Lasso and Ridge is that Lasso will have fewer predictors in its model whereas Ridge will have all the predictors in its model although there coefficient will be small. Therefore, Lasso is more interpretable when 'p' is large.

The tuning parameter 'lambda' is selected using cross-validation approach.

### Dimension Reduction Methods.

In this approach we reduces the number of predictors by capturing the relationship of predictors in new variables which are less than the number of predictors.

The relationship can be captured in two ways:
1. Principal Component Analysis -> Similar to Ridge therefor scaling of predictor is required. While deriving components target variable are not considered which make this step a unsupervised learning.

2. Partial Least Square -> Unlike PCA the components are derived by considering target variable which makes it a supervised learning.


A pretty implementation of this in python can be found here:

http://www.science.smith.edu/~jcrouser/SDS293/labs/lab8-py.html 
