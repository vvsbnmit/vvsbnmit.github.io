---
layout:     post
title:      "Statistical Learning"
subtitle:   "ISLR using Python"
date:       2018-12-17 10:00:00
author:     "Vikram Vishal"
header-img: "assets/tfe/colony.jpg"
summary:    "This Chapter covers the basic of Statistical Learning"
---


## Statistical Learning:

Input Variable denoted by Xi

Target Variable denoted by Y

Y = f(X)+e  Where 'e' has a mean of zero.

Statistical learning is a set of approaches for estimating 'f'

Reasons for estimating f
1. Prediction
2. Inference

The inaccuracy of a model is due to two factors:
1. Reducible error : These type of error can be reduced by estimating our function 'f' as close as possible.
2. Irreducible error: These type of error is due to 'e' from above equation and cannot be reduced.

Estimation of function 'f' can be done using two ways:
1. Parametric Methods - We make assumption about the shape of the function for e.g. Linear, Quadratic etc and estimates the coefficient using methods like OLS and other.

2. Non-Parametric Methods - Do not make assumption about the functional form of 'f'. A very large number of observation is required to estimate 'f' and coefficient unlike Parametric estimates.

Prediction Accuracy and Interpretability are inversely proportional to each other.

Categories of Statistical Modelling

1. Supervised Learning - Target varible is given which is of our interest and prediction.

2. Unsupervised Learning - No Target variable is present. We want to understand the relationship between variables or observation.


Regression - When target variable is quantitative
Classification - When target variable is Qualitative(classes)

Logistic regression is called regression although it is used in classification of two-class is because of the probability estimates it gives out for each class which are quantitative or continuous value in nature. 

Methods such as KNN and Boosting(Trees) can be use for both regression and classification problems.


Accessing Model Accuracy - MSE (Mean Squared Error) for regression and Training error rate for Classification.

Flexibility of Model(Flexibility of Curve) ~ Degree of Freedom

Cross Validation is used to estimate Test MSE and to estimate the shape of 'f' and coefficient that doesn't overfit the data.

Bias-Variance trade off: U-Shape curve of Test MSE.

MSE of x0(test observation) consists of 3 fundamental quantities
1. Var f'(x0)
2. [Bias(f'(x0))]^2
3. Var(e)  where f' is the estimated function which we fit using our Training Dataset.

Variance - By what amount the f' will change if we estimate it using different training data sets.
Bias - It arrises when we fit a simple model for complex model.

Bayes Classifier produces the lowest possible test error rate called the Bayes error rate. Bayes error rate is analogous to irreducible error rate i.e. Var(e)


As for regression degree of freedom defines the flexibility of the model in KNN the number of K defines the flexibility of the model when K=1 model has high variance hence low training error and high test error.


```python
import numpy as np

x = np.random.normal(0,0.1,100)
y = np.random.normal(0,0.1,100)
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.scatter(x,y)
```




    <matplotlib.collections.PathCollection at 0x2217f5f2c18>




![png](Chapter%20-%202_files/Chapter%20-%202_2_1.png)



```python
#Reding Auto file
#We have mentioned that the first row to be treated as the header and the first column should be treated as index
import pandas as pd

Auto = pd.read_csv(r'C:\Users\Vikram\Desktop\ISLR\Data\Auto.csv',header='infer', index_col='Column1')
```


```python
# equivalent to dim(Auto) in R. However in R the answer will be (392,9).

Auto.shape
```




    (392, 9)




```python
# Similar to names(Auto) in R

list(Auto)
```




    ['mpg',
     'cylinders',
     'displacement',
     'horsepower',
     'weight',
     'acceleration',
     'year',
     'origin',
     'name']




```python
# Similar to plot(Auto$cylinders,Auto$mpg) in R

plt.scatter(x = Auto.cylinders,y = Auto.mpg)
plt.xlabel('Cylinders')
plt.ylabel('mpg')
```




    Text(0,0.5,'mpg')




![png](Chapter%20-%202_files/Chapter%20-%202_6_1.png)



```python
# Boxplot in R requires you to convert cylinder into categorical variable using as.fator(cylinder). Python's seaborn library auto
#-matically takes care of it.

import seaborn as sns
sns.boxplot(x = Auto.cylinders,y = Auto.mpg)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2217f661e48>




![png](Chapter%20-%202_files/Chapter%20-%202_7_1.png)



```python
# Equivalent to hist(Auto$mpg in R)
plt.hist(Auto.mpg)
```




    (array([13., 78., 73., 58., 53., 48., 37., 22.,  4.,  6.]),
     array([ 9.  , 12.76, 16.52, 20.28, 24.04, 27.8 , 31.56, 35.32, 39.08,
            42.84, 46.6 ]),
     <a list of 10 Patch objects>)




![png](Chapter%20-%202_files/Chapter%20-%202_8_1.png)



```python
# Similar to pairs(~mpg+displacement+horsepower+weight+acceleration, Auto) in R

sns.pairplot(Auto, 
             vars=['mpg','displacement','horsepower','weight','acceleration'])
```




    <seaborn.axisgrid.PairGrid at 0x2217f773908>




![png](Chapter%20-%202_files/Chapter%20-%202_9_1.png)



```python
# Equivalent to summary(Auto) in R

Auto.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>392.000000</td>
      <td>392.000000</td>
      <td>392.000000</td>
      <td>392.000000</td>
      <td>392.000000</td>
      <td>392.000000</td>
      <td>392.000000</td>
      <td>392.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.445918</td>
      <td>5.471939</td>
      <td>194.411990</td>
      <td>104.469388</td>
      <td>2977.584184</td>
      <td>15.541327</td>
      <td>75.979592</td>
      <td>1.576531</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.805007</td>
      <td>1.705783</td>
      <td>104.644004</td>
      <td>38.491160</td>
      <td>849.402560</td>
      <td>2.758864</td>
      <td>3.683737</td>
      <td>0.805518</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>68.000000</td>
      <td>46.000000</td>
      <td>1613.000000</td>
      <td>8.000000</td>
      <td>70.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.000000</td>
      <td>4.000000</td>
      <td>105.000000</td>
      <td>75.000000</td>
      <td>2225.250000</td>
      <td>13.775000</td>
      <td>73.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.750000</td>
      <td>4.000000</td>
      <td>151.000000</td>
      <td>93.500000</td>
      <td>2803.500000</td>
      <td>15.500000</td>
      <td>76.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29.000000</td>
      <td>8.000000</td>
      <td>275.750000</td>
      <td>126.000000</td>
      <td>3614.750000</td>
      <td>17.025000</td>
      <td>79.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.600000</td>
      <td>8.000000</td>
      <td>455.000000</td>
      <td>230.000000</td>
      <td>5140.000000</td>
      <td>24.800000</td>
      <td>82.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Equivalent to summary(Auto$mpg) in R

Auto.mpg.describe()
```




    count    392.000000
    mean      23.445918
    std        7.805007
    min        9.000000
    25%       17.000000
    50%       22.750000
    75%       29.000000
    max       46.600000
    Name: mpg, dtype: float64


