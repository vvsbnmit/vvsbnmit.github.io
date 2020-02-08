
## Resampling Method

There are 2 type of resampling method

1. Cross-Validation
2. Bootstrap

Resampling is done to perform models-assessment and model-selection

### Cross-Validation

In cross validation we randomly divide the data sets into two halves one is used to fit the statistical model known as training data sets and the other half is used to asses the model known as validation data set.

There are two drawbacks of cross-validation

1. Since we randomly divide the data sets it is possible that for different run we will get different Test MSE which will not give us a clear picture of our model performnce.

2. Since we are only using half of our data to fit our model it does not capture the real form of the data and in that case the validation set tends to overestimate the data sets.

#### Leave-one-out cross validation

Here we fit the model using (n-1) data sets and validate on the one left out observation. We perform this task n times and take the average of all MSE in the end which gives us Test MSE. It addresses the two concerns mentioned above. But there is one drawback in this method that is it is cost expensive since we have to fit the model n times and in case where n is a large number.

#### k-fold cross validation

In this approach we divide our data sets into k equal buckets and fit our model using (k-1) data and test on the left out chunk of data. We repeat this approach k times and averages the MSE from each run to estimates the Test MSE. The obvious benefit is that it doesn't suffer from the limitation of LOOC and is computationally cheap and also takes care of Variance with which LOOC suffers.

### Bootstrap

In Bootstrap we draw samples with replacement to generate different data sets on which we perform our statistical analysis.


```python
# importing required libraries
import pandas as pd
import random
```


```python
# reading the Auto data set

Auto = pd.read_csv(r'C:\Users\Vikram\Desktop\ISLR\Data\Auto.csv',header='infer', index_col='Column1')
```


```python
#setting up seed to randomly draw data for cross validation from Auto data set
random.seed(100)
```


```python
# defining the independent variable.

Auto_x = Auto.drop(['mpg','name'],axis=1)
```


```python
# randomly splitting data into training and test data set.

Auto_x_train = Auto.drop(['mpg','name'],axis=1).sample(196)
Auto_y_train = Auto.mpg.sample(196)
```


```python
# columns in Auto data set
Auto.columns, Auto.shape
```




    (Index(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration', 'year', 'origin', 'name'],
           dtype='object'), (392, 9))




```python
# importing Linear Regression from sklear. We will use Linear regression to study the impact of resampling.

from sklearn.linear_model import LinearRegression
```


```python
# fitting the Linear Regression model.

reg = LinearRegression()
reg.fit(Auto_x_train, Auto_y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
# importing numpy in order to calculate the mean error produced by our model. As we can see that MSE is -1.072
import numpy as np

np.mean(Auto.mpg - reg.predict(Auto_x))
```




    0.9661936266354951




```python
# generating different random seed to demonstrate the issues of random resampling
random.seed(200)
```


```python
# drawing our independent and dependent variable randomly with new seed

Auto_x_train1 = Auto.drop(['mpg','name'],axis=1).sample(196)
Auto_y_train1 = Auto.mpg.sample(196)
```


```python
# fitting Linear Regression on the new data sets

reg1 = LinearRegression()
reg1.fit(Auto_x_train1, Auto_y_train1)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
# calculating the mean error and as visible there is a huge diffrence from -1.07 to 0.144. This tells us the issue which 
# cross validation suffers from which is variablility.

np.mean(Auto.mpg - reg1.predict(Auto_x))
```




    0.5194935571596861




```python
# implementing the Leave-One-Out cross validation from sklearn

from sklearn.model_selection import LeaveOneOut

Auto_looc_X = Auto.drop(['name','mpg'], axis=1)

looc = LeaveOneOut()

looc.get_n_splits(Auto_looc_X)
```




    392




```python
# below is the different combination of indices from our data sets which were split into training and test using LOOC
# implementation

#for train_index, test_index in looc.split(Auto_looc_X):
   #print("TRAIN:", train_index, "TEST:", test_index)
   #X_train, X_test = np.array(Auto_looc_X)[train_index], np.array(Auto_looc_X)[test_index]
   #y_train, y_test = np.array(Auto.mpg)[train_index], np.array(Auto.mpg)[test_index]
   #print(y_train, y_test)
```


```python
# implemnting the k-cross validation with k=5 which will split the data set in 5 equal samples and will evaluate the performance.
# As visible the estimates are more stable now and k-cross validation has addressed the issue of variability to some extent.

from sklearn.model_selection import cross_val_score, cross_val_predict

reg_k = LinearRegression()
scores = cross_val_score(reg_k, Auto_x, Auto.mpg, cv=5)
scores
```




    array([0.55691895, 0.68950582, 0.82212138, 0.6795006 , 0.2250594 ])




```python
#LOOC
%matplotlib inline
import matplotlib.pyplot as plt

meg_pred =  cross_val_predict(reg_k, Auto_x, Auto.mpg, cv=392)

fig, ax = plt.subplots()
ax.scatter(Auto.mpg, meg_pred, edgecolors=(0,0,0))
ax.plot([Auto.mpg.min(), Auto.mpg.max()], [Auto.mpg.min(),Auto.mpg.max()], 'k--',lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plt.show()
```


![](/assets/Chapter%20-%205_files/Chapter%20-%205_17_0.png)



```python
#k-cross validation with k=5
%matplotlib inline
import matplotlib.pyplot as plt

meg_pred =  cross_val_predict(reg_k, Auto_x, Auto.mpg, cv=5)

fig, ax = plt.subplots()
ax.scatter(Auto.mpg, meg_pred, edgecolors=(0,0,0))
ax.plot([Auto.mpg.min(), Auto.mpg.max()], [Auto.mpg.min(),Auto.mpg.max()], 'k--',lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plt.show()
```


![](/assets/Chapter%20-%205_files/Chapter%20-%205_18_0.png)


###### For Bootstrap we can use the random procedure used above with a small addition of 'replacement=True'
