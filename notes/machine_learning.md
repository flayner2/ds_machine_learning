# Machine Learning

Link to slides: [**Introduction to Machine Learning**](https://docs.google.com/presentation/d/1WurfW8OWRqjiSmzmwOW71iN6CEShkbfOnyETqTvf6BE/edit#slide=id.g5c6d0d8677_0_85)

Link to book: [**Introduction to Statistical Learning**](https://drive.google.com/file/d/1ke7z4bL_5eSwXulTkYC1f8owJRnb2Onl/view?usp=sharing)

```toc
```

## 1. What is Machine Learning?

**Machine Learning** is a method of data analysis that automates analytical model building. Using algorithms that iteratively learn from data, machine learning allows computers to find hidden insights without being explicitly programmed where to look. Some uses for machine learning are:

- classifying e-mails (Spam, Offers, etc.);
- extracting sentiment from text;
- targeting ads;
- identifying network security breaches;
- rating survivorship based on traits;
- and many more...

## 2. What are Neural Networks and Deep Learning?

**Neural Networks** are a way of modeling biological neuron systems mathematically. These networks can then be used for solving tasks that many other types of algorithms can not (e.g., image classification). **Deep Learning** refers to neural networks with more than one hidden layer.

## 3. Types of Machine Learning implementations

### 3.1 Supervised Learning

**Supervised learning** algorithms are trained using *labeled examples*, such as an input where the desired output is known. For example:

- classifying email as *spam* vs *legitimate*;
- classifying movie reviews as *positive* vs *negative*.

The network receives a set of inputs alongside the corresponding correct outputs, and the algorithm learns by comparing its actual result with correct ones to find errors. It then modifies the model accordingly. Supervised learning is used commonly in applications where historical data predicts likely future events.

#### 3.1.1 Steps in Machine Learning projects

1. **Data Acquisition**: get the data you will work with (customers, sensors, prebuilt datasets, sequencing data, etc.);
2. **Data Cleaning**: clean and format your data to use within your programming environment (usually done with `pandas` in Python;
3. **Split the data into *Training* and *Test* data**: take some portion of the data (e.g., 30%) to be the actual *test* data, and the rest (the majority) to be the *training* data;
4. **Model Training & Building**: use the *training* data to build and train your Machine Learning model;
5. **Model Testing**: run the *test* data through the trained and built model and compare the predictions to the actual label the data has (for **Supervised Learning**);
6. **Adjust Model Parameters (Optional)**: go back one step and adjust the model, depending on the previous results being satisfactory or not;
7. **Model Deployment**: deploy the built model to the real world.

There is an inherent problem with this simplified methodology: *is it fair to use our single split of the data to evaluate our model's performance? After all, we were given the chance to update the model parameters again and again.* Because of that, data is usually split into **3** **sets**:

- **Training Data**: used to train model parameters;
- **Validation Data**: used to determine what model [**hyperparameters**](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) to adjust (this is the *test* data from before);
- **Test Data**: used to get some final performance metric.

#### 3.1.2 Model Evaluation - Classification

There are some key classification metrics to understand:

- Accuracy;
- Recall;
- Precision;
- F1-Score.

Typically in any classification task your model can only achieve two results:

- Either your model was **correct** in its prediction;
- Or your model was **incorrect** in its prediction.

*Correct* or *incorrect* apply to situations in which there are multiple classes, for example: predicting the species of an animal, based on an image, between dog, cat, or goat. The prediction for a particular animal (dog, cat, or goat) may be either *correct* or *incorrect*:

- Image of a cat > model prediction = cat > *correct*;
- Image of a cat > model prediction = dog > *incorrect*.

In the real world, not all incorrect or correct matches hold equal value! Also in the real world, a single metric won’t tell the complete story! The 4 metrics mentioned before may be summarized in a [**confusion matrix**](https://en.wikipedia.org/wiki/Confusion_matrix).

![https://miro.medium.com/max/2102/1*fxiTNIgOyvAombPJx5KGeA.png](https://miro.medium.com/max/2102/1*fxiTNIgOyvAombPJx5KGeA.png)

Image 1: Confusion matrix.

**Accuracy** in classification problems is the *number of correct predictions* made by the model divided by the *total number of predictions*. 

$$Accuracy = \frac{tp + tn}{tp + fp + tn + fn}$$

**Accuracy** is useful when the target classes are well balanced (all classes have roughly the same amount of observations, e.g., the same amount o dog, cat, and goat images). It is not a good choice with *unbalanced* classes:

- We have 100 images, and 99 are dogs;
- We train a model that *only predicts dogs;*
- Model accuracy will be 99%, but it will only be able to predict dogs;
- If the same model would be used on a dataset with 30 dogs and 70 cats, its accuracy would be 30%.

**Recall** is the ability of a model to find all the relevant cases within a dataset. It is the number of true positives *divided by* the number of true positives *plus* the number of true negatives.

$$Recall = \frac{tp}{tp + tn}$$

**Precision** is the ability of a classification model to identify only the relevant data points. It is defined as the number of true positives *divided by* the number of true positives *plus* the number of false positives.

$$Precision = \frac{tp}{tp + fp}$$

Often there's a trade-off between **recall** and **precision**. While **recall** expresses the ability to find *all relevant instances in a dataset*, **precision** expresses *the proportion of the data points our model says was relevant that actually were relevant*.

**F1-Score** is used in cases where we want to find an optimal blend of **precision** and **recall** by combining the two metrics. It is defined as the [**harmonic mean**](https://en.wikipedia.org/wiki/Harmonic_mean) **of **precision** and **recall**:

$$F_1 = 2 \times \frac{precision * recall}{precision + recall}$$

We use the harmonic mean instead of a simple average because it punishes extreme values.

We can also quantify the *error rate* for a classificator by using the following equation:

$$\frac{1}{n}\sum^n_{i=1}I(y_i\ne \hat{y_i})$$

In this equation, $I$ is an *indicator variable* that equals 1 if $y_i\ne \hat{y_i}$ and 0 if $y_i= \hat{y_i}$. This equation is refered to as the *training error rate* because it is computed based on the data that was used to train our classifier.

#### 3.1.3 Model Evaluation - Regression

**Regression** is a task when a model attempts to predict continuous values instead of categorical values (which is **classification**). For example, attempting to predict the price of a house given its features is a **regression** task. Attempting to predict the country of a house given its features would be a **classification** task.

The most common evaluation metrics for regression are:

- Mean Absolute Error;
- Mean Squared Error;
- Root Mean Squared Error.

**Mean Absolute Error** is the *mean* of the *absolute value of errors*. However, it won't punish large errors. For the below and beyond, $y$ is the actual value and $\hat{y}$ is the predicted value.

$$\frac{1}{n}\sum_{i=1}^n|y_i - \hat{y_i}|$$

**Mean Squared Error** is the mean of the squared errors. Larger errors are noted more than with **MAE**, making **MSE** more popular. The problem is that the squaring of the values also squares the units, making the results difficult to interpret.

$$\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y_i})^2$$

**Root Mean Square Error** is the root of the mean of the squared errors, and it's the most popular error metric, as it has the same unit as $y$.

$$\sqrt{\frac{1}{n}\sum^n_{i=1} (y_i - \hat{y_i})^2}$$

When we're calculating the error (usually MSE) for a model, we're trying to minimize the *test* MSE, and no the *training* MSE. Most machine learning models already try to minimze the *training* MSE, but a low *training* MSE doesn't guarantee a low *test* MSE or a good fit for the *test* data. Rather, if pursuing to reduce the *training* MSE in hopes of imporving the model's predictions for the *test* data, one may fall into what is known as [**overfitting**](https://en.wikipedia.org/wiki/Overfitting). In short, **overfitting** happens when we try to fit a model so well into the *training* set that, at some point, it will be a perfect fit for that dataset but will, in turn, be unable to correctly predict or fit over previously unseen data.

![Machine%20Learning%2019b2b0877b3745a7954b45efa1a38488/2.10-1.png](2.10-1.png)

In the left panel, the green curve has a large flexibility and low *training* MSE, but it is not very close to the true function that describes the data (black line). Green is a function that's overfitted.

### 3.2. Unsupervised Learning

In some situations, we have a collection of predictor measurements $x_i, i=1...n$, but no response variables $y$ to *supervise* our model. For example, we could have a housing dataset for which we want to predict house prices, but the dataset itself doesn't contain any house prices. In this case, we only have access to the variables we can use to predict the prices, but no way of comparing our predictions to real values. That is a situation in which the applied models would fit in the category of **unsupervised learning**.

A common example of unsupervised learning techniques is **cluster analysis** (aka **clustering**), and its goal is to check if our data can be discriminated into a certain ammount of groupings, or *clusters* of data points that are relatively distinct from each other.

![image](https://community.alteryx.com/t5/image/serverpage/image-id/42991i9D95EADC2C491B3C/image-size/large?v=1.0&px=999)

A visual representation of a cluster analysis using the famous *iris* dataset available on the standard distribution of the R statistical language.

## 4. Machine Learning with Python

We will use the [**scikit-learn**](https://scikit-learn.org/stable/) package as it is the most popular **Machine Learning** package for Python and has a lot of algorithms built-in.

Every algorithm is exposed in `scikit-learn` via an "**Estimator**". First you'll import the model:

```python
# The generic form is
from sklearn.family import Model
# An actual example
from sklearn.linear_model import LinearRegression
```

**Estimator parameters**: all the parameters of an estimator can be set when it is instantiated, and have suitable default values. For example:

```python
model = LinearRegression(normalize=True)
print(model)

"""
Output:
LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
"""
```

After the model is defined with its parameters, you should feed it some data. Of course, the data has first to be split into the *test* and *training* sets.

```python
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)
print(X)

"""
Output:
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
"""

print(list(y))

"""
Output:
[0, 1, 2, 3, 4]
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train)

"""
Output:
array([[4, 5],
       [0, 1],
       [6, 7]])
"""

print(y_train)

"""
Output:
[2, 0, 3]
"""

print(X_test)

"""
Output:
array([[2, 3],
       [8, 9]])
"""

print(y_test)

"""
Output:
[1, 4]
"""
```

Now that the data is split, we can train/fit the model on the *training* data:

```python
model.fit(X_train, y_train)
```

Now, the model is ready to predict labels or values on the *test* set. We get predicted values using the `predict` method:

```python
predictions = model.predict(X_test)
```

Then we evaluate our model by comparing our predictions to the correct values. The evaluation method depends on what sort of **Machine Learning** algorithm we are using.

## 5. Linear Regression

In the words of [**Francis Galton**](https://en.wikipedia.org/wiki/Francis_Galton), "a father's son's height tends to **regress** (or drift towards) the mean (average) height". Galton is referring to the fact that if a particularly tall man has a son, that son is probabaly also going to be quite tall, but he's probably not gonna be as tall as his father, and his height is likely gonna shift toward's the population's average. This is the phenomenon Galton named **regression**. 

![image](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1200px-Linear_regression.svg.png)

The central line is the regression line.

All we're trying to do when we calculate our regression line is draw a line that's as close to every dot (data point) as possible. For **classic linear regression**, or the "**Least Squares Method**", you only measure the closeness in the "*up and down*" direction.

By applying this concept to a graph with many data points, we could do things like tell a man how tall we expect his son to be *before he even has a son*.

Our goal with linear regression is to minimize the vertical distance between all the data points and our line. There are many different ways to minimize this:

- sum of squared errors;
- sum of absolute errors;
- etc.

Overall, all these methods have a general goal of minimizing this distance. One of the most popular methods is the **[least squares method](https://en.wikipedia.org/wiki/Least_squares)**. This method is fitted by minimizing the *sum of the squares of the residuals*.

The **residuals** for an observation is the difference between the observation (the $y$ value) and the fitted line.

![image](https://cdn.kastatic.org/googleusercontent/Ebu4-AAwd4Z3irAQ9-AVyvA2abB-rb8cvQBjy60N42qD7JcDyd81bvz8DRiX6y2op9w2ryROslzP9OFtJ5PO9i6s)

Residuals are marked by the arrows.

### 5.1 Linear Regression with Python

First step is to get the features and the data we're trying to predict. This example uses a fictional housing prices dataset:

```python
# The features for the price prediction. We won't use text data such as 'Address'
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]
# The data we're trying to predict
y = df['Price']
```

Then we import `sklearn.model_selection.train_test_split` to split our data into *training* and *test* sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
																										random_state=101)
```

`X` and `y` are our dataframes, `test_size` defines the proportion of the data that is going to be used for the *test* set, and `random_state` is a seed for the random splitting.

Now, we need to import, create and train the model:

```python
from sklearn.linear_model import LinearRegression

# Instantiate the model
lm = LinearRegression()

# Train or 'fit' the model on the training data. This happens inplace
lm.fit(X_train, y_train)

# Check the values for the linear model
# Intercept !NOTE: the trailing '_' is intentional
print(lm.intercept_)

# Coefficients for each feature
print(lm.coef_)

# Create a dataframe to organize the coefficients
cdf = pd.DataFrame(lm.coef_, X_train.columns, columns=['Coeff'])
```

The interpretation for these coefficients is simplified as something like:

> If you hold the value for each other coefficient still, an increase of one unit in, for example, *Avg. Area Income*, means an increase of $y$ (in this case, around$21.528276$) in *Price*.
> 

**NOTE**: If you want to do a similar analysis on real data, follow the steps below:

```python
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.keys())
```

Now, to get *price* predictions with our model:

```python
# We only pass the features, not the target
# In this case, we want to use a dataset our model has never seen
# So, we pass X_test to the model
predictions = lm.predict(X_test)
print(predictions)
```

To check the correctness of the predictions, we could use a scatter plot:

```python
plt.scatter(y_test, predictions)
```

![Machine%20Learning%2019b2b0877b3745a7954b45efa1a38488/house_price_pred.png](house_price_pred.png)

A perfectly straight line would mean our predictions were perfect. A slight slope like this one tells us we've done a pretty good job.

We could also plot a histogram of the residuals:

```python
# The residuals are the difference between the actual values and our predictions
sns.displot((y_test - predictions), kde=True, alpha=.4, edgecolor=None)
```

![Machine%20Learning%2019b2b0877b3745a7954b45efa1a38488/house_price_residuals.png](house_price_residuals.png)

We can also calculate our **[loss functions](https://en.wikipedia.org/wiki/Loss_function)** to see how good our predictions were:

```python
from sklearn import metrics

# MAE
print(metrics.mean_absolute_error(y_test, predictions))
# MSE
print(metrics.mean_squared_error(y_test, predictions))
# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

Each of these functions take our *test targets* (`y_test`) and our predicted values (`predictions`) as parameters.

## 6. The bias-variance trade-off

The expected **MSE** for a given value of $x_0$ can be decomposed into:

$$E(y_0-\hat{f}(x_0))^2=Var(\hat{f}(x_0)+[Bias(\hat{f}(x_0)]^2+Var(\epsilon)$$

If that's the case, to reduce the **MSE** for a particular model, one should seek to reduce the terms $Var(\hat{f}(x_0))$ and $[Bias(\hat{f}(x_0)]^2$, both of which are positive by definition; in other terms, the expected *test* **MSE** cannot be lower than $Var(\epsilon)$, where $\epsilon$ is the irreducible error term. 

For a statistical learning method, **variance** refers to the ammount by which $\hat{f}$ would change if we estimated it using a different *training* data set. If a method has high variance then small changes in the *training* data can result in large changes in $\hat{f}$. In general, *more flexible statistical methods have higher variance*.

**Bias** refers to the error that is introduced by approximating a potentially very complicated real-life problem by a much simpler model. Generally, *more flexible methods result in less bias*.

As a general rule, as we use more flexible methods, the variance will increase and the bias will decrease.