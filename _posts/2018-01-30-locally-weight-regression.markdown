---
layout: post
categories: [python,maths,machine-learning]
title: "Locally Weighted Regression"
image: /assets/images/locally-weighted-regression.png 
twitter_text: "Locally Weighted Regression"
tags: [Maths,Python,regression, lwr]
status: published
---


A couple of weeks back, I started a review of the linear models I've used over the years and and I realized that I never really understood how the locally weighted regression algorithm works. This and the fact that `sklearn` had no support for it, encouraged me to do an investigation into the working principles of the algorithm. In this post, I would attempt to provide an overview of the algorithm using mathematical inference and list some of the implementations available in Python. 

The rest of this article will be organised as follows:

* Regression 
    * Regression Function
    * Regression Assumptions
    * The Linear Regression Algorithm
* Locally Weighted Regression
    * Python Implementation
        * StatsModel Implementation
        * Alexandre Gramfort's implementation
    * Benchmark
* Conclusion
* Resources


## Notations


The following notations would be used throughout this article

| Symbol          | Meaning       |
| -------------   |:-------------:| 
| $$ y $$         |Target Variable|
| $$ X $$         |Features         |  
| $$ (X, y)   $$  |Training set  |
| $$ n $$         |Number of features|
| $$ X_i, y_i $$  |<sup>ith</sup> index of X and y |
| $$ m $$         |Number of training examples|



## Regression
Regression is the estimation of a continuous response variable based on the values of some other variable. The variable to be estimated is dependent on the other variable(s) in the function space. It is parametric in nature because it makes certain assumptions on the available data. If the data follows these [assumptions](#regression-assumptions), regression gives incredible results. Otherwise, it struggles to provide convincing accuracy. 

### Regression Assumptions

As was mentioned above, regression works best when the assumptions made about the available data are true. 
Some of these assumptions are:

* There exists a linear relationship between $$X$$ and $$y$$. 

    This assumes that a change in $$X$$ would lead to a corresponding change in $$y$$.
    This assumptions is particularly important in linear regression of which locally weighted regression is a form.

* There must be no correlation in the set of data in $$X$$. 

    The presence of correlation in $$X$$ leads to a concept known as [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity). 
    If variables are correlated, it becomes extremely difficult for the model to determine the true effect of $$X$$ on $$Y$$.

* Independence of errors

     This assumes that the errors of the response variables are uncorrelated with each other i.e the error at $$ h(x)_i $$ should not indicate the error at any other point. $$h(x)_i$$ is the estimate of the true function between Y and X. It's discussed in more context below.

* $$y$$ must have a normal distribution to the error term


### Regression Function
The regession function is a parametric function used for estimating the target variable. This function can either be linear or non-linear. Now, since this article's main focus is the locally weighted regression which is a form of the linear regression, there would be a little more focus on linear regression.

Linear regression is an approach for modelling the linear relationship between a scalar $$y$$ and a set of variables $$X$$.

![Unfitted scatter plot.](/assets/images/scatter.png) Figure 1


Given a function whose scatter plot is above, a linear regression can be modeled to it by finding the line of best fit. Finding the **line of best fit** in simple terms is really just getting the best function to represent the relationship between the $$X$$ and $$y$$ variables. This function, mostly called the `linear regression function` or more generally the `hypothesis` is a linear function which includes a dependent variable(target), a set of independent variables(features) and an unknown parameter. It's represented as:

$$ h_{\theta}(x) = \theta_{0} + \theta_{1} X_{1} + \theta_{2} X_{2}+ ... + \theta_{n} X_{n} \label{a}\tag{1}$$

When evaluated, $$h_{\theta}(x)$$ in $$\ref{a}$$ above becomes $$h(x)$$. The regression function is called `simple linear regression` when only one independent variable $$(X)$$ is involved. In such cases, it becomes $$ h(x) = \theta_{0} + \theta_{1} X_{1} + \epsilon_{1} \label{b}\tag{2}$$ It's called `multivariate linear regression` when there is more than a single value for $$X$$.

Additionally, the equation above is the equation of a line, more formally represented as $$y = mx + c$$ . Given this, to simplify the function, the intercept $$X_{0}$$ at $$\theta_{0}$$,is assumed to be $$1$$ so that it becomes a summation equation expressed as:
 
$$ h(x) = \sum_{i=0}^{n} \theta_{i} X_{i} + \epsilon_{i} \label{c}\tag{3}$$ 


An alternative representation of $$\ref{c}$$ when expressed in vector form is given as:

$$ h(x) = θ^{{T}} x_{i} \label{d}\tag{4}$$ 


###The Linear Regression Algorithm
The linear regression algorithm applies the regression function in one form or another in predicting values using `real` data.
Since this prediction can never really be totally accurate, an error (represented as $$\epsilon$$ in $$\ref{b}$$ above) is generated.
This error, formulated as $$\epsilon = |y - h(x)|$$ is the vertical distance from the actual $$y$$ value to our prediction ($$h(x)$$) for the same $$x$$.  The error has a direct relationship with the accuracy of the algorithm. This means the smaller the error, the higher the model accuracy and vice versa. As such, the algorithm attempts to minimize this error.     

The process of minimizing the error involves selecting the most appropriate features($$\theta$$) to include in fitting the algorithm. 
A popular approach for selecting $$\theta$$'s is making $$h(x)$$ as close to to $$y$$ as possible for each item in $$(X, y)$$. To do this, a function caled the `cost function` is used. The `cost function` measures the closeness of each $$h(x)$$ to its corresponding $$y$$.

> The cost function calculates the `cost` of your regression algorithm.


It's represented mathematically as:

$$ J(\theta) = (\frac{1}{2}) \sum_{i=1}^{m} \left| \left(h(x)_i - y_i \right)^2\right|  \label{e}\tag{5}$$


So, essentially, the linear regression algorithm tries to choose the best $$\theta$$ to minimize $$J(\theta)$$ which would in turn reduce the measure of error.
To do this, the algorithm starts by :

1. Choosing a base value for $$\theta$$. 
2. Updating this value to make $$J(\theta)$$ smaller. 

This goes on for many iterations until the value of $$J(\theta)$$ converges to it's local minima. An implementation of the above steps is called the `gradient descent` algorithm. 

The working principles of the gradient descent are beyond the scope of this article and would be covered in a different article in the near future. Alternatively, a very good resource on how they work is available in Sebastian Ruder's paper [here](http://ruder.io/optimizing-gradient-descent/index.html).

In summary, to evaluate $$h(x)$$, i.e make a prediction, the linear regression algorithm:

1. Fits $$\theta$$ to minimize $$\sum_{i}(y_i - \theta^T x_i)^2 \label{f}\tag{6}$$. 

    Upon successful fitting, the graph of the function above becomes
    ![Fitted scatter plot.](/assets/images/fitted_scatter.png)

2. Ouputs $$\theta^T x$$.



## Locally Weighted Regression


![Unfitted LWR.](/assets/images/unfitted_lwr.png) Figure 3
![Unfitted LWR.](/assets/images/fitted_lwr.png)   Figure 4

Compared to Figure 1, Figure 3 above, has a relatively higher number of mountains in the input/output relationship. Attempting to fit this with linear regression would result in getting a very high error and a line of best fit that does not optimally fit the data as shown in Figure 4. This error results from the fact that linear regression generally struggles in fitting functions with non-linear relationships. These difficulties introduce a new approach for fitting non-linear multivariate regression functions called "locally weighted regression".

Locally weighted regression is a non-parametric variant of the linear regression for fitting data using multivariate smoothing. Often called `LOWESS` (locally weighted scatterplot smoothing), this algorithm is a mix of multiple local regression models on a meta `k-nearest-neighor`.
It's mostly used in cases where linear regression does not perform well i.e finds it very hard to find a line of best fit. 

It works by fitting simple models to localized subsets ,say $$x$$, of the data to build up a function that describes the deterministic part of the variation in the data. The points covered by each point (i.e neighorhood) $$x$$ is calculated using k-nearest-neighors.

> For each selected $$x_i$$, LWR selects a point $$x$$ that acts as a neighorhood within which a local linear model is fitted.

LOWESS while fitting the data, gives more weight to points within the neighorhood of $$x$$ and less weight to points further away from it. A user dependent value, called the **bandwidth** determines the size of the data to fit each local subset. 
The given weight $$w$$ at each point $$i$$ is calculated using:

$$w_i = \exp(- \frac{(x_i - x ) ^ 2}{2 \tau ^ 2}) \label{g}\tag{7}$$



Since, a small $$|x_i − x|$$ yields a $$w(i)$$ close to 1 and a large  $$|x_i − x|$$ yields a very small $$w(i)$$, 
the parameter ($$\theta$$) is calculated for the LOWESS by giving more weight to the points within the 
neighorhood of $$x$$ than the points outside it.


Essentially, this algorithm makes a prediction by:

1. Fitting $$\theta$$ to minimize $$\sum_{i}w_i(y_i - \theta^T x_i)^2  \tag{8}$$.       

    The fitting is done using either **weighted linear least squares** or the **weighted quadratic least squares**. The algorithm is called the LOESS when it's fitted using the **weighted quadratic least square** regression.

2. Ouputs $$\theta^T x$$.



### Python Implementation 


The support for LOWESS in Python is rather poor. This is primarily because the algorithm is computationally intensive given that it has to fit $$j$$ number of lines at every point $$x_i$$ within the neighorhood of $$x_i$$.

Regardless of this challenge, there are currently 2 implementations of the LOWESS algorithm in Python that I have come across. These are:

1. Statsmodel Implementation
[http://www.statsmodels.org/devel/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html](http://www.statsmodels.org/devel/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html)

    Statsmodel is a python package that provides a range of tools for carrying out statistical computation in Python.

    It provides support for LOWESS in it's `statsmodel.nonparametric` module. Statsmodel supports

    >A lowess function that outs smoothed estimates of endog at the given exog values from points (exog, endog)

    The *exog* and *endog* expressions in the quote above represent 1-D numpy arrays.These arrays denote $$x_i$$ and $$y_i$$ from the equation

    This function takes input $$y$$ and $$x$$ and estimates each smooth $$y_i$$ closest to $$(x_i, y_i)$$ based on their values of $$x$$. It essentially uses a **weighted linear least square** approach to fitting the data. 
    A downside of this is that statsmodels combines fit and predict methods into one, and so doesn't allow prediction on new data.

2. Alexandre Gramfort's implementation

    [https://gist.github.com/agramfort/850437](https://gist.github.com/agramfort/850437)

    This implementation is quite similar to the statsmodel implementation in that it supports only 1-D numpy arrays.
    
    The function is:

``` python
import numpy as np
from scipy import linalg


def agramfort_lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

```
    

### Benchmark
To benchmark the 3 implementations, let's declare the following constants:
1. Let $$x$$ be a set of 1000 random float between $$-\tau$$ and $$\tau$$.

```python
x = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=1000)
```

2. Let $$y$$ be a function of the sine of x.

``` python
y =  np.sin(x)
```

A scatter plot of the relationship between $$x$$ and $$y$$ is shown below:

   ![Unfitted LWR.](/assets/images/statsmodel_case.png) Figure 3

Now, predicting with lowess using :

Statsmodel LOWESS:
```python
>>> from statsmodels.api.nonparametric import lowess
>>> %timeit lowess(y, x, frac=2./3)
1 loop, best of 3: 303 ms per loop
```

Alexandre Gramfort's implementation
```python
>>> %timeit agramfort_lowess(y, x)
1 loop, best of 3: 837 ms per loop
```



### Conclusion

Though the pure LOWESS functions are hardly used in Python, I hope I've been able to provide a little intuition into how they work and possible implementation. 

On why this is maths intensive, while I believe we can make-do with black-box implementations of fundamental tools constructed by our more algorithmically-minded colleagues, I am a firm believer that the more understanding we have about the low-level algorithms we're applying to our data, the better practitioners we'll be.


### Resources

1. [https://en.wikipedia.org/wiki/Linear_regression](https://en.wikipedia.org/wiki/Linear_regression)
1. [https://jeremykun.com/2013/08/18/linear-regression/](https://jeremykun.com/2013/08/18/linear-regression/)
1. [https://stackoverflow.com/questions/26804656/why-do-we-use-gradient-descent-in-linear-regression](https://stackoverflow.com/questions/26804656/why-do-we-use-gradient-descent-in-linear-regression)
1. [https://en.wikipedia.org/wiki/Regression_analysis#Regression_models](https://en.wikipedia.org/wiki/Regression_analysis#Regression_models)
1. [https://www.quora.com/In-what-situation-should-I-use-locally-weighted-linear-regression-when-I-do-predictions](https://www.quora.com/In-what-situation-should-I-use-locally-weighted-linear-regression-when-I-do-predictions)
1. [https://www.quora.com/Why-is-that-in-locally-weighted-learning-models-we-tend-to-use-linear-regression-and-not-non-linear-ones](https://www.quora.com/Why-is-that-in-locally-weighted-learning-models-we-tend-to-use-linear-regression-and-not-non-linear-ones)