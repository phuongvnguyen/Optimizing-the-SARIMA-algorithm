#!/usr/bin/env python
# coding: utf-8

# $$\large \color{green}{\textbf{Optimizing the SARIMA algorithm}}$$ 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# 
# This Machine Learning program was written by Phuong V. Nguyen, based on the $\textbf{Anacoda 1.9.7}$ and $\textbf{Python 3.7}$.
# 
# 
# $$\underline{\textbf{Main Contents}}$$
# 
# $\text{1. Main Job:}$ The SARIMA algorithm is one of the most common forecasting tools. An optimal SARIMA would make a better forecast for a number of interesting time-series variables.  Thus,  the main purpose of this project is to introduce how to optimize SARIMA algorithm. Three criteria, such as Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), and Log-Likelihood, are used to choose the optimal SARIMA model.
# 
# $\text{2. Dataset:}$ 
# 
# One can download the dataset used to replicate my project at my Repositories on the Github site below
# 
# https://github.com/phuongvnguyen/Optimizing-the-SARIMA-algorithm
# 

# # Preparing Problem
# 
# ##  Loading Libraries
# 
# 

# In[2]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# ## Defining some varibales for printing the result

# In[3]:


Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'


# ##  Loading Dataset

# In[24]:


data = pd.read_excel("data.xlsx")
data.head(5)


# In[25]:


closePrice = data[['DATE','CLOSE']]
closePrice.head(5)


# In[26]:


closePrice =closePrice.set_index('DATE')
closePrice.head()


# In[27]:


closePrice.index


# It is $NOT$ good. Please, find an appropriate solution for fulfilling missing data after using the above function $\textbf{asfreq()}$

# # Optimizing model 

# We are going to apply one of the most commonly used method for time-series forecasting, known as ARIMA, which stands for Autoregressive Integrated Moving Average.
# 
# ARIMA models are denoted with the notation $ARIMA(p, d, q)$. These three parameters account for seasonality, trend, and noise in data:

# ## Find the Optimal ARIMA Model

# This step is parameter Selection for our furniture’s sales ARIMA Time Series Model. Our goal here is to use a “grid search” to find the optimal set of parameters that yields the best performance for our model.

# ## Setting a set of hyperparameters

# In[41]:


p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
pdq


# In[42]:


print(Bold + 'A Number of combinations: {}'.format(len(pdq)))


# In[43]:


print(Bold + 'Examples of parameter combinations for ARIMA...' + End)
print('ARIMA: {}'.format(pdq[1]))
print('ARIMA: {}'.format(pdq[2]))
print('ARIMA: {}'.format(pdq[3]))
print('ARIMA: {}'.format(pdq[4]))


# ## Finding the Optimal Set of Hyperparameters 

# In[44]:


AIC=list()
BIC=list()
para=list()
Lihood=list()
print(Bold + 'Training ARIMA with a Number of Configuration:'+ End)
for param in pdq:
    mod=sm.tsa.statespace.SARIMAX(closePrice,order=param,seasonal_order=(0, 0, 0, 0), 
                                               enforce_stationarity=False, enforce_invertibility=False)
    results= mod.fit()
    para.append(param)
    AIC.append(results.aic)
    BIC.append(results.bic)
    Lihood.append(results.llf)
    print('ARIMA{} - AIC:{} - BIC:{} - Log likehood: {}'.format(param,results.aic, 
                                                                results.bic,results.llf))
    
print(Bold +'The Optimal Choice Suggestions:'+End)
print('The minimum value of Akaike Information Criterion (AIC):{}'.format(min(AIC)))
print('The minimum value of Bayesian Information Criterion (BIC): {}'.format(min(BIC)))
print('The maximum value of Log likehood: {}'.format(min(Lihood)))
print(Bold + 'Descending the Values of AIC and BIC:'+End)
ModSelect=pd.DataFrame({'Hyperparameters':para,'AIC':AIC,'BIC':BIC, 'Log likehood':Lihood}).sort_values(by=['AIC','BIC','Log likehood'],ascending=False)
ModSelect


# # Fitting the optimal SARIMA model

# In[46]:


mod = sm.tsa.statespace.SARIMAX(closePrice, order=(2, 1, 1),enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(Bold + 'The estimated ARIMA(2,1,1) Model'+ End)
#print(results.summary().tables[1])
print(results.summary())


# # Performing Model Diagnostics

# We should always run model diagnostics to investigate any unusual behavior.

# In[49]:


results.plot_diagnostics(figsize=(17,15))
plt.show()


# Our primary concern is to ensure that the residuals of our model are $\textbf{uncorrelated}$ and $\textbf{normally distributed with zero-mean}$. If the seasonal ARIMA model does not satisfy these properties, it should be further improved.
# 
# In this case, our model diagnostics suggests that the model residuals are not normally distributed based on the following:
# 
# 1. The residuals over time (top left plot) seem to display an obvious seasonality (downward trend) and moight not a obvious white noise process.
# 
# 
# 2. In the top right plot, we see that the red KDE line is far from the N(0,1) line (where N(0,1)) is the standard notation for a normal distribution with mean 0 and standard deviation of 1). This is a good indication that the residuals are not normally distributed. 
# 
# 3. The qq-plot on the bottom left shows that the ordered distribution of residuals (blue dots) do not follow the linear trend of the samples taken from a standard normal distribution with N(0, 1). Again, this is a strong indication that the residuals are not normally distributed.
# 
# see more $\text{Q-Q Plot}$ at https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot
# 
# 4. The autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have low correlation with lagged versions of itself, but several espisodes have high correlation with their own lagged values.
# 
# Those observations lead us to conclude that our model produces a satisfactory fit that could help us understand our time series data and forecast future values.
# 
# Although we have a satisfactory fit, some parameters of our seasonal ARIMA model could be changed to improve our model fit. For example, our grid search only considered a restricted set of parameter combinations, so we may find better models if we widened the grid search.
# 
# It is not good enough since our model diagnostics suggests that the model residuals are not near normally distributed.
