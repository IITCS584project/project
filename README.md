# project


## Things to do
* Collecting assets' factor data. 
* Design the methods, and try to predict data 
    * Traditional way to find the factors
    * Lienar Regression
    * Time Series Analysis
    * Logistic Regression
    * Neural Network
        * Design the neural network
        * Find the loss function
* Backtest system

## Descriptions
### What is the multi-factor investment?
Factors are the elements that affects the market price, and it varies at every moment, but there is always some factors which can describe the current asset's price. So if I can find the some effective factors, and I invest into these factors, theoeritically I can get the expected return.
### Traditional way to find factors
The most intuitive way is to order the assets by the factors. For example

### Linear Regression
So there is another way to find out whether a specific factor affects the asset's price.

### Time Series Analysis
The time series analysis consider the prices in the last serveral epochs as factors, and it tries to find the regressional relationships between current price and last several prices. 
There are two parts in the time sereis analysis, and one of them is ARMA model. Actually this model contains two models: AR(p) model and MA(q) model.  
AR(p) model:
$$
y_t = c + \phi_1y_{t-1} + \phi_2y_{t-2} + ... + \phi_py_{t-p} + \epsilon_t
$$
It shows that the price in time t has a linear relationship with prices of last epochs.

MA(q) model:
$$
y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} + ... + \theta_q\epsilon_{t-q}
$$
It shows the price is flucating around the $\mu$ by the residuals of last epochs.  
ARMA model combines the AR(p) and MA(q) model
$$
y_t = c + \phi_1y_{t-1} + \phi_2y_{t-2} + ... + \phi_py_{t-p} + \epsilon_t + \theta_1\epsilon_{t-1} + ... + \theta_q\epsilon_{t-q}
$$
So we can say that ARMA(p,q) model predicts the expected $y_t$. But there is still an important part not mentioned in this model which is how to model the residual: $\epsilon_t$.

Here I use pandas to do the time series analysis.

### Logistic Regression

### Neural Network

## What are the factors data?
I use two kinds of data here:
* Cross-section data
* Time series data

The cross-section data contains 3 kinds of data:
* Aseets character data, here we can use the stock's fundamental data of financial report.
* Technical indicators
* Economics data


## Backtest system
The most important thing in the investment is to gain the excess return with minimum risk. So if I make the investment decision by the strategy, I need to a good criteria to evaluate whether this strategy is good or not. Sharpe ratio is what I want.
$$
S = \frac{E(R-r_f)}{\sigma}
$$
* S: Sharpe ratio
* R: yield of asset
* $r_f$: risk-free rate
* $\sigma$: standard deviation of asset's yield
