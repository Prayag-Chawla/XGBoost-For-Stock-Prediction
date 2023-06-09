
## XGBoost for stock trend & prices prediction
Using technical indicators as features, I use XGBRegressor from XGBoost library to predict future stock prices.
The project works on the yahoo finance dataset and multiple algorithms and functions along with XGBoost were also used.

Data science libraries such as NumPy, Pandas, Matplotlib, and XGBoost comprise the first import block. NumPy and Pandas are used for data manipulation and analysis. Machine learning algorithms such as XGBoost are commonly used in regression, classification, and ranking. Visualizations can be created with Matplotlib, a plotting library.

## XGBRegressor
XGBoost is a powerful approach for building supervised regression models. The validity of this statement can be inferred by knowing about its (XGBoost) objective function and base learners. The objective function contains loss function and a regularization term. It tells about the difference between actual values and predicted values, i.e how far the model results are from the real values. The most common loss functions in XGBoost for regression problems is reg:linear, and that for binary classification is reg:logistics. Ensemble learning involves training and combining individual models (known as base learners) to get a single prediction, and XGBoost is one of the ensemble learning methods. XGBoost expects to have the base learners which are uniformly bad at the remainder so that when all the predictions are combined, bad predictions cancels out and better one sums up to form final good predictions


## Functions used
* OHLC Chart
* Decomposition
* Relative Strength Index
*  Moving Average Convergence Divergence (MACD)


## Exponential smoothing
Now, let's see what happens if we start weighting all available observations while exponentially decreasing the weights as we move further back in time. There exists a formula for exponential smoothing that will help us with this:

y^t=α⋅yt+(1−α)⋅y^t−1
 
Here the model value is a weighted average between the current true value and the previous model values. The  α
  weight is called a smoothing factor. It defines how quickly we will "forget" the last available true observation. The smaller  α
  is, the more influence the previous observations have and the smoother the series is.

Exponentiality is hidden in the recursiveness of the function -- we multiply by  (1−α)
  each time, which already contains a multiplication by  (1−α)
  of previous model values.
  
## Autoregressive Integrated Moving Average Model (ARIMA)
This acronym is descriptive, capturing the key aspects of the model itself. Briefly, they are:

AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.






## Usage and Installation

```
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
# Show charts when running kernel
init_notebook_mode(connected=True)
# Change default background color for all visualizations
layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(250,250,250,0.8)')
fig = go.Figure(layout=layout)
templated_fig = pio.to_templated(fig)
pio.templates['my_template'] = templated_fig.layout.template
pio.templates.default = 'my_template'

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose as sm
# import decompose


```

## Output Snapshots
![image](https://github.com/Prayag-Chawla/XGBoost-Fro-Stock-Prediction/assets/92213377/07952a70-6336-4b8c-835b-62ecabd07f25)
![image](https://github.com/Prayag-Chawla/XGBoost-Fro-Stock-Prediction/assets/92213377/cae59b3b-3f75-4056-8eb2-5b6e5017272e)
![image](https://github.com/Prayag-Chawla/XGBoost-Fro-Stock-Prediction/assets/92213377/483df038-87b0-4895-8bdc-29ac94b75339)
![image](https://github.com/Prayag-Chawla/XGBoost-Fro-Stock-Prediction/assets/92213377/ab232e60-cb3f-4cca-ae29-53bb458956e1)

Note - There is a descripancy in final output vs predicted graph. This discrepncy is due to the difference in dataset used, and as the stock is still active, and the trained model was for an inactive dataset, we have this difference. I am still working to fix this issue, and it will get resolved soon.






## Used By
The project is used by various comapanies involving statistical approaches towards their customer proposition. The fields include finance, stock prediction analysis, and related fields.
## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

