#Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

#Import all the libraries to be used
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
from ipywidgets import widgets 
from IPython.display import display, clear_output, Image
from plotly.graph_objs import *
from plotly.widgets import GraphWidget
from plotly.graph_objs import Scatter

#Import .csv file to pandas
df = pd.read_csv('Auto.csv')

#Remove Make-Model as it is string and not very useful for data visualization
df = df.drop('Make-Model',1)

#Removing the ? from the given data and replacing it with NaN
df =df.replace('?', np.nan)

#filling NaNs with the mean values
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values= 'NaN', strategy = 'median' , axis = 0)
imp.fit(df)
df = pd.DataFrame(imp.transform(df), columns=df.columns)


#preparing data for plotting
y1 = df['mpg']
x1 = df['horsepower']
x2 = df['displacement']
x3 = df['weight']
x4 = df['year']

# Create traces
trace0 = go.Scatter(
    x = x1,
    y = y1,
    mode = 'markers',
    name = 'mpg vs bhp'
)
trace1 = go.Scatter(
    x = x2,
    y = y1,
    mode = 'markers',
    name = 'mpg vs displacement'
)
trace2 = go.Scatter(
    x = x3,
    y = y1,
    mode = 'markers',
    name = 'mpg vs weight'
)
trace3 = go.Scatter(
    x = x4,
    y = y1,
    mode = 'markers',
    name = 'mpg vs year'
)

data = [trace0, trace1, trace2, trace3]
py.iplot(data, filename='scatter-mode')

#
data = Data([trace0, trace1, trace2,trace3])
layout = Layout(
    title='AUTO DATA',
    updatemenus=list([
        dict(
            x=-0.05,
            y=1,
            yanchor='top',
            buttons=list([
                dict(
                    args=['visible', [True, True, True, True]],
                    label='All',
                    method='restyle'
                ),
                dict(
                    args=['visible', [True, False, False, False]],
                    label='mpg vs horsepower',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, True, False, False]],
                    label='mpg vs displacement',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, True, False]],
                    label='mpg vs weight',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, False, True]],
                    label='mpg vs year',
                    method='restyle'
                ),
                
            ]),
        )
    ]),
)

fig = Figure(data=data, layout=layout)
url = py.plot(fig, filename='Auto Data')

#create a graph widget for the plot which plotted above to be modified with the widget value
g = GraphWidget(url)

# Creating a slider widget for nuber of cylinders 
Cylinders = widgets.FloatSlider(
    value=1.0,
    min=3.0,
    max=8.0,
    step=1.0,
    description='Cylinders:',
    continuous_update = False
)

#creating a container for the widget
container = widgets.HBox(children=[Cylinders] )

# Define a response function to capture the slider movement and updating the plot 
def response(change):
    filter_list = [i for i in df['cylinders'] == Cylinders.value]
    temp_df = df[filter_list]
    y = temp_df['mpg']
    x1 = temp_df['horsepower']
    x2 = temp_df['displacement']
    x3 = temp_df['weight']
    x4 = temp_df['year']
 
    g.restyle({'x':[x1], 'y':[y]}, indices=[0])
    g.restyle({'x':[x2], 'y':[y]}, indices=[1])
    g.restyle({'x':[x3], 'y':[y]}, indices=[2])
    g.restyle({'x':[x4], 'y':[y]}, indices=[3])

Cylinders.observe(response, names="value")

display(container)
display(g)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import OneHotEncoder

#Import .csv file to pandas
df = pd.read_csv('Auto.csv')

#drop 
df = df.drop('Make-Model',1)
#df = df.drop('year',1)
df = df.drop('origin',1)

df =df.replace('?', np.nan)

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values= 'NaN', strategy = 'median' , axis = 0)
imp.fit(df)
df = pd.DataFrame(imp.transform(df), columns=df.columns)

#Create input and output data sets
y1 = df.mpg
x1 = df.horsepower

X = df.drop('mpg',1)
Y = df.mpg

from sklearn import preprocessing

# Scatter plot between mpg and horsepower 
plt.plot(X.horsepower,Y,'ro')
plt.ylabel('MPG')
plt.xlabel('Horsepower')
plt.show()

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X.horsepower[:,np.newaxis],Y)

#Plotting linear regression line for mpg vs horsepower(since horsepower seems to be the most important factor) 
plt.plot(X.horsepower,Y,'ro')
plt.plot(X.horsepower, lm.predict (X.horsepower[:,np.newaxis]), color='blue')
plt.ylabel('mpg')
plt.xlabel('Horsepower')
plt.show()
lm.score(X[:,np.newaxis],Y)

from sklearn.metrics import r2_score

#normalizing data
min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

#Splitting the training and test data (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1)

#Lasso Regression Model
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train,Y_train)

#Calculating model scores (r^2)
print(clf.coef_)
print (clf.score(X_train, Y_train, sample_weight=None))
print (clf.score(X_test, Y_test, sample_weight=None))
pd.DataFrame(zip(X.columns, np.transpose(clf.coef_)))

#Import .csv file to pandas
df = pd.read_csv('Auto.csv')
df = df.drop('Make-Model',1)

df =df.replace('?', np.nan)

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values= 'NaN', strategy = 'median' , axis = 0)
imp.fit(df)
df = pd.DataFrame(imp.transform(df), columns=df.columns)

#Create a new binary output variable Target, Target = 1 if acceleration>=15, else 0 
df['Target'] = [1 if x>15 else 0 for x in df['acceleration'] ]

#creating the input and output datasets 
X = df.drop('acceleration',1)
X = X.drop('Target',1)
Y = df.Target

#Normalizing data
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

#Splitting traing and test datasets (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1)

#feature selection using chi squared method
select = sklearn.feature_selection.SelectKBest(score_func=chi2,k=4)
selected_features = select.fit(X_train, Y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]
colnames_selected
X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]

# Function to calculate AUC using roc_auc_score
def model_score(x_train, y_train, x_test, y_test):
    model = LogisticRegression(penalty='l2', C=100)
    model.fit(x_train, Y_train)
    y_hat = [x[1] for x in model.predict_proba(x_test)]
    auc = roc_auc_score(y_test, y_hat)
    return auc

auc_processed = model_score(X_train, Y_train, X_test, Y_test)
print auc_processed

#Caluculate Model score by applying logistic regression
model = LogisticRegression(penalty='l2', C=10)
model.fit(X_train, Y_train)
model.score(X_train, Y_train, sample_weight=None)

#Calculate Model score for test data using the same model
model.score(X_test, Y_test, sample_weight=None)

#Calculate 10 fold cross validation score
scores = cross_val_score(LogisticRegression(), X, Y, scoring='accuracy', cv=10)
print scores.mean()

#Using statsmodels.api to get the summary result
import statsmodels.api 

logit = sm.Logit(Y, X)
result = logit.fit()
print result.summary()
