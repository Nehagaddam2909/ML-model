#import response from rest framework
from rest_framework.response import Response
from rest_framework.decorators import api_view
from datetime import datetime
import csv
import numpy as np
import pandas as pd
from fbprophet import Prophet

#import file system
# import FileSystemStorage
from django.core.files.storage import FileSystemStorage
fs=FileSystemStorage(location='data/')

@api_view(['POST'])  
def getData(request):
    get_path=fs.path('CovidData.csv')
    train = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv')
    train.rename(columns = {"Province/State": "state", "Country/Region":"country"}, inplace = True)
    train.drop('Lat', axis = 1, inplace = True)
    train.drop('Long', axis = 1, inplace = True)
    train.drop('Recovered', axis = 1, inplace = True)

    train.state = train.state.fillna('Not Available')
    
    train_new = train[train.Date == train.Date.max()].groupby(by='country')
    train_new = train_new.aggregate(np.sum)
    train_new.reset_index(level=0, inplace=True)
    
    train_n2 = train.drop(['state'],axis=1)

    train_n2.sort_values('Date').reset_index().drop('index', axis =1)


    train_n2.groupby('Date')[['Confirmed','country','Deaths']].sum().reset_index()

    #TRAINING THE MODEL TO GET A PREDICTION MODEL CURVE FOR INDIA AND BASED PREDICTIONS
    train_ind = train_n2.loc[train_n2['country']=='India'].copy()

    train_ind['Date'] = pd.to_datetime(train_ind['Date'])
    train_ind = train_ind.set_index('Date')
    # train_ind.head()

    train_ind2 = train_ind.drop(['country','Deaths'],axis = 1)
    train_ind2 = train_ind2.loc[train_ind2['Confirmed']>0]
    train_ind3 = train_ind2
    # train_ind2.head()

    # #DATA VISUALIZATION
    # pd.plotting.register_matplotlib_converters()
    # from statsmodels.tsa.seasonal import seasonal_decompose
    # result = seasonal_decompose(train_ind2, model='multiplicative')
    # result.plot()
    # plt.show()

    new_colname = 'y'
    train_ind2.index.rename('ds', inplace=True)
    train_ind2.rename(columns = {'Confirmed' : 'y'},inplace=True)
    train_ind2.reset_index(level=0, inplace=True)
    train_ind2.head()


    # instantiate the model and set parameters
    model = Prophet(
        interval_width=0.95,
        holidays = pd.DataFrame({'holiday': 'lockdown','ds': pd.to_datetime(['2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31','2020-04-01'
        ,'2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10','2020-04-11','2020-04-12','2020-04-13','2020-04-14'])}),
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )

    # fit the model to historical data

    model.fit(train_ind2)

    future_pd = model.make_future_dataframe(
        periods=60,
        freq='d',
        include_history=True
    )

    # predict over the dataset
    forecast_pd = model.predict(future_pd)

    # #PREDICTION CURVE OF COVID19 INDIA FOR UPCOMING MONTHS
    # #CURVE IS EXACTLY SIMILAR AS PROVIDED BY RESEARCH OF SINGAPORE UNIVERSITY
    # predict_fig = model.plot(forecast_pd, xlabel='date', ylabel='confirmed cases')

    # #MONTH WISE PREDICTION INDIA
    # from fbprophet.plot import plot_plotly
    # import plotly.offline as py

    # fig = plot_plotly(model, forecast_pd)  # This returns a plotly Figure
    # py.iplot(fig)

    # forecast_pd[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[55:63]

    # fpd = pd.DataFrame(forecast_pd[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # fpd.rename(columns= {'ds':'Date', 'yhat':'Predicted Cases'},inplace=True)

    # fpd.drop(['yhat_lower','yhat_upper'],axis=1,inplace=True)

    # #PREDICTION OF COVID CASES INDIA IN UPCOMING MONTHS AND DAYS JUST CHANGE THE DATE AND RUN THE SCRIPT FOR UPDATED RESULTS
    # fpd[(fpd['Date']>= '2020-04-29') & (fpd['Date']<= '2020-04-30')]



    person={'name':'John','age':23}
    return Response(future_pd.to_json())

