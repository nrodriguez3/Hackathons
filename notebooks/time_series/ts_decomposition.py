import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
 
df=pd.read_csv("AirPassengers.csv", sep=',')

df.set_index('Month',inplace=True)
df.index=pd.to_datetime(df.index)
#drop null values
df.dropna(inplace=True)

result=seasonal_decompose(df['#Passengers'], model='multiplicable', period=12)
result.plot()

result2=STL(df['#Passengers'], model='multiplicable', period=12)
result2.plot()