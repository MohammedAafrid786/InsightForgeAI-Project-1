from sklearn.linear_model import LinearRegression
import numpy as np

def simple_forecast(df):

    numeric = df.select_dtypes(include="number")

    if len(numeric.columns) == 0:
        return None

    col = numeric.columns[0]

    y = numeric[col].values

    x = np.arange(len(y)).reshape(-1,1)

    model = LinearRegression()
    model.fit(x,y)

    future = model.predict([[len(y)+1]])

    return round(float(future[0]),2)