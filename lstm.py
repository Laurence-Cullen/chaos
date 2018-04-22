from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import math
import pandas as pd
import mackey_glass

# Developing general purpose LSTM model to predict the future values of univariate time series.
# Drawn heavily from https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/


def get_data():
    generator = mackey_glass.MackeyGlassGenerator(mu=0.1, beta=0.2, tau=30, n=9.65)
    series = generator.generate_series(20000)

    cut_step = 18000

    train_series = pd.DataFrame(series[0:cut_step:1])  # .transpose()
    test_series = pd.DataFrame(series[cut_step:-1:1])  # .transpose()

    return train_series, test_series


def fit_lstm(batch_size, train_data):
    # TODO something wrong with array dimensions and batch_input_shape :'(

    print('train data shape = %s' % str(train_data.shape))
    print(train_data)

    x, y = train_data[:, 0:-1], train_data[:, -1]
    x = x.reshape(x.shape[0], 1, x.shape[1])

    model = Sequential()
    model.add(LSTM(units=10,
              activation='tanh',
              stateful=True,
              batch_input_shape=(batch_size, x.shape[1], x.shape[2])))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')

    epochs = 10

    for epoch in range(0, epochs):
        model.fit(x=x, y=y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
        print('epoch = %i' % epoch)

    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, x):
    x = x.reshape(1, 1, len(x))
    yhat = model.predict(x, batch_size=batch_size)
    return yhat[0, 0]


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def main():
    train_series, test_series = get_data()

    print('train series shape = %s' % str(train_series.shape))

    train_data = timeseries_to_supervised(train_series).values
    test_data = timeseries_to_supervised(test_series).values

    model = fit_lstm(batch_size=1, train_data=train_data)

    # forecast the entire training dataset to build up state for forecasting
    x, y = train_data[:, 0:-1], train_data[:, -1]
    x = x.reshape(x.shape[0], 1, x.shape[1])
    model.predict(x=x, batch_size=1)

    # walk-forward validation on the test data
    predictions = []

    for i in range(len(test_data)):
        # make one-step forecast
        x, y = test_data[i, 0:-1], test_data[i, -1]
        yhat = forecast_lstm(model, 1, x)

        # store forecast
        predictions.append(yhat)

        # expected = raw_values[len(train) + i + 1]
        # print('step=%d, predicted=%f, expected=%f' % (i + 1, yhat, expected))

    # report performance
    rmse = math.sqrt(mean_squared_error(test_series, predictions))
    print('Test RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    plt.plot(test_series, label='test series')
    plt.plot(predictions, label='predictions')
    plt.show()


if __name__ == '__main__':
    main()
