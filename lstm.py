from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import math
import pandas as pd
import mackey_glass


def get_data():
    generator = mackey_glass.MackeyGlassGenerator()
    series = generator.generate_series(20000)

    cut_step = 15000

    train_series = pd.DataFrame(series[0:cut_step:1])  # .transpose()
    test_series = pd.DataFrame(series[cut_step:-1:1])  # .transpose()

    return train_series, test_series


def fit_lstm(batch_size, train_data):
    print(train_data.shape)

    X = train_data.values[:, 0:-1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    y = train_data.values[:, -1]

    print(X.shape)
    print(y.shape)

    model = Sequential()
    model.add(LSTM(units=100,
              activation='tanh',
              stateful=True,
              batch_input_shape=(batch_size, X.shape[1], X.shape[2])))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    epochs = 50

    for epoch in range(0, epochs):
        model.fit(x=X, y=y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()

    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def main():
    train_series, test_series = get_data()

    model = fit_lstm(batch_size=1, train_data=train_series)

    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_series.values[:, 0:-1]
    train_reshaped = train_reshaped.reshape(train_reshaped.shape[0], 1, train_reshaped.shape[1])
    model.predict(x=train_reshaped, batch_size=1)

    # walk-forward validation on the test data
    predictions = list()
    for i in range(test_series.shape[1]):
        # make one-step forecast
        X, y = test_series.values[i, 0:-1], test_series.values[i, -1]
        yhat = forecast_lstm(model, 1, X)

        # store forecast
        predictions.append(yhat)

        # expected = raw_values[len(train) + i + 1]
        # print('step=%d, predicted=%f, expected=%f' % (i + 1, yhat, expected))

    # report performance
    rmse = math.sqrt(mean_squared_error(test_series, predictions))
    print('Test RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    plt.plot(test_series)
    plt.plot(predictions)
    plt.show()


if __name__ == '__main__':
    main()
