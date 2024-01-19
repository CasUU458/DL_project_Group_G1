from keras import Sequential
from keras.src.layers import LSTM, Dense

def train_lstm(train_x, train_y, test_x, test_y):
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_x.shape[1], train_x.shape[2])))

    model.add(Dense(units=4))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=7, batch_size=16, validation_data=(test_x, test_y))
    model.save("temp")

    return model