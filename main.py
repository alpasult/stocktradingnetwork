# Press the green button in the gutter to run the script.
import random
import numpy as np
import tensorflow as tf
import csv
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf


# Neural Network
def getTickerList():
    file = open('nasdaq_screener_1634980838717.csv')
    rows = []
    csvreader = csv.reader(file)
    for row in csvreader:
        rows.append(row[0])
    rows.remove('Symbol')
    return rows


# App Functionality
if __name__ == '__main__':
    while True:
        choice = input("1) Train (DO NOT RUN WITHOUT COMPILING DATASET FIRST)\n"
                       "2) Compile Dataset\n"
                       "3) Quit\n")
        choice = int(choice)
        if choice == 1:
            x_train = np.loadtxt('x_train.csv', delimiter=',')
            y_train = np.loadtxt('y_train.csv', delimiter=',')
            inputs = tf.keras.Input(shape=(30,))
            x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(inputs)
            outputs = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            try:
                model = keras.models.load_model('checkpoint')
            except OSError:
                pass
            model.fit(x_train, y_train, epochs=3)
            model.save('checkpoint')
        if choice == 2:
            tickerList = getTickerList()
            # Approximately 10 seconds per training batch
            batch_size = int(input("How many training batches? "))
            x_train = np.ndarray(shape=(batch_size, 30), dtype=float)
            y_train = np.ndarray(shape=(batch_size, 1), dtype=float)
            for i in range(batch_size):
                hist = 0
                while True:
                    while True:
                        tick = yf.Ticker(tickerList[random.randint(0, len(tickerList))])
                        if tick.info['regularMarketPrice'] is not None:
                            break
                    hist = tick.history(period="max")
                    if len(hist) > 100:
                        break
                index = random.randint(30, len(hist) - 30)
                hist = hist['Open'].to_numpy()
                sig_hist = hist[index - 30: index]
                for j in range(len(sig_hist)):
                    sig_hist[j] = 1 / (1 + np.exp(-sig_hist[j]))
                x_train[i] = sig_hist
                print(i)
                if hist[index + 30] > hist[index]:
                    y_train[i] = np.array([0.99])
                else:
                    y_train[i] = np.array([0])
            np.savetxt('x_train.csv', x_train, delimiter=',')
            np.savetxt('y_train.csv', y_train, delimiter=',')
        if choice == 3:
            break
