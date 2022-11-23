import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
from sklearn import svm
import seaborn as sns


def creat_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i: (i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    print(np.array(dataX).shape)
    print(np.array(dataY).shape)
    return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
    url = "/Users/AUBREY/Desktop/SDG_Data_File_Daily.csv"
    df = pd.read_csv(url)
    # 1.nan设为0
    df = df.fillna(0)

    # 2.筛选Company_Name为apple inc
    df = df[df['Company_Name'] == 'apple inc']
    print(df)

    # 3.'SDG_1_News_Volume', 'SDG_2_News_Volume', 'SDG_3_News_Volume' heatmap
    df['year'] = df.apply(lambda x: int(datetime.datetime.strptime(x['Timestamp'], '%Y-%m-%d').strftime('%Y')), axis=1)
    data = df.groupby('year', as_index=False).mean()
    data = data[['SDG_1_News_Volume', 'SDG_2_News_Volume', 'SDG_3_News_Volume']].values
    with sns.axes_style("white"):
        sns.heatmap(data,
                    cmap="YlGnBu",
                    annot=True,
                    )
    plt.title('heatmap')
    plt.ylabel('time')
    plt.xlabel('feature')
    plt.show()

    # 4.SDG_1_STD 图
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.plot(df['SDG_1_STD'])
    plt.xlabel("SDG_1_STD")
    plt.show()

    # 5.SDG_1 svm
    dataset1 = df[['SDG_1']]
    dataset1 = dataset1.values
    clf = svm.SVR()
    clf.fit([[x] for x in range(len(df['SDG_1']))], dataset1)
    trainPredict = clf.predict([[x] for x in range(len(df['SDG_1']))])
    plt.plot(trainPredict, color='blue', label='pred')
    plt.plot(dataset1, color='red', label='true')
    plt.show()

    # 6.LTS_1 图
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.plot(df['LTS_1'])
    plt.xlabel("LTS_1")
    plt.show()

    # 7.SDG_4 lstm预测
    dataset = df[['SDG_4']]
    dataset = dataset.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    train_size = int(len(dataset) * 1)
    test_size = len(dataset) - train_size
    train, test = dataset[0: train_size], dataset[train_size: len(dataset)]
    print(dataset)

    look_back = 1
    trainX, trainY = creat_dataset(train, look_back)

    model = Sequential()
    model.add(LSTM(input_dim=1, units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(input_dim=50, units=100, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(input_dim=100, units=200, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(300, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dense(units=1))
    model.add(Activation('relu'))
    start = time.time()
    model.compile(loss='mean_squared_error', optimizer='Adam')
    model.summary()

    history = model.fit(trainX, trainY, batch_size=64, epochs=100,
                        validation_split=0.1, verbose=2)
    print('compilatiom time:', time.time() - start)

    trainPredict = model.predict(trainX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)

    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
    print('Train Sccore %.2f RMSE' % (trainScore))

    plt.plot(trainPredict, color='blue', label='pred')
    plt.plot(trainY, color='red', label='true')
    plt.show()
    print("预测数据\n", trainPredict)
    print("实际数据\n", trainY)





