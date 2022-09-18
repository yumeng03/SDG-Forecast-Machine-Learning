import pandas as pd
import ppscore as pps
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def creat_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i: (i+look_back)]
        # print(a.reshape(-1))
        dataX.append(a.reshape(-1))
        dataY.append(dataset[i+look_back])
        # print("----------")
        # print(dataset[i+look_back][0])
    return pd.DataFrame(dataX), pd.DataFrame(dataY)


if __name__ == '__main__':
    url = "/Users/xiaojun/Desktop/SDG_Data_File_Daily.csv"
    df = pd.read_csv(url)
    # 1.nan设为0
    df = df.fillna(0)

    # 2.筛选Company_Name为apple inc
    df = df[df['Company_Name'] == 'apple inc']
    print(df)

    df1 = df[['SDG_1', 'SDG_2', 'SDG_3', 'SDG_4', 'SDG_5', 'SDG_6', 'SDG_7', 'SDG_8', 'SDG_9', 'SDG_10',
              'SDG_11', 'SDG_12', 'SDG_13', 'SDG_14', 'SDG_15', 'SDG_16', 'SDG_17']]
    df2 = df[['SDG_1_News_Volume', 'SDG_2_News_Volume', 'SDG_3_News_Volume', 'SDG_4_News_Volume',
              'SDG_5_News_Volume', 'SDG_6_News_Volume', 'SDG_7_News_Volume', 'SDG_8_News_Volume',
              'SDG_9_News_Volume', 'SDG_10_News_Volume', 'SDG_11_News_Volume', 'SDG_12_News_Volume',
              'SDG_13_News_Volume', 'SDG_14_News_Volume', 'SDG_15_News_Volume', 'SDG_16_News_Volume', 'SDG_17_News_Volume']]

    # 20190401-20190531
    df1_45 = df[(df['Timestamp'] >= '2019-04-01') & (df['Timestamp'] <= '2019-05-31')]
    df1_45 = df1_45.reset_index()
    print(df1_45)

    # 20190601-20190731
    df1_67 = df[(df['Timestamp'] >= '2019-06-01') & (df['Timestamp'] <= '2019-07-31')]
    df1_67 = df1_67.reset_index()
    print(df1_67)

    # SDG_1-SDG_17
    df_sdg_1 = pd.concat([df1_45['SDG_1'], df1_67['SDG_1']], axis=1)
    df_sdg_1.columns = ['x', 'y']
    df_sdg_2 = pd.concat([df1_45['SDG_2'], df1_67['SDG_2']], axis=1)
    df_sdg_2.columns = ['x', 'y']
    df_sdg_3 = pd.concat([df1_45['SDG_3'], df1_67['SDG_3']], axis=1)
    df_sdg_3.columns = ['x', 'y']
    df_sdg_4 = pd.concat([df1_45['SDG_4'], df1_67['SDG_4']], axis=1)
    df_sdg_4.columns = ['x', 'y']
    df_sdg_5 = pd.concat([df1_45['SDG_5'], df1_67['SDG_5']], axis=1)
    df_sdg_5.columns = ['x', 'y']
    df_sdg_6 = pd.concat([df1_45['SDG_6'], df1_67['SDG_6']], axis=1)
    df_sdg_6.columns = ['x', 'y']
    df_sdg_7 = pd.concat([df1_45['SDG_7'], df1_67['SDG_7']], axis=1)
    df_sdg_7.columns = ['x', 'y']
    df_sdg_8 = pd.concat([df1_45['SDG_8'], df1_67['SDG_8']], axis=1)
    df_sdg_8.columns = ['x', 'y']
    df_sdg_9 = pd.concat([df1_45['SDG_9'], df1_67['SDG_9']], axis=1)
    df_sdg_9.columns = ['x', 'y']
    df_sdg_10 = pd.concat([df1_45['SDG_10'], df1_67['SDG_10']], axis=1)
    df_sdg_10.columns = ['x', 'y']
    df_sdg_11 = pd.concat([df1_45['SDG_11'], df1_67['SDG_11']], axis=1)
    df_sdg_11.columns = ['x', 'y']
    df_sdg_12 = pd.concat([df1_45['SDG_12'], df1_67['SDG_12']], axis=1)
    df_sdg_12.columns = ['x', 'y']
    df_sdg_13 = pd.concat([df1_45['SDG_13'], df1_67['SDG_13']], axis=1)
    df_sdg_13.columns = ['x', 'y']
    df_sdg_14 = pd.concat([df1_45['SDG_14'], df1_67['SDG_14']], axis=1)
    df_sdg_14.columns = ['x', 'y']
    df_sdg_15 = pd.concat([df1_45['SDG_15'], df1_67['SDG_15']], axis=1)
    df_sdg_15.columns = ['x', 'y']
    df_sdg_16 = pd.concat([df1_45['SDG_16'], df1_67['SDG_16']], axis=1)
    df_sdg_16.columns = ['x', 'y']
    df_sdg_17 = pd.concat([df1_45['SDG_17'], df1_67['SDG_17']], axis=1)
    df_sdg_17.columns = ['x', 'y']

    # SDG_1_News_Volume-SDG_17_News_Volume
    df_sdg_news_1 = pd.concat([df1_45['SDG_1_News_Volume'], df1_67['SDG_1_News_Volume']], axis=1)
    df_sdg_news_1.columns = ['x', 'y']
    df_sdg_news_2 = pd.concat([df1_45['SDG_2_News_Volume'], df1_67['SDG_2_News_Volume']], axis=1)
    df_sdg_news_2.columns = ['x', 'y']
    df_sdg_news_3 = pd.concat([df1_45['SDG_3_News_Volume'], df1_67['SDG_3_News_Volume']], axis=1)
    df_sdg_news_3.columns = ['x', 'y']
    df_sdg_news_4 = pd.concat([df1_45['SDG_4_News_Volume'], df1_67['SDG_4_News_Volume']], axis=1)
    df_sdg_news_4.columns = ['x', 'y']
    df_sdg_news_5 = pd.concat([df1_45['SDG_5_News_Volume'], df1_67['SDG_5_News_Volume']], axis=1)
    df_sdg_news_5.columns = ['x', 'y']
    df_sdg_news_6 = pd.concat([df1_45['SDG_6_News_Volume'], df1_67['SDG_6_News_Volume']], axis=1)
    df_sdg_news_6.columns = ['x', 'y']
    df_sdg_news_7 = pd.concat([df1_45['SDG_7_News_Volume'], df1_67['SDG_7_News_Volume']], axis=1)
    df_sdg_news_7.columns = ['x', 'y']
    df_sdg_news_8 = pd.concat([df1_45['SDG_8_News_Volume'], df1_67['SDG_8_News_Volume']], axis=1)
    df_sdg_news_8.columns = ['x', 'y']
    df_sdg_news_9 = pd.concat([df1_45['SDG_9_News_Volume'], df1_67['SDG_9_News_Volume']], axis=1)
    df_sdg_news_9.columns = ['x', 'y']
    df_sdg_news_10 = pd.concat([df1_45['SDG_10_News_Volume'], df1_67['SDG_10_News_Volume']], axis=1)
    df_sdg_news_10.columns = ['x', 'y']
    df_sdg_news_11 = pd.concat([df1_45['SDG_11_News_Volume'], df1_67['SDG_11_News_Volume']], axis=1)
    df_sdg_news_11.columns = ['x', 'y']
    df_sdg_news_12 = pd.concat([df1_45['SDG_12_News_Volume'], df1_67['SDG_12_News_Volume']], axis=1)
    df_sdg_news_12.columns = ['x', 'y']
    df_sdg_news_13 = pd.concat([df1_45['SDG_13_News_Volume'], df1_67['SDG_13_News_Volume']], axis=1)
    df_sdg_news_13.columns = ['x', 'y']
    df_sdg_news_14 = pd.concat([df1_45['SDG_14_News_Volume'], df1_67['SDG_14_News_Volume']], axis=1)
    df_sdg_news_14.columns = ['x', 'y']
    df_sdg_news_15 = pd.concat([df1_45['SDG_15_News_Volume'], df1_67['SDG_15_News_Volume']], axis=1)
    df_sdg_news_15.columns = ['x', 'y']
    df_sdg_news_16 = pd.concat([df1_45['SDG_16_News_Volume'], df1_67['SDG_16_News_Volume']], axis=1)
    df_sdg_news_16.columns = ['x', 'y']
    df_sdg_news_17 = pd.concat([df1_45['SDG_17_News_Volume'], df1_67['SDG_17_News_Volume']], axis=1)
    df_sdg_news_17.columns = ['x', 'y']

    # SDG_1-SDG_17 pps
    print("SDG_1:", pps.score(df_sdg_1, 'x', 'y'))
    print("SDG_2:", pps.score(df_sdg_2, 'x', 'y'))
    print("SDG_3:", pps.score(df_sdg_3, 'x', 'y'))
    print("SDG_4:", pps.score(df_sdg_4, 'x', 'y'))
    print("SDG_5:", pps.score(df_sdg_5, 'x', 'y'))
    print("SDG_6:", pps.score(df_sdg_6, 'x', 'y'))
    print("SDG_7:", pps.score(df_sdg_7, 'x', 'y'))
    print("SDG_8:", pps.score(df_sdg_8, 'x', 'y'))
    print("SDG_9:", pps.score(df_sdg_9, 'x', 'y'))
    print("SDG_10:", pps.score(df_sdg_10, 'x', 'y'))
    print("SDG_11:", pps.score(df_sdg_11, 'x', 'y'))
    print("SDG_12:", pps.score(df_sdg_12, 'x', 'y'))
    print("SDG_13:", pps.score(df_sdg_13, 'x', 'y'))
    print("SDG_14:", pps.score(df_sdg_14, 'x', 'y'))
    print("SDG_15:", pps.score(df_sdg_15, 'x', 'y'))
    print("SDG_16:", pps.score(df_sdg_16, 'x', 'y'))
    print("SDG_17:", pps.score(df_sdg_17, 'x', 'y'))

    # SDG_1_News_Volume-SDG_17_News_Volume pps
    print("SDG_1_News_Volume:", pps.score(df_sdg_news_1, 'x', 'y'))
    print("SDG_2_News_Volume:", pps.score(df_sdg_news_2, 'x', 'y'))
    print("SDG_3_News_Volume:", pps.score(df_sdg_news_3, 'x', 'y'))
    print("SDG_4_News_Volume:", pps.score(df_sdg_news_4, 'x', 'y'))
    print("SDG_5_News_Volume:", pps.score(df_sdg_news_5, 'x', 'y'))
    print("SDG_6_News_Volume:", pps.score(df_sdg_news_6, 'x', 'y'))
    print("SDG_7_News_Volume:", pps.score(df_sdg_news_7, 'x', 'y'))
    print("SDG_8_News_Volume:", pps.score(df_sdg_news_8, 'x', 'y'))
    print("SDG_9_News_Volume:", pps.score(df_sdg_news_9, 'x', 'y'))
    print("SDG_10_News_Volume:", pps.score(df_sdg_news_10, 'x', 'y'))
    print("SDG_11_News_Volume:", pps.score(df_sdg_news_11, 'x', 'y'))
    print("SDG_12_News_Volume:", pps.score(df_sdg_news_12, 'x', 'y'))
    print("SDG_13_News_Volume:", pps.score(df_sdg_news_13, 'x', 'y'))
    print("SDG_14_News_Volume:", pps.score(df_sdg_news_14, 'x', 'y'))
    print("SDG_15_News_Volume:", pps.score(df_sdg_news_15, 'x', 'y'))
    print("SDG_16_News_Volume:", pps.score(df_sdg_news_16, 'x', 'y'))
    print("SDG_17_News_Volume:", pps.score(df_sdg_news_17, 'x', 'y'))

    ts = df[['SDG_7']]
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(ts)
    print(dataset)
    train = dataset[0: int(len(dataset))]
    # test = dataset[int(len(dataset)*0.8):]

    look_back = 10
    trainX_df, trainY_df = creat_dataset(train, look_back)
    # testX, testY = creat_dataset(test, look_back)
    trainX, testX, trainY, testY = train_test_split(trainX_df, trainY_df, test_size=0.2, random_state=0)
    print(trainX.shape, testX.shape)
    print(trainY.shape, testY.shape)

    # svr model
    # dataset1 = ts.values
    clf = svm.SVR()
    clf.fit(trainX, trainY)
    trainPredict = clf.predict(trainX)
    testPredict = clf.predict(testX)

    # 反归一化
    trainPredict1 = scaler.inverse_transform(np.array(trainPredict).reshape(-1, 1))
    trainY1 = scaler.inverse_transform(np.array(trainY).reshape(-1, 1))
    testPredict1 = scaler.inverse_transform(np.array(testPredict).reshape(-1, 1))
    testY1 = scaler.inverse_transform(np.array(testY).reshape(-1, 1))

    trainPredict_df1 = pd.DataFrame(trainPredict1)
    trainY_df1 = pd.DataFrame(trainY1)
    testPredict_df1 = pd.DataFrame(testPredict1)
    testY_df1 = pd.DataFrame(testY1)
    # print(trainPredict_df1)
    # print(trainY_df1)
    # print(testPredict_df1)
    # print(testY_df1)

    # train画图
    plt.plot(trainPredict_df1, color='blue', label='pred')
    plt.plot(trainY_df1, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # test画图
    plt.plot(testPredict_df1, color='blue', label='pred')
    plt.plot(testY_df1, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # 评价
    print("svr train mse:", mean_squared_error(trainPredict_df1.values.tolist(), trainY_df1.values.tolist()))
    print("svr train r2:", r2_score(trainPredict_df1.values.tolist(), trainY_df1.values.tolist()))
    print("svr test mse:", mean_squared_error(testPredict_df1.values.tolist(), testY_df1.values.tolist()))
    print("svr test r2:", r2_score(testPredict_df1.values.tolist(), testY_df1.values.tolist()))

    # LinearR
    lr = LinearRegression()
    lr.fit(trainX, trainY)
    trainPredict = lr.predict(trainX)
    testPredict = lr.predict(testX)

    # 反归一化
    trainPredict2 = scaler.inverse_transform(np.array(trainPredict).reshape(-1, 1))
    trainY2 = scaler.inverse_transform(np.array(trainY).reshape(-1, 1))
    testPredict2 = scaler.inverse_transform(np.array(testPredict).reshape(-1, 1))
    testY2 = scaler.inverse_transform(np.array(testY).reshape(-1, 1))

    trainPredict_df2 = pd.DataFrame(trainPredict2)
    trainY_df2 = pd.DataFrame(trainY2)
    testPredict_df2 = pd.DataFrame(testPredict2)
    testY_df2 = pd.DataFrame(testY2)
    # print(trainPredict_df2)
    # print(trainY_df2)
    # print(testPredict_df2)
    # print(testY_df2)

    # train画图
    plt.plot(trainPredict_df2, color='blue', label='pred')
    plt.plot(trainY_df2, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # test画图
    plt.plot(testPredict_df2, color='blue', label='pred')
    plt.plot(testY_df2, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # 评价
    print("LinearRegression train mse:", mean_squared_error(trainPredict_df2.values.tolist(), trainY_df2.values.tolist()))
    print("LinearRegression train r2:", r2_score(trainPredict_df2.values.tolist(), trainY_df2.values.tolist()))
    print("LinearRegression test mse:", mean_squared_error(testPredict_df2.values.tolist(), testY_df2.values.tolist()))
    print("LinearRegression test r2:", r2_score(testPredict_df2.values.tolist(), testY_df2.values.tolist()))

    # DecisionTreeRegressor
    regr = DecisionTreeRegressor(max_depth=2)
    regr.fit(trainX, trainY)
    trainPredict = regr.predict(trainX)
    testPredict = regr.predict(testX)

    # 反归一化
    trainPredict6 = scaler.inverse_transform(np.array(trainPredict).reshape(-1, 1))
    trainY6 = scaler.inverse_transform(np.array(trainY).reshape(-1, 1))
    testPredict6 = scaler.inverse_transform(np.array(testPredict).reshape(-1, 1))
    testY6 = scaler.inverse_transform(np.array(testY).reshape(-1, 1))

    trainPredict_df6 = pd.DataFrame(trainPredict6)
    trainY_df6 = pd.DataFrame(trainY6)
    testPredict_df6 = pd.DataFrame(testPredict6)
    testY_df6 = pd.DataFrame(testY6)
    # print(trainPredict_df6)
    # print(trainY_df6)
    # print(testPredict_df6)
    # print(testY_df6)

    # train画图
    plt.plot(trainPredict_df6, color='blue', label='pred')
    plt.plot(trainY_df6, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # test画图
    plt.plot(testPredict_df6, color='blue', label='pred')
    plt.plot(testY_df6, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # 评价
    print("DecisionTreeRegressor train mse:",
          mean_squared_error(trainPredict_df6.values.tolist(), trainY_df6.values.tolist()))
    print("DecisionTreeRegressor train r2:", r2_score(trainPredict_df6.values.tolist(), trainY_df6.values.tolist()))
    print("DecisionTreeRegressor test mse:",
          mean_squared_error(testPredict_df6.values.tolist(), testY_df6.values.tolist()))
    print("DecisionTreeRegressor test r2:", r2_score(testPredict_df6.values.tolist(), testY_df6.values.tolist()))

    # alter param
    regr = DecisionTreeRegressor(max_depth=5)
    regr.fit(trainX, trainY)
    trainPredict = regr.predict(trainX)
    testPredict = regr.predict(testX)

    # 反归一化
    trainPredict3 = scaler.inverse_transform(np.array(trainPredict).reshape(-1, 1))
    trainY3 = scaler.inverse_transform(np.array(trainY).reshape(-1, 1))
    testPredict3 = scaler.inverse_transform(np.array(testPredict).reshape(-1, 1))
    testY3 = scaler.inverse_transform(np.array(testY).reshape(-1, 1))

    trainPredict_df3 = pd.DataFrame(trainPredict3)
    trainY_df3 = pd.DataFrame(trainY3)
    testPredict_df3 = pd.DataFrame(testPredict3)
    testY_df3 = pd.DataFrame(testY3)
    # print(trainPredict_df3)
    # print(trainY_df3)
    # print(testPredict_df3)
    # print(testY_df3)

    # train画图
    plt.plot(trainPredict_df3, color='blue', label='pred')
    plt.plot(trainY_df3, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # test画图
    plt.plot(testPredict_df3, color='blue', label='pred')
    plt.plot(testY_df3, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # 评价
    print("DecisionTreeRegressor train mse:", mean_squared_error(trainPredict_df3.values.tolist(), trainY_df3.values.tolist()))
    print("DecisionTreeRegressor train r2:", r2_score(trainPredict_df3.values.tolist(), trainY_df3.values.tolist()))
    print("DecisionTreeRegressor test mse:", mean_squared_error(testPredict_df3.values.tolist(), testY_df3.values.tolist()))
    print("DecisionTreeRegressor test r2:", r2_score(testPredict_df3.values.tolist(), testY_df3.values.tolist()))

    # RandomForestRegressor
    rf_before = RandomForestRegressor(n_estimators=30)
    rf_before.fit(trainX, trainY)
    trainPredict = rf_before.predict(trainX)
    testPredict = rf_before.predict(testX)

    # 反归一化
    trainPredict4 = scaler.inverse_transform(np.array(trainPredict).reshape(-1, 1))
    trainY4 = scaler.inverse_transform(np.array(trainY).reshape(-1, 1))
    testPredict4 = scaler.inverse_transform(np.array(testPredict).reshape(-1, 1))
    testY4 = scaler.inverse_transform(np.array(testY).reshape(-1, 1))

    trainPredict_df4 = pd.DataFrame(trainPredict4)
    trainY_df4 = pd.DataFrame(trainY4)
    testPredict_df4 = pd.DataFrame(testPredict4)
    testY_df4 = pd.DataFrame(testY4)
    # print(trainPredict_df4)
    # print(trainY_df4)
    # print(testPredict_df4)
    # print(testY_df4)

    # train画图
    plt.plot(trainPredict_df4, color='blue', label='pred')
    plt.plot(trainY_df4, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # test画图
    plt.plot(testPredict_df4, color='blue', label='pred')
    plt.plot(testY_df4, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # 评价
    print("RandomForestRegressor train mse:", mean_squared_error(trainPredict_df4.values.tolist(), trainY_df4.values.tolist()))
    print("RandomForestRegressor train r2:", r2_score(trainPredict_df4.values.tolist(), trainY_df4.values.tolist()))
    print("RandomForestRegressor test mse:", mean_squared_error(testPredict_df4.values.tolist(), testY_df4.values.tolist()))
    print("RandomForestRegressor test r2:", r2_score(testPredict_df4.values.tolist(), testY_df4.values.tolist()))

    # RandomForestRegressor alter param
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(trainX, trainY)
    trainPredict = rf.predict(trainX)
    testPredict = rf.predict(testX)

    # 反归一化
    trainPredict5 = scaler.inverse_transform(np.array(trainPredict).reshape(-1, 1))
    trainY5 = scaler.inverse_transform(np.array(trainY).reshape(-1, 1))
    testPredict5 = scaler.inverse_transform(np.array(testPredict).reshape(-1, 1))
    testY5 = scaler.inverse_transform(np.array(testY).reshape(-1, 1))

    trainPredict_df5 = pd.DataFrame(trainPredict5)
    trainY_df5 = pd.DataFrame(trainY5)
    testPredict_df5 = pd.DataFrame(testPredict5)
    testY_df5 = pd.DataFrame(testY5)
    # print(trainPredict_df5)
    # print(trainY_df5)
    # print(testPredict_df5)
    # print(testY_df5)

    # train画图
    plt.plot(trainPredict_df5, color='blue', label='pred')
    plt.plot(trainY_df5, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # test画图
    plt.plot(testPredict_df5, color='blue', label='pred')
    plt.plot(testY_df5, color='red', label='true')
    plt.legend(loc='upper left')
    plt.show()

    # 评价
    print("RandomForestRegressor train mse:", mean_squared_error(trainPredict_df5.values.tolist(), trainY_df5.values.tolist()))
    print("RandomForestRegressor train r2:", r2_score(trainPredict_df5.values.tolist(), trainY_df5.values.tolist()))
    print("RandomForestRegressor test mse:", mean_squared_error(testPredict_df5.values.tolist(), testY_df5.values.tolist()))
    print("RandomForestRegressor test r2:", r2_score(testPredict_df5.values.tolist(), testY_df5.values.tolist()))