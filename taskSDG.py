import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


if __name__ == '__main__':
    # 读取数据
    url = "/Users/aubrey/Desktop/SDG_Data_File_Daily.csv"
    df = pd.read_csv(url)
    # 剔除na值
    df = df.dropna()
    print(df)
    # 描述性分析
    print(df.describe().T)
    print(df.info())

    # 可视化分析 柱状图 根据Ticker维度，求SDG_1的平均值
    df1 = df.groupby(['Ticker'], as_index=False).mean()
    print(df1)
    plt.figure(figsize=(10, 5))
    # plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.bar([x for x in range(len(df1['SDG_1']))], height=df1['SDG_1'].values.tolist(), color="indigo", linewidth=1.5)

    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend()  # 显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
    plt.show()

    # 根据Ticker维度，求SDG_1的中位数
    df2 = df.groupby(['Ticker'], as_index=False).median()
    print(df2)
    plt.figure(figsize=(10, 5))
    # plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.bar([x for x in range(len(df2['SDG_1']))], height=df2['SDG_1'].values.tolist(), color="indigo", linewidth=1.5)

    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend()  # 显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
    plt.show()

    # k-means,根据'SDG_1', 'SDG_2', 'STS_1', 'STS_2'字段做聚类分析
    df3 = df[['SDG_1', 'SDG_2', 'STS_1', 'STS_2']]
    x = np.array(df3.values.tolist())
    k_means = KMeans(n_clusters=3, random_state=10)
    k_means.fit(x)
    y_predict = k_means.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_predict)
    plt.show()
    print(k_means.cluster_centers_)
    print(k_means.inertia_)

