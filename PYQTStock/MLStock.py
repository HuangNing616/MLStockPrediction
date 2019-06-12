"""
    Author：黄宁
    Date: June-11-2019
    Function: 机器学习预测股票价格
"""

from PYQTStock.StockIndex import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from xgboost.sklearn import XGBRegressor
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



class StockModel:

    def __init__(self):
        self.flag = True
        self.start_date = '2007-01-04'
        self.sep_date = '2018-12-31'
        self.end_date = '2019-05-23'

    def data_processing(self, FileReadRoute, CSVRoute):
        """
            新构造数据的频率为天, 并且没有时间列, 最后保存成csv
        :return:
        """
        data = pd.read_csv(FileReadRoute, header=None)  # 不将第一列作为列名
        data.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volumn', 'amount']

        df1 = data[data['time'] == '10:00:00']    # 取出10:00的数据集
        list1 = df1.index                         # 取出10:00的数据集的索引，即第几行
        df2 = data[data['time'] == '09:31:00']    # 注意时间列里面的数:9:31:00取出就变成了09:31：00
        list2 = df2.index                         # index 是可以直接按照[*]来取值的
        df3 = data[data['time'] == '11:30:00']
        data_len = len(list1)
        df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volumn', 'amount', 'post_close'])

        for i in range(data_len):

            df4 = data.iloc[list2[i]:list1[i]+1]  # 取出09:31：00到10：00:00之间的取值，左闭右开

            _date = df4['date'].iloc[0]
            _open = df4['open'].iloc[0]
            _high = df4['high'].max()
            _low = df4['low'].min()
            _close = df1['close'].iloc[i]
            _volumn = df4['volumn'].sum()
            _amount = df4['amount'].sum()
            _preclose = df3['close'].iloc[i]

            li1 = [_date, _open, _high, _low, _close, _volumn, _amount, _preclose]

            # 相比于字典，用列表插入更简单，并且有顺序！！！
            df_temp = pd.DataFrame([li1], columns=['date', 'open', 'high', 'low', 'close',
                                                   'volumn', 'amount', 'post_close'])
            df = pd.concat([df, df_temp], axis=0)
        df = df.reset_index(drop=True)
        df.to_csv(CSVRoute, index=None)
        print('新频率数据已保存完成')

    def load_data1(self):  # 将处理后的数据读入并且将要预测的post_close放到第二列

        post_close_trans = self.data['post_close']
        self.data.drop(labels='post_close', axis=1, inplace=True)
        self.data.insert(1, 'post_close', post_close_trans)  # 插入操作必须是不存在的列才可以
        self.X = self.data.iloc[:, 2:]

    def load_data2_delay(self):  # 将post_close这一列向上串一位,最后一位通过随机生成

        self.close = self.data['post_close']
        self.close = self.close.shift(-1)
        self.close.iloc[-1] = random.uniform(2300, 2400)
        self.data['post_close'] = self.close

    def split_train_test(self):

        self.data['date'] = pd.to_datetime(self.data['date'].astype(str), format="%Y-%m-%d")  #将时间戳格式转化成标准时间格式
        data_ = self.data.set_index('date')

        #  时间上左闭右开, 并且将没有该日期的数据删除
        data_train = pd.DataFrame(data=data_, index=np.arange(self.start_date, self.sep_date, dtype='datetime64[D]')).dropna()
        data_test = pd.DataFrame(data=data_, index=np.arange(self.sep_date, self.end_date, dtype='datetime64[D]')).dropna()

        # trainset 为2007-2018年数据
        self.x_train = data_train.drop('post_close', axis=1)
        self.y_train = data_train.post_close

        # testset 为2018-201904的数据
        self.x_test = data_test.drop('post_close', axis=1)
        self.y_test = data_test.post_close
        self.y_test = self.y_test.reset_index(drop=True)

    def randomforest_model(self):

        forest_clf = RandomForestRegressor(n_estimators=25, max_features=6, random_state=42)  #训练数据最多就6个特征
        forest_clf.fit(self.x_train, self.y_train)
        y_randomforest_pred = forest_clf.predict(self.x_test)
        y_randomforest_pred = pd.DataFrame(list(y_randomforest_pred), columns=['RandomForestPred'])
        forest_mse = mean_squared_error(self.y_test, y_randomforest_pred)
        forest_rmse = np.sqrt(forest_mse)
        ratio_rf = pd.DataFrame(self.y_test/y_randomforest_pred['RandomForestPred'], columns=['RandomForestError'])  # 利用Series除以数据框 实际值比预测值

        return forest_rmse, y_randomforest_pred, ratio_rf

    def xgb_model(self):

        xgb_clf = XGBRegressor(learning_rate=0.1, n_estimators=75)
        xgb = xgb_clf.fit(self.x_train, self.y_train)
        self.y_xgboost_pred = xgb.predict(self.x_test)
        y_xgboost_pred = pd.DataFrame(list(self.y_xgboost_pred))
        error = mean_squared_error(self.y_test, self.y_xgboost_pred)
        xgb_rmse = np.sqrt(error)
        ratio_xgb = self.y_test/y_xgboost_pred
        return xgb_rmse, y_xgboost_pred, ratio_xgb

    def ridge_model(self):

        ridge_clf = Ridge(alpha=1.9)
        ridge_clf.fit(self.x_train, self.y_train)
        y_pred_ridge = ridge_clf.predict(self.x_test)
        y_pred_ridge = pd.DataFrame(list(y_pred_ridge), columns=['RidgePred'])
        ridge_mse = mean_squared_error(self.y_test, y_pred_ridge)
        ridge_rmse = np.sqrt(ridge_mse)
        ratio_ridge = pd.DataFrame(self.y_test/y_pred_ridge['RidgePred'], columns=['RidgeError'])
        return ridge_rmse, y_pred_ridge, ratio_ridge

    def randomforest_single_pred(self):

        x_train = self.x_train
        y_train = self.y_train

        x_test = self.x_test
        y_test = self.y_test

        y_pred_all_rf = []
        y_train = list(y_train)
        forest_clf = RandomForestRegressor(n_estimators=25, max_features=6, random_state=42)
        for i in range(len(x_test)):

            forest_clf.fit(x_train, y_train)
            x_test_one = x_test.iloc[i:i+1]
            y_pred_one = forest_clf.predict(x_test_one)
            y_pred_all_rf.append(list(y_pred_one)[0])  # 预测值一个一个求

            x_train = x_train.append(x_test_one)   # 更新x_train 进行下一轮拟合
            y_train.append(y_test[i])              # 更新y_train 进行下一轮拟合

        forest_mse = mean_squared_error(self.y_test, y_pred_all_rf)
        forest_rmse = np.sqrt(forest_mse)
        y_pred_all_rf = pd.DataFrame(y_pred_all_rf, columns=['RandomForestSinglePred'])
        ratio_single_rf = pd.DataFrame(self.y_test/y_pred_all_rf['RandomForestSinglePred'], columns=['RandomForestSingleError'])
        return forest_rmse, y_pred_all_rf, ratio_single_rf

    def xgboost_single_pred(self):

        x_train = self.x_train
        y_train = self.y_train

        x_test = self.x_test
        y_test = self.y_test

        self.y_pred_all_xgb = []
        y_train = list(y_train)
        xgboost_clf = XGBRegressor(learning_rate=0.1, n_estimators=75)

        for i in range(len(x_test)):
            xgboost_clf.fit(x_train, y_train)
            x_test_one = x_test.iloc[i:i+1]
            y_test_one = xgboost_clf.predict(x_test_one)
            self.y_pred_all_xgb.append(list(y_test_one)[0])
            x_train = x_train.append(x_test_one)
            y_train.append(y_test[i])

        xgboost_mse = mean_squared_error(self.y_test, self.y_pred_all_xgb)
        xgboost_rmse = np.sqrt(xgboost_mse)
        y_pred_all_xgb = pd.DataFrame(list(self.y_pred_all_xgb))
        ratio_single_xgb = pd.DataFrame(list(self.y_test))/y_pred_all_xgb
        return xgboost_rmse, y_pred_all_xgb, ratio_single_xgb

    def ridge_single_pred(self):

        x_train = self.x_train
        y_train = self.y_train

        x_test = self.x_test
        y_test = self.y_test

        y_pred_all_ridge = []
        y_train = list(y_train)
        ridge_clf = Ridge(alpha=1.9)
        for i in range(len(x_test)):

            ridge_clf.fit(x_train, y_train)
            x_test_one = x_test.iloc[i:i+1]

            y_pred_one = ridge_clf.predict(x_test_one)
            y_pred_all_ridge.append(list(y_pred_one)[0])

            x_train = x_train.append(x_test_one)
            y_train.append(y_test[i])

        ridge_mse = mean_squared_error(self.y_test, y_pred_all_ridge)
        ridge_rmse = np.sqrt(ridge_mse)
        y_pred_all_ridge = pd.DataFrame(y_pred_all_ridge, columns=['RidgeSinglePred'])
        ratio_single_ridge = pd.DataFrame(self.y_test/y_pred_all_ridge['RidgeSinglePred'], columns=['RidgeSingleError'])
        return ridge_rmse, y_pred_all_ridge, ratio_single_ridge


    def main1(self):
        Filedir = './/SH000016-1min-2007-201901.csv'
        CSVRoute = './StockDay.csv'
        StockIndexRoute = './StockIndex.csv'
        # self.data_processing(Filedir, CSVRoute)  # 产生StockDay数据
        # StockMain(CSVRoute, StockIndexRoute)  # 产生股票指标数据集

        self.data = pd.read_csv(StockIndexRoute)
        self.load_data1()
        self.split_train_test()

        # nday = 4  # 运行一次，串n个交易日
        # for iday in range(0, nday):
        #     self.load_data2_delay()

        # Ridge回归
        rmse_ridge, y_ridge_pred, ratio_ridge = self.ridge_model()
        rmse_ridge_single, y_ridge_pred_single, ratio_ridge_single = self.ridge_single_pred()

        # 随机森林
        rmse_rf, y_randomforest_pred, ratio_rf = self.randomforest_model()
        rmse_rf_single, y_randomforest_pred_single, ratio_rf_single = self.randomforest_single_pred()

        # 通过随机森林比较整体预测和单次预测的差异
        fig = plt.figure(facecolor='white', figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))        # 指定X轴的以日期格式显示
        ax.xaxis.set_major_locator(mdates.DayLocator())                     # X轴的间隔为日

        line1, = plt.plot(self.x_test.index, self.y_test, label='Actual Close-Price', color='k')
        line2, = plt.plot(self.x_test.index, y_randomforest_pred, label='RandomForest Prediction', color='r')
        line3, = plt.plot(self.x_test.index, y_randomforest_pred_single, label='rfsingle Prediction', color='b')

        plt.grid()
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close-Price', fontsize=18)
        plt.title('Actual RandomForest and RF-Single predicted close-price', fontsize=20)
        plt.legend([line1, line2, line3], ['Actual Close-Price', 'RandomForest Prediction', 'RF-Single Prediction'], fontsize=18)
        plt.gcf().autofmt_xdate()  # 自动旋转x轴标签
        plt.show()
        plt.close()

        Final_data = pd.concat([pd.DataFrame(list(self.y_test.index), columns=['date']),
                                pd.DataFrame(list(self.y_test), columns=['actual']),
                                y_randomforest_pred,
                                ratio_rf,
                                y_randomforest_pred_single,
                                ratio_rf_single,
                                y_ridge_pred,
                                ratio_ridge,
                                y_ridge_pred_single,
                                ratio_ridge_single], axis=1)

        Final_data.to_csv('./CompareResult.csv', index=False)


if __name__ == '__main__':

    stock = StockModel()
    stock.main1()


