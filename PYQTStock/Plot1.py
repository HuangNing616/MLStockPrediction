import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PYQTStock.StockIndex import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 目的为显示中文标题


class Visualization:

    def __init__(self, file_dir, par_list):
        """
        读取本地文件(相对路径)
        :return:
        """
        self.combine_data = pd.read_csv(file_dir)
        self.length = len(self.combine_data)   # 数据的长度
        self.num = par_list[0]
        self.type = par_list[1]
        self.result1 = -1
        self.result2 = -1
        self.result3 = -1
        self.result4 = -1

    def DateToDatetime(self, str_i):
        """
        日期标准化，方便后续制作K线图
        :param str_i: 需要转换的字符串
        :return:
        """
        list_i = list(str_i)
        list_i.insert(4, '/')
        list_i.insert(7, '/')
        str_i = ''.join(list_i)
        date_ = pd.to_datetime(str_i, format="%Y/%m/%d")
        return date_

    def DrawMap(self):
        """
        绘制图
        :return:
        """
        dates = []  # 用于存储日期格式的交易日期
        for item in list(self.combine_data['trade_date']):
            value = self.DateToDatetime(str(item))
            dates.append(value)
        self.combine_data['trade_date'] = dates
        self.combine_data.index = self.combine_data.trade_date   # 将日期赋值给列索引

        ax1 = plt.subplot(4, 1, 1)
        ax1.cla()

        data = self.combine_data.drop(['trade_date'], axis=1)    # 删除该日期列, 并将处理后的数据记作data
        data_open = data.loc[:, 'open']
        data_high = data.loc[:, 'high']
        data_low = data.loc[:, 'low']
        data_close = data.loc[:, 'close']
        self.result1 = pd.concat([data_open, data_close, data_high, data_low], axis=1)
        x_data = self.result1.index  # 索引可以直接作为画图的横坐标

        for i in range(self.length):
            ax1.plot([x_data[i], x_data[i]], [data_high[i], data_low[i]], color='b', linewidth=1)
        for i in range(self.length):
            if data_close[i] > data_open[i]:  # 若收盘比开盘高，说明涨, 为红色，反之为绿色
                ax1.plot([x_data[i], x_data[i]], [data_open[i], data_close[i]], color='r', linewidth=3)
            else:
                ax1.plot([x_data[i], x_data[i]], [data_close[i], data_open[i]], color='g', linewidth=3)

        MA_3day = MA_Index({"period": 3},self.result1)[0]
        MA_5day = MA_Index({"period": 5},self.result1)[0]
        MA_7day = MA_Index({"period": 7},self.result1)[0]

        ax1.plot(x_data, MA_3day, color='blue', linewidth=1)
        ax1.plot(x_data, MA_5day, color='yellow', linewidth=1)
        ax1.plot(x_data, MA_7day, color='k', linewidth=1)

        ax1.set_xticks([])  # 关闭坐标轴刻度

        # ---------------------------------------------- #

        # 绘制默认的第二张MACD图
        ax2 = plt.subplot(4, 1, 2)
        ax2.cla()
        self.result2 = pd.concat([data['MACD_DIF_value'], data['MACD_DEA_value'], data['MACD_MACD_value']], axis=1)

        bar_1 = []
        bar_2 = []
        num_1 = []
        num_2 = []
        for id, item in enumerate(list(self.result2['MACD_MACD_value'])):
            if item >= 0:
                num_1.append(self.result2.index[id])
                bar_1.append(list(self.result2['MACD_MACD_value'])[id])
            else:
                num_2.append(self.result2.index[id])
                bar_2.append(list(self.result2['MACD_MACD_value'])[id])

        ax2.plot(self.combine_data.index, self.result2['MACD_DIF_value'], 'k')
        ax2.plot(self.combine_data.index, self.result2['MACD_DEA_value'], 'm')
        ax2.bar(num_1, bar_1, color='r', width=0.8, lw=0)  # 涨为红色的条形图
        ax2.bar(num_2, bar_2, color='g', width=0.8, lw=0)  # 跌为绿色的条形图
        ax2.set_xticks([])

        # ------------------------------------------------------ #
        # 第三张默认图是KDJ
        ax3 = plt.subplot(4, 1, 3)
        ax3.cla()

        self.result3 = pd.concat([data['KDJ_K_value'], data['KDJ_D_value'], data['KDJ_J_value']], axis=1)

        ax3.plot(self.combine_data.index, self.result3['KDJ_K_value'], 'k')
        ax3.plot(self.combine_data.index, self.result3['KDJ_D_value'], 'm')
        ax3.plot(self.combine_data.index, self.result3['KDJ_J_value'], 'g')
        ax3.set_xticks([])

        # ------------------------------------------------------ #
        # 第四张默认图是布林线
        ax4 = plt.subplot(4, 1, 4)
        ax4.cla()
        self.result4 = pd.concat([data['BOLL_upper_value'], data['BOLL_middle_value'], data['BOLL_lower_value']], axis=1)
        ax4.set_xticks(dates)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))     # 指定X轴的以日期格式显示, 如果不指定，就显示不出来!!!
        ax4.plot(self.combine_data.index, self.result4['BOLL_upper_value'], 'k')
        ax4.plot(self.combine_data.index, self.result4['BOLL_middle_value'], 'm')
        ax4.plot(self.combine_data.index, self.result4['BOLL_lower_value'], 'g')

        if self.num != 0:
            if self.num == 1:
                ax = ax1
            if self.num == 2:
                ax = ax2
            if self.num == 3:
                ax = ax3
            if self.num == 4:
                ax = ax4
            if self.num<0 or self.num > 4:
                print('您输入的子图个数超出了图片范围，自动退出')
                exit()
            ax.cla()  # 清空当前子图，保存其他子图
            if self.type == 'Kline':
                for i in range(self.length):
                    ax.plot([self.combine_data.index[i], self.combine_data.index[i]], [self.combine_data.high[i], self.combine_data.low[i]], color='b', linewidth=1)
                for i in range(self.length):
                    if data_close[i] > data_open[i]:  # 若收盘比开盘高，说明涨, 为红色，反之为绿色
                        ax.plot([self.combine_data.index[i], self.combine_data.index[i]], [self.combine_data.open[i], self.combine_data.close[i]], color='r', linewidth=3)
                    else:
                        ax.plot([self.combine_data.index[i], self.combine_data.index[i]], [self.combine_data.close[i], self.combine_data.open[i]], color='g', linewidth=3)
                self.new_data = pd.concat([self.combine_data.open, self.combine_data.close, self.combine_data.high, self.combine_data.low], axis=1)
                MA_3day = MA_Index({"period": 3}, self.new_data)[0]
                MA_5day = MA_Index({"period": 5}, self.new_data)[0]
                MA_7day = MA_Index({"period": 7}, self.new_data)[0]

                ax.plot(x_data, MA_3day, color='blue', linewidth=1)
                ax.plot(x_data, MA_5day, color='yellow', linewidth=1)
                ax.plot(x_data, MA_7day, color='k', linewidth=1)

            elif self.type == 'MACD':
                bar_1 = []
                bar_2 = []
                num_1 = []
                num_2 = []
                for id, item in enumerate(list(self.result2['MACD_MACD_value'])):
                    if item >= 0:
                        num_1.append(self.combine_data.index[id])
                        bar_1.append(list(self.result2['MACD_MACD_value'])[id])
                    else:
                        num_2.append(self.combine_data.index[id])
                        bar_2.append(list(self.result2['MACD_MACD_value'])[id])
                ax.plot(self.combine_data.index, self.result2['MACD_DIF_value'], 'k')
                ax.plot(self.combine_data.index, self.result2['MACD_DEA_value'], 'm')
                ax.bar(num_1, bar_1, color='r', width=0.8, lw=0)  # 涨为红色的条形图
                ax.bar(num_2, bar_2, color='g', width=0.8, lw=0)  # 跌为绿色的条形图
                ax.set_xticks([])
                self.new_data = self.result2

            else:  # 找出符合该类型的列名
                name_list = list(data.columns)
                color_list = ['k', 'r', 'b', 'y', 'g', 'm', 'c']
                count = 0
                newdata = pd.DataFrame()
                for name in name_list:
                    if self.type in name:  # 将符合条件的name列存储到数据框
                        ax.plot(self.combine_data.index, data[name], color_list[count%7])
                        newdata = pd.concat([newdata, data[name]], axis=1)
                        count += 1
                if count == 0:
                    print('您输入的指标名称不在查找范围内，自动退出')
                    exit()
                ax.set_xticks([])
                self.new_data = newdata

            if self.num == 1:
                self.result1 = self.new_data
            if self.num == 2:
                self.result2 = self.new_data
            if self.num == 3:
                self.result3 = self.new_data
            if self.num == 4:
                ax.set_xticks(self.combine_data.index)
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))  # 指定X轴的以日期格式显示, 如果不指定，就显示不出来!!!
                self.result4 = self.new_data
        for label in ax4.xaxis.get_ticklabels():
                    label.set_rotation(45)
        # plt.show()
        return self.result1, self.result2, self.result3, self.result4


if __name__ == "__main__":

    file_dir = '.Data/newdata.csv'
    # par_list = [0, 'no']
    # par_list = [3, 'Kline']

    # 类的实例化,传入文件路径和要查询的代码
    # visualizationv = Visualization(file_dir, par_list)
    # visualizationv.DrawMap()


