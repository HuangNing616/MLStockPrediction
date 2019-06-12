"""
    Author: 黄宁
    Date: June-12-2019
    Function: 股票指标
"""

import pandas as pd
import numpy as np
import talib as ta
import datetime
import math


# MA(Moving Average):计算移动平均指标, 指数平滑方法
# 此函数不需要对特征进行容错，因为还会进入新的列名进行相应计算
def MA_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"feature": "close", "period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:  # 针对没有输入的参数赋默认值
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    sum_value = df_dataframe[arg_dict['feature']].rolling(arg_dict['period'], min_periods=1).sum()
    mean_value = df_dataframe[arg_dict['feature']].rolling(arg_dict['period'], min_periods=1).mean()  # 前两个数分别是与前面的数计算平均
    mean_value = pd.DataFrame(mean_value)
    sum_value = pd.DataFrame(sum_value)
    mean_value.columns = ['Mean_value']
    sum_value.columns = ['Sum_value']
    return mean_value, sum_value


# 我认为指数移动平均和周期没有太大关系，周期只决定了平滑系数
# 问: 昨日移动平均指的是少一天的移动平均吗????直到一天的时候返回当天的值?????平滑系数一直按九天算
# EXPMA(Exponential Moving Average):指数平滑移动平均线, 对移动平均线进行了取长补短，同时又具备了KDJ指标和MACD指标的"金叉"和"死叉"等功能
# 计算公式:EMA（X，N）=2*X/(N+1)+(N-1)*EMA(N-1)/(N+1)
# 当日指数平均值=平滑系数*（当日指数值-昨日指数平均值）+ 昨日指数平均值; 平滑系数=2/（周期单位+1）
def EMA_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"feature": "close", "period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:  # 针对没有输入的参数赋默认值
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    price_list = list(df_dataframe[arg_dict['feature']])  # 将序列变成列表
    smooth_factor = 2/(arg_dict['period']+1)
    EMA_list = []  # 用于存储指数移动平均
    data_len = len(price_list)
    for i in range(data_len):
        if i == 0:
            EMA_list.append(price_list[i])
        else:
            EMA_list.append((1 - smooth_factor) * EMA_list[i - 1] + smooth_factor * price_list[i])
    mean_value = pd.DataFrame(EMA_list)
    sum_value = pd.DataFrame(df_dataframe[arg_dict['feature']].rolling(arg_dict['period'], min_periods=1).sum())

    mean_value.columns = ['Mean_value']
    sum_value.columns = ['Sum_value']
    return mean_value, sum_value


# MACD（Moving Average Convergence and Divergence): 异同移动平均线, 中长期指标
# DIF(离差) = 今日EMA（12）－今日EMA（26）
# DEA(离差平均值)：DIF的9日EMA, 即用今日DIF×2/10+前一日DEA×8/10
# MACD: DIF与它自己的移动平均之间差距的大小一般bar = 2(DIF-DEA), 但是talib中MACD的计算是bar = DIF-DEA
# DIF - DEA均为正, 买入信号参考, 向上突破为金叉
# DIF - DEA均为负, 卖出信号参考, 向下跌破为死叉
# def MACD_Index(*arg_method, df_dataframe, feature, short=12, long=26, period=9):
def MACD_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"feature": "close", "isuser": "user", "period": 9, "short": 12, "long": 26}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    # 特征容错
    feature_list = list(df_dataframe.columns)
    if arg_dict["feature"] not in feature_list:
        print('特征输入有误')
        return -1, -1, -1

    if arg_dict["isuser"] is "user":

        EMA_short = EMA_Index({'feature': arg_dict["feature"], 'period': arg_dict["short"]}, df_dataframe)[0]
        EMA_long = EMA_Index({'feature': arg_dict["feature"], 'period': arg_dict["long"]}, df_dataframe)[0]
        DIF = EMA_short - EMA_long
        DIF.columns = [arg_dict["feature"]]  # 给新构造的快线离差赋予列名,数据框做差必须要列名相同
        DEA = EMA_Index({"feature": arg_dict["feature"], 'period': arg_dict["period"]}, DIF)[0]
        DEA.columns = [arg_dict["feature"]]
        MACD = 2*(DIF-DEA)

    else:
        column_value = df_dataframe[arg_dict["feature"]].values
        DIF, DEA, MACD = ta.MACD(np.array(column_value), fastperiod=6, slowperiod=12, signalperiod=9)
        DIF = pd.DataFrame(DIF)
        DEA = pd.DataFrame(DEA)
        MACD = pd.DataFrame(2*MACD)

    DIF.columns = ['DIF_value']
    DEA.columns = ['DEA_value']
    MACD.columns = ['MACD_value']

    return DIF, DEA, MACD

# KDJ:随机指标, 是一种短期指标，主要是研究最高价、最低价和收盘价之间的关系，衡量股价偏离正常水平的程度
# 即未成熟随机值RSV，计算公式为:RSV=（C－L9）÷（H9－L9）×100
# Cn为第n日收盘价, Ln为n日内的最低价, Hn为n日内的最高价
# K:股价在近期行情中的位置
# D:代表平均位置
# J:反映了KD两线之间的距离
def KDJ_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"period": 9, "M1": 2/3, "M2": 1/3}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    datalen = len(df_dataframe)
    k_list = []     # 用于存储每一天的K值
    d_list = []     # 用于存储每一天的D值
    j_list = []     # 用于存储每一天的J值

    for i in range(datalen):
        if i - arg_dict['period'] < 0:
            b = 0
        else:
            b = i - arg_dict['period'] + 1
        rsv_head = df_dataframe.iloc[b:i + 1]
        close_value = df_dataframe.iloc[i]['close']
        low_value = rsv_head['low'].values.min()
        high_value = rsv_head['high'].values.max()
        rsv_value = (close_value - low_value)/(high_value - low_value)*100

        if i == 0:  # 表示第一天的K值和D值就是RSV的值
            k_value = rsv_value
            d_value = rsv_value

        else:  # 之后的每一天都要借助前一天求解k值和d值
            k_value_yesterday = k_list[i-1]
            d_value_yesterday = d_list[i-1]
            k_value = arg_dict['M1'] * k_value_yesterday + arg_dict['M2'] * rsv_value
            d_value = arg_dict['M1'] * d_value_yesterday + arg_dict['M2'] * k_value
        j_value = 3 * float(k_value) - 2 * float(d_value)
        k_list.append(k_value)
        d_list.append(d_value)
        j_list.append(j_value)

    k_dataframe = pd.DataFrame(k_list)
    d_dataframe = pd.DataFrame(d_list)
    j_dataframe = pd.DataFrame(j_list)

    k_dataframe.columns = ['K_value']
    d_dataframe.columns = ['D_value']
    j_dataframe.columns = ['J_value']

    return k_dataframe, d_dataframe, j_dataframe

# 问：MB是N-1天的是什么意思？？？？？？？？？？
# 布林线：一种短期指标，无法判断长期走势的股价底部和顶部
# 日BOLL指标的计算公式:
# 中轨线=N日的移动平均线
# 上轨线=中轨线＋两倍的标准差
# 下轨线=中轨线－两倍的标准差
# 计算MA
# MA=N日内的收盘价之和÷N
# 计算标准差MD
# MD=平方根N日的（C－MA）的两次方之和除以N
# 计算MB、UP、DN线
# MB=（N－1）日的MA
# UP=MB+2×MD
# DN=MB－2×MD
def BOLL_Index(arg_dict, df_dataframe):


    # 默认参数字典
    arg_dic_default = {"feature": 'close', "period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    # 特征容错
    feature_list = list(df_dataframe.columns)
    if arg_dict["feature"] not in feature_list:
        print('特征输入有误')
        return -1, -1, -1

    middle = MA_Index({'feature': arg_dict['feature'], 'period': arg_dict['period']}, df_dataframe=df_dataframe)[0]

    MD_list = []  # 用于存储标准差的列表
    for i in range(len(middle)):
        if i < arg_dict['period']:
            b = 0
        else:
            b = i - arg_dict['period'] + 1
        diff = df_dataframe.iloc[b:i+1]['close']-middle.iloc[i]['Mean_value']
        MD_list.append((sum(diff**2)/(i+1-b))**0.5)
    MD_df = pd.Series(MD_list)

    upper = middle['Mean_value'] + 2 * MD_df
    lower = middle['Mean_value'] - 2 * MD_df

    upper = pd.DataFrame(upper)
    middle = pd.DataFrame(middle)
    lower = pd.DataFrame(lower)

    upper.columns = ['upper_value']
    middle.columns = ['middle_value']
    lower.columns = ['lower_value']


    return upper, middle, lower


# RSI(Relative Strength Index)：相对强弱指数, 在某一阶段价格上涨所产生的波动占整个波动的百分比。
# 计算公式为：
# RS = X天的平均上涨点数/X天的平均下跌点数
# RSI = RS/(1+RS)*100
# RSI = [上升平均数÷(上升平均数＋下跌平均数)]×100%
# 经过化简之后就是n天上涨的总数比n天上涨的总数和下跌总数之和*100
# 设置阈值低于或高于0.618%算下跌或上涨
# 第一天的RSI默认为前九天平均
def RSI_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"feature": 'close', "period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    # 特征容错
    feature_list = list(df_dataframe.columns)
    if arg_dict["feature"] not in feature_list:
        print('特征输入有误')
        return -1, -1, -1

    column = df_dataframe[arg_dict['feature']]
    data_length = len(column)
    RSI_list = []
    for i in range(data_length-1):  # 从第一行数据开始算起
        ascend_list = []
        descend_list = []

        if i < arg_dict['period']-1:
            b = 0
        else:
            b = i - arg_dict['period'] + 1

        for j in range(b, i+1):
            diff = column[j+1]-column[j]
            threshold = column[j] * 0.00618
            if diff > threshold:
                ascend_list.append(diff)
            if diff < -threshold:
                descend_list.append(abs(diff))
        if len(ascend_list) == 0 and len(descend_list) == 0:  # 如果上涨数和下跌数都为0，那么该天的RSI为0
            RSI_list.append(0)
        else:

            percentage = sum(ascend_list)/(sum(descend_list)+sum(ascend_list))*100
            RSI_list.append(percentage)

    RSI_first = sum(RSI_list[0:arg_dict['period']])/arg_dict['period']
    RSI_list.insert(0, RSI_first)
    RSI_df = pd.DataFrame(RSI_list)
    RSI_df.columns = ['RSI_value']
    return RSI_df


# A=∣当天最高价-前一天收盘价∣
# B=∣当天最低价-前一天收盘价∣
# C=∣当天最高价-前一天最低价∣
# D=∣前一天收盘价-前一天开盘价∣
# 2.比较A、B、C三数值：
# 若A最大，R=A+1/2B+1/4D
# 若B最大，R=B+1/2A+1/4D
# 若C最大，R=C+1/4D
# ASI值第一天默认取前九天的平均
def ASI_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    df_length = len(df_dataframe)
    ASI_list = []   # 用于存储ASI的值
    for i in range(1, df_length):
        A = abs(df_dataframe.iloc[i]['high'] - df_dataframe.iloc[i-1]['close'])
        B = abs(df_dataframe.iloc[i]['low'] - df_dataframe.iloc[i-1]['close'])
        C = abs(df_dataframe.iloc[i]['high'] - df_dataframe.iloc[i-1]['low'])
        D = abs(df_dataframe.iloc[i-1]['close'] - df_dataframe.iloc[i-1]['open'])
        if A > B and A > C:
            R = A + 0.5*B + 0.25*D
        elif B > C:
            R = B + 0.5*A + 0.25*D
        else:
            R = C + 0.25*D
        ASI_list.append(R)
    ASI_first = sum(ASI_list[:arg_dict['period']])/arg_dict['period']
    ASI_list.insert(0, ASI_first)
    ASI_df = pd.DataFrame(ASI_list)
    ASI_df.columns = ['ASI_value']
    return ASI_df


# 8、WR——威廉指标：度量市场处于超买还是超卖状态
# WR(N) = 100 * [ HIGH(N)-C ] / [ HIGH(N)-LOW(N) ]
# C：当日收盘价
# HIGH(N)：N日内最高值的最高价
# LOW(n)：N日内最低值的最低价
def WR_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    length = len(df_dataframe)
    WR_list = []  # 用于存储WR值
    for i in range(length):
        if i - arg_dict['period'] < 0:  # 对于前9天数据进行小窗口处理
            b = 0
        else:
            b = i - arg_dict['period'] + 1
        WR_head = df_dataframe.iloc[b:i+1]
        C = df_dataframe.iloc[i]['close']
        HIGH = WR_head['high'].values.max()
        LOW = WR_head['low'].values.min()
        WR_value = 100 * (HIGH - C)/(HIGH - LOW)
        WR_list.append(WR_value)
    WR_dataframe = pd.DataFrame(WR_list)
    WR_dataframe.columns = ['WR_value']

    return WR_dataframe


# 9、CR:能量指标, 又叫中间意愿指标、价格动量指标, 是中长期的工具
# 计算公式:
# CR（N日）=P1÷P2×100
# P1=Σ（H－YM），表示N日以来多方力量的总和
# P2=Σ（YM－L），表示N日以来空方力量的总和
# H表示今日的最高价，L表示今日的最低价, YM表示昨日（上一个交易日）的中间价
# YM中间价也是一个计算指标（Yesterday Medium）
# M=（2C+H+L）÷4 (中间价的定义有很多)
# C为收盘价，H为最高价，L为最低价，O为开盘价
# 第一天的CR默认取前九天平均
def CR_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    data_length = len(df_dataframe)
    CR_list = []
    for i in range(data_length-1):
        if i - arg_dict['period'] < 0:
            b = 0
        else:
            b = i - arg_dict['period'] + 1
        H = df_dataframe.iloc[b+1:i+2]['high'].values
        L = df_dataframe.iloc[b+1:i+2]['low'].values
        HY = df_dataframe.iloc[b:i+1]['high'].values
        LY = df_dataframe.iloc[b:i+1]['low'].values
        CY = df_dataframe.iloc[b:i+1]['close'].values
        YM = (2*CY+HY+LY)/4  #

        P1 = sum(H - YM)
        P2 = sum(YM - L)
        CR_value = P1/P2*100
        CR_list.append(CR_value)
    CR_first = sum(CR_list[0:arg_dict['period']])/arg_dict['period']
    CR_list.insert(0, CR_first)
    CR_dataframe = pd.DataFrame(CR_list)
    CR_dataframe.columns = ['CR_value']

    return CR_dataframe


# 10、RT: 变差指标
# RT = (lnPt-lnPt-1)*100
def RT_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"feature": 'close', "period": 9}
    arg_list = list(arg_dic_default.keys())
    for arg in arg_list:
        if arg not in arg_dict:
            arg_dict[arg] = eval(arg, arg_dic_default)

    li = list(map(lambda x, y: (math.log(x) - math.log(y))*100, df_dataframe[arg_dict['feature']], df_dataframe[arg_dict['feature']].shift()))
    mean_value = sum(li[1:arg_dict['period']+1])/arg_dict['period']
    li[0] = mean_value
    RT_value = pd.DataFrame(li)
    RT_value.columns = ['RT_value']
    return RT_value

# 11、WVAD：威廉变异离散量
# 计算公式:
# A=当天收盘价-当天开盘价
# B=当天最高价-当天最低价
# C=A/B*成交量
# WVAD=N日ΣC
# WVAD:(收盘价－开盘价)/(最高价－最低价)×成交量
def WVAD_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    A = df_dataframe['close'] - df_dataframe['open']
    B = df_dataframe['high'] - df_dataframe['low']
    C = A/B*df_dataframe['vol']
    C = pd.DataFrame(C)
    C.columns = ['close']
    WVAD = MA_Index({'period': arg_dict['period']}, df_dataframe=C)[1]
    WVAD = pd.DataFrame(WVAD)
    WVAD.columns = ['WVAD_value']
    return WVAD


# 计算公式中分母的成交额指的是amount？？？？？？？？？？？？？？？
# EMV:简易波动指标, 将价格与成交量的变化结合在一起的指标
# 1.A=（今日最高+今日最低）/2
#   B =（前日最高+前日最低）/2
#   C = 今日最高-今日最低
# 2.EM=（A-B）*C/今日成交额
# 3.EMV=N日内EM的累和
# EM值的第一天默认为9日均值
def EMV_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    total = (df_dataframe['high'] + df_dataframe['low'])/2
    diff = df_dataframe['high'] - df_dataframe['low']
    A = total.iloc[1:].values  # 数组结构
    B = total.iloc[:-1].values
    C = diff.iloc[1:].values
    vol = df_dataframe.iloc[1:]['amount'].values

    EM = (A - B) * C / vol
    EM_list = EM.tolist()
    EM_df = pd.DataFrame(EM_list)
    EM_df.columns = ['close']
    EMV = MA_Index({'period': arg_dict['period']}, df_dataframe=EM_df)[1]
    EMV.columns = ['EMV_value']
    EM_first = sum(EMV.EMV_value[0:arg_dict['period']]) / arg_dict['period']
    EMV = pd.concat([pd.DataFrame([EM_first], columns=['EMV_value']), EMV], axis=0)
    EMV = EMV.reset_index(drop=True)
    return EMV


# 13、CCI: 顺势指标，没有上下界
# 计算公式：
# 第一种计算过程如下：
# CCI（N日）=（TP－MA）÷MD÷0.015
# TP=（最高价+最低价+收盘价）÷3,
# MA=近N日收盘价的累计之和÷N,
# MD=近N日（MA－收盘价）的累计之和÷N
# 0.015为计算系数，period为计算周期
# 根据第一天的收盘和MA相同, 所以第一天的MA-MD为接下来9天的平均值
def CCI_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    TP = df_dataframe['high'] + df_dataframe['low'] + df_dataframe['close']
    MA = MA_Index({'feature': 'close'}, df_dataframe=df_dataframe)[0]
    diff = MA['Mean_value'] - df_dataframe['close']
    diff[0] = sum(diff[1:arg_dict['period']+1])/arg_dict['period']
    diff = pd.DataFrame(diff)
    diff.columns = ['diff_value']
    MD = MA_Index({'feature': 'diff_value'}, df_dataframe=diff)[0]
    CCI = (TP - MA['Mean_value'])/(MD['Mean_value']*0.015)
    CCI = pd.DataFrame(CCI)
    CCI.columns = ['CCI_value']
    return CCI


# 14、ROC(Price Rate of Change)又称变动率指标
# 计算公式：
# ROC = (今天的收盘价 - N日前的收盘价) / N日前的收盘价 * 100
def ROC_Index(arg_dict, df_dataframe):


    # 默认参数字典
    arg_dic_default = {"period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)

    data = df_dataframe['close']
    data_len = len(df_dataframe)
    ROC_list = []  # 存储ROC值
    for i in range(data_len):
        if i < arg_dict['period']:
            b = 0
        else:
            b = i - arg_dict['period'] + 1
        cal_data = data[b:i+1]
        ROC_list.append((cal_data[i]-cal_data[b])/cal_data[b]*100)
    ROC_df = pd.DataFrame(ROC_list)
    ROC_df.columns = ['ROC_value']
    return ROC_df


# 1.初始价（TYP）=（当日最高价+当日最低价+当日收盘价）/3
# 2.HH=N日内最高价的最高值
#   LL=N日内最低价的最低值
# 3.压力线
# 初级压力线（WEKR）=TYP+(TYP-LL)
# 中级压力线（MIDR）=TYP+(HH-LL)
# 强力压力线（STOR）=2*HH-LL
# 4.支撑线
# 初级支撑线（WEKS）=TYP-(HH-TYP)
# 中级支撑线（MIDS）=TYP-(HH-LL)
# 强力支撑线（STOS）=2*LL-HH
def MIKE_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)
    data_length = len(df_dataframe)
    TYP = (df_dataframe['high']+df_dataframe['low']+df_dataframe['close'])/3  # 序列可以直接做四则运算
    WEKR_list = []
    MIDR_list = []
    STOR_list = []
    WEKS_list = []
    MIDS_list = []
    STOS_list = []
    for i in range(data_length):
        if i - arg_dict['period'] < 0:
            b = 0
        else:
            b = i - arg_dict['period'] + 1
        cal_data = df_dataframe.iloc[b:i + 1]  # 计算未成熟随机指标值, 返回前几行的数据
        LL = cal_data['low'].values.min()
        HH = cal_data['high'].values.max()
        WEKR = TYP.iloc[i] + (TYP.iloc[i]-LL)
        MIDR = TYP.iloc[i] + (HH - LL)
        STOR = 2*HH-LL
        WEKS = TYP.iloc[i] - (HH - TYP[i])
        MIDS = TYP.iloc[i] - (HH - LL)
        STOS = 2 * LL - HH
        WEKR_list.append(WEKR)
        MIDR_list.append(MIDR)
        STOR_list.append(STOR)
        WEKS_list.append(WEKS)
        MIDS_list.append(MIDS)
        STOS_list.append(STOS)

    WEKR_df = pd.DataFrame(WEKR_list)
    MIDR_df = pd.DataFrame(MIDR_list)
    STOR_df = pd.DataFrame(STOR_list)
    WEKS_df = pd.DataFrame(WEKS_list)
    MIDS_df = pd.DataFrame(MIDS_list)
    STOS_df = pd.DataFrame(STOS_list)

    WEKR_df.columns = ['WEKR_value']
    MIDR_df.columns = ['MIDR_value']
    STOR_df.columns = ['STOR_value']
    WEKS_df.columns = ['WEKS_value']
    MIDS_df.columns = ['MIDS_value']
    STOS_df.columns = ['STOS_value']

    return WEKR_df, MIDR_df, STOR_df, WEKS_df, MIDS_df, STOS_df


# 16、BIAS: 乖离率，是用百分比来表示价格与MA间的偏离程度(差距率)。
# 乖离率=[(当日收盘价-N日平均价)/N日平均价]*100%
def BIAS_Index(arg_dict, df_dataframe):

    # 默认参数字典
    arg_dic_default = {"period": 9}
    arg_list = list(arg_dic_default.keys())

    for arg1 in arg_list:
        if arg1 not in arg_dict.keys():
            arg_dict[arg1] = eval(arg1, arg_dic_default)
    mean_value = MA_Index({'feature': 'close', 'period': arg_dict['period']}, df_dataframe=df_dataframe)[0]
    bias_value = abs(df_dataframe['close'] - mean_value['Mean_value'])/mean_value['Mean_value']*100
    bias_value = pd.DataFrame(bias_value)
    bias_value.columns = ['BIAS_value']
    return bias_value

def LogFunc(text_list):

    with open('E:/Python/Data_simu/log4.txt', 'a') as f:
        time_now = datetime.datetime.now()
        text_str = ' '.join(text_list)
        time_str = time_now.strftime("%Y-%m-%d %H:%M:%S")
        record = time_str + ' ' + text_str + ' ' + '这组数据存在问题' + '\n'
        f.writelines(record)  # 将数据写到了缓存中


def FeatureFunc(df_dataframe, Index_dic):

    data_df = pd.DataFrame()
    index_len = len(Index_dic)
    key_list = ['MA', 'EMA', 'MACD', 'KDJ', 'BOLL', 'RSI', 'ASI', 'WR',
                'CR', 'RT', 'WVAD', 'EMV', 'CCI', 'ROC', 'MIKE', 'BIAS']
    count = 0 # 用于统计已经完成指标的个数
    for key in Index_dic:
        if key in key_list:  # 对列名进行判断
            index_value = eval(key+'_Index(Index_dic[key], df_dataframe=df_dataframe)')
            if isinstance(index_value, tuple):  # 如果返回数据不是一列, 即数据类型为tuple，将其拆开再合并
                value_len = len(index_value)
                for column in range(value_len):
                    feature_newname = key+'_'+list(index_value[column].columns)[0]
                    data_temp = index_value[column]
                    data_temp.columns = [feature_newname]
                    data_df = pd.concat([data_df, data_temp], axis=1)
            else:
                feature_newname = key + '_' + list(index_value.columns)[0]
                data_temp = index_value
                data_temp.columns = [feature_newname]
                data_df = pd.concat([data_df, data_temp], axis=1)
        else:
            print('\r{}指标还未完成，敬请期待...'.format(key))
        count += 1
        percent = count/index_len
        print('\r当前{0:}指标计算完毕，已完成{1:.2%}'.format(key, percent), end='')
    print()
    return data_df


def StockMain(file_dir, StockIndexRoute):

    stock_data = pd.read_csv(file_dir)   # 该数据2985行, 11列
    stock_data.columns = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'post_close']
    # stock_data.sort_values(by='trade_date', ascending=True, inplace=True)

    # li = [['error','000016SH'], ['warning', '000017SH']]
    # for txt in li:
    #     LogFunc(txt)

    # 新生成的指标构成的数据
    Index_value = FeatureFunc(stock_data, {'MA': {}, 'EMA': {}, 'MACD': {}, 'KDJ': {}, 'BOLL': {}, "RSI": {}, 'ASI': {},
                                           'WR': {}, 'CR': {}, 'RT': {}, 'WVAD': {}, 'EMV': {}, 'CCI': {}, 'ROC': {},
                                           'MIKE': {}, 'BIAS': {}})

    combine_data = pd.concat([stock_data, Index_value], axis=1)
    combine_data.to_csv(StockIndexRoute, index=None)


if __name__ == '__main__':
    StockMain('./000016SH.csv')

