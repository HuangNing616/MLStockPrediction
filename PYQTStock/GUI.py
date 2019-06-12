"""
    Author: 黄宁
    Date: June-9-2019
    补充：subplots_adjust:(Tune the subplot layout)调整子图布局, 调整边距和子图的间距
    # wspace 为子图之间的空间保留的宽度，平均轴宽的一部分
    # hspace 为子图之间的空间保留的高度，平均轴高度的一部分
    # right 图片中子图的右侧距离左边的距离
    # left 图片中子图左侧距离左边的距离
    # 待完成功能: 将每次设置新图片的时候保存之前的图片，但是几乎是做不到的
    # PYQT框架下每次都有更新操作, 不会保存之前的图，因此只能是一次一更新，并且设置4个图不能只做一张图
"""

import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from win32api import GetSystemMetrics

# 时间轴刻度为时间，但是位置标注显示不了，因为字符无法对应数字
# from PYQT_index.Plot1 import *
from PYQTStock.Plot2 import *  # 时间轴为数字
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MainUi(QtWidgets.QMainWindow):

    """
        https://zmister.com/archives/793.html
    """

    def __init__(self):
        super().__init__()                      # super() 函数是用于调用父类(超类)的一个方法, 这里调用父类的构造函数
        screenwidth = GetSystemMetrics(0)-200   # 设定显示的屏幕宽度
        screenheight = GetSystemMetrics(1)-100  # 设定显示的屏幕高度
        self.fig_num = 0                        # 表示最开始的默认四张图
        self.fig_name = 'no'                    # 表示没有名字

        # 从屏幕上（100，50）位置开始，显示一个width宽度，height高的界面
        self.setGeometry(100, 50, screenwidth, screenheight)
        self.setWindowTitle("股票指标预测")

        # Qt::CustomizeWindowHint://自定义窗口标题栏,以下标志必须与这个标志一起使用才有效,否则窗口将有默认的标题栏
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint)
        self.setWindowFlag(Qt.WindowTitleHint)

        self.figure = plt.figure(facecolor='white')  # 可选参数,facecolor为背景颜色
        self.ax1 = self.figure.add_subplot(4, 1, 1)
        self.ax2 = self.figure.add_subplot(4, 1, 2)
        self.ax3 = self.figure.add_subplot(4, 1, 3)
        self.ax4 = self.figure.add_subplot(4, 1, 4)

        self.figure.subplots_adjust(left=0.001, bottom=0.05, right=0.999, top=1, wspace=0, hspace=0)
        self.canvas = FigureCanvas(self.figure)

        # Widget是用户界面最基础的原子，它接收鼠标、键盘产生的事件，然后回应。
        self.main_widget = QtWidgets.QWidget()

        # QGridLayout：格栅布局，也被称作网格布局（多行多列）。
        self.main_layout = QtWidgets.QGridLayout()

        # # 将格栅布局添加到组件中
        self.main_widget.setLayout(self.main_layout)

        # Qt程序中的主窗口通常具有一个中心窗口部件。从理论上来讲，
        # 任何继承自QWidget的类的派生类的实例，都可以作为中心窗口部件使用。
        self.setCentralWidget(self.main_widget)

        # 单行文本编辑控件
        self.stock_code = QtWidgets.QLineEdit()
        self.stock_code.setText('start, newdata.csv')  # 将文本框的输入内容保存

        # # 将上述部件添加到布局层中
        self.main_layout.addWidget(self.stock_code, 1, 0, 1, 1)
        self.main_layout.addWidget(self.canvas, 0, 0, 1, 0)
        self.show()

    def Draw(self):

        file_dir = './newdata.csv'
        par_list = [self.fig_num, self.fig_name]
        visualizationv = Visualization(file_dir, par_list)
        result1, result2, result3, result4 = visualizationv.DrawMap()
        def scroll(event):
            """
            控制图形的缩放函数
            """
            axtemp = event.inaxes
            x_min, x_max = axtemp.get_xlim()
            y_min, y_max = axtemp.get_ylim()

            fanwei = (x_max - x_min) / 10

            if event.button == 'up':
                self.ax1.set(xlim=(x_min + fanwei, x_max - fanwei))
                self.ax2.set(xlim=(x_min + fanwei, x_max - fanwei))
                self.ax3.set(xlim=(x_min + fanwei, x_max - fanwei))
                self.ax4.set(xlim=(x_min + fanwei, x_max - fanwei))

            elif event.button == 'down':
                self.ax1.set(xlim=(x_min - fanwei, x_max + fanwei))
                self.ax2.set(xlim=(x_min - fanwei, x_max + fanwei))
                self.ax3.set(xlim=(x_min - fanwei, x_max + fanwei))
                self.ax4.set(xlim=(x_min - fanwei, x_max + fanwei))

        def motion(event):
            try:
                #######################################
                temp1 = y1[int(np.round(event.xdata))]
                for i in range(len_y1):
                    _y1[i] = temp1
                self.line_x1.set_ydata(_y1)
                self.line_y1.set_xdata(event.xdata)
                self.text1.set_position((event.xdata, temp1))

                # 更新显示纵坐标轴的值
                self.text1.set_text(str(temp1))
                #######################################
                temp2 = y2[int(np.round(event.xdata))]
                for i in range(len_y2):
                    _y2[i] = temp2
                self.line_x2.set_ydata(_y2)
                self.line_y2.set_xdata(event.xdata)

                self.text2.set_position((event.xdata, temp2))
                self.text2.set_text(str(temp2))
                #######################################
                temp3 = y3[int(np.round(event.xdata))]
                for i in range(len_y3):
                    _y3[i] = temp3
                self.line_x3.set_ydata(_y3)
                self.line_y3.set_xdata(event.xdata)

                self.text3.set_position((event.xdata, temp3))
                self.text3.set_text(str(temp3))
                #######################################
                temp4 = y4[int(np.round(event.xdata))]
                for i in range(len_y4):
                    _y4[i] = temp4
                self.line_x4.set_ydata(_y4)
                self.line_y4.set_xdata(event.xdata)
                self.text4.set_position((event.xdata, temp4))
                self.text4.set_text(str(temp4))
                #######################################
            except:
                pass
            self.canvas.draw_idle()  # 绘图动作实时反映在图像上

        def button(event):
            axtemp = event.inaxes
            x_min, x_max = axtemp.get_xlim()
            fanwei = (x_max - x_min) / 10
            if event.button == 1:
                self.ax1.set(xlim=(x_min - fanwei, x_max - fanwei))
                self.ax2.set(xlim=(x_min - fanwei, x_max - fanwei))
                self.ax3.set(xlim=(x_min - fanwei, x_max - fanwei))
                self.ax4.set(xlim=(x_min - fanwei, x_max - fanwei))
            else:
                self.ax1.set(xlim=(x_min + fanwei, x_max + fanwei))
                self.ax2.set(xlim=(x_min + fanwei, x_max + fanwei))
                self.ax3.set(xlim=(x_min + fanwei, x_max + fanwei))
                self.ax4.set(xlim=(x_min + fanwei, x_max + fanwei))
        # -------------------------------------------- #

        y1 = list(result1[list(result1.columns)[-1]])  # 取最后一个列名作为时刻的交叉点
        len_y1 = len(y1)
        x1 = list(result1.index)
        _y1 = [y1[-1]] * len_y1

        self.line_x1 = self.ax1.plot(x1, _y1, color='skyblue')[0]
        self.line_y1 = self.ax1.axvline(x=x1[-1], color='skyblue')
        self.text1 = self.ax1.text(x1[-1], y1[-1], str(y1[-1]), fontsize=10)

        # -------------------------------------------- #
        y2 = list(result2[list(result2.columns)[-1]])
        len_y2 = len(y2)
        x2 = list(result2.index)
        _y2 = [y2[-1]] * len_y2
        self.line_x2 = self.ax2.plot(x2, _y2, color='skyblue')[0]
        self.line_y2 = self.ax2.axvline(x=x2[-1], color='skyblue')
        self.text2 = self.ax2.text(x2[-1], y2[-1], str(y2[-1]), fontsize=10)

        # -------------------------------------------- #
        y3 = list(result3[list(result3.columns)[-1]])
        len_y3 = len(y3)
        x3 = list(result3.index)
        _y3 = [y3[-1]] * len_y3
        self.line_x3 = self.ax3.plot(x3, _y3, color='skyblue')[0]
        self.line_y3 = self.ax3.axvline(x=x3[-1], color='skyblue')
        self.text3 = self.ax3.text(x3[-1], y3[-1], str(y3[-1]), fontsize=10)

        # -------------------------------------------- #
        y4 = list(result4[list(result4.columns)[-1]])
        len_y4 = len(y4)
        x4 = list(result4.index)
        _y4 = [y4[-1]] * len_y4
        self.line_x4 = self.ax4.plot(x4, _y4, color='skyblue')[0]       # 控制横线
        self.line_y4 = self.ax4.axvline(x=x4[-1], color='skyblue')  # 控制竖线
        self.text4 = self.ax4.text(x4[-1], y4[-1], str(y4[-1]), fontsize=10)

        # -------------------------------------------- #
        # 使用fig.canvas.mpl_connect()函数来绑定相关fig的滚轮事件
        # 利用事件event的inaxes属性获取当前鼠标所在坐标系ax

        self.canvas.mpl_connect('scroll_event', scroll)
        self.canvas.mpl_connect('motion_notify_event', motion)
        self.canvas.mpl_connect('button_press_event', button)
        self.canvas.draw()  # 用于反馈回车之后的第一章初始图片

    def keyPressEvent(self, event):
        if (event.key() == QtCore.Qt.Key_Enter) | (event.key() == QtCore.Qt.Key_Return):
            text = self.stock_code.text()               # 如果没有输入就用默认的set，如果有输入就更新输入
            split_text = text.split(',')                # 返回一个指令列表
            if 'set' in split_text[0]:
                self.fig_num = int(split_text[0][-1])   # 获取更新第几张图
                self.fig_name = split_text[1]           # 获取更新图片的名字
                print(self.fig_name)
            self.Draw()
            self.stock_code.clear()                     # 清除文本框内容

        if event.key() == QtCore.Qt.Key_Escape:
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    QtWidgets.QApplication.processEvents()
    dispatch = MainUi()
    sys.exit(app.exec_())

