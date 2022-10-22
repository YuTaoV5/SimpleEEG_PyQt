'''
DEAP 数据库

EEG_feature.txt 包含了1216个脑电信号样本的160维特征,每行为一个样本,每列为一种特征。特征从左至右分别是每个脑电电极的theta(1-32列)、slow alpha(33-64列)、alpha(65-96列)、beta(1-97128列)、gamma(129-160列)波段的脑电特征。

subject_video.txt 包含了1216个脑电信号对应的32名被试和38段视频信息,其中包含两列。第一列是对应的被试编号,第二列是对应的视频编号。

EEG_feature.txt 与 subject_video.txt和valence_arousal_label.txt中每行都是一一对应的,例如subject_video.txt的第二行就是EEG_feature.txt中第二个样本(第二行)的被试和视频信息；valence_arousal_label.txt的第二行也是EEG_feature.txt中第二个样本(第二行)的愉悦度和唤醒度标签。valence_arousal_label.txt中第一列为愉悦度标签,1代表positive,2代表negative；第二列为唤醒度标签,1代表high,2代表low。

DEAP数据库并未提供情感类别标签。

'''


import sys
import serial
import serial.tools.list_ports
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QGridLayout
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QMutex, QUrl
from qt_material import apply_stylesheet
from gui import Ui_Form
import pynput
from pynput import keyboard
import vtk
import netron
from PyQt5 import QtCore, QtGui
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTKWidget
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PyQt5.QtWidgets import QApplication,QMainWindow
import os
from pathlib import Path
import os.path as osp
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from scipy.interpolate import make_interp_spline
import logging
import time
import json
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# data 中心化
def MaxMinNormalization(x):
    x = np.transpose(x)
    return np.transpose(np.array([(x[i] - np.average(x[i])) / np.std(x[i]) for i in range(x.shape[0])]))

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev = 0.2)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize = [1,2,2, 1],strides = [1,1,1, 1], padding='SAME')

class Pyqt5_Serial(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(Pyqt5_Serial, self).__init__()
        self.control = keyboard.Controller()
        self.Com_Dict = None
        self.sess = None
        self.setupUi(self)
        self.pushButton_train.setStyleSheet('border-radius: 10px')
        self.pushButton_save.setStyleSheet('border-radius: 10px')
        self.pushButton_label.setStyleSheet('border-radius: 10px')
        self.pushButton_model.setStyleSheet('border-radius: 10px')
        self.pushButton_dataset.setStyleSheet('border-radius: 10px')
        self.init()
        self.trainflag=0
        self.setWindowTitle("EEG上位机")
        self.setWindowIcon(QIcon("mylogo.jpg"))
        self.ser = serial.Serial()
        self.port_check()
        self.pushButton_train.setEnabled(False)
        self.pushButton_save.setEnabled(False)
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))

        # 模型
        self.vtk_widget = QVTKWidget()
        self.vl.addWidget(self.vtk_widget)

        self.render_window = self.vtk_widget.GetRenderWindow()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(49/255, 54/255, 59/255)

        self.render_window.AddRenderer(self.renderer)
        self.render_window.Render()
        self.iren = self.render_window.GetInteractor()
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)
        self.original_model = vtk.vtkOBJReader()
        self.original_model.SetFileName("../Model/F450x.obj")
        self.original_model.Update()

        self.original_mapper = vtk.vtkPolyDataMapper()
        self.original_mapper.SetInputConnection(self.original_model.GetOutputPort())

        self.original_actor = vtk.vtkActor()
        self.original_actor.SetMapper(self.original_mapper)
        self.original_actor.GetProperty().SetColor(0.5, 0.5, 0.5)

        self.renderer.AddActor(self.original_actor)
        self.renderer.ResetCamera()
    def init(self):

        # 串口检测按钮
        self.s1__box_1.clicked.connect(self.port_check)

        # 串口信息显示
        self.s1__box_2.currentTextChanged.connect(self.port_imf)

        # 打开串口按钮
        self.open_button.clicked.connect(self.port_open)

        # 关闭串口按钮
        self.close_button.clicked.connect(self.port_close)

        # 发送数据按钮
        self.s3__send_button.clicked.connect(self.data_send)

        # 定时发送数据
        self.timer_send = QTimer()
        self.timer_send.timeout.connect(self.data_send)
        self.timer_send_cb.stateChanged.connect(self.data_send_timer)

        # 定时器接收数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.data_receive)

        # 清除发送窗口
        self.s3__clear_button.clicked.connect(self.send_data_clear)

        # 清除接收窗口
        self.s2__clear_button.clicked.connect(self.receive_data_clear)

        # 选择数据集
        self.pushButton_dataset.clicked.connect(self.open_dataset)
        # 选择标签
        self.pushButton_label.clicked.connect(self.open_label)

        # 训练
        self.pushButton_train.clicked.connect(self.start_train)

        # 保存模型
        self.pushButton_save.clicked.connect(self.save_model)

        # 方向
        self.pushButton_W.clicked.connect(self.pressW)
        self.pushButton_A.clicked.connect(self.pressA)
        self.pushButton_S.clicked.connect(self.pressS)
        self.pushButton_D.clicked.connect(self.pressD)

        # 显示模型结构
        self.pushButton_model.clicked.connect(self.my_netron)

    def my_netron(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.h5')
        if fileName:
            netron.start(fileName)
            self.browser1.load(QUrl("http://localhost:8080/"))
    def pressW(self):
        self.pushButton_W.setEnabled(True)
        self.control.press('w')
        time.sleep(0.3)
        self.control.release('w')
        self.pushButton_W.setEnabled(False)

    def pressA(self):
        self.pushButton_A.setEnabled(True)
        self.control.press('a')
        time.sleep(0.3)
        self.control.release('a')
        self.pushButton_A.setEnabled(False)

    def pressS(self):
        self.pushButton_S.setEnabled(True)
        self.control.press('s')
        time.sleep(0.3)
        self.control.release('s')
        self.pushButton_S.setEnabled(False)

    def pressD(self):
        self.pushButton_D.setEnabled(True)
        self.control.press('d')
        time.sleep(0.3)
        self.control.release('d')
        self.pushButton_D.setEnabled(False)
    # 串口检测
    def port_check(self):
        # 检测所有存在的串口,将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        self.s1__box_2.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.s1__box_2.addItem(port[0])
        if len(self.Com_Dict) == 0:
            self.state_label.setText(" 无串口")


    # 串口信息
    def port_imf(self):
        # 显示选定的串口的详细信息
        imf_s = self.s1__box_2.currentText()
        if imf_s != "":
            self.state_label.setText(self.Com_Dict[self.s1__box_2.currentText()])

    # 打开串口
    def port_open(self):
        self.ser.port = self.s1__box_2.currentText()
        self.ser.baudrate = int(self.s1__box_3.currentText())
        self.ser.bytesize = int(self.s1__box_4.currentText())
        self.ser.stopbits = int(self.s1__box_6.currentText())
        self.ser.parity = self.s1__box_5.currentText()

        try:
            self.ser.open()
        except:
            QMessageBox.critical(self, "Port Error", "此串口不能被打开！")
            return None

        # 打开串口接收定时器,周期为20ms
        self.timer.start(20)

        if self.ser.isOpen():
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)
            #self.formGroupBox1.setTitle("串口状态(已开启)")

    # 关闭串口
    def port_close(self):
        self.timer.stop()
        self.timer_send.stop()
        try:
            self.ser.close()
        except:
            pass
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.lineEdit_3.setEnabled(True)
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))
        #self.formGroupBox1.setTitle("串口状态(已关闭)")

    # 发送数据
    def data_send(self):
        if self.ser.isOpen():
            input_s = self.s3__send_text.toPlainText()
            if input_s != "":
                # 非空字符串
                if self.hex_send.isChecked():
                    # hex发送
                    input_s = input_s.strip()
                    send_list = []
                    while input_s != '':
                        try:
                            num = int(input_s[0:2], 16)
                        except ValueError:
                            QMessageBox.critical(self, 'wrong data', '请输入十六进制数据,以空格分开!')
                            return None
                        input_s = input_s[2:].strip()
                        send_list.append(num)
                    input_s = bytes(send_list)
                else:
                    # ascii发送
                    input_s = (input_s + '\r\n').encode('utf-8')

                num = self.ser.write(input_s)
                self.data_num_sended += num
                self.lineEdit_2.setText(str(self.data_num_sended))
        else:
            pass

    # 接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()
        except:
            self.port_close()
            return None
        if num > 0:
            data = self.ser.read(num)
            num = len(data)
            # hex显示
            if self.hex_receive.checkState():
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '{:02X}'.format(data[i]) + ' '
                self.s2__receive_text.insertPlainText(out_s)
            else:
                # 串口接收到的字符串为b'123',要转化成unicode字符串才能输出到窗口中去
                self.s2__receive_text.insertPlainText(data.decode('iso-8859-1'))

            # 统计接收字符的数量
            self.data_num_received += num
            self.lineEdit.setText(str(self.data_num_received))

            # 获取到text光标
            textCursor = self.s2__receive_text.textCursor()
            # 滚动到底部
            textCursor.movePosition(textCursor.End)
            # 设置光标到text中去
            self.s2__receive_text.setTextCursor(textCursor)
            cal_data = data.decode('iso-8859-1')
            if not self.checkBox_EEG.isChecked() and not self.checkBox_IoT.isChecked():
                head = cal_data.find('[')
                tail = cal_data.find(']',head)
                num = cal_data.count(',',head,tail)
                if head != -1 and tail != -1 and num == 12:
                    cal_data = cal_data[head+1:tail]
                    my_data = cal_data.split(",")
                    if my_data[1] == '1':
                        self.label_bow.setText("已开弓")
                    else:
                        self.label_bow.setText("未开弓")
                    if my_data[0] == '1':
                        self.label_control.setText("已控制")
                    else:
                        self.label_control.setText("未控制")

                    if my_data[3] == '1':#W
                        self.pushButton_W.setEnabled(True)
                    else:
                        self.pushButton_W.setEnabled(False)
                    if my_data[4] == '1':#A
                        self.pushButton_A.setEnabled(True)
                    else:
                        self.pushButton_A.setEnabled(False)
                    if my_data[5] == '1':#S
                        self.pushButton_S.setEnabled(True)
                    else:
                        self.pushButton_S.setEnabled(False)
                    if my_data[2] == '1':#D
                        self.pushButton_D.setEnabled(True)
                    else:
                        self.pushButton_D.setEnabled(False)
                    self.lcdNumber_ax.display(my_data[7])
                    self.lcdNumber_ay.display(my_data[8])
                    self.lcdNumber_az.display(my_data[9])
                    self.lcdNumber_gx.display(my_data[10])
                    self.lcdNumber_gy.display(my_data[11])
                    self.lcdNumber_gz.display(my_data[12])

                #print(my_data)
                # 鼠标初始化
                mouse = pynput.mouse.Controller()
                heading = madgwickahrs.MadgwickAHRS()
                accel = [float(my_data[7]), float(my_data[8]), float(my_data[9])]
                gyro = [float(my_data[10]), float(my_data[11]), float(my_data[12])]
                heading.update_imu(gyro, accel)
                ahrs = heading.quaternion.to_euler_angles()
                roll = ahrs[0]*1000
                pitch = ahrs[1]*1000
                yaw = ahrs[2]*1000
                self.lcdNumber_heading.display(yaw)
                self.lcdNumber_roll.display(roll)
                self.lcdNumber_pitch.display(pitch)
                print(roll, pitch, yaw)
                if my_data[0] == '1':
                    mouse.move(-20*float(my_data[10]), 20*float(my_data[11]))
                else:
                    pass
            if self.checkBox_EEG.isChecked() and not self.checkBox_IoT.isChecked():
                head = cal_data.find('[')
                tail = cal_data.find(']', head)
                num = cal_data.count(', ', head, tail)
                if head != -1 and tail != -1 and num == 159:
                    cal_data = cal_data[head+1:tail]
                    #my_data = cal_data.split(", ")
                #test_data = np.fromstring(cal_data, sep=', ')
                '''
                sess = tf.compat.v1.Session()
                saver = tf.compat.v1.train.Saver()
                model_path = 'P:\\Item\\EEG-Drone\\PyQt\\Model\\model.ckpt'
                saver.restore(sess, model_path)
                '''
                #y = sess.run(y, feed_dict={x: data})
            if self.checkBox_EEG.isChecked() and self.checkBox_IoT.isChecked():
                if len(cal_data) == 1:
                    if cal_data == "w":
                        self.pressW()
                    if cal_data == "a":
                        self.pressA()
                    if cal_data == "s":
                        self.pressS()
                    if cal_data == "d":
                        self.pressD()



    # 定时发送数据
    def data_send_timer(self):
        if self.timer_send_cb.isChecked():
            self.timer_send.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)
        else:
            self.timer_send.stop()
            self.lineEdit_3.setEnabled(True)

    # 清除显示
    def send_data_clear(self):
        self.s3__send_text.setText("")

    def receive_data_clear(self):
        self.s2__receive_text.setText("")

    def open_dataset(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.txt')
        if fileName:
            self.train_data = np.loadtxt(fileName)
            self.trainflag=1
            rem_text = "数据集已加载"+" , "+"请添加标签文件"
            self.label_dital.setText(rem_text)
            self.pushButton_dataset.setEnabled(False)
    def open_label(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.txt')
        if fileName:
            self.train_label = np.loadtxt(fileName)
            if self.trainflag == 1:
                self.pushButton_train.setEnabled(True)
                data_dital = "数据集大小为: " + " " + "   " +str(self.train_data.shape) + "\r\n" + "标签集大小为: " + " " + "   "+ str(self.train_label.shape)
                self.label_dital.setText(data_dital)
            self.pushButton_label.setEnabled(False)

    def save_model(self):
        with tf.compat.v1.Session() as self.sess:
            tf.compat.v1.global_variables_initializer().run()
            saver = tf.compat.v1.train.Saver()
            model_path, type = QFileDialog.getSaveFileName(self, "文件保存", "/",
                                                         'ckpt(*.ckpt)')  # 前面是地址，后面是文件类型,得到输入地址的文件名和地址txt(*.txt*.xls);;image(*.png)不同类别
            #model_path = 'P:\\Item\\EEG-Drone\\PyQt\\Model\\model.ckpt'
            saver.save(self.sess, model_path)
            self.label_dital.setText("模型已保存!")


    def start_train(self):

        train_data = self.train_data
        train_label = self.train_label
        train_label = np.transpose(train_label)
        # 将label转换成类别矩阵
        train_label = np.transpose([2 - train_label[1], train_label[1] - 1])
        train_data = MaxMinNormalization(train_data)


        learning_rate = float(self.lineEdit_learningrate.text())
        n_input = 160
        n_classes = int(self.lineEdit_classes.text())
        # 占位符
        # tf.compat.v1.disable_eager_execution()#消错
        tf.compat.v1.disable_eager_execution()
        x_ = tf.compat.v1.placeholder(tf.float32, [None, n_input])  # 改成v1版本compat.v1
        y_ = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
        x = tf.reshape(x_, [-1, 32, 5, 1])
        keep_prob = tf.compat.v1.placeholder(tf.float32)
        # 建立网络
        # 卷积层1
        W_conv1 = weight_variable([2, 2, 1, 2])
        b_conv1 = bias_variable([2])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        # 卷积层2
        W_conv2 = weight_variable([2, 2, 2, 2])
        b_conv2 = bias_variable([2])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        # 全连接层
        W_fc1 = weight_variable([2 * 32 * 5, 25])
        b_fc1 = bias_variable([25])
        h_pool2_flat = tf.reshape(h_conv2, [-1, 2 * 32 * 5])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_dropout = tf.nn.dropout(h_fc1, rate=1 - keep_prob)
        # 输出层
        W_fc2 = weight_variable([25, n_classes])
        b_fc2 = bias_variable([n_classes])
        y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

        # 训练和评估#

        # 均方误差MSE
        if self.comboBox_loss.currentIndex() == 0:
            loss=tf.reduce_mean(tf.square(y_conv-y_))
            print("均方误差MSE")
        # 交叉熵
        if self.comboBox_loss.currentIndex() == 1:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
            print("交叉熵")
        # 优化器
        if self.comboBox_optimizer.currentIndex() == 0:
            train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            print("梯度下降")
        if self.comboBox_optimizer.currentIndex() == 1:
            train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)#
            print("ADAM")

        if self.comboBox_optimizer.currentIndex() == 2:
            train_step = tf.compat.v1.train.AdagradOptimizer(learning_rate).minimize(loss)
            print("Adagrad")
        if self.comboBox_optimizer.currentIndex() == 3:
            train_step = tf.compat.v1.train.MomentumOptimizer(learning_rate).minimize(loss)
            print("动量优化")
        if self.comboBox_optimizer.currentIndex() == 4:
            train_step = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(loss)
            print("RMSProp")
        # 准确度
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))

        # 训练
        with tf.compat.v1.Session() as self.sess:
            tf.compat.v1.global_variables_initializer().run()
            train_accuracy_all = []
            test_accuracy_all = []
            train_loss_all = []
            test_loss_all = []
            epoch_times = int(self.lineEdit_epoch.text())
            # rand=random.sample(range(1216),215)
            for i in range(epoch_times):
                train_loss = loss.eval(feed_dict={x_: train_data[0:1000], y_: train_label[0:1000], keep_prob: 1.})
                train_accuracy = accuracy.eval(
                    feed_dict={x_: train_data[0:1000], y_: train_label[0:1000], keep_prob: 1.})
                train_accuracy_all.append(train_accuracy)
                train_loss_all.append(train_loss)
                test_loss = loss.eval(feed_dict={x_: train_data[1001:1215], y_: train_label[1001:1215], keep_prob: 1.})
                test_accuracy = accuracy.eval(
                    feed_dict={x_: train_data[1001:1215], y_: train_label[1001:1215], keep_prob: 1.})
                test_accuracy_all.append(test_accuracy)
                test_loss_all.append(test_loss)
                self.progressBar.setMaximum(epoch_times)
                self.progressBar.setValue(i+1)
                if (i % 50 == 0):
                    print(i)
                    print('setp {},the train accuracy: {},the train MSE: {}'.format(i, train_accuracy, train_loss))
                    print('setp {},the test accuracy: {},the test MSE: {}'.format(i, test_accuracy, test_loss))
                # rand=np.random.randint(0,high=1000, size=500)
                # add noise
                # noise=np.random.normal(0,1/500,[1216,160])
                # train_data_noised=np.add(train_data,noise)
                train_step.run(feed_dict={x_: train_data[0:1000], y_: train_label[0:1000], keep_prob: .25})
            self.pushButton_save.setEnabled(True)
            '''
            saver = tf.compat.v1.train.Saver()
            model_path = 'P:\\Item\\EEG-Drone\\PyQt\\Model\\model.ckpt'
            saver.save(sess, model_path)
            
            builder = tf.compat.v1.saved_model.builder.SaveModelBuilder(model_path)  # PATH是保存路径
            builder.add_meta_graph_and_variables(sess,[
                tf.compat.v1.saved_model.tag_constants.TRAINING])#保存整张网络及其变量,这种方法是可以保存多张网络的,在此不作介绍,可自行了解
            builder.save()#完成保存
            '''

            # 绘制模型分类结果的性能图
            filepath = "train_data.txt"
            file = open(filepath, 'w')
            #data_str = ' '.join(map(str, train_data.ravel().tolist()))
            #file.write(data_str)
            data_str = train_data.tolist()  # [1,2,3]
            data_str = str(data_str)  # '[1,2,3]'
            file.write(data_str)
            plt.figure(0)
            plt.plot(train_accuracy_all)
            plt.plot(test_accuracy_all)
            plt.hlines(0.55, 0, i, colors='green', linestyles="dashed")
            plt.hlines(1.0, 0, i, colors='green', linestyles="dashed")
            plt.xlabel('number')
            plt.ylabel('classification accuracy')
            plt.title('Deep : positive & negative')
            plt.tight_layout()

            #plt.show()
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            plt.figure(1)

            ax1 = plt.subplot(221)
            ax1 = plt.plot(train_loss_all)
            ax1 = plt.title('训练集误差')

            ax2 = plt.subplot(222)
            ax2 = plt.plot(train_accuracy_all)
            ax2 = plt.title('训练集准确度')

            ax3 = plt.subplot(223)
            ax3 = plt.title('测试集误差')
            ax3 = plt.plot(test_loss_all)

            ax4 = plt.subplot(224)
            ax4 = plt.title('测试集准确度')
            ax4 = plt.plot(test_accuracy_all)
            plt.tight_layout()

            #plt.show()
            self.pushButton_dataset.setEnabled(True)
            self.pushButton_label.setEnabled(True)
            # 创建figure
            figure1 = plt.figure(0)
            figure2 = plt.figure(1)
            # 绑定figure到canvas上

            canvas1 = FigureCanvas(figure1)
            canvas2 = FigureCanvas(figure2)

            # 更新canvas画布
            canvas1.draw()
            canvas2.draw()
            # 显示至pyqt主界面
            self.tabWidget_3.setCurrentIndex(2)
            self.gridlayout = QGridLayout(self.plot_view1)  # GroupBox的name
            self.gridlayout.addWidget(canvas1)
            self.gridlayout = QGridLayout(self.plot_view2)  # GroupBox的name
            self.gridlayout.addWidget(canvas2)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = Pyqt5_Serial()
    # setup stylesheet
    apply_stylesheet(app, theme='dark_amber.xml')#dark_amber
    stylesheet = app.styleSheet()
    with open('custom.css') as file:
        app.setStyleSheet(stylesheet + file.read().format(**os.environ))
    # setup stylesheet
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # or in new API
   # app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    myshow.show()
    sys.exit(app.exec_())
