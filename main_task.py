import sys
import random
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore ,uic ,QtWidgets
from PyQt5.QtWidgets import QApplication, QSizePolicy, QFileDialog
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PyQt5.uic import loadUiType
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


# figure canvas classes to use them in UI
class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        # self.axes.hold(False)
        self.compute_initial_figure()
        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

# A static figure Canvas
class phantomMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def compute_initial_figure(self):
        # Open the image file
        img = Image.open('images/Phantom512.png')
        # Get the pixels array as a 2D list
        pixel_data = list(img.getdata())
        # Convert the pixel data to a NumPy array
        t1_img_array = np.array(pixel_data)
        # Reshape the array to match the image dimensions
        width, height = img.size
        t1_img_array = t1_img_array.reshape((height, width, 3))
        self.axes.imshow(t1_img_array, cmap='gray')
        


# Create a class for your main window that inherits from Ui_MainWindow and QMainWindow
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # -------------link ui file------------------------------#
        uic.loadUi(r'UI/Task1.ui', self)
        
        #--------------Adding Canvas figures to layouts-----------#
        phantomLayout = self.verticalLayout_13 
        phantomCanvas = phantomMplCanvas(self.centralwidget, width=5, height=4, dpi=100)
        phantomLayout.addWidget(phantomCanvas)# phantom Canvas
        

        self.sequenceLayout = self.verticalLayout_12
        self.sequenceCanvas = MyMplCanvas(self.centralwidget, width=5, height=4, dpi=100)
        self.sequenceLayout.addWidget(self.sequenceCanvas)# sequence Canvas
        #self.sequenceCanvas.draw()


        
        # ---------------------Global variables----------------------#
        self.Rf_line = 20
        self.Gz_line = 15
        self.Gy_line = 10
        self.Gx_line = 5
        self.Ro_line = 0

        # -----------------Connect buttons with functions--------------#
        phantomCanvas.mpl_connect('button_press_event', self.phantom_onClick)
        self.actionOpen.triggered.connect(lambda:self.read_file())
        

    # -----------------------functions defination------------------------#
    def phantom_onClick(self , event):
        # x, y = int(event.xdata), int(event.ydata)
        # print(f'Clicked on pixel ({x}, {y})')
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        ('double' if event.dblclick else 'single', event.button,
        event.x, event.y , event.xdata, event.ydata))
        self.T1value_label.setText(str(event.xdata))
        self.T2value_label.setText(str(event.ydata))
        self.PDvalue_label.setText(str(event.ydata))

    def read_file(self):# BROWSE TO READ THE FILE
        self.File_Path = QFileDialog.getOpenFileName(self, "Open File", "This PC",
            "All Files (*);;JSON Files(*.json)")

        with open(self.File_Path[0], 'r') as handle:
            json_data = [json.loads(line) for line in handle]
        self.df = pd.DataFrame(json_data)

        #calling the function that used in plotting mri sequence
        self.Draw_Sequence(self.df)

    def Draw_Sequence(self, df):
        # plotting constant lines for Rf,Gz,Gy,Gx,Ro
        [self.sequenceCanvas.axes.axhline(y=i, color='r', linestyle='-') for i in [self.Ro_line, self.Gx_line, self.Gy_line, self.Gz_line, self.Rf_line]]

        # plotting functions of Rf,Gz,Gy,Gx,Ro
        x1 = np.linspace(df["RF_Ts"].values[0], df["RF_Te"].values[0], 1000)
        y1 = self.Rf_line + ((df["RF_value"].values[0]) * np.sinc(x1 - 10))

        x5 = np.linspace(df["Ro_Ts"].values[4], df["Ro_Te"].values[4], 1000)
        y5 = self.Ro_line + ((df["Ro_value"].values[4]) * np.sinc(x5 - 55))

        self.sequenceCanvas.axes.plot(x1, y1, color='maroon', marker='o')
        self.sequenceCanvas.axes.step(x=[df["Gz_Ts"].values[1], df["Gz_Te"].values[1], df["Gz_Te"].values[1]],
                 y=[self.Gz_line, (self.Gz_line + 1) * df["Gz_value"].values[1], self.Gz_line])
        self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[2], df["Gy_Te"].values[2], df["Gy_Te"].values[2]],
                 y=[self.Gy_line, (self.Gy_line + 1) * df["Gy_value"].values[2], self.Gy_line])
        self.sequenceCanvas.axes.step(x=[df["Gx_Ts"].values[3], df["Gx_Te"].values[3], df["Gx_Te"].values[3]],
                 y=[self.Gx_line, (self.Gx_line + 1) * df["Gx_value"].values[3], self.Gx_line])
        self.sequenceCanvas.axes.plot(x5, y5, color='maroon', marker='o')

        # Plotting repeat of Gy if it exists
        if (df["Gy_repeated"].values[2] == "True"):
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[2], df["Gy_Te"].values[2], df["Gy_Te"].values[2]],
                     y=[(self.Gy_line + 1), (self.Gy_line + 2) * df["Gy_value"].values[2], (self.Gy_line + 1)])
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[2], df["Gy_Te"].values[2], df["Gy_Te"].values[2]],
                     y=[(self.Gy_line + 2), (self.Gy_line + 3) * df["Gy_value"].values[2], (self.Gy_line + 2)])

        # Plotting reverse of Gy if it exists
        if (df["Gy_reversed"].values[2] == "True"):
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[2], df["Gy_Te"].values[2], df["Gy_Te"].values[2]],
                     y=[self.Gy_line, ((self.Gy_line + 1) * df["Gy_value"].values[2] * -1) + (self.Gy_line + 1) + 9, self.Gy_line])
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[2], df["Gy_Te"].values[2], df["Gy_Te"].values[2]],
                     y=[(self.Gy_line + 1), ((self.Gy_line + 2) * df["Gy_value"].values[2] * -1) + (self.Gy_line + 1) + 9,
                        (self.Gy_line + 1)])
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[2], df["Gy_Te"].values[2], df["Gy_Te"].values[2]],
                     y=[(self.Gy_line + 2), ((self.Gy_line + 3) * df["Gy_value"].values[2] * -1) + (self.Gy_line + 1) + 9,
                        (self.Gy_line + 2)])

        self.sequenceCanvas.axes.set_xlabel('t (msec)')
        self.sequenceCanvas.axes.set_yticklabels([ 0,'Ro', 'Gx', 'Gy', 'Gz', 'Rf'])

        self.sequenceCanvas.draw()


    def plotting(self,GRAPHICSINDEX,X_ARRAY,Y_ARRAY,COLORLIST):
        self.GraphicsView[GRAPHICSINDEX].plot(X_ARRAY, Y_ARRAY, pen=COLORLIST)
        self.GraphicsView[GRAPHICSINDEX].plotItem.setLabel("bottom", text="Time (ms)")
        self.GraphicsView[GRAPHICSINDEX].plotItem.showGrid(True, True, alpha=1)
        self.GraphicsView[GRAPHICSINDEX].plotItem.setLimits(xMin=0, xMax=10, yMin=-20, yMax=20)


if __name__ == '__main__':
    # Instantiate the main window class and show it
    app = QApplication([])
    window = MainWindow()
    window.show()
    # Run the application
    app.exec_()