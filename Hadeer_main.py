import sys
import random
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore ,uic ,QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget,QInputDialog, QLineEdit, QFileDialog
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pathlib
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
class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        img = nib.load('./data/images/BRATS_002.nii.gz')
        imgArray = img.get_fdata()
        imgArrayShape = imgArray.shape
        self.axes.imshow(imgArray[:,:,imgArrayShape[2]//3 , 0], cmap='gray')
        self.axes.plot()

# A Dynamic figure canvas 
class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [random.randint(0, 10) for i in range(4)]
        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()


# Create a class for your main window that inherits from Ui_MainWindow and QMainWindow
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # -------------link ui file------------------------------#
        uic.loadUi(r'UI/Task1.ui', self)
        
        #--------------Adding Canvas figures to layouts-----------#
        phantomLayout = self.verticalLayout_13
        phantomCanvas = MyMplCanvas(self.centralwidget, width=5, height=4, dpi=100)
        phantomLayout.addWidget(phantomCanvas)

        self.sequenceLayout = self.verticalLayout_12
        self.sequenceCanvas = MyMplCanvas(self.centralwidget, width=5, height=4, dpi=100)
        self.sequenceLayout.addWidget(self.sequenceCanvas)
        #self.sequenceCanvas.draw()


        
        # ---------------------Global variables----------------------#
        self.Rf_line = 20
        self.Gz_line = 15
        self.Gy_line = 10
        self.Gx_line = 5
        self.Ro_line = 0
        self.JSON_List = []
        #with open('json_data.json', 'w') as f:
        #    json.loads("{}")

        # -----------------Connect buttons with functions--------------#
        self.actionOpen.triggered.connect(lambda:self.read_file())
        self.Sequence_Combobox.activated.connect(self.Generate_Sequence)
        self.Flag_Line_Edit.textChanged.connect(lambda: self.get_Flag())
        #self.Te_Line_Edit.textChanged.connect(lambda: self.DrawTR_TE())
        self.Value_Line_Edit.textChanged.connect(lambda: self.get_Value())
        self.Ts_Line_Edit.textChanged.connect(lambda: self.get_Ts())
        self.Te_Line_Edit.textChanged.connect(lambda: self.get_Te())
        self.Export_Button.clicked.connect(self.DrawTR_TE)

        


    # -----------------------functions defination------------------------#
    def read_file(self):# BROWSE TO READ THE FILE
       # path = QFileDialog.getOpenFileName()[0]
       #if pathlib.Path(path).suffix == ".csv":
        #   self.data = np.genfromtxt(path, delimiter=',')

        self.File_Path = QFileDialog.getOpenFileName(self, "Open File", "This PC",
            "All Files (*);;JSON Files(*.json)")

        with open(self.File_Path[0], 'r') as handle:
            json_data = [json.loads(line) for line in handle]
        self.df = pd.DataFrame(json_data)

        #calling the function that used in plotting mri sequence
        self.Draw_Sequence(self.df)

    def get_Value(self):
        if self.Value_Line_Edit.text() != "":
            self.value = self.Value_Line_Edit.text()
        return float(self.value)

    def get_Flag(self):
        if self.Flag_Line_Edit.text() != "":
            self.Flag = self.Flag_Line_Edit.text()
        return float(self.Flag)

    def get_Ts(self):
        if self.Ts_Line_Edit.text() != "":
            self.Ts = self.Ts_Line_Edit.text()
        return float(self.Ts)

    def get_Te(self):
        if self.Te_Line_Edit.text() != "":
            self.Te = self.Te_Line_Edit.text()
        return float(self.Te)

    def Clear_Line_Edits(self):
        self.Value_Line_Edit.clear()
        self.Flag_Line_Edit.clear()
        self.Ts_Line_Edit.clear()
        self.Te_Line_Edit.clear()

    def Generate_Sequence(self):
        val = self.get_Value()
        # Flag = self.get_Flag()
        Ts = self.get_Ts()
        Te = self.get_Te()
        if self.Sequence_Combobox.currentIndex()==0:
            self.Draw_RF(val, Ts, Te)
        elif self.Sequence_Combobox.currentIndex()==1:
            self.Draw_Gradients(val, Ts, Te, self.Gz_line)
        elif self.Sequence_Combobox.currentIndex()==2:
            self.Draw_Gradients(val, Ts, Te, self.Gy_line)
        elif self.Sequence_Combobox.currentIndex()==3:
            self.Draw_Gradients(val, Ts, Te, self.Gx_line)
        else:
            self.Draw_Ro(val, Ts, Te)
        self.Clear_Line_Edits()

    def Draw_RF(self, val, Ts, Te):
        self.plot_Const_Lines()
        x1 = np.linspace(Ts, Te, 1000)
        y1 = self.Rf_line + (val * np.sinc(x1 - 10))
        self.sequenceCanvas.axes.plot(x1, y1, color='maroon', marker='o')
        self.sequenceCanvas.draw()
        data_1 = {"RF_Value": val, "RF_Ts": Ts, "RF_Te": Te},
        self.JSON_List.append(data_1)

    def Draw_Ro(self, val, Ts, Te):
        x1 = np.linspace(Ts, Te, 1000)
        y1 = self.Ro_line + (val * np.sinc(x1 - 55))
        self.sequenceCanvas.axes.plot(x1, y1, color='maroon', marker='o')
        self.sequenceCanvas.draw()
        data_1 = {'Value': val, 'Ts': Ts, 'Te': Te},
        self.JSON_List.append(data_1)

    def Draw_Gradients(self, val, Ts, Te, line):
        self.sequenceCanvas.axes.step(x=[Ts, Te, Te], y=[line, (line + 1) * val, line])
        self.sequenceCanvas.draw()
        data_1 = {'Value': val, 'Ts': Ts, 'Te': Te},
        self.JSON_List.append(data_1)

    def write_file(self):
        with open('Data_Json.json', 'w') as f:
            json.dump(self.JSON_List, f)

    def plot_Const_Lines(self):
        self.sequenceCanvas.axes.cla()
        # plotting constant lines for Rf,Gz,Gy,Gx,Ro
        [self.sequenceCanvas.axes.axhline(y=i, color='r', linestyle='-') for i in
         [self.Ro_line, self.Gx_line, self.Gy_line, self.Gz_line, self.Rf_line]]
        self.sequenceCanvas.axes.set_xlabel('t (msec)')
        self.sequenceCanvas.axes.set_yticklabels([0, 'Ro', 'Gx', 'Gy', 'Gz', 'Rf'])
        self.sequenceCanvas.draw()

    def DrawTR_TE(self):
        #if self.Flag_Line_Edit.text() != "" and self.Te_Line_Edit.text() != "":
        TR = self.get_Flag()
        TE = self.get_Te()
        for p,l in zip([TR, TE], ['TE', 'TR']):
            self.sequenceCanvas.axes.axvline(p, ls='--')
            self.sequenceCanvas.axes.annotate(l, xy= (p, 23)) 
        self.sequenceCanvas.draw()


         
    def Draw_Sequence(self, df):
        self.plot_Const_Lines()
        # plotting functions of Rf,Gz,Gy,Gx,Ro
        x1 = np.linspace(df["RF1_Ts"].values[0], df["RF1_Te"].values[0], 1000)
        y1 = self.Rf_line + ((df["RF1_value"].values[0]) * np.sinc(x1 - 10))

        x5 = np.linspace(df["Ro_Ts"].values[4], df["Ro_Te"].values[4], 1000)
        y5 = self.Ro_line + ((df["Ro_value"].values[4]) * np.sinc(x5 - 55))

        x6 = np.linspace(df["RF2_Ts"].values[5], df["RF2_Te"].values[5], 1000)
        y6 = self.Rf_line + ((df["RF2_value"].values[5]) * np.sinc(x1 - 10))

        self.sequenceCanvas.axes.plot(x1, y1, color='maroon', marker='o')
        self.sequenceCanvas.axes.step(x=[df["Gz_Ts"].values[1], df["Gz_Te"].values[1], df["Gz_Te"].values[1]],
                 y=[self.Gz_line, (self.Gz_line + 1) * df["Gz_value"].values[1], self.Gz_line])
        self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[2], df["Gy_Te"].values[2], df["Gy_Te"].values[2]],
                 y=[self.Gy_line, (self.Gy_line + 1) * df["Gy_value"].values[2], self.Gy_line])
        self.sequenceCanvas.axes.step(x=[df["Gx_Ts"].values[3], df["Gx_Te"].values[3], df["Gx_Te"].values[3]],
                 y=[self.Gx_line, (self.Gx_line + 1) * df["Gx_value"].values[3], self.Gx_line])
        self.sequenceCanvas.axes.plot(x5, y5, color='maroon', marker='o')
        self.sequenceCanvas.axes.plot(x6, y6, color='maroon', marker='o')


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

        # Plotting the TR and TE Lines
        #TR = df["Ro_Ts"].values[4]+(df["Ro_Te"].values[4]-df["Ro_Ts"].values[4])/2
        #TE = df["RF2_Ts"].values[5]+(df["RF2_Te"].values[5]-df["RF2_Ts"].values[5])/2
        #x_points= [df["RF1_Te"].values[0]/2, TR, TE]
        #label= ['', 'TE', 'TR']
        #for p,l in zip(x_points, label):
         #   self.sequenceCanvas.axes.axvline(p, ls='--')
          #  self.sequenceCanvas.axes.annotate(l, xy= (p, 23)) 
        
        self.sequenceCanvas.axes.set_xlabel('t (msec)')
        self.sequenceCanvas.axes.set_yticklabels([0,'Ro', 'Gx', 'Gy', 'Gz', 'Rf'])
        
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