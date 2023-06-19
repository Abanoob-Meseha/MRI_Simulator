from phantominator import shepp_logan
import matplotlib

matplotlib.use("Qt5Agg")
from PyQt5 import QtCore, uic, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSizePolicy, QFileDialog, QGraphicsScene, QGraphicsView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.uic import loadUiType
import numpy as np
import json
import pandas as pd
import matplotlib.patches as patches
from PIL import Image, ImageEnhance
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from threading import Thread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QMouseEvent
import math


# Main Figure Canvas class to use them in UI
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


# A Phantom figure canvas with ploting function
class phantomMplCanvas(MyMplCanvas, QtWidgets.QMainWindow):
    """Simple canvas with a sine plot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_initial_figure(self, contrastFactor=float(1), imageSizeIndex=0, imageTypeIndex=0,
                               clickedData={"clicked": False, "X": 0, "Y": 0}):
        # generate phantom of specific size
        imageSize = [16, 32, 64]
        phantomImg = shepp_logan(imageSize[imageSizeIndex])
        # MR phantom (returns proton density, T1, and T2 maps)
        PD, T1, T2 = shepp_logan((imageSize[imageSizeIndex], imageSize[imageSizeIndex], 20), MR=True)
        imageType = [phantomImg, T1[:, :, 15], T2[:, :, 15], PD[:, :, 15]]
        # onclick adding a pixel rectangle around the pixel
        if clickedData["clicked"] == True:
            # Create a Rectangle patch
            x = clickedData["X"]
            y = clickedData["Y"]
            rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            self.axes.add_patch(rect)
        # save the image to be easy to control contrast
        plt.imsave('images/tempPhantom.png', imageType[imageTypeIndex], cmap='gray')
        img = Image.open("images/tempPhantom.png")
        img_contr_obj = ImageEnhance.Contrast(img)
        factor = contrastFactor
        e_img = img_contr_obj.enhance(factor)
        arrayImg = np.array(e_img)
        self.axes.imshow(arrayImg, cmap='gray')
        return (str(T1[int(clickedData["X"]), int(clickedData["Y"]), 15]),
                str(T2[int(clickedData["X"]), int(clickedData["Y"]), 15]),
                str(PD[int(clickedData["X"]), int(clickedData["Y"]), 15]))


# -------------------------------------< MAINWINDOW Code >-----------------------------------------------------

# Create a class for your main window that inherits from Ui_MainWindow and QMainWindow
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # -------------link ui file------------------------------#
        uic.loadUi(r'UI/MRI_Simulator_2.ui', self)

        # --------------Adding Phantom figure to layouts-----------#
        self.phantomLayout = self.verticalLayout_26
        self.phantomCanvas = phantomMplCanvas(self.centralwidget, width=3, height=4, dpi=100)
        self.phantomLayout.addWidget(self.phantomCanvas)  # phantom Canvas
        self.phantomCanvas.mpl_connect('button_press_event', self.phantom_onClick)
        self.phantomCanvas.mpl_connect('button_release_event', self.phantom_contrast)

        # --------------Adding Sequence figure to layouts-----------#
        self.sequenceLayout = self.verticalLayout_25
        self.sequenceCanvas = MyMplCanvas(self.centralwidget, width=7, height=3, dpi=100)
        self.sequenceLayout.addWidget(self.sequenceCanvas)  # sequence Canvas

        # --------------Adding Reconstucted image figure to layouts-----------#
        self.Reconstructedimage_graph_layout_1 = self.verticalLayout_17
        self.Reconstructedimage_graph_1 = MyMplCanvas(self.centralwidget, width=3, height=3.5, dpi=100)
        self.Reconstructedimage_graph_layout_1.addWidget(self.Reconstructedimage_graph_1)

        # --------------Adding K-sapce figure to layouts-----------#
        self.KspaceLayout_1 = self.verticalLayout_18
        self.Kspace_graph_1 = MyMplCanvas(self.centralwidget, width=3, height=3.5, dpi=100)
        self.KspaceLayout_1.addWidget(self.Kspace_graph_1)

        # --------------Adding Reconstucted image figure to layouts-----------#
        self.Reconstructedimage_graph_layout_2 = self.verticalLayout_19
        self.Reconstructedimage_graph_2 = MyMplCanvas(self.centralwidget, width=3, height=3.5, dpi=100)
        self.Reconstructedimage_graph_layout_2.addWidget(self.Reconstructedimage_graph_2)

        # --------------Adding K-sapce figure to layouts-----------#
        self.KspaceLayout_2 = self.verticalLayout_20
        self.Kspace_graph_2 = MyMplCanvas(self.centralwidget, width=3, height=3.5, dpi=100)
        self.KspaceLayout_2.addWidget(self.Kspace_graph_2)

        # ---------------------Global variables----------------------#
        # sequence variables
        self.Rf_line = 20
        self.Gz_line = 15
        self.Gy_line = 10
        self.Gx_line = 5
        self.Ro_line = 0
        self.JSON_List = []

        # contrast variables
        self.contrastFactor = float(1)
        self.minContrast = 0.1
        self.maxContrast = 10
        self.clicked_point = None

        # -----------------Connect buttons with functions--------------#
        self.phantomSize_comboBox.activated.connect(lambda: self.phantomImageDraw())
        self.Prep_pulse_comboBox.activated.connect(lambda: self.choose_prep_pulse())
        self.Choose_display_Combobox.activated.connect(lambda: self.make_threading(self.choose_where_to_display))
        self.imageTypeCombobox.activated.connect(lambda: self.phantomImageDraw())
        self.actionOpen.triggered.connect(lambda: self.read_file())
        self.Send_Button.clicked.connect(lambda: self.calculate_ernst_angle())
        self.Sequence_Combobox.activated.connect(self.Generate_Sequence)
        self.Value_Line_Edit.textChanged.connect(lambda: self.get_Value())
        self.Ts_Line_Edit.textChanged.connect(lambda: self.get_Ts())
        self.Te_Line_Edit.textChanged.connect(lambda: self.get_Te())
        self.Export_Button.clicked.connect(self.write_file)
        self.TR_Line_Edit.textChanged.connect(lambda: self.get_ReptitionTime())
        self.TEcho_Line_Edit.textChanged.connect(lambda: self.get_EchoTime())
        self.Add_Button.clicked.connect(self.DrawTR_TE)
        self.FA_Line_Edit.textChanged.connect(lambda: self.get_Flip_angle())
        self.T1_Line_Edit.textChanged.connect(lambda: self.get_T1_ernst())
        self.TR_Line_Edit_2.textChanged.connect(lambda: self.get_Tr_ernst())

    # -----------------------functions defination-----------------------------------#
    def phantom_onClick(self, event: QMouseEvent):
        print(event.button)
        if event.dblclick:
            T1, T2, PD = self.phantomImageDraw(clicked={"clicked": True, "X": event.xdata, "Y": event.ydata})
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            self.T1value_label.setText(T1)
            self.T2value_label.setText(T2)
            self.PDvalue_label.setText(PD)

        if event.button == 1:  # Left mouse button
            self.clicked_point = (event.xdata, event.ydata)

    def phantom_contrast(self, event):
        if event.button == 1 and self.clicked_point is not None:  # Left mouse button and a point has been clicked
            released_point = (event.xdata, event.ydata)
            dy = released_point[1] - self.clicked_point[1]
            print(f"Released mouse button on point ({released_point[0]:.2f}, {released_point[1]:.2f}), dy = {dy:.2f}")
            self.clicked_point = None
            if (dy < 0):
                print("UP")
                self.contrastFactor -= dy / 10
            elif (dy > 0):
                print("DOWN")
                self.contrastFactor -= dy / 10

            if self.contrastFactor <= self.minContrast:
                self.contrastFactor = self.minContrast
            elif self.contrastFactor >= self.maxContrast:
                self.contrastFactor = self.maxContrast
            print("factor is :", self.contrastFactor)
            self.phantomImageDraw()
            print("done")

    def phantomImageDraw(self, clicked={"clicked": False, "X": 0, "Y": 0}):
        # current indeces of the phantom size combobox and phantom image combobox
        self.imageSizeIndex = self.phantomSize_comboBox.currentIndex()
        self.imageTypeIndex = self.imageTypeCombobox.currentIndex()
        self.phantomLayout.removeWidget(self.phantomCanvas)  # phantom Canvas
        self.phantomCanvas = phantomMplCanvas(self.centralwidget, width=3, height=4, dpi=100)
        T1, T2, PD = self.phantomCanvas.compute_initial_figure(imageSizeIndex=self.imageSizeIndex,
                                                               imageTypeIndex=self.imageTypeIndex,
                                                               clickedData=clicked, contrastFactor=self.contrastFactor)
        self.phantomLayout.addWidget(self.phantomCanvas)  # phantom Canvas
        self.phantomCanvas.mpl_connect('button_press_event', self.phantom_onClick)
        self.phantomCanvas.mpl_connect('button_release_event', self.phantom_contrast)
        return (T1, T2, PD)

    def read_file(self):  # BROWSE TO READ THE FILE
        self.File_Path = QFileDialog.getOpenFileName(self, "Open File", "This PC",
                                                     "All Files (*);;JSON Files(*.json)")

        with open(self.File_Path[0], 'r') as handle:
            json_data = [json.loads(line) for line in handle]
        self.df = pd.DataFrame(json_data)

        # calling the function that used in plotting mri sequence
        self.Draw_Sequence(self.df)

    def get_Value(self):
        if self.Value_Line_Edit.text() != "":
            self.value = self.Value_Line_Edit.text()
        return float(self.value)

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
        self.Ts_Line_Edit.clear()
        self.Te_Line_Edit.clear()

    def Generate_Sequence(self):
        val = self.get_Value()
        Ts = self.get_Ts()
        Te = self.get_Te()
        if self.Sequence_Combobox.currentIndex() == 0:
            self.Draw_RF(val, Ts, Te)
        elif self.Sequence_Combobox.currentIndex() == 1:
            self.Draw_Gradients(val, Ts, Te, self.Gz_line)
        elif self.Sequence_Combobox.currentIndex() == 2:
            self.Draw_Gradients(val, Ts, Te, self.Gy_line)
        elif self.Sequence_Combobox.currentIndex() == 3:
            self.Draw_Gradients(val, Ts, Te, self.Gx_line)
        else:
            self.Draw_Ro(val, Ts, Te)
        self.Clear_Line_Edits()

    def Draw_RF(self, val, Ts, Te):
        if (val == 90):
            Rf_amplitude = 3
        elif (val > 90):
            Rf_amplitude = 5
        elif (val < 90):
            Rf_amplitude = 1
        self.plot_Const_Lines()
        x1 = np.linspace(Ts, Te, 1000)
        y1 = self.Rf_line + (Rf_amplitude * np.sinc(x1 - 10))
        self.sequenceCanvas.axes.plot(x1, y1, color='maroon', marker='o')
        self.sequenceCanvas.draw()
        data_1 = {"Value": val, "Ts": Ts, "Te": Te},
        self.JSON_List.append(data_1)

    def Draw_Ro(self, val, Ts, Te):
        if (val == 90):
            Ro_amplitude = 3
        elif (val > 90):
            Ro_amplitude = 5
        elif (val < 90):
            Ro_amplitude = 1
        x1 = np.linspace(Ts, Te, 1000)
        y1 = self.Ro_line + (Ro_amplitude * np.sinc(x1 - 55))
        self.sequenceCanvas.axes.plot(x1, y1, color='maroon', marker='o')
        self.sequenceCanvas.draw()
        data_1 = {'Value': val, 'Ts': Ts, 'Te': Te},
        self.JSON_List.append(data_1)

    def Draw_Gradients(self, val, Ts, Te, line):
        if (val == 90):
            Gradient_amplitude = 1
        elif (val > 90):
            Gradient_amplitude = 1.06
        elif (val < 90):
            Gradient_amplitude = 0.98
        self.sequenceCanvas.axes.step(x=[Ts, Te, Te], y=[line, (line + 1) * Gradient_amplitude, line])
        self.sequenceCanvas.draw()
        data_1 = {'Value': val, 'Ts': Ts, 'Te': Te},
        self.JSON_List.append(data_1)

    def write_file(self):
        with open('Data_Json.json', 'w') as f:
            json.dump(self.JSON_List, f)

    def plot_Const_Lines(self):
        self.sequenceCanvas.axes.clear()
        self.sequenceCanvas.draw()
        [self.sequenceCanvas.axes.axhline(y=i, color='r', linestyle='-') for i in
         [self.Ro_line, self.Gx_line, self.Gy_line, self.Gz_line, self.Rf_line]]
        self.sequenceCanvas.axes.set_xlabel('t (msec)')
        self.sequenceCanvas.axes.set_yticklabels([0, 'Ro', 'Gx', 'Gy', 'Gz', 'Rf'])
        self.sequenceCanvas.draw()

    def get_ReptitionTime(self):
        if self.TR_Line_Edit.text() != "":
            self.TR = self.TR_Line_Edit.text()
        return float(self.TR)

    def get_EchoTime(self):
        if self.TEcho_Line_Edit.text() != "":
            self.TEcho = self.TEcho_Line_Edit.text()
        return float(self.TEcho)

    def DrawTR_TE(self):
        for line in self.sequenceCanvas.axes.lines:
            if line.get_color() == 'green':
                line.remove()
        for label in self.sequenceCanvas.axes.texts:
            if label.get_color() == 'green':
                label.remove()
        TR = self.get_ReptitionTime()
        TE = self.get_EchoTime()
        for p, l in zip([TR, TE], ['TR', 'TE']):
            self.sequenceCanvas.axes.axvline(p, color='green', ls='--')
            self.sequenceCanvas.axes.text(p, 23, l, color='green')
        self.sequenceCanvas.draw()

    def T1_prep(self):
        x_rf = np.linspace(-30, -10, 1000)
        y_rf = self.Rf_line + ((5) * np.sinc(x_rf + 20))
        self.sequenceCanvas.axes.step(x=[-30, -10, -10], y=[self.Gz_line, (self.Gz_line + 1) * 1.06, self.Gz_line])
        self.sequenceCanvas.axes.axvline(x=-5, ls='--')
        self.sequenceCanvas.axes.plot(x_rf, y_rf, color='maroon', marker='o')
        self.sequenceCanvas.draw()

    def T2_prep(self):
        x_rf1 = np.linspace(-60, -40, 1000)
        y_rf1 = self.Rf_line + ((3) * np.sinc(x_rf1 + 50))
        x_rf2 = np.linspace(-30, -10, 1000)
        y_rf2 = self.Rf_line + ((3) * np.sinc(x_rf2 + 20))
        self.sequenceCanvas.axes.axvline(x=-5, ls='--')
        self.sequenceCanvas.axes.plot(x_rf1, y_rf1, color='maroon', marker='o')
        self.sequenceCanvas.axes.plot(x_rf2, y_rf2, color='maroon', marker='o')
        self.sequenceCanvas.draw()

    def Tagging_prep(self):
        x_rf1 = np.linspace(-60, -40, 1000)
        y_rf1 = self.Rf_line + ((3) * np.sinc(x_rf1 + 50))
        self.sequenceCanvas.axes.step(x=[-40, -30, -30], y=[self.Gx_line, (self.Gx_line + 1) * 1.2, self.Gx_line])
        x_rf2 = np.linspace(-30, -10, 1000)
        y_rf2 = self.Rf_line + ((3) * np.sinc(x_rf2 + 20))
        self.sequenceCanvas.axes.axvline(x=-5, ls='--')
        self.sequenceCanvas.axes.plot(x_rf1, y_rf1, color='maroon', marker='o')
        self.sequenceCanvas.axes.plot(x_rf2, y_rf2, color='maroon', marker='o')
        self.sequenceCanvas.draw()

    # choose prep pulse function
    def choose_prep_pulse(self):
        if self.Prep_pulse_comboBox.currentIndex() == 0:
            self.plot_Const_Lines()
        elif self.Prep_pulse_comboBox.currentIndex() == 1:
            self.plot_Const_Lines()
            self.T1_prep()
        elif self.Prep_pulse_comboBox.currentIndex() == 2:
            self.plot_Const_Lines()
            self.T2_prep()
        elif self.Prep_pulse_comboBox.currentIndex() == 3:
            self.plot_Const_Lines()
            self.Tagging_prep()

    def choose_where_to_display(self):
        if self.Choose_display_Combobox.currentIndex() == 0:
            self.Kspace_graph_1.axes.clear()
            self.Kspace_graph_1.draw()
            self.Reconstructedimage_graph_1.axes.clear()
            self.Reconstructedimage_graph_1.draw()
            self.Kspace_graph_2.axes.clear()
            self.Kspace_graph_2.draw()
            self.Reconstructedimage_graph_2.axes.clear()
            self.Reconstructedimage_graph_2.draw()
        elif self.Choose_display_Combobox.currentIndex() == 1:
            self.reconstruct_image(self.Kspace_graph_1, self.Reconstructedimage_graph_1)
        elif self.Choose_display_Combobox.currentIndex() == 2:
            self.reconstruct_image(self.Kspace_graph_2, self.Reconstructedimage_graph_2)

    def get_T1_ernst(self):
        if self.T1_Line_Edit.text() != "":
            self.T1_ernst = self.T1_Line_Edit.text()
        return float(self.T1_ernst)

    def get_Tr_ernst(self):
        if self.TR_Line_Edit_2.text() != "":
            self.Tr_ernst = self.TR_Line_Edit_2.text()
        return float(self.Tr_ernst)

    def calculate_ernst_angle(self):
        T1_ernst = self.get_T1_ernst()
        TR_ernst = self.get_Tr_ernst()
        Ernst_angle = round(math.degrees(math.acos(math.exp(-TR_ernst / T1_ernst))), 3)
        self.Ernst_Angle_label.setText("Ernst Angle: " + str(Ernst_angle))

    def Draw_Sequence(self, df):
        self.choose_prep_pulse()
        # plotting functions of Rf,Gz,Gy,Gx,Ro
        x_rf = np.linspace(df["RF1_Ts"].values[0], df["RF1_Te"].values[0], 1000)
        y_rf = self.Rf_line + ((df["RF1_value"].values[0]) * np.sinc(x_rf - 10))

        x_ro = np.linspace(df["Ro_Ts"].values[4], df["Ro_Te"].values[4], 1000)
        y_ro = self.Ro_line + ((df["Ro_value"].values[4]) * np.sinc(x_ro - 55))

        x_rf2 = np.linspace(df["RF2_Ts"].values[5], df["RF2_Te"].values[5], 1000)
        y_rf2 = self.Rf_line + ((df["RF2_value"].values[5]) * np.sinc(x_rf2 - 100))

        self.sequenceCanvas.axes.plot(x_rf, y_rf, color='maroon', marker='o')
        self.sequenceCanvas.axes.step(x=[df["Gz_Ts"].values[1], df["Gz_Te"].values[1], df["Gz_Te"].values[1]],
                                      y=[self.Gz_line, (self.Gz_line + 1) * df["Gz_value"].values[1], self.Gz_line])
        self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[2], df["Gy_Te"].values[2], df["Gy_Te"].values[2]],
                                      y=[self.Gy_line, (self.Gy_line + 1) * df["Gy_value"].values[2], self.Gy_line])
        self.sequenceCanvas.axes.step(x=[df["Gx_Ts"].values[3], df["Gx_Te"].values[3], df["Gx_Te"].values[3]],
                                      y=[self.Gx_line, (self.Gx_line + 1) * df["Gx_value"].values[3], self.Gx_line])
        self.sequenceCanvas.axes.step(
            x=[df["dephase_Ts"].values[6], df["dephase_Te"].values[6], df["dephase_Te"].values[6]],
            y=[self.Gz_line, ((self.Gz_line + 1) * df["dephase_value"].values[6] * -1) + (self.Gz_line + 1) + 14,
               self.Gz_line])
        self.sequenceCanvas.axes.step(
            x=[df["Noise_eff_Ts"].values[7], df["Noise_eff_Te"].values[7], df["Noise_eff_Te"].values[7]],
            y=[self.Gx_line, ((self.Gx_line + 1) * df["Noise_eff_val"].values[7] * -1) + (self.Gx_line + 1) + 4,
               self.Gx_line])

        self.ploting_Repitition_And_Reversion(df, 2)

        if (df["SE_180_pulse"].values[8] == "True"):
            x_rephase = np.linspace(df["Rephase_Ts"].values[8], df["Rephase_Te"].values[8], 1000)
            y_rephase = self.Rf_line + ((df["Rephase_val"].values[8]) * np.sinc(x_rephase - 55))
            self.sequenceCanvas.axes.plot(x_rephase, y_rephase, color='maroon', marker='o')
            self.sequenceCanvas.axes.step(
                x=[df["Gz_rephase_Ts"].values[9], df["Gz_rephase_Te"].values[9], df["Gz_rephase_Te"].values[9]],
                y=[self.Gz_line, ((self.Gz_line + 1) * df["Gz_rephase_val"].values[9]), self.Gz_line])
            y_ro = self.Ro_line + ((df["Ro_value"].values[4]) * np.sinc(x_ro - 80))

        if (df["ssfp_Noise_eff"].values[8] == "True"):
            self.sequenceCanvas.axes.step(
                x=[df["Noise_eff_2_Ts"].values[8], df["Noise_eff_2_Te"].values[8], df["Noise_eff_2_Te"].values[8]],
                y=[self.Gx_line, ((self.Gx_line + 1) * df["Noise_eff_val_2"].values[8] * -1) + (self.Gx_line + 1) + 4,
                   self.Gx_line])
            self.sequenceCanvas.axes.step(
                x=[df["Gz_Reversed_Ts"].values[10], df["Gz_Reversed_Te"].values[10], df["Gz_Reversed_Te"].values[10]],
                y=[self.Gz_line, ((self.Gz_line + 1) * df["Gz_Reversed_val"].values[10] * -1) + (self.Gz_line + 1) + 14,
                   self.Gz_line])
            self.ploting_Repitition_And_Reversion(df, 9)

        self.sequenceCanvas.axes.plot(x_ro, y_ro, color='maroon', marker='o')
        self.sequenceCanvas.axes.plot(x_rf2, y_rf2, color='maroon', marker='o')
        self.sequenceCanvas.axes.axvline(x=(df["RF1_Te"].values[0] + df["RF1_Ts"].values[0]) / 2, ls='--')

        self.sequenceCanvas.axes.set_xlabel('t (msec)')
        self.sequenceCanvas.axes.set_yticklabels([0, 'Ro', 'Gx', 'Gy', 'Gz', 'Rf'])
        self.sequenceCanvas.draw()

    def ploting_Repitition_And_Reversion(self, df, i):
        # Plotting repeat of Gy_Reversed if it exists
        if (df["Gy_repeated"].values[i] == "True"):
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[i], df["Gy_Te"].values[i], df["Gy_Te"].values[i]],
                                          y=[(self.Gy_line + 1), (self.Gy_line + 2) * df["Gy_value"].values[i],
                                             (self.Gy_line + 1)])
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[i], df["Gy_Te"].values[i], df["Gy_Te"].values[i]],
                                          y=[(self.Gy_line + 2), (self.Gy_line + 3) * df["Gy_value"].values[i],
                                             (self.Gy_line + 2)])

        # Plotting reverse of Gy if it exists
        if (df["Gy_reversed"].values[i] == "True"):
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[i], df["Gy_Te"].values[i], df["Gy_Te"].values[i]],
                                          y=[self.Gy_line, ((self.Gy_line + 1) * df["Gy_value"].values[i] * -1) + (
                                                      self.Gy_line + 1) + 9, self.Gy_line])
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[i], df["Gy_Te"].values[i], df["Gy_Te"].values[i]],
                                          y=[(self.Gy_line + 1),
                                             ((self.Gy_line + 2) * df["Gy_value"].values[i] * -1) + (
                                                         self.Gy_line + 1) + 9,
                                             (self.Gy_line + 1)])
            self.sequenceCanvas.axes.step(x=[df["Gy_Ts"].values[i], df["Gy_Te"].values[i], df["Gy_Te"].values[i]],
                                          y=[(self.Gy_line + 2),
                                             ((self.Gy_line + 3) * df["Gy_value"].values[i] * -1) + (
                                                     self.Gy_line + 1) + 9,
                                             (self.Gy_line + 2)])

    # Reconstrucing the image

    # getting the value of the flip angle
    def get_Flip_angle(self):
        if self.FA_Line_Edit.text() != "":
            self.FA = self.FA_Line_Edit.text()
        else:
            self.FA = 90
        return float(self.FA)

    # normalizing the image to put the minimum and maximum pixel values between 0 and 255
    def normalize_image(self, image):
        # Find the minimum and maximum pixel values
        min_val = np.min(image)
        max_val = np.max(image)

        # Normalize the image using the formula (image - min) / (max - min)
        normalized_image = (image - min_val) / (max_val - min_val)

        return normalized_image

    # moddifying the image to reconstruct it
    def modify_image(self, Phantom_img):
        normalized_img = self.normalize_image(Phantom_img)
        final_image = np.zeros((Phantom_img.shape[0], Phantom_img.shape[1], 3))
        final_image[:, :, 2] = normalized_img
        return final_image

    # matrix of rotation z
    def equ_of_Rotation_z(self, theta):
        rotation_z = np.array(
            [[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
             [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0], [0, 0, 1]])
        return rotation_z

    # matrix of rotation x
    def equ_of_Rotation_x(self, theta):
        rotation_x = np.array([[1, 0, 0], [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                               [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
        return rotation_x

    # applying rotation x to the modified image with the flip angle we want
    def Rotation_x(self, Image, phase_X):
        rotated_image = np.zeros(Image.shape)
        for i in range(0, Image.shape[0]):
            for j in range(0, Image.shape[1]):
                rotated_image[i, j] = np.dot(self.equ_of_Rotation_x(phase_X), Image[i, j])

        return rotated_image

    # Reconstrucing the image and generating kspace
    # you will find the number of the step in the function
    # step 1 : select the phantom size and modify image and get the flip angle
    # step 2 : start looping over rows and columns of the phantom image and calling the function of rotation x that means we hit rf signal with our flip angle
    # step 3 : we made the phase to make the rotation z (gradient x and y) with it and apply gradient on rows and columns of the image
    # step 4 : we get gradient image, make sumation of values x and y and make the complex value (kspace)
    # step 5 : we plot kspace and image after reconstruction
    def reconstruct_image(self, Kspace_graph, Reconstructedimage_graph):
        # step 1
        Kspace_graph.axes.clear()
        Kspace_graph.draw()
        Reconstructedimage_graph.axes.clear()
        Reconstructedimage_graph.draw()
        # choosing size of phantom
        if self.phantomSize_comboBox.currentIndex() == 0:
            phantomImg = shepp_logan(16)
        elif self.phantomSize_comboBox.currentIndex() == 1:
            phantomImg = shepp_logan(32)
        elif self.phantomSize_comboBox.currentIndex() == 2:
            phantomImg = shepp_logan(64)
        else:
            phantomImg = shepp_logan(16)

        kSpace = np.zeros((phantomImg.shape[0], phantomImg.shape[1]), dtype=np.complex_)
        modified_img = self.modify_image(phantomImg)
        Phase_of_X = self.get_Flip_angle()
        # step 2
        for R in range(0, modified_img.shape[0]):
            rotated_matrix = self.Rotation_x(modified_img, Phase_of_X)
            for C in range(0, modified_img.shape[1]):
                # step 3
                step_of_Y = (360 / modified_img.shape[0]) * C
                step_of_X = (360 / modified_img.shape[1]) * R
                Final_matrix = np.zeros(modified_img.shape)
                # Applying rotation z in x&y plane
                for i in range(0, modified_img.shape[0]):
                    for j in range(0, modified_img.shape[1]):
                        phase = step_of_Y * j + step_of_X * i
                        Final_matrix[i, j] = np.dot(self.equ_of_Rotation_z(phase), rotated_matrix[i, j])
                # step 4
                # Getting the value of kspace
                gradient_image = Final_matrix
                sum_of_x = np.sum(gradient_image[:, :, 0])
                sum_of_y = np.sum(gradient_image[:, :, 1])
                complex_value = np.complex(sum_of_x, sum_of_y)
                kSpace[R, C] = complex_value

            Final_img = np.zeros((phantomImg.shape[0], phantomImg.shape[1], 3))
            Final_img[:, :, 2] = phantomImg
            # step 5
            Kspace_shifted = np.fft.fftshift(kSpace)
            Kspace_graph.axes.imshow(np.abs(Kspace_shifted), cmap='gray')
            Kspace_graph.draw()
            Kspace_graph.start_event_loop(0.0005)
            Reconstructed_image = np.fft.fft2(kSpace)
            Reconstructedimage_graph.axes.imshow(np.abs(Reconstructed_image), cmap='gray')
            Reconstructedimage_graph.draw()
            Reconstructedimage_graph.start_event_loop(0.0005)
            print(R)

    def make_threading(self, any_function):
        # create a thread
        thread = Thread(target=any_function)
        # run the thread
        thread.start()


if __name__ == '__main__':
    # Instantiate the main window class and show it
    app = QApplication([])
    window = MainWindow()
    window.show()
    # Run the application
    app.exec_()