from phantominator import shepp_logan
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore ,uic ,QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSizePolicy, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.uic import loadUiType
import numpy as np
import json
import pandas as pd
import matplotlib.patches as patches
from PIL import Image,ImageEnhance
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt




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
class phantomMplCanvas(MyMplCanvas , QtWidgets.QMainWindow):
    """Simple canvas with a sine plot."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_initial_figure(self ,contrastFactor=float(1), imageSizeIndex = 0 ,imageTypeIndex = 0 , clickedData = {"clicked":False , "X":0 , "Y":0}):
        #generate phantom of specific size
        imageSize = [16 , 32 , 64]
        phantomImg = shepp_logan(imageSize[imageSizeIndex])
        # MR phantom (returns proton density, T1, and T2 maps)
        PD, T1, T2 = shepp_logan((imageSize[imageSizeIndex], imageSize[imageSizeIndex], 20), MR=True)
        imageType = [phantomImg , T1[:,:,15] , T2[:,:,15] , PD[:,:,15]]
        # onclick adding a pixel rectangle around the pixel
        if clickedData["clicked"] == True:
            # Create a Rectangle patch
            x = clickedData["X"]
            y = clickedData["Y"]
            rect = patches.Rectangle((x,y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
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
        return (str(T1[int(clickedData["X"]) , int(clickedData["Y"]) , 15]) , 
                str(T2[int(clickedData["X"]) , int(clickedData["Y"]) , 15]),
                str(PD[int(clickedData["X"]) , int(clickedData["Y"]) , 15]))
        


# Create a class for your main window that inherits from Ui_MainWindow and QMainWindow
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # -------------link ui file------------------------------#
        uic.loadUi(r'UI/Task1.ui', self)
        
        #--------------Adding Canvas figures to layouts-----------#
        self.phantomLayout = self.verticalLayout_13 
        self.phantomCanvas = phantomMplCanvas(self.centralwidget, width=3, height=4, dpi=100)
        self.phantomLayout.addWidget(self.phantomCanvas)# phantom Canvas
        self.phantomCanvas.mpl_connect('button_press_event', self.phantom_onClick)

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

        # contrast Global variables
        self.contrastFactor = float(1)
        self.minContrast = 0.1
        self.maxContrast = 10

        # -----------------Connect buttons with functions--------------#
        self.phantomSize_comboBox.activated.connect(lambda:self.phantomImageDraw())
        self.imageTypeCombobox.activated.connect(lambda:self.phantomImageDraw())
        self.actionOpen.triggered.connect(lambda:self.read_file())
        

    # -----------------------functions defination-----------------------------------#
    def phantom_onClick(self , event ):
        print(event.button)
        if event.dblclick :
            T1 ,T2 , PD = self.phantomImageDraw(clicked={"clicked":True , "X":event.xdata , "Y":event.ydata})
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y , event.xdata, event.ydata))
            self.T1value_label.setText(T1)
            self.T2value_label.setText(T2)
            self.PDvalue_label.setText(PD)
            
        #controlling Contrast via leftmouse and Rightmouse
        elif str(event.button) == "MouseButton.RIGHT":
            print("Right Mouse clicked")
            self.contrastFactor = self.contrastFactor + 0.1
            self.phantom_contrast()

        elif str(event.button) == "MouseButton.LEFT":
            print("left Mouse clicked")
            self.contrastFactor = self.contrastFactor - 0.1
            self.phantom_contrast()


    def phantom_contrast(self ):
        if self.contrastFactor <= self.minContrast:
            self.contrastFactor = self.minContrast
        elif self.contrastFactor >= self.maxContrast:
            self.contrastFactor = self.maxContrast
        print("factor is :" , self.contrastFactor)
        self.phantomImageDraw()
        
    def phantomImageDraw(self , clicked = {"clicked":False , "X":0 , "Y":0}):
        #current indeces of the phantom size combobox and phantom image combobox
        self.imageSizeIndex = self.phantomSize_comboBox.currentIndex()
        self.imageTypeIndex = self.imageTypeCombobox.currentIndex()
        self.phantomLayout.removeWidget(self.phantomCanvas)# phantom Canvas
        self.phantomCanvas = phantomMplCanvas(self.centralwidget, width=3, height=4, dpi=100)
        T1 ,T2 , PD = self.phantomCanvas.compute_initial_figure(imageSizeIndex = self.imageSizeIndex ,imageTypeIndex = self.imageTypeIndex ,
                                                clickedData = clicked,  contrastFactor=self.contrastFactor)
        self.phantomLayout.addWidget(self.phantomCanvas)# phantom Canvas
        self.phantomCanvas.mpl_connect('button_press_event', self.phantom_onClick)
        return(T1 ,T2 , PD)

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


    # def plotting(self,GRAPHICSINDEX,X_ARRAY,Y_ARRAY,COLORLIST):
    #     self.GraphicsView[GRAPHICSINDEX].plot(X_ARRAY, Y_ARRAY, pen=COLORLIST)
    #     self.GraphicsView[GRAPHICSINDEX].plotItem.setLabel("bottom", text="Time (ms)")
    #     self.GraphicsView[GRAPHICSINDEX].plotItem.showGrid(True, True, alpha=1)
    #     self.GraphicsView[GRAPHICSINDEX].plotItem.setLimits(xMin=0, xMax=10, yMin=-20, yMax=20)


    # Reconstrucing the image

    #getting the value of the flip angle
    def get_Flip_angle(self):
        if self.FA_Line_Edit.text() != "":
            self.FA = self.FA_Line_Edit.text()
        else:
            self.FA = 90
        return self.FA

    # normalizing the image to put the minimum and maximum pixel values between 0 and 255
    def normalize_image(self,image):
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

    def equ_of_Rotation_z(self, theta):
        rotation_z = np.array(
            [[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
             [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0], [0, 0, 1]])
        return rotation_z

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
    def reconstruct_image(self):
        self.Reconstructedimage_graph.cla()
        self.Kspace_graph.cla()
        # choosing size of phantom
        if self.phantomSize_comboBox.currentIndex()==0:
            phantomImg = shepp_logan(16)
        elif self.phantomSize_comboBox.currentIndex()==1:
            phantomImg = shepp_logan(32)
        elif self.phantomSize_comboBox.currentIndex()==2:
            phantomImg = shepp_logan(64)
        else:
            phantomImg = shepp_logan(16)

        kSpace = np.zeros((phantomImg.shape[0], phantomImg.shape[1]), dtype=np.complex_)
        modified_img = self.modify_image(phantomImg)
        Phase_of_X = self.get_Flip_angle()
        for A in range(0, modified_img.shape[0]):
            rotated_matrix = self.Rotation_x(modified_img, Phase_of_X)
            for B in range(0, modified_img.shape[1]):
                step_of_Y = (360 / modified_img.shape[0]) * B
                step_of_X = (360 / modified_img.shape[1]) * A
                Final_matrix = np.zeros(modified_img.shape)
                #Applying rotation z in x&y plane
                for i in range(0, modified_img.shape[0]):
                    for j in range(0, modified_img.shape[1]):
                        phase = step_of_Y * j + step_of_X * i
                        Final_matrix[i, j] = np.dot(self.equ_of_Rotation_z(phase), rotated_matrix[i, j])
                #Getting the value of kspace
                gradient_image = Final_matrix
                sum_of_x = np.sum(gradient_image[:, :, 0])
                sum_of_y = np.sum(gradient_image[:, :, 1])
                complex_value = np.complex(sum_of_x, sum_of_y)
                kSpace[A, B] = complex_value

            Final_img = np.zeros((phantomImg.shape[0], phantomImg.shape[1], 3))
            Final_img[:, :, 2] = phantomImg
            self.Kspace_graph.axes.imshow(np.abs(kSpace), cmap='gray')
            self.Kspace_graph.draw(block=False)
            self.Kspace_graph.pause(0.5)
            self.Kspace_graph.close()
            print(A)

        Reconstructed_image = np.fft.fft2(kSpace)
        self.Reconstructedimage_graph.axes.imshow(np.abs(Reconstructed_image), cmap='gray')
        self.Reconstructedimage_graph.draw()
        self.Kspace_graph.axes.imshow(np.abs(kSpace), cmap='gray')
        self.Kspace_graph.draw()





if __name__ == '__main__':
    # Instantiate the main window class and show it
    app = QApplication([])
    window = MainWindow()
    window.show()
    # Run the application
    app.exec_()