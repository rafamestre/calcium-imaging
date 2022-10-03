# -*- coding: utf-8 -*-
"""
CIA: Calcium Imaging Analyser

v1.0

22/01/2022    

@author: Rafael Mestre; r.mestre@soton.ac.uk; rafa.mestrec@gmail.com

"""

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'auto')
import sys
# import traceback
from PyQt5.QtWidgets import QWidget, QRubberBand, QSlider, QLabel, QApplication, QDesktopWidget
from PyQt5.QtWidgets import QVBoxLayout, QTabWidget, QSizePolicy, QPushButton, QLineEdit, QGridLayout
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage
# from PyQt5.QtGui import qRed, qGreen, qBlue
# from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, QRect, Qt
import cv2
import os
import numpy as np
# import struct
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')  
from scipy import signal
# import pickle
import matplotlib.ticker as plticker
import seaborn as sns
from pathlib import Path


#######PLOTTING ARGUMENTS
labelFont = {'fontname':'sans-serif', 'fontsize':34}
ticksFont= {'fontsize':28}
pltArgs = {'linewidth' : 3}
figsize = (12,9)
sns.set_context("talk", font_scale=2, rc={"lines.linewidth": 2})
sns.set_style("ticks")
sns.set_palette(sns.color_palette("colorblind", 10))
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.width'] = 3


######GLOBAL VARIABLES

videoNameFull = None
currentFrame = None
dirVideos = None
videoFrames = None
length = None
width = None
height = None
fps = None
cropped = None


class cropPopUp(QWidget):
    
    #Opens a window for cropping
    
    def __init__(self):
        QWidget.__init__(self)
        self.paintEvent()
        
    def paintEvent(self):
        #Puts the picture in the window
        self.picture = QLabel(self)
        self.picture.setPixmap(currentFrame)

    def mousePressEvent (self, eventQMouseEvent):
        #When the mouse is clicked and dragged, it creates a rectangle
        self.originQPoint = eventQMouseEvent.pos()
        self.currentQRubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.currentQRubberBand.setGeometry(QRect(self.originQPoint, QSize()))
        self.currentQRubberBand.show()

    def mouseMoveEvent (self, eventQMouseEvent):
        self.currentQRubberBand.setGeometry(QRect(self.originQPoint, eventQMouseEvent.pos()).normalized())

    def mouseReleaseEvent (self, eventQMouseEvent):
        global cropped
        self.currentQRubberBand.hide()
        currentQRect = self.currentQRubberBand.geometry()
        cropped = currentQRect
        self.currentQRubberBand.deleteLater()
        self.close()
        #The image in the rectangle is taken to crop
        
    def closeEvent(self, event):
        self.ClearInputs()
        self.emit(SIGNAL("closed()"))
        
        
 

class mainWindow(QWidget):
    
    
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
       
    def initUI(self):
        
        global currentFrame
        global dirVideos
        # dirVideos = None
        
        self.setGeometry(100, 100, 1000, 600)
        self.setFixedSize(1000,600)
        self.setWindowTitle('Calcium imaging analyser v1.0')
        #self.setWindowIcon(QIcon('web.png'))        
        self.center()

        self.layout = QVBoxLayout(self)
 
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()	
        self.tab2 = QWidget()
        #self.tabs.resize(300,200) 
 
        # Add tabs
        self.tabs.addTab(self.tab1,"Viewer")
        self.tabs.addTab(self.tab2, "Analyser")
 
        # Create first tab
#        self.tab1.layout = QVBoxLayout(self)
#        self.pushButton1 = QPushButton("PyQt5 button")
#        self.tab1.layout.addWidget(self.pushButton1)
#        self.tab1.setLayout(self.tab1.layout)
# 
#        # Add tabs to widget        
#        self.layout.addWidget(self.tabs)
#        self.setLayout(self.layout)



        
        self.main_frame = QWidget()
        
        # Creates the canvas with the PlotCanvas class
        # They are the left and right plots, respectively
        self.canvas = PlotCanvas(self)
        self.canvas.setParent(self.main_frame)  
        self.canvas2 = PlotCanvas(self)
        self.canvas2.setParent(self.main_frame)  
    
        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(self.close)
        qbtn.resize(qbtn.sizeHint())
        #qbtn.move(500, 500)  
        
        # Button to crop the video
        popUpBtn = QPushButton('Crop',self)
        popUpBtn.clicked.connect(self.cropVideo)
        #popUpBtn.move(600,600)
        
        analyzeBtn = QPushButton('Analyze',self)
        analyzeBtn.clicked.connect(self.analyze)
        #cropBtn.move(700,600)
        
        # Button to select the video file
        selectBtn = QPushButton('Select File',self)
        selectBtn.clicked.connect(self.selectFile)
        
        self.rangeEdit = QLineEdit(self)    
        self.rangeEdit2  = QLineEdit(self)
        
        # Button to change the range from both canvases
        rangeBtn = QPushButton('Change range',self)
        rangeBtn.clicked.connect(self.change_range_both_canvases)

        # Button to clear the data from the plots, resets them
        clearBtn = QPushButton('Clear data',self)
        clearBtn.clicked.connect(self.clear_both_canvases)
        
        # Button to reset the limits of the plots
        resetLimitsBtn = QPushButton('Reset limits',self)
        resetLimitsBtn.clicked.connect(self.reset_limits_both_canvases)
        
        # Button to save the plot 1
        saveBtn1 = QPushButton('Save plot 1',self)
        saveBtn1.clicked.connect(self.selectDirectory1)
        
        # Button to save the plot 2
        saveBtn2 = QPushButton('Save plot 2',self)
        saveBtn2.clicked.connect(self.selectDirectory2)
        
        openBtn1 = QPushButton('Open figure 1',self)
        openBtn1.clicked.connect(self.canvas.open_figure)
        
        openBtn2 = QPushButton('Open figure 2',self)
        openBtn2.clicked.connect(self.canvas2.open_figure)
        
#        self.titleEdit = QLineEdit(self)    
#        self.xLabelEdit  = QLineEdit(self)     
#        self.yLabelEdit  = QLineEdit(self)     
#        self.fontEdit  = QLineEdit(self)     
       
        ## TODO: USE PATH LIBRARY FOR THIS, USE CURRENT DIRECTORY
        # directory = ('C:\\Users\\rmestre\\OneDrive - IBEC\\PhD\\'+ \
        #                       'C2C12\\032417 Day 7 calcium\\' + \
        #                       'Training3_signals.png')

        #self.getVideo()
        
        self.vid = QLabel(self)
        self.vid.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        # Time slider that updates the current frame to be shown
        self.videoSl = QSlider(Qt.Horizontal)
#        self.videoSl.setMinimum(0)
#        self.videoSl.setMaximum(length)
#        self.videoSl.setValue(0)
        self.videoSl.setTickInterval(1)
        self.videoSl.setTickPosition(QSlider.TicksBelow)
        # When the value is changed, the frame is updated
        self.videoSl.valueChanged.connect(self.convertVideoAndShow)
        
#        self.convertVideoAndShow() 
#        self.cropFrame.setPixmap(currentFrame)


        #### Grid: positining of the buttons and other stuff
        # Tab 1
        self.tab1.layout = QGridLayout()
        #self.tab1.layout.setColumnMinimumWidth(1,1)
        #self.layout.setSpacing(1)
        self.tab1.layout.addWidget(self.vid,1,0,1,2)
        self.tab1.layout.addWidget(self.videoSl,2,0,1,2)
        self.tab1.layout.addWidget(popUpBtn,3,0)
        self.tab1.layout.addWidget(qbtn,3,1)
        self.tab1.layout.addWidget(analyzeBtn,4,1)  
        self.tab1.layout.addWidget(selectBtn,4,0)

#        self.layout.addWidget(self.cropFrame)
        # Layout is added
        self.tab1.setLayout(self.tab1.layout)
        
        # Tab 2
        self.tab2.layout = QGridLayout()
        self.tab2.layout.addWidget(self.canvas,1,0,1,2)
        self.tab2.layout.addWidget(self.canvas2,1,2,1,2)   
        self.tab2.layout.addWidget(self.rangeEdit,2,0)
        self.tab2.layout.addWidget(self.rangeEdit2,3,0)
        self.tab2.layout.addWidget(rangeBtn,2,1)
        self.tab2.layout.addWidget(resetLimitsBtn,3,1)
        self.tab2.layout.addWidget(saveBtn1,6,0)
        self.tab2.layout.addWidget(saveBtn2,6,1)
        self.tab2.layout.addWidget(openBtn1,7, 0)
        self.tab2.layout.addWidget(openBtn2,7, 1)
        self.tab2.layout.addWidget(clearBtn,5,0,1,2)
        
        #self.tab2.layout.setColumnStretch(0,0)
        self.rangeEdit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.rangeEdit2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        #self.layout.addWidget(self.toolbar)
        
        # Layout is added
        self.tab2.setLayout(self.tab2.layout)
        
        # Final layout is set
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # Show everything
        self.show()


    
    def clear_both_canvases(self):
        # Clears the data from both canvases (right and left)
        self.canvas.clear_data()
        self.canvas2.clear_data()
        
    def reset_limits_both_canvases(self):
        # Resets the limits of both canvases (right and left)
        self.canvas.reset_limits()
        self.canvas2.reset_limits()
        
        
    def change_range_both_canvases(self):
        # Changes the range of the plots in both canvases (right and left)
        self.canvas.change_range()
        self.canvas2.change_range()
        

    def QImageToCvMat(self,incomingImage):
        # Converts a QImage into an opencv MAT format
        incomingImage = incomingImage.convertToFormat(QImage.Format_RGB32)
        
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.constBits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr
   
    def analyzeAllVideo(self):
        # Analyse the whole video
        
        
        
        # global length
        # global videoFrames
        global fps
        
        # sumaTotal stores the total intensity in the cropped image
        # over time (length)
        sumaTotal = list()
        for x in range(0,length):
            # Loop through the length of the video
            frame = videoFrames[x]
   
            image = QImage(frame, frame.shape[1],\
                            frame.shape[0], frame.shape[1] * 3,QImage.Format_RGB888)
            # Image is cropped
            QPixmapCrop = QPixmap(image).copy(cropped)
            image = QPixmapCrop.toImage()
            width = image.width()
            height = image.height()      
            
            # Image is converted to opencv MAT format to sum the intensity
            arr = self.QImageToCvMat(image)
            intensity_image = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            suma = 0
            for y in range(0,height):
                for x in range(0,width):
                    suma += intensity_image[y,x,2]
            sumaTotal.append(suma)
        
        # An offset in the y direction for the plots is calculated
        # and applied. This is an offset that's forcefully applied to move
        # the plots a bit in the y-axis to avoid that they crash with each
        # other. This is not problematic because the y-axis has arbitrary units
        maxSumaLog = np.log10(np.max(sumaTotal))
        offset = 0
        if self.canvas.nbPlots > 0:
            offset = 10*(self.canvas.nbPlots) + np.max(sumaTotal)/(10**(maxSumaLog-1))
        sumaTotal = [offset+sumaTotal[i]/(10**(maxSumaLog-1)) for i in range(len(sumaTotal))]
        
        # TODO: figure out this
        exception = True
        
        if exception:
            
            fps = 7.6416
            fps = 50
            fps = 12.5
        
        time = np.linspace(0,len(sumaTotal)/fps,len(sumaTotal))
#        # discards the old graph
#        self.figure.clear()
#        # create an axis
#        ax = self.figure.add_subplot(111)
#        fps = 12.5
#        ax.plot(time,np.asarray(sumaTotal))
#        # refresh canvas
#        self.canvas.draw()
#        print(fps)
        
        # If this is the first plot, create the axes
        if self.canvas.nbPlots == 0:
            self.canvas.create_axes()
            self.canvas2.create_axes()
            
        # Detrend the signal for canvas 2
        # and apply the offset for this plot
        detrended = signal.detrend(np.asarray(sumaTotal))
        maxDetrendedLog = np.log10(np.max(detrended))
        if self.canvas.nbPlots > 0:
            offset = 10*(self.canvas.nbPlots) + np.max(detrended)/(10**(maxDetrendedLog-1))
        detrended = [offset+(detrended[i] - np.min(detrended))/(10**(maxDetrendedLog-1)) for i in range(len(detrended))]
        self.canvas2.plot_data(time,detrended)
        self.canvas.plot_data(time,
                              np.asarray(sumaTotal))
        self.tabs.setCurrentIndex(1)



    def analyze(self):
        # When analyze button is pressed, the whole video is analysed
        # TODO: add an exception if the crop has not been selected
        
        # global cropped
        global width
        global height
        #TODO: check if this gives errors

        a = currentFrame.copy(cropped)

#        self.cropFrame.setPixmap(a)
        b = a.toImage()
        width = b.width()
        height = b.height()
        
        self.analyzeAllVideo()
   
    

    def cropVideo(self):
        # Crops the video
        
        print("Opening a new popup window...")
        self.w = cropPopUp()
        if width > 700 or height > 700:
            self.w.setGeometry(50, 50, 700, 700)
        else:
            self.w.setGeometry(100, 100, width, height)
        
        self.w.show()




    def getVideo(self):
        # Loads the video after selecting a file
        
        global videoNameFull
        global currentFrame
        global dirVideos
        global dirVideos
        global videoFrames
        videoFrames = list()
        global length
        global width
        global height
        global fps
        
#        dirVideos = 'C:\\Users\\rmestre\\OneDrive - IBEC\\PhD\\C2C12\\290517 Stimulation super good\\'
        print('here')
        videoName = videoNameFull.split("\\\\")[-1]
        dirVideos = videoNameFull[:(len(videoNameFull)-len(videoName))]
        videoName = videoName.split(".avi")[0]
        #TODO: CHANGE THIS WITH PATH
        
        os.chdir(dirVideos)
        
        cap = cv2.VideoCapture(videoNameFull)
        print(videoName)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(fps)

        aux = 0
        
        while (cap.isOpened()):
            
            ret,frame = cap.read()
            videoFrames.append(frame)
            aux += 1
            
            if aux == length:
                break
        
        cap.release()
        print('done')
        
        self.videoSl.setMinimum(0)
        self.videoSl.setMaximum(length)
        self.videoSl.setValue(0)
        
        frameNb = self.videoSl.value()
        frame = videoFrames[frameNb]
        
        image = QImage(frame, frame.shape[1],\
                            frame.shape[0], frame.shape[1] * 3,QImage.Format_RGB888)
        currentFrame = QPixmap(image)
        currentFrame2 = currentFrame.scaledToWidth(360)
        self.vid.setPixmap(currentFrame2)   
                
    def selectFile(self):
        # Selects the video file
        
        #TODO: do this with Path
        global videoNameFull
        print('Selecting file...')
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
                            'c:\\',"Image files (*.avi)")
        print(type(fname))
        videoNameFull = str(fname[0])
        print(type(videoNameFull))
        videoNameFull = videoNameFull.replace("/","\\\\")
        print(videoNameFull)
        self.getVideo()   
        
    def selectDirectory1(self):
        # Saves the plot number 1
        
        #TODO: change this with Path
        if dirVideos is None:
            startingDir = 'C:\\'
        else:
            startingDir = dirVideos
                
        dname = QFileDialog.getSaveFileName(self, 'Save', 
                            startingDir,filter='(*.png)')
        dname = dname[0].replace("/","\\\\")
        if self.canvas.nbPlots > 0:
            self.canvas.save_canvas(dname)
        else:
            print('Error: Figure 1 cannot be saved.')
        
        
        
    def selectDirectory2(self):
        # Saves the plot number 2
        
        #TODO: CHANGE THIS WITH PATH
        if dirVideos is None:
            startingDir = 'C:\\'
        else:
            startingDir = dirVideos
        
        dname = QFileDialog.getSaveFileName(self, 'Save', 
                            startingDir,'*.png, *.avi')
        dname = dname[0].replace("/","\\\\")
        if self.canvas2.nbPlots > 0:
            self.canvas2.save_canvas(dname)
        else: 
            print('Error: Figure 2 cannot be saved.')
        
        
    def convertVideoAndShow(self):
        # When the time slider is modified, the current frame
        # that is shown is updated
        
        global currentFrame
        frameNb = self.videoSl.value()
        frame = videoFrames[frameNb]
        
        image = QImage(frame, frame.shape[1],\
                            frame.shape[0], frame.shape[1] * 3,QImage.Format_RGB888)
        currentFrame = QPixmap(image)
        currentFrame2 = currentFrame.scaledToWidth(360)
        self.vid.setPixmap(currentFrame2)   
        
    
        
    def center(self):
        
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        
    def closeEvent(self, event):
        
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()        
                
            
    # initialize the list of reference points and boolean indicating
    # whether cropping is being performed or not

     
       
class PlotCanvas(FigureCanvas):
    def __init__(self, parent = None, w = 9, h = 6):
        
        self.fig = Figure(figsize = (w, h),tight_layout=True)
        self.axes = None
        #self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        #self.axes.hold(False)

        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        self.nbPlots = 0
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Minimum)
        FigureCanvas.updateGeometry(self)
        self.figure.clear()
#        self.fig.tight_layout()

        self.dataX = list()
        self.dataY = list()



    def plot_data(self, x,y):
        
        self.axes.plot(x,y)
        self.store_data(x,y)
        self.axes.yaxis.grid(False)
        self.axes.xaxis.grid(False)

        
        # refresh canvas
        if self.nbPlots == 0:
            self.originalLimit = self.axes.get_xlim()
            self.axes.set_xlabel('Time (s)', fontsize = 14)
            self.axes.set_ylabel('Intensity (a.u.)', fontsize = 14)
            self.axes.tick_params(labelsize=14)
            self.axes.tick_params(labelsize=14)
            
        self.axes.set_aspect(1./self.axes.get_data_ratio())
        
        
#        self.fig.tight_layout()
        self.draw()
        self.nbPlots += 1
        
        
    def store_data(self,x,y):
        
        self.dataX.append(x)
        self.dataY.append(y)
        
        
    def clear_data(self):
        # Clears all the data in the plots: resets data stored
        # in dataX and dataY, sets nb of plots to 0 and erases figure
        self.dataX = list()
        self.dataY = list()
        self.figure.clear()
        self.draw()
        self.nbPlots = 0
    
    def create_axes(self):
        
        self.dataX = list()
        self.dataY = list()
        print('Axes created.')
        self.axes = self.fig.add_subplot(111)
        self.nbPlots = 0
#        self.fig.tight_layout()
        
    def change_range(self):
        # Changes the range of the plot according to the numbers given
        
        # Gets the text stored in the rangeEdit elements
        # which are actually numbers
        text1 = ex.rangeEdit
        text2 = ex.rangeEdit2
        print(text1.text())
        
        if text1.text().lstrip('-+').isdigit() and text2.text().lstrip('-+').isdigit():
            # First, it checks they are actually numbers
            if float(text1.text()) < float(text2.text()):
                # If the first number is less than the second
                # it changes the limits
                self.axes.set_xlim([float(text1.text()),float(text2.text())])
                self.axes.set_aspect(1./self.axes.get_data_ratio())
#                self.fig.tight_layout()
                self.draw()
            else:
                # If the first number is larger than the second, the
                # limits don't make sense
                print('Not possible to change range.')
        else:
            # If they are not numbers, it's also not possible to continue
            # but no error is thrown
            print('Not possible to change range.')

    def reset_limits(self):
        # Resets the limits of the axes to the original limit
        # it also resets the aspect ratio of the plots
        if self.axes:
            self.axes.set_xlim(self.originalLimit)
            self.axes.set_aspect(1./self.axes.get_data_ratio())
            self.fig.tight_layout()
            self.draw()
        else:
            print('Not possible to resent limits.')
                
    def save_canvas(self,dname):
        

        
        plt.ioff() 
        fig = plt.figure(figsize = figsize)
        for i in range(0,len(self.dataX)):
            plt.plot(self.dataX[i],self.dataY[i],**pltArgs)
           
        ax = plt.gca()
        ax.set_xlim(self.axes.get_xlim())
        plt.xlabel('Time (s)',**labelFont)
        plt.ylabel('Intensity (a.u.)',**labelFont)
        plt.xticks(**ticksFont)
        plt.yticks(**ticksFont)
        plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
        loc = plticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
#        ax.xaxis.set_major_locator(loc)
#        plt.tight_layout()
        
        
#        aspectRatio = self.axes.get_data_ratio()
#        size = self.fig.get_size_inches()
#        self.fig.set_size_inches(9,6)
        fig.savefig(dname + '.png',dpi=500)
        fig.savefig(dname + '.svg',format='svg',dpi=1200) 
#        self.axes.set_aspect(1./aspectRatio)
#        self.fig.set_size_inches(size)
#
#        self.draw()
        
        plt.close()
        plt.ion() 
        
        filename = dname+'.txt'
                
        with open(filename,'w') as f:
            f.write('Time (s)\tIntensity (a.u.)\n')
            
            for i in range(0,len(self.dataX)):
                f.write('Plot number '+str(i)+'\n')
                for j in range(0,len(self.dataX[i])):
                    f.write(str(self.dataX[i][j])+'\t'+str(self.dataY[i][j])+'\n')
                        
        for i in range(0,len(self.dataX)):
            
            if i == 0:
                
                dataAll = np.column_stack((self.dataX[i],self.dataY[i]))
                
            else:
                
                dataAll = np.column_stack((dataAll,self.dataX[i]))
                dataAll = np.column_stack((dataAll,self.dataY[i]))

            
        #np.savetxt(filename, dataAll, newline=os.linesep)

#        self.fig.tight_layout()
        
    def open_figure(self):
        
        # figOpen = plt.figure(figsize=figsize)
        plt.figure(figsize=figsize)

        for i in range(0,len(self.dataX)):
            plt.plot(self.dataX[i],self.dataY[i])
           
        ax = plt.gca()
        ax.set_xlim(self.axes.get_xlim())
        plt.xlabel('Time (s)',**labelFont)
        plt.ylabel('Intensity (a.u.)',**labelFont)
        plt.xticks(**ticksFont)
        plt.yticks(**ticksFont)
        plt.tight_layout()
        plt.show()
        
            
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = mainWindow()

    
    sys.exit(app.exec_())  
