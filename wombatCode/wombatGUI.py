"""
Set of functions and classes used to build and run the WOMBAT GUI

The main function is releaseTheWombat, which is likely the only function
one would need to call in an external program

obsFiles, nWFs=1, overviewPlot=False, reloadDict=None, logFile=None, tRes=20, doFigLabs=True

Inputs:
    obsFiles: nested lists of maps and headers that releaseTheWombat uses to
              set up the background images in the form [inst1, inst2, ...] 
              where each insts is an array of [[RDmaps], [BDmaps], [hdrs]] where maps and 
              hdrs are time series of the obs maps and their corresponding headers 
              (e.g. [[[COR2A_RDmap1, COR2A_RDmap2, ...], [COR2A_BDmaps..] [COR2Ahdr1, 
                     COR2Ahdr2, ...]] [[C2_RDmaps], [C2_BDmaps], [C2 hdrs]]
                     [[AIA171maps], [AIA171maps] [AIA171hdrs]]])
              *** note we will pass two sets of EUV maps that are most often the same 
                  since we default to not taking a difference but this makes it easier
                  to run consistent code across all instruments

Optional Inputs:
    nWFs:         number of wireframes. Currently set an upper limit of 5 to keep
                  GUI from becoming overloaded
                  defaults to 1

    overviewPlot: flag to include the polar/top-down overview panel showing the relative
                  locations of the Sun, Earth, satellites, and wireframes
                  defaults to False
      
    reloadDict:   option to pass a reloadDictionary (from processReload in
                  wombatWrapper.py) to relaunch the GUI from a previous state
                  defaults to None (aka no reload)

    logFile:      name of the log file used to load a recon. Will be put into the
                  text box in the param window
                  defaults to None
    
    tRes:         time resolution (in mins) to use for the main slider. The pickled data may
                  be in higher or lower resolution but this will be mapped to the slider
                  values (potentially downselecting if data is higher res)
                  defaults to 20 mins    

    doFigLabs:    flag to include labels when saving figs (instrument name + time) 
                  defaults to True

Outputs:
    No outputs automatically generated. If the save button is hit then figures will be saved
    within wbOutputs/ as wombat_SAT+INST_YYYY-MM-DD-THHMMSS.png 

    If the log button is hit lines will be appended to the log file. The
    default log file (if none was provided) is wbOutputs/WomBlog.txt and each line contains:
        1.    time of fit  
        2.    instrument
        3.    time of observation
        4.    WF type + panel number (e.g. GCS1). Adding panel number allows multiple of same type
        5-13. WF parameter values. If a type has <9 parameters the extras are filled with None
        14.   the WOMBAT pickle
        15.   the index of the obs time in this pickle
        16.   the background difference mode (0 running diff, 1 base diff)
        17.   the scaling mode (1 linear, 2 log, 3 sqrt)
        18.   the min brightness setting (0-256)
        19.   the max brightness setting (0-256)
    
    The amount of figures saved/lines logged depends on whether one clicks a save/log button or the 
    active window when one hits the s/l short cut keys

External Calls:
    everything from wombatWF, wombatLoadCTs, wombatMass
    fitshead2wcs, wcs_get_pixel, wcs_get_coord from wcs_funs in the prep code

"""

import sys, os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QGridLayout, QTabWidget, QSlider, QComboBox, QLineEdit, QDoubleSpinBox, QPushButton, QRadioButton
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QPainter
import pyqtgraph as pg
import datetime
from itertools import pairwise
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u


# |------------------------------------|
# |------- Import Wombat Friends ------|
# |------------------------------------|
import wombatWF as wf
import wombatMass as wM
from wombatLoadCTs import *

sys.path.append('prepCode/') 
from wcs_funs import fitshead2wcs, wcs_get_pixel, wcs_get_coord

# |------------------------------------------------------------|
# |----------------- Suppress QGrid Warnings ------------------|
# |------------------------------------------------------------|
import logging
logging.basicConfig(level='INFO')
slogger = logging.getLogger('QGridLayout')
slogger.setLevel(logging.ERROR)

# |------------------------------------------------------------|
# |------------------ Parameter Window Class ------------------|
# |------------------------------------------------------------|
class ParamWindow(QMainWindow):
    """
    Class for the window with parameter settings. It sets up the layout
    and widgets and connects them to the global variables defining the
    wireframe shapes.
    
    Inputs:
        nTabs: the number of panels within the window (one for each
               wireframe object). We limit this to a maximum number of 10
               bc the program slows as the number increase and there
               aren't many use cases for 10+ wireframs
    
    Optional Input:
        tlabs: labels for the time slider (instead of printing an index)
               defaults to none
        
    """
    def __init__(self, nTabs, tlabs=None):
        """
        Intial setup for the param window class.
    
        Inputs:
            nTabs: the number of panels within the window (one for each
                   wireframe object)
    
        Optional Input:
            tlabs: labels for the time slider (strings of date time). This
                   also tells the window how many time steps to have. Passing
                   none indicates a single time step
                   defaults to None
     
        """
        super().__init__()
        
        # Suggest that more than 10 windows is a bad idea
        if nTabs > 10:
            sys.exit('Do you really need to fit >10 wireframes at once? If so figure out where the upper limit of 10 is hardcoded in the ParamWindow class in wombatGUI.py')
        self.nTabs = nTabs
        
        # Check how many times we have using the name strings
        # If multi time set up the slider
        if type(tlabs) != type(None):    
            self.nTsli = len(tlabs) - 1 # slider goes 0 to val so subtract 1
            self.tlabs = tlabs
            self.Tlabels = []
            self.Tsliders = []
        # Hide the time slider if we only have one time
        else:
            self.nTsli = 0 # random number to make it happy
            self.tlabs = ['']
        
        # Flag to keep widgets from overplotting
        self.holdIt = False
        
        # Window size and name    
        self.setWindowTitle('Wombat Parameters')
        self.setGeometry(10, 10, 300, 950)
        self.setFixedSize(300, 950) 

        # Create a QTabWidget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Create individual tabs (pages)
        self.tabs =[]
        
        # Create holder for the WF types
        self.WFtypes = np.zeros(nTabs)
        self.WFnum2type = ['None', 'GCS', 'Torus', 'Sphere', 'Half Sphere', 'Ellipse', 'Half Ellipse', 'Slab', 'Tube', 'GCS*']
        self.WFshort = {'GCS':'GCS', 'Torus':'Tor', 'Sphere':'Sph', 'Half Sphere':'HSph', 'Ellipse':'Ell', 'Half Ellipse':'HEll', 'Slab':'Slab', 'Tube':'Tube', 'GCS*':'GCS*'}
        
        # Create holder for the WF params
        self.WFparams = np.array([np.zeros(10) for i in range(nTabs)])
        
        # Holders for the param widgets so we can rm them if turn off a wf
        self.WFLays = []
        self.widges = [None for i in range(nTabs)]
        self.layouts = []
        self.cbs = []
        self.radButs = []
        self.textBoxes = []
        
        # log File name
        self.saveName = ''
        
        # Number of points in the parameter sliders
        self.nSliders = 201
        
        # Set up the layout within each tab
        for i in range(nTabs):
            aTab = QWidget()
            layout, WFlay = self.paramLayout(i)
            aTab.setLayout(layout)
            self.tabs.append(aTab)
            self.tab_widget.addTab(aTab, "Tab" + str(i))
            self.layouts.append(layout)
            self.WFLays.append(WFlay)
        
    #|-------------------------------------| 
    #|--------- Layout Functions ----------|
    #|-------------------------------------| 
    def paramLayout(self, i):
        """
        Layout set up for a full tab panel
    
        Inputs:
            i: tab number
     
        """
        # |------ Start Layout ------|
        # Start with fake massive label to setup grid size
        layout = QGridLayout()
        label = QLabel('')
        layout.addWidget(label, 0,0,40,11,alignment=QtCore.Qt.AlignCenter)
        
        # |------ Time Label ------|
        if self.nTsli > 0:
            myTlab = ('Time selection: '+self.tlabs[0]).ljust(65)
            Timelabel = QLabel(myTlab)
        else:
            Timelabel = QLabel('Single Time Given')
        # stay the same size ffs. want to avoid resizing
        # screwing with the parameter sliders
        Timelabel.setMaximumWidth(250)
        Timelabel.setMaximumHeight(30)
        Timelabel.setMinimumWidth(250)
        Timelabel.setMinimumHeight(30)
        
        self.Tlabels.append(Timelabel)
        layout.addWidget(Timelabel,0,0,1,11,alignment=QtCore.Qt.AlignLeft)
        
        # |------ Time Slider ------|
        Tslider1 = QSlider()
        Tslider1.setOrientation(QtCore.Qt.Horizontal)
        self.Tsli_dragging = False # State flag
        # Slider doesn't like the number 1 for no apparent reason
        # so easiest just to avoid
        Tslider1.setRange(2,self.nTsli+2)
        Tslider1.sliderPressed.connect(self.dragOn)
        Tslider1.valueChanged.connect(lambda x: self.update_tidx(x, i))
        Tslider1.sliderReleased.connect(self.tsli_release)
        layout.addWidget(Tslider1, 1,0,1,11)
        Tslider1.setMaximumWidth(250)
        Tslider1.setMaximumHeight(30)
        Tslider1.setMinimumWidth(250)
        Tslider1.setMinimumHeight(30)
        self.Tsliders.append(Tslider1)


        # |------ WF Type Label ------|
        label = QLabel('WF Type')
        layout.addWidget(label, 2,0,1,4,alignment=QtCore.Qt.AlignLeft)
        
        # |----- WF Drop Down Box ----|
        cbox = self.wfComboBox(i)
        self.cbs.append(cbox)
        layout.addWidget(cbox,2,4,1,7,alignment=QtCore.Qt.AlignCenter)
                
        # |----- Log Fit Button ----|
        logBut = QPushButton('Log WF Fit')
        logBut.released.connect(self.LBclicked)
        layout.addWidget(logBut, 3, 0, 1,5)
        
        # |----- Show/Hide WF Button ----|
        hideBut = QPushButton('Show/Hide WF')
        hideBut.released.connect(lambda: self.HBclicked(i))
        layout.addWidget(hideBut, 3, 5, 1,6)
        
        # |----- Output Name Label ----|
        label = QLabel('Output Name')
        layout.addWidget(label, 4,0, 1, 5,alignment=QtCore.Qt.AlignLeft)
        
        # |----- Output Name Text Box ----|
        oBox = QLineEdit(self)
        oBox.setText(self.saveName)
        oBox.editingFinished.connect(lambda: self.updateSaveName(i))
        self.textBoxes.append(oBox)
        layout.addWidget(oBox, 4,5,1,6)
        
        # |----- Nested parameter layout ----|       
        # Put a layout within the layout for the slider friends
        # It's like inception but without Elliot Page explaining everything
        WFLay = QGridLayout()
        layout.addLayout(WFLay, 7,0,25,11)
        
        
        # |----- Background Diff Button ----|
        label = QLabel('Background Diff')
        layout.addWidget(label, 41,0,1,6,alignment=QtCore.Qt.AlignLeft)
        radBut1 = QRadioButton('Run')
        radBut1.setChecked(True)
        layout.addWidget(radBut1, 41,5,1,5,alignment=QtCore.Qt.AlignCenter)
        radBut2 = QRadioButton('Base')
        layout.addWidget(radBut2, 41,8,1,5,alignment=QtCore.Qt.AlignCenter)
        radBut1.clicked.connect(lambda:self.btnstate(radBut1, isMain=True))
        radBut2.clicked.connect(lambda:self.btnstate(radBut1, isMain=True))
        self.radButs.append([radBut1, radBut2])

        # |----- Background Drop Down Box ----|
        # Background mode drop down box
        label = QLabel('Background Scaling')
        layout.addWidget(label, 42,0,1,8,alignment=QtCore.Qt.AlignLeft)
        cbox = self.bgComboBox()
        layout.addWidget(cbox,42,7,1,6,alignment=QtCore.Qt.AlignCenter)
        self.Bcbox = cbox
        
        # |----- Happy Little Space Padding ----|
        #label = QLabel('')
        #layout.addWidget(label, 42,0,2,11,alignment=QtCore.Qt.AlignCenter)
        
        # |----- Save Button ----|
        saveBut = QPushButton('Save')
        saveBut.released.connect(self.SBclicked)
        layout.addWidget(saveBut, 43, 0, 1,3,alignment=QtCore.Qt.AlignCenter)

        # |----- Mass Button ----|
        massBut = QPushButton('Mass')
        massBut.released.connect(self.MBclicked)
        layout.addWidget(massBut, 43, 4, 1,3,alignment=QtCore.Qt.AlignCenter)

        # |----- Exit Button ----|
        exitBut = QPushButton('Exit')
        exitBut.released.connect(self.EBclicked)
        exitBut.setStyleSheet("background-color: red")
        layout.addWidget(exitBut, 43, 8, 1,3,alignment=QtCore.Qt.AlignCenter)
        
        return layout, WFLay
    
    def wfComboBox(self,i):
        """
        Combo box for the wireframe type
    
        Inputs:
            i: tab number
        
        Outputs:
            cbox: the widget
     
        """
         # |----- Make Combo Box ----|
        cbox = QComboBox()
        
        # |----- Add Items ----|
        cbox.addItem('|-----None-----|')
        cbox.addItem('GCS')
        cbox.addItem('Torus')
        cbox.addItem('Sphere')
        cbox.addItem('Half Sphere')
        cbox.addItem('Ellipse')
        cbox.addItem('Half Ellipse')
        cbox.addItem('Slab')
        cbox.addItem('Tube')
        cbox.addItem('GCS*')
        
         # |----- Connect Event ----|
        cbox.currentIndexChanged.connect(lambda x: self.cb_index_changed(x,i))
        return cbox
        
    def bgComboBox(self):
        """
        Combo box for the wireframe type
    
        Inputs:
            i: tab number
        
        Outputs:
            cbox: the widget
     
        """
        # |----- Make Combo Box -----|
        cbox = QComboBox()
        
        # |----- Add Items -----|
        cbox.addItem('Linear')
        cbox.addItem('Log')
        cbox.addItem('SQRT')
        
        # |----- Connect Event -----|
        cbox.activated.connect(lambda x:self.back_changed(x, doItAll=True))
        return cbox
        
    def WFparamLayout(self, myWF):
        """
        Layout set up for the WF parameter portion, this is nested within
        the full parameter layout. The number of sliders and all of their
        properties will be set by values in myWF
    
        Inputs:
            myWF: the wireframe object we attach to these parameters
        
        Outputs:
            WFlay: the layout 
        
            widges: an array of widgets for all the parameters, organized as
                    [[TextBox1, TextBox2, ...], [Slider1, Slider2, ...]]
                    yes the var name is widges not widgets 
     
        """
        # |---------------------------------------|
        # |------------ Set up Widgets -----------| 
        # |---------------------------------------|
       
        # |------- Time Label -------|
        WFLay = QGridLayout()
        
        # |------ Make widgets ------|
        # Make a label, text box, and slider for each parameter
        # The number of parameters varies based on WF type but this
        # will pull out the appropriate value from myWF
        widges = [[], []]
        i2f = [] # integer to float for slider to value
        nSliders = self.nSliders # number of points within a slider
        
        # We have a maximum of 9 parameters so loop through all possible
        # ones and add/hide as necessary 
        for i in range(9):
            # Compare to number of labels from myWF
            # Add if we have a param for i
            if i < len(myWF.labels):
                # Get the conversion factor between the slider integers and float vals
                myRng = myWF.ranges[i]
                i2f.append((myRng[1] - myRng[0]) / (nSliders - 1))
                
                # |------ Label ------|
                myDef = myWF.params[i]
                label = QLabel(myWF.labels[i]) 
                WFLay.addWidget(label, 3*i,1,1,3)   
                
                # |------ Label ------|
                wBox = QDoubleSpinBox()
                wBox.setKeyboardTracking(False)
                wBox.setRange(myRng[0], myRng[1])
                if myWF.labels[i] in ['kappa', 'deltaAx', 'deltaCS', 'ecc1', 'ecc2']:
                    wBox.setDecimals(3)
                    wBox.setSingleStep(0.02)
                else:
                    wBox.setDecimals(2)
                    wBox.setSingleStep(0.5)
                WFLay.addWidget(wBox, 3*i,7,1,3)  
                widges[0].append(wBox)
                
                # |------ Slider ------|
                slider = QSlider()
                slider.setOrientation(QtCore.Qt.Horizontal)
                slider.setRange(0,nSliders)
                WFLay.addWidget(slider, 3*i+1,1,1,9)  
                widges[1].append(slider)
                
            # If we don't have this many labels throw some blank widgets of the same
            # size in the GUI to keep the layout positioned the same    
            else:
                # |------ Blank Label ------|
                label = QLabel('')
                WFLay.addWidget(label, 3*i,1,1,3)   
                label = QLabel('')
                WFLay.addWidget(label, 3*i,1,1,3)   
                
                # |------ Hidden Slider ------|
                slider = QSlider()
                slider.setOrientation(QtCore.Qt.Horizontal)
                # Setting these to zero makes it disappear
                slider.setMinimum(0)
                slider.setMaximum(0)
                WFLay.addWidget(slider, 3*i+1,1,1,9)  
        
        # |---------------------------------------|
        # |------ Connect Widgets to Events ------| 
        # |---------------------------------------|
        # Need to do this explicit for each one for some reason other wise gets
        # upset about the looped index variable
        
        # |------ Parameters 1 - 4 ------|
        # All wftype have 4+ parameters so these always included
        widges[1][0].valueChanged.connect(lambda x: self.s2b(x, widges[0][0], i2f[0], myWF.ranges[0][0], myWF, widges))  
        widges[0][0].valueChanged.connect(lambda: self.b2s(widges[1][0], widges[0][0], i2f[0], myWF.ranges[0][0],nSliders, myWF, widges, 0))     
        widges[1][1].valueChanged.connect(lambda x: self.s2b(x, widges[0][1], i2f[1], myWF.ranges[1][0], myWF, widges))  
        widges[0][1].valueChanged.connect(lambda: self.b2s(widges[1][1], widges[0][1], i2f[1], myWF.ranges[1][0],nSliders, myWF, widges, 1))
        widges[1][2].valueChanged.connect(lambda x: self.s2b(x, widges[0][2], i2f[2], myWF.ranges[2][0], myWF, widges))  
        widges[0][2].valueChanged.connect(lambda: self.b2s(widges[1][2], widges[0][2], i2f[2], myWF.ranges[2][0],nSliders, myWF, widges, 2))
        widges[1][3].valueChanged.connect(lambda x: self.s2b(x, widges[0][3], i2f[3], myWF.ranges[3][0], myWF, widges))  
        widges[0][3].valueChanged.connect(lambda: self.b2s(widges[1][3], widges[0][3], i2f[3], myWF.ranges[3][0],nSliders, myWF, widges, 3))
        # |-------- Parameters 5+ -------|
        # Need to check each of the remaining bc depends on wftype
        myNP = len(myWF.labels)
        # At least 5 params
        if myNP > 4:
            widges[1][4].valueChanged.connect(lambda x: self.s2b(x, widges[0][4], i2f[4], myWF.ranges[4][0], myWF, widges))  
            widges[0][4].valueChanged.connect(lambda: self.b2s(widges[1][4], widges[0][4], i2f[4], myWF.ranges[4][0],nSliders, myWF, widges, 4))
        # At least 6 params    
        if myNP > 5:
            widges[1][5].valueChanged.connect(lambda x: self.s2b(x, widges[0][5], i2f[5], myWF.ranges[5][0], myWF, widges))  
            widges[0][5].valueChanged.connect(lambda: self.b2s(widges[1][5], widges[0][5], i2f[5], myWF.ranges[5][0],nSliders, myWF, widges, 5))
        # At least 7 params    
        if myNP > 6:
            widges[1][6].valueChanged.connect(lambda x: self.s2b(x, widges[0][6], i2f[6], myWF.ranges[6][0], myWF, widges))  
            widges[0][6].valueChanged.connect(lambda: self.b2s(widges[1][6], widges[0][6], i2f[6], myWF.ranges[6][0],nSliders, myWF, widges, 6))
        # At least 8 params           
        if myNP > 7:
            widges[1][7].valueChanged.connect(lambda x: self.s2b(x, widges[0][7], i2f[7], myWF.ranges[7][0], myWF, widges))  
            widges[0][7].valueChanged.connect(lambda: self.b2s(widges[1][7], widges[0][7], i2f[7], myWF.ranges[7][0],nSliders, myWF, widges, 7))
        # At least 9 params    
        if myNP > 8:
            widges[1][8].valueChanged.connect(lambda x: self.s2b(x, widges[0][8], i2f[8], myWF.ranges[8][0], myWF, widges))  
            widges[0][8].valueChanged.connect(lambda: self.b2s(widges[1][8], widges[0][8], i2f[8], myWF.ranges[8][0],nSliders, myWF, widges, 8))

        # |---------------------------------------|
        # |------- Initiate Widget Values --------| 
        # |---------------------------------------|
        # Set things to the values the WF has
        inParams = np.copy(myWF.params)
        for i in range(myNP):
            myVal = inParams[i]
            if type(myVal) != type(None):
                slidx = int((myVal - myWF.ranges[i][0])/ i2f[i])
                if slidx > nSliders -1:
                    slidx = nSliders -1
                    myVal = myWF.ranges[i][1]
                elif slidx < 0:
                    slidx = 0
                    myVal = myWF.ranges[i][0]
                widges[1][i].setValue(slidx)
                widges[0][i].setValue(myVal)
        
        # An attempt to make things always stay the same size
        # This isn't exact same across all WF but close
        for i in range(WFLay.rowCount()):
                WFLay.setRowStretch(i, 1)
        return WFLay, widges        
        
   
    #|------------------------------------| 
    #|--------- Event Functions ----------|
    #|------------------------------------| 
    def keyPressEvent(self, event):
        """
        Event for key press events. 
        
        Actions (based on key):
            return = replot (pulls out of param text box)
            q      = close a window
            esc    = close everything
            left   = move time slider to earlier time
            right  = move time slider to later time
            b      = switch this time to base difference
            r      = switch this time to running difference
            s      = save figs for this time 
            l      = log current time wf/background params
            m      = calculate mass
            h      = show/hide wfs
            1      = switch this time to linear scaling
            2      = switch this time to log scaling
            3      = switch this time to sqrt scaling
            
            # Shift actions (shift + key) - do it for everyone
            B      = switch all times to base difference
            R      = switch all times to running difference
            S      = save figures for all times (this will be slowish)
            L      = log all times (using the paramLog/curSet)
            1      = switch all times to linear scaling
            2      = switch all times to log scaling
            3      = switch all times to sqrt scaling
            7 (&)  = propagate WF params back in time
            8 (*)  = propagate WF params forward in time
            9 (()  = propagate min/max values back in time
            0 ())  = propagate min/max values forward in time
        
     
        """
        #|--- Pull Params/Plot ---|
        if event.key() == QtCore.Qt.Key_Return:
            for iii in range(nwfs):
                if type(self.widges[iii]) != type(None):
                    self.updateWFpoints(wfs[iii], self.widges[iii])
                    focused_widget = self.focusWidget()
                    try:
                        focused_widget.deselect()
                    except:
                        pass
                    tabIndex = self.tab_widget.currentIndex()
                    self.Tsliders[tabIndex].setFocus()
        #|--- Closing things ---|
        elif event.key() == QtCore.Qt.Key_Q: 
            self.close()    
        elif event.key() == QtCore.Qt.Key_Escape:
            sys.exit()
        #|--- Time Slider ---| 
        # Can't hit this from mainwindow since t slider already
        # owns left/right keys but will hit since ovw just throws
        # its event to this function
        elif event.key()== QtCore.Qt.Key_Right:
            Tval = self.Tsliders[0].value()
            self.Tsliders[0].setValue(Tval+1)
            self.tsli_release() 
        elif event.key()== QtCore.Qt.Key_Left:
            Tval = self.Tsliders[0].value()
            self.Tsliders[0].setValue(Tval-1) 
            self.tsli_release()          
        #|--- Mass ---|
        elif event.key() == QtCore.Qt.Key_M:
            self.MBclicked()       
        #|--- Upper case for save/log/diff/scl ---|
        elif event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:   
            #|--- LOG ALL!!! ---|
            if event.key() ==  QtCore.Qt.Key_L:
                self.LBclicked(doItAll=True)
            #|--- SAVE ALL!!! ---|
            elif event.key() == QtCore.Qt.Key_S:
                self.SBclicked(doItAll=True)
            #|--- Diff mode ---|
            elif event.key() in [QtCore.Qt.Key_B, QtCore.Qt.Key_R]:
                self.updateDiffMode(event.key(), doItAll=True)
            #|--- Scaling mode ---|
            elif event.key() in [33, 64, 35]:
                if event.key() == 33: #shift 1
                    key = QtCore.Qt.Key_1
                elif event.key() == 64: #shift 2
                    key = QtCore.Qt.Key_2
                elif event.key() == 35: #shift 3
                    key = QtCore.Qt.Key_3
                self.updateScaleMode(key, doItAll=True)
            #|--- Propagate settings forward/back ---|
            elif event.key() in [38, 42, 40, 41]: # shift 7, 8, 9, 0
                self.propagateVals(event.key())
        #|--- Scaling mode ---|
        elif event.key() in [QtCore.Qt.Key_1, QtCore.Qt.Key_2, QtCore.Qt.Key_3]:
            self.updateScaleMode(event.key(), doItAll=False)
        #|--- Difference mode ---|
        elif event.key() in [QtCore.Qt.Key_B, QtCore.Qt.Key_R]:
            self.updateDiffMode(event.key())    
        #|--- Save fig ---|    
        elif event.key() == QtCore.Qt.Key_S:
            self.SBclicked()
        #|--- Log it ---|
        elif event.key() == QtCore.Qt.Key_L:
            self.LBclicked()
        #|--- Show/Hide ---|
        elif event.key() == QtCore.Qt.Key_H:
            for i in range(self.nTabs):
                self.HBclicked(i)
                
    def updateScaleMode(self, key, doItAll=False, justSat=None):
        """
        Helper function to take a scale mode event and adjust
        the main window combo box and replot the background as
        needed. Mostly passes to self.back_changed
        """
        
        if key == QtCore.Qt.Key_1:
            val = 0
        elif key == QtCore.Qt.Key_2:
            val = 1
        elif key == QtCore.Qt.Key_3:
            val = 2
        self.Bcbox.setCurrentIndex(val)
        self.back_changed(val, doItAll=doItAll, justSat=justSat)
            
    def updateDiffMode(self, key, doItAll=False, justSat=None):
        """
        Helper function to take a diff mode event and adjust
        the main window combo box and replot the background as
        needed. It does update the log files as needed
        """
        # Check if doing all instruments or single one
        if type(justSat) == type(None):
            justSat = range(nSats)
            allSats = True
        else:
            allSats = False
            
        # Figure out if moving to running or base 
        if key == QtCore.Qt.Key_B:
            setidx = 1
        elif key == QtCore.Qt.Key_R:
            setidx = 0
        # Can't just set offidx to the oppo bc sometimes people press the
        # same button twice and that will erase min/max vals. 
        # CK is some people
               
        # Switch radio button if doing all
        if doItAll:
            offidx = np.abs(setidx-1) # ok just for the toggle
            for ff in range(self.nTabs):
                self.radButs[ff][offidx].setChecked(False)
                self.radButs[ff][setidx].setChecked(True)
        
        # |--- Loop through whatever inst we have ---|        
        for iii in justSat:
            aPW = pws[iii] 
            if doItAll:
                allThisTidx = range(len(aPW.t2p))
            else:
                tidx = self.Tsliders[0].value() - 2
                # Switch all tidx for the pidx of this inst
                pidx = aPW.t2p[tidx]
                allThisTidx = aPW.p2t[pidx]
            
            # Get current values 
            myscl = curSet[aPW.instTag][1][aPW.tslIdx]
            offidx = curSet[aPW.instTag][0][aPW.tslIdx] # could be same as setidx
            setLog[aPW.instTag][offidx][myscl][0][allThisTidx] = np.copy(curSet[aPW.instTag][2][allThisTidx])
            setLog[aPW.instTag][offidx][myscl][1][allThisTidx] = np.copy(curSet[aPW.instTag][3][allThisTidx])
            curSet[aPW.instTag][0][allThisTidx] = setidx
            curSet[aPW.instTag][2][allThisTidx] = np.copy(setLog[aPW.instTag][setidx][myscl][0][allThisTidx])
            curSet[aPW.instTag][3][allThisTidx] = np.copy(setLog[aPW.instTag][setidx][myscl][1][allThisTidx])                
            
            aPW.MinSlider.setValue(curSet[aPW.instTag][2][aPW.tslIdx])  
            aPW.MaxSlider.setValue(curSet[aPW.instTag][3][aPW.tslIdx])
            aPW.plotBackground()                
        
    def propagateVals(self, key, justSat=None):
        """
        Helper function to propagate wireframe or settings 
        values back or forward in time based on key press event
        
        Shift + 7 copy all params backward
        Shift + 7 copy all params forward
        Shift + 8 copy min backward
        Shift + 9 copy max forward
        
        """        
        # Figure out if going forward or back
        # get the corresponding idx
        tidx = self.Tsliders[0].value() - 2
        if key in [38, 40]: # back
            allThisTidx = range(tidx)
        else:
            allThisTidx = range(tidx,self.nTsli+1)
        
        # |--- Propagate wf parameters ---|
        if key in [38, 42]:
            # Get current tab
            myTab = str(self.tab_widget.currentIndex() +1)
            myWF = wfs[self.tab_widget.currentIndex()]
            # Loop through params
            for ii in range(len(myWF.params)):
                paramLog[myWF.WFtype+myTab][ii][allThisTidx] = paramLog[myWF.WFtype+myTab][ii][tidx]
            
            
        # |--- Propagate plot min/max ---|
        elif key in [40, 41]:
            if type(justSat) == type(None):
                justSat = range(nSats)
                allSats = True
            else:
                allSats = False
            
            for iii in justSat:
                aPW = pws[iii] 
                # Change it
                curSet[aPW.instTag][2][allThisTidx] = curSet[aPW.instTag][2][tidx]
                curSet[aPW.instTag][3][allThisTidx] = curSet[aPW.instTag][3][tidx]
                                    
    def updateSaveName(self, i):
        self.saveName = self.textBoxes[i].text()
        for iii in range(nwfs):
            self.textBoxes[iii].setText(self.saveName)
    
    def s2b(self, x=None, b=None, dx=None, x0=None, myWF=None, widges=None):
        """
        Event for slider changes. Triggered on enter press, not just a text change
        
        Inputs:
            x:      the integer slider value
            b:      the box friend for this slider
            dx:     the spacing between slider points in parameter units
            x0:     the parameter value for slider = 0
            myWF:   the associated wireframe
            widges: the array with all the widges
            * Parameters aren't optional but get passed through a lambda function like this
        
        Actions:
            Sets the corresponding box to the new parameter value based on slider index
            Updates the parameter in the wf structure, recalculates wf, and updates figs
     
        """
        # Convert slider idx to parameter value
        myVal = x0+dx*x
        # Make nice string formatting
        if dx < 0.005:
            myStr = '{:.3f}'.format(myVal) 
        else:
            myStr = '{:.2f}'.format(myVal) 
        # Set the box value
        if b.text() == '':
            b.setValue(myVal)
        if np.abs(float(b.text().replace(',','.')) - myVal) > dx:
            b.setValue(myVal)
            # Update the wirefram
            #self.updateWFpoints(myWF, widges) # not needed now with value change on spin box
        

    def b2s(self,s,b, dx=None, x0=None, nSli=None, myWF=None, widges=None, myOrd=None):
        """
        Event for slider changes
        
        Inputs:
            x:      the text box value
            b:      this box
            s:      the slider friend for this box
            dx:     the spacing between slider points in parameter units
            x0:     the parameter value for slider = 0
            nSli:   the number of slider points
            myWF:   the associated wireframe
            widges: the array with all the widges
            myOrd:  the order of this parameter in the full list
            * Parameters aren't optional but get passed through a lambda function like this
        
        Actions:
            Sets the corresponding box to the appropriate index based on value
            Updates the parameter in the wf structure, recalculates wf, and updates figs
     
        """
        temp = b.text()
        # Convert parameter value to slider idx 
        slidx = int((float(b.text().replace(',', '.')) - x0)/dx)
        # Make sure it is in range
        # Readjust and print warning if not
        if slidx > nSli -1:
            slidx = nSli -1
            temp = str(x0 + (nSli-1) * dx)
            print('Value too high, maximum value allowed is ', temp)
        elif slidx < 0:
            slidx = 0
            temp = str(x0)
            print('Value too low, minimum value allowed is ', x0)
        # Set slider value    
        s.setValue(slidx)
        
        if type(paramLog) != type(None):
            tidx = self.Tsliders[0].value() - 2
            toSwitch = range(200)
            # Loop through and find the shortest p2t range to switch
            #for jj in range(nSats):
            for aPW in pws:
                pidx = aPW.t2p[tidx]
                #allThisTidx = np.where(self.tmap[jj] == pidx)[0]
                allThisTidx = aPW.p2t[pidx]
                #print (pidx, allThisTidx)
                if len(allThisTidx) < len(toSwitch):
                    toSwitch = allThisTidx
            
            newPs = myWF.params
            newPs[myOrd] = float(temp)
            myTab = str(self.tab_widget.currentIndex() +1)
            
            for ii in range(len(myWF.params)):
                paramLog[myWF.WFtype+myTab][ii][toSwitch] = newPs[ii]
                
        # Update the wirefram
        if not self.holdIt:
            self.updateWFpoints(myWF, widges)
        
    def cb_index_changed(self, a='None',idx=-10):
        """
        Event for wireframe combo box changes
        
        Inputs:
            a:      the text box (integer) value
            idx:    index for this wireframe
        
        Actions:
            Established a new wireframe object if needed, otherwise 
            changes the existing one to the new type
        
        """
        self.WFtypes[idx] = a

        # |---------------------------------------|
        # |---------- Case of no prior WF --------| 
        # |---------------------------------------|
        global paramsBuilt
        if type(wfs[idx].WFtype) == type(None):
            paramsBuilt = False
            # Make the new WF
            myType = self.WFnum2type[a]
            wfs[idx] = wf.wireframe(myType, WFidx=idx+1)
            
            # Change the tab name to this type
            self.tab_widget.setTabText(idx,self.WFshort[myType])
            # Set up a new param layout
            WFLay, widges = self.WFparamLayout(wfs[idx])
            paramsBuilt = True
            # Add everything to holders            
            self.layouts[idx].addLayout(WFLay, 7,0,30,11)
            self.WFLays[idx] = WFLay
            self.widges[idx] = widges
        
        # |---------------------------------------|
        # |-------- Case of turning WF off -------| 
        # |---------------------------------------|    
        elif a == 0:
            # Create an empty none type WF
            wfs[idx] = wf.wireframe(None)
            
            # Change the tab name
            self.tab_widget.setTabText(idx,'None')
            
            # Clean the parameter layout
            thisLay = self.cleanLayout(self.WFLays[idx])
            
            # Turn off this wf in all plot window
            for aPW in pws:
                aPW.scatters[idx].setData([])
            # Turn off this arrow in the overview window    
            if type(ovw) != type(None):
                ovw.arrows[idx].setStyle(angle=0, headWidth=0, headLen=0, tailLen=0, tailWidth=0, pxMode=False,  pen={'color': color, 'width': 0}, brush=color)    
        
        # |---------------------------------------|
        # |------- Switching existing type -------| 
        # |---------------------------------------|    
        else:
            myType = self.WFnum2type[a]
            newWF = wf.wireframe(self.WFnum2type[a], WFidx=idx+1)
            paramsBuilt = False
            
            # Check if have a reload file first
            doneReloadedIt = False
            tidx = self.Tsliders[0].value() - 2
            myTab = str(self.tab_widget.currentIndex() +1)
            checkIt = paramLog[myType+myTab][0][tidx]
            if type(checkIt) != type(None):
                nps = len(paramLog[myType+myTab])
                newPs = np.zeros(len(paramLog[myType+myTab]))
                for iii in range(nps):
                    newPs[iii] = paramLog[myType+myTab][iii][tidx]
                newWF.params = newPs
                doneReloadedIt = True
            # Otherwise just match what params we can
            if not doneReloadedIt:
                # Create a new wf object but pass it any matching
                # parameters from the previous version
                ogLabs = wfs[idx].labels
                ogParams = wfs[idx].params
                newLabs = newWF.labels
                for iii in range(len(ogLabs)):
                    aLab = ogLabs[iii]
                    if aLab in newLabs:
                        # find matching parameter labels
                        pidx = np.where(newLabs == aLab)[0]
                        newWF.params[pidx] = ogParams[iii]
                        
                # save the new values in param log
                # Grab range to switch
                toSwitch = range(200)
                for aPW in pws:
                    pidx = aPW.t2p[tidx]
                    allThisTidx = aPW.p2t[pidx]
                    if len(allThisTidx) < len(toSwitch):
                        toSwitch = allThisTidx
                        
                for iii in range(len(newLabs)):
                    paramLog[myType+myTab][iii][toSwitch] = newWF.params[iii]
                    
            # Change the tab text        
            self.tab_widget.setTabText(idx,self.WFshort[self.WFnum2type[a]])
            
            # Update the slider layout
            thisLay = self.cleanLayout(self.WFLays[idx])
            WFLay, widges = self.WFparamLayout(newWF)
            self.layouts[idx].addLayout(WFLay, 7,0,30,11)
            self.WFLays[idx] = WFLay
            self.widges[idx] = widges
            paramsBuilt = True
            
            # Give the structure the new wf
            wfs[idx] = newWF
            self.updateWFpoints(newWF, widges)
            
        # |--------------------------------------------|
        # |------- Check for existing same type -------| 
        # |--------------------------------------------| 
        if a !=0:
            allTypes = [wfs[i].WFtype for i in range(len(wfs))]
            allColors = []
            for i in range(len(allTypes)):
                if type(allTypes[i]) != type(None):
                    allColors.append(wfs[i].WFcolor)
            allColors = np.array(allColors)
            if len(np.where(allColors == wfs[idx].WFcolor)[0]) > 1:
                icol = 0
                while icol < len(wf.bonusColors):
                    if wf.bonusColors[icol] not in allColors:
                        wfs[idx].WFcolor = wf.bonusColors[icol]
                        icol = 9999
                    else:
                        icol += 1
                    
            
         
        # |---------------------------------------|
        # |------- Update the plot windows -------| 
        # |---------------------------------------|    
        for aPW in pws:
            # need to do background too bc it can get cleaned
            aPW.plotBackground()
            aPW.plotWFs(justN=idx)
            
    def back_changed(self,text, doItAll=False, justSat=None):
        """
        Event for background combo box changes. The plot window combo box
        event handles most of the heavy lifting. This just passed the change
        along to each of the windows that do the majority of the work.
        
        Inputs:
            text: the text box (integer) value
        
        Actions:
            Changes the background scaling for each plot window
        
        """
        if type(justSat) == type(None):
            justSat = range(nSats)
            allSats = True
        else:
            allSats = False
           
        for iii in justSat:
            aPW = pws[iii]
            
            # Set just the inst slider without plotting yet
            self.holdIt = True
            aPW.cbox.setCurrentIndex(text)       
            self.holdIt = False
            # Do background update with doItAll set as needed
            aPW.back_changed(text, doItAll=doItAll)   
    
    def dragOn(self):
        self.Tsli_dragging = True
    
    def update_tidx(self, tval, myId=None):
        """
        Event for time slider changes. It changes the time index and updates 
        the wireframe parameters/main window sliders and the background based
        on any previous values used in the new time step. It also saves the 
        previous values into the logs
        
        The background observations will update as the time slider is dragged
        but the WF points are only replotted upon release.
        
        Inputs:
            tval: the time slider integer value
        
       
        """
        # Cannot for the life of me figure out why having tval = 1
        # makes the parameter sliders appear at 0 (values and WFs ok tho)
        # Just avoid 1 so the slider starts at 2 and shift what is passed
        # Reset all time sliders to the same value
        for ff in range(self.nTabs):
            if self.Tsliders[ff].value() != tval:
                self.Tsliders[ff].setValue(tval) 

        # Only do the full update process once
        if self.tab_widget.currentIndex() == myId:
            tidx = tval - 2 # this is what everyone who is not a slider should use   
            ff = 0  
            for aPW in pws:
                aPW.tslIdx = tidx
                aPW.pickIdx = aPW.t2p[tidx]
                # Check if we have existing aesthetics
                if type(curSet) != type(None):
                    myInst = aPW.instTag
                    myDif = curSet[myInst][0][tidx]
                    myscl = curSet[myInst][1][tidx]
                    myMin = curSet[myInst][2][tidx]
                    myMax = curSet[myInst][3][tidx]

                    self.radButs[0][myDif].setChecked(True)
                    self.radButs[0][np.abs(myDif-1)].setChecked(False)
                    aPW.cbox.setCurrentIndex(myscl)    
                    
                    aPW.MinSlider.setValue(myMin)
                    aPW.MaxSlider.setValue(myMax)
            
                    aPW.plotBackground()   

            for aTlab in self.Tlabels: 
                aTlab.setText('Time selection: '+self.tlabs[tidx])
                
            if ovw:
                ovw.updateFoV()
            
            # If not dragging the time slider (which will hit here)
            # update the wf points if the parameters have changed
            if not self.Tsli_dragging:
                isDiff = False
                if paramsBuilt:
                     for ff in range(self.nTabs):  
                        # Make sure wf is defined 
                        if type(wfs[ff].WFtype) != type(None):
                            theKey = wfs[ff].WFtype+str(ff+1)
                            
                            # Check if paramLog has None for this time, if so copy current vals
                            toSwitch = range(200)
                            for aPW in pws:
                                pidx = aPW.t2p[tidx]
                                allThisTidx = aPW.p2t[pidx]
                                if len(allThisTidx) < len(toSwitch):
                                    toSwitch = allThisTidx
                            for i in range(len(wfs[ff].params)):                               
                                if type(paramLog[theKey][i][tidx]) == type(None):
                                    paramLog[theKey][i][toSwitch] = wfs[ff].params[i]
                                    # Back fill as needed
                                    if None in paramLog[theKey][i][:toSwitch[-1]+1]:
                                        for ii in range(toSwitch[0]):
                                            if type(paramLog[theKey][i][ii]) == type(None):
                                                paramLog[theKey][i][ii] = wfs[ff].params[i]
                                    
                                # Check if paramLog val diff from current sliders
                                else:
                                    sumDiff = 0
                                    nowVals = [] # collect vals for test printing
                                    for jj in range(len(wfs[ff].params)):
                                        sumDiff += np.abs(wfs[ff].params[jj] - paramLog[theKey][jj][tidx])
                                        nowVals.append(wfs[ff].params[jj])
                                    # Difference found
                                    if sumDiff != 0:     
                                        for i in range(len(wfs[ff].params)):                       
                                            wfs[ff].params[i] = np.copy(paramLog[theKey][i][tidx]) # no pointer
                                        wfs[ff].getPoints()
                                        self.holdIt = True
                                        for j in range(len(wfs[ff].params)):
                                            self.widges[ff][0][j].setValue(paramLog[theKey][j][tidx])
                                        self.holdIt = False
                                        self.updateWFpoints(wfs[ff], self.widges[ff])
                                        isDiff = True
                                    
                # Replot the wfs if we need to                                                                    
                if isDiff:
                    for ipw in range(nSats):
                        pws[ipw].plotWFs()
                        
    def tsli_release(self):
        """
        Action for release of the T slider. Essentially the 
        same behavior as single step on slider but called on release
        not during drag
        """
        if type(paramLog) != type(None):
            # Turn off dragging flag
            self.Tsli_dragging = False
            for ff in range(self.nTabs):  
                if type(wfs[ff].WFtype) != type(None):
                    theKey = wfs[ff].WFtype+str(ff+1)
                    tval = self.Tsliders[ff].value()
                    prevT = pws[0].tslIdx
                    tidx = tval - 2
                    # Check if param log is empty for this WF
                    # Grab range to switch
                    toSwitch = range(200)
                    for aPW in pws:
                        pidx = aPW.t2p[tidx]
                        allThisTidx = aPW.p2t[pidx]
                        if len(allThisTidx) < len(toSwitch):
                            toSwitch = allThisTidx
                    
                    #|---- Existing params are none ----|
                    # Fill with current values of sliders
                    for i in range(len(wfs[ff].params)): 
                        if type(paramLog[theKey][i][tidx]) == type(None):
                            paramLog[theKey][i][toSwitch] = np.copy(wfs[ff].params[i])
                            if None in paramLog[theKey][i][:toSwitch[-1]+1]:
                                for ii in range(toSwitch[0]):
                                    if type(paramLog[theKey][i][ii]) == type(None):
                                        paramLog[theKey][i][ii] = wfs[ff].params[i]
                                                                            
                    #|--- Prev values exist ---|
                    else:
                        for i in range(len(wfs[ff].params)): 
                            wfs[ff].params[i] = np.copy(paramLog[theKey][i][tidx])
                        
                    #|--- Update the widget values ---|
                    self.holdIt = True
                    for j in range(len(wfs[ff].params))[::-1]:
                        self.widges[ff][0][j].setValue(paramLog[theKey][j][tidx])
                    self.holdIt = False
                    self.updateWFpoints(wfs[ff], self.widges[ff])     
   
            for ipw in range(nSats):
                pws[ipw].plotWFs()
        
    def EBclicked(self):
        """
        Event for clicking the exit button
        
        Actions:
            Everything goes bye-bye
        
        """
        if type(self.logFile) == type(None):
            self.logFile.close()
        sys.exit()

    def HBclicked(self, i):
        """
        Event for clicking the show/hide button
        
        Inputs:
            i: the wireframe index
        
        Actions:
            Toggles showing/hiding a wireframe
        
        """
        # If its on, turn it off
        if wfs[i].showMe: 
            wfs[i].showMe = False
            # Hide it by setting scatter data to empty
            for aPW in pws:
                aPW.scatters[i].setData([])
        # If its off, turn it on
        else:
            wfs[i].showMe = True
            # Just recalc and show the scatter points
            for aPW in pws:
                aPW.plotWFs(justN=i)


    def LBclicked(self, singleSat=None, doItAll=False):
        """
        Event for clicking the log button. If called by the parameter
        window it will log the wf parameters for each of the plot panels. 
        If called by the plot panel it will only log that panels timestamp
        It is set to append to existing files rather than rewrite. It also
        is being difficult about using a global variable for a file name 
        so have resorted to open/close file on each click bc not noticeably
        slow. AKA the Robin's nest.
        
        Optional Inputs:
            singleSat: the index of a single plot window
        
            doItAll: a flag to save all the current WF fits for all times and all
                     instruments. defaults to False
        
        Actions:
            Adds a line in wbOutputs/oBoxText.txt where oBox text is the 
            current value of the output text box
        
            The line contains
                Time fit is made (real world time)
                Instrument
                Time of observation used
                WFtype
                Parameters (filled with Nones up to 9 values as needed)
                Pickle name
                Time index for that sat
                Base mode for that sat (R or B)
                Scale mode for that sat (1 2 3 for lin, log, sqrt)
                Levels min (int)
                Levels max
                
        
        """
        # Get file name from window object
        nameIt = self.saveName
        if nameIt == '':
            nameIt = 'WomBlog'
        logFile = open('wbOutputs/'+nameIt+'.txt', 'a')

        # Check if doing one or all fits
        if type(singleSat) != type(None):
            toDo = [singleSat]
        else:
            toDo = range(nSats)
        
                
        # Grab the current real world time
        nowTime = datetime.datetime.now() 
        
        # Loop through sats we want to log 
        for j in toDo:
            aPW = pws[j]
            tidx2do = [] # uniform time index (for param log)
            pidx2do = [] # plot window (for sat stuff)
            
            # Grab idx to log
            if doItAll:
                #print (aPW.p2t)
                for ii in range(len(aPW.satStuff[0])):
                    #myLogId = np.where(aPW.st2obs == ii)[0][0]
                    if len(aPW.p2t[ii]) > 0:
                        # Take middle of the p ranges, should be closest to
                        # the obs time
                        #myLogId = int(np.median(aPW.p2t[ii]))
                        myLogId = aPW.p2tBF[ii]
                        pidx2do.append(ii)
                        tidx2do.append(myLogId)
            else:
                pidx2do = [aPW.pickIdx]
                tidx2do = [aPW.p2tBF[aPW.pickIdx]]
            
            # |--- Loop through wfs ---|
            for k in range(nwfs):
                aWF = wfs[k]
                # Track the first params, only print it at the latest time
                firstLine = None
                firstParams = None
                if type(aWF.WFtype) != type(None):
                    for iii in range(len(tidx2do)):
                        pidx = pidx2do[iii]
                        tidx = tidx2do[iii]
                        tag = aPW.satStuff[0][0]['KEY']
                        obsT = aPW.satStuff[0][pidx]['DATEOBS']
                        
                        # |--- Make an output line ---|
                        # Observer and time of obs
                        outStr = nowTime.strftime("%Y-%m-%dT%H:%M:%S")
                        outStr += ' ' + tag + ' ' + obsT + ' ' + aWF.WFtype.replace(' ', '') + str(k+1) +' '
                        # Dump all the params and fill with Nones as needed
                        myPs = paramLog[aWF.WFtype+str(k+1)]
                        paramStr = ''
                        for ii in range(9):
                            if ii < (len(myPs)):
                                paramStr += str(myPs[ii][tidx]) + ' '
                            else:
                                paramStr += 'None '
                        outStr += paramStr
                            
                        # Add the name of the background pickle and the time index for that data
                        outStr += bkgpkl + ' ' + str(pidx)    
                    
                        # Add the background info
                        didx, sclidx = curSet[aPW.instTag][0][tidx], curSet[aPW.instTag][1][tidx]
                        smin, smax = curSet[aPW.instTag][2][tidx], curSet[aPW.instTag][3][tidx]
                        outStr += ' ' + str(didx) + ' ' + str(sclidx+1)     
                        #svals = aPW.slidervals[aPW.didx, aPW.sclidx] # diff, scale time, min/max 
                        outStr += ' ' + str(smin) + ' '+ str(smax)
                        
                        
                        #|--- Check on adding lines ---|
                        addLine = False
                        # May have times before the first real fit. Only add the latest
                        # time with the params matching those at t=0
                        if type(firstLine) == type(None):
                            firstParams = paramStr
                            firstLine = outStr
                        elif firstLine != 'Done':
                            if paramStr != firstParams:
                                addLine = True
                                firstLine = 'Done'
                                
                            else:
                                firstLine = outStr
                        else:
                            # Take the first time hit new params if not at the beginning
                            if paramStr != firstParams:
                                addLine = True    
                                
                        # And ignore it all for single line        
                        if not doItAll:
                            addLine = True
                                
                        #|--- Print to screen and log ---|                                    
                        if addLine:
                            firstParams = paramStr
                            print(outStr)
                            logFile.write( outStr+ '\n')
                else:
                    print('Cannot log WF', k, 'no parameters defined')
        logFile.close()

    def SBclicked(self, singleSat=None, doItAll=False):
        """
        Event for clicking the save button. If called by the parameter
        window it will save the wf parameters/reload file and images 
        for each of the plot panels. If called by the plot panel it will
        only doing one figure. 
        
        This is a little bit of a hack where the function sorts out which
        tslider indices need to be saved and just sets the GUI to each one
        then calls a separate function that converts the windows to pngs
        
        Optional Inputs:
            singleSat: the index of a single plot window

            doItAll: a flag to save all the current WF fits for all times 
                     and all instruments. defaults to False
                
        Actions:
            Saves a png for each plot window (only at current time index)
            Saves a png of the overview window
        
        """

        #|------------------------------------| 
        #|---------- Save Figures ------------|
        #|------------------------------------|
        # Sort out sats
        if type(singleSat) != type(None):
            toDo = [singleSat]
        else:
            toDo = range(nSats)
        
        # Sort out times
        if doItAll:
            alltidx = []    
            for j in toDo:
                aPW = pws[j]
                for ii in range(len(aPW.satStuff[0])):
                    #myLogId = np.where(aPW.st2obs == ii)[0][0]
                    myLogId = aPW.p2t[ii][0]
                    if myLogId not in alltidx:
                        alltidx.append(myLogId)
            counter = 1
            for aIdx in alltidx:
                self.Tsliders[0].setValue(aIdx+2) 
                print ('--- Saving time step', counter, 'out of',len(alltidx),'---')
                self.makeFigs(toDo)
                counter += 1
        else:
            self.makeFigs(toDo) 
                               
    def MBclicked(self):
        """
        Event for clicking the mass button. Print the mass
        for each WF in each instrument panel to the terminal.
        
        It uses pts2mask from the wf class to generate a mask
        surrounding the projected points, which plotBackground
        uses to sum the appropriate region of the corresponding
        mass file for that observation.
                
        """
        # Check if wireframes are turned on
        for aPW in pws:
            didx = curSet[aPW.instTag][0][self.Tsliders[0].value() - 2]
            if not (aPW.satStuff[didx][0]['OBSTYPE'] == 'EUV'):
                for j in range(len(aPW.scatters)):
                    aScat = aPW.scatters[j]
                    xs, ys = aScat.getData()
                    mask = wf.pts2mask(aPW.mIms[0].shape, [xs,ys])
                    aPW.WFmasks[j] = mask
                     
                if aPW.nowMass:
                    aPW.nowMass = False
                else:
                    aPW.nowMass = True                     
                aPW.plotBackground()
            else:
                print ('No mass calc for EUV images')
                    
    def btnstate(self,b, isMain=False):
        """
        Action for hitting the difference mode radio button.
        Requires actually clicking the button, not activated 
        by other changes to it.
        
        """
        # Swap the radio button
        if b.isChecked():
            setidx = 0
            oidx = 1
        else:
            setidx = 1
            oidx = 0

        # Update the plot windows and log
        for aPW in pws:            
            myInst = instNames[aPW.winidx]
            
            allThisTidx = range(len(aPW.t2p))
            
            myscl = curSet[aPW.instTag][1][allThisTidx[0]]
            setLog[aPW.instTag][oidx][myscl][0][allThisTidx] = np.copy(curSet[aPW.instTag][2][allThisTidx])
            setLog[aPW.instTag][oidx][myscl][1][allThisTidx] = np.copy(curSet[aPW.instTag][3][allThisTidx])
            curSet[aPW.instTag][0][allThisTidx] = setidx
            curSet[aPW.instTag][2][allThisTidx] = np.copy(setLog[aPW.instTag][setidx][myscl][0][allThisTidx])
            curSet[aPW.instTag][3][allThisTidx] = np.copy(setLog[aPW.instTag][setidx][myscl][1][allThisTidx])
            
            aPW.MinSlider.setValue(curSet[aPW.instTag][2][aPW.tslIdx])  
            aPW.MaxSlider.setValue(curSet[aPW.instTag][3][aPW.tslIdx])
            aPW.plotBackground()
            
    #|------------------------------| 
    #|----------- Others -----------|
    #|------------------------------| 
    def cleanLayout(self,lay):
        """
        Function for clearing out a layout when a wf is changed/removed
        
        Inputs:
            lay: a layout
        
        Actions:
            Removes all the widgets from a layout so it appears
            blank
        
        """
        for i in reversed(range(lay.count())): 
            item = lay.takeAt(i)
            widget = item.widget()
            widget.deleteLater()
        return lay
    
    def updateWFpoints(self, aWF, widges):
        """
        Function to trigger wf update based on a parameter change
        
        Inputs:
            aWF:    the wireframe of interest
        
            widges: the widgets that correspond to this wireframe
        
        Actions:
            Updates the wireframe points/plot points using the 
            new parameter values
        
        """
        # Got to check if all the points are set or this
        # will blow up on the first run through before panel is built
        if paramsBuilt:
            for i in range(len(widges[0])):
                if widges[0][i].text() != '':
                    aWF.params[i] = float(widges[0][i].text().replace(',','.'))
            aWF.getPoints()
                        
            for ipw in range(nSats):
                pws[ipw].plotWFs(justN=aWF.WFidx-1)
            if ovw:
                ovw.updateArrow(aWF.WFidx-1,color=aWF.WFcolor)
                
    def makeFigs(self, toDo, silent=False):
        """
        Function to save pngs of the current plot windows, both
        the observation panels and the overview panel. It uses
        the global variable figLabels to determine whether to 
        write the instrument and time on top of the observations.
        The results are saved as wombat_TAG_SAT_INST_YYYY-MM-DDTHHMMSS.png
        
        Inputs:
            toDo:   a list of plot window indices
        
        Optional Inputs:
            silent: flag to not print updates to screen
        
        """
        #|--------- Save plot windows --------|         
        for j in toDo:
            aPW = pws[j]
            
            # Get the file name
            pidx = aPW.pickIdx
            if self.saveName == '':
                pref = 'wombat'
            else:
                pref = 'wombat_'
            figName = pref+ self.saveName + '_' +  aPW.satStuff[0][pidx]['MYTAG'].replace(' ','_') + '_' + aPW.satStuff[0][pidx]['DATEOBS'].replace(':','')  +'.png'
            
            # Grab the window image
            figGrab = aPW.pWindow.grab()
            
            # Make sure figLabels is set (should be, but doesn't
            # hurt to check)
            if 'figLabels' in globals():
                showLabs = figLabels
            else:
                showLabs = True
            
            # Add labels    
            if showLabs:
                painter = QPainter(figGrab)
                painter.setPen(pg.mkPen('w', width=1.75)) 
                #painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
                painter.drawText(5, 395, aPW.satStuff[0][pidx]['MYTAG'].replace('_',' ')) # X, Y coordinates
                painter.drawText(285, 395, aPW.satStuff[0][pidx]['DATEOBS']) # X, Y coordinates
                painter.end()
                QApplication.processEvents()
            
            # Save it
            figGrab.save('wbOutputs/'+figName)
            if not silent:
                print ('Saving figure in wbOutputs/'+figName )
            
        #|------- Save overview window -------|   
        if ovw:
            figName = 'wombat_'+ self.saveName + '_overview_' + pws[0].satStuff[0][0]['DATEOBS'].replace(':','') + '.png'
            figGrab = ovw.pWindow.grab()
            figGrab.save('wbOutputs/'+figName)
            if not silent:
                print ('Saving figure in wbOutputs/'+figName )
    
            
# |------------------------------------------------------------|
# |------------------- Figure Window Class --------------------|
# |------------------------------------------------------------|
class FigWindow(QWidget):
    """
    Class for the plot window showing the background image and the
    projected wireframe points. It also sets up a limited number of 
    widgets mostly related to scaling the background.
    
    Inputs:
        myObs:    an array containing two times series, one of the unscaled 
                  image maps and another of the corresponding headers
                  e.g. [[map1, map2,...], [hdr1, hdr2, ...]]
    
        myScls:   an array of three times series the data scaled using diffent
                  methods (linear, logarithmic, square root). This data is in 
                  array form, not maps.
                  e.g. [[lin1, log1, sqrt1], [lin2, log2, sqrt2], ...]
    
        satStuff: the header like structure created by getSatStuff.
    
        massIms:  the data converted into mass imgs
                  e.g. [mIm1, mIm2, ...]
    
    Optional Inputs:
        myNum:    an index number for this plot window, useful when creating
                  multiple windows, but unnecessary for single window
                  defaults to 0
        
        tmap:     an array with [t2p and p2t] where t2p is an array mapping from 
                  t slider to pickle idx and p2t is a dictionary for the reverse
                  the t2p maps are arrays (e.g. [0, 1, 1, 2, 3, 4, 4])
                  where the array index is t and the value is p
                  the p2t maps are dicts (e.g. {0:[0], 1:[1,2], ...})
                  where the key is the p index and the array is all the
                  matching t idxs
         
        screenXY: size of the computer display in pixels [x,y]. used to help place windows
    
     
    """
    def __init__(self, myObs, myScls, satStuff, massIms, myNum=0, tmap=[[0],{0:0}], screenXY=None, mouseEnabled=False):
        """
        Intial setup for the figure window class.
    
        Inputs:
            myObs:    an array containing two times series, one of the unscaled 
                      image maps and another of the corresponding headers
                      e.g. [[map1, map2,...], [hdr1, hdr2, ...]]
    
            myScls:   an array of three times series the data scaled using diffent
                      methods (linear, logarithmic, square root). This data is in 
                      array form, not maps.
                      e.g. [[lin1, log1, sqrt1], [lin2, log2, sqrt2], ...]
    
            satStuff: the header like structure created by getSatStuff.
    
        Optional Inputs:
            myNum:    an index number for this plot window, useful when creating
                      multiple windows, but unnecessary for single window
                      defaults to 0
        
            tmap:     an array with [t2p and p2t] where t2p is an array mapping from 
                      t slider to pickle idx and p2t is a dictionary for the reverse
                      the t2p maps are arrays (e.g. [0, 1, 1, 2, 3, 4, 4])
                      where the array index is t and the value is p
                      the p2t maps are dicts (e.g. {0:[0], 1:[1,2], ...})
                      where the key is the p index and the array is all the
                      matching t idxs
        
            screenXY: resolution of the computer monitor. used to place windows nicer
        
            mouseEnabled: allow for scrolling and dragging of the plot data within a window.
                          Defaults to disabled because tends to cause more harm than good.
        
        External Calls:
            check4CT from wombatLoadCTs
     
        """
        super().__init__()
        
        #|---- Setup variables ----|
        self.winidx = myNum # index number for multi mode
        self.satStuff = satStuff 
        self.satName = satStuff[0][0]['OBS'] +' '+ satStuff[0][0]['INST']
        self.instTag = satStuff[0][0]['KEY']
        self.OGims = myObs[0]
        self.mIms  = massIms
        self.hdrs = myObs[1]
        self.myScls2 = myScls # the scaled images
        self.pickIdx = 0 # index within the pickle
        self.tslIdx = 0 # index within the time slider
        self.t2p = tmap[0] # slider time to pickle index
        self.p2t = tmap[1] # pickle index to slider time
        self.p2tBF = tmap[2] # pickle index to single closest slider time
        self.nowMass = False # show the region used to calc mass
        self.WFmasks = [np.zeros(myObs[0][0].data.shape, dtype=int) for i in range(nwfs)]
        
        #|---- Set up/name window ----|
        if type(screenXY) == type(None):
            self.setGeometry(550*(myNum+1), 350, 450, 450) 
        else:
            pWinWid = 10 + 300 + 10
            remWid  = screenXY[0] - pWinWid
            myWid = 465
            # Check if we can fit across
            if remWid > nSats * myWid:
                myx = pWinWid + myWid * myNum
                self.setGeometry(myx, 100, 350, 450) 
            # Else start stacking
            else:
                # Figure out max we can put in a row
                nRow = int(remWid / myWid)
                if myNum < (2*nRow -1):
                    # Fill up top row
                    if myNum < nRow:
                        myx = pWinWid + myWid * myNum
                        self.setGeometry(myx, 10, 350, 450)
                
                    # Add remaining to bottom
                    else:
                        subNum = myNum - nRow
                        myx = pWinWid + myWid * subNum
                        myy = int(screenXY[1] / 2)
                        self.setGeometry(myx, myy, 350, 450)
                # Too many to organize nicely, give up
                else:
                    self.setGeometry(550*(myNum+1), 350, 450, 450) 
                    
        myTitle = self.satName.replace('_',' ')    
        # Clean up some title things
        myTitle = myTitle.replace('WISPR HI1', 'WISPR Inner')    
        myTitle = myTitle.replace('WISPR HI1', 'WISPR Outer')    
        self.setWindowTitle(myTitle)        
        
        #|---- Make a layout ----|
        layoutP =  QGridLayout()
        
        #|---- Add a plot widget ----|
        self.pWindow = pg.PlotWidget()
        self.pWindow.setMinimumSize(400, 400)
        self.pWindow.scene().sigMouseClicked.connect(self.mouse_clicked)
        if not mouseEnabled:
            self.pWindow.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        layoutP.addWidget(self.pWindow,0,0,11,11,alignment=QtCore.Qt.AlignCenter)

        #|---- Make an image item ----|
        self.image = pg.ImageItem(axisOrder='row-major')

        #|---- Mass contour image item ----|
        self.MCimage = pg.ImageItem(axisOrder='row-major')
        
        #|---- Check for color table ----|
        hasCT = check4CT(satStuff[0][0])
        if type(hasCT) != type(None):
             self.image.setLookupTable(hasCT)
        
        #|---- Add the image ----|
        self.pWindow.addItem(self.image)
        self.pWindow.addItem(self.MCimage)
        # shape is [rows, columns] = [y,x]
        self.pWindow.setRange(xRange=(0,myObs[0][0].data.shape[1]), yRange=(0,myObs[0][0].data.shape[0]), padding=0)
        self.pWindow_circle = None
        self.pWindow_north  = None
        
        #|---- Hide the axes ----|
        self.pWindow.hideAxis('bottom')
        self.pWindow.hideAxis('left')
        
        #|---- Set up WF scatters ----|
        # Need to do this here so can adjust them
        # on the fly without clearing
        self.scatters = []
        for i in range(nwfs):
            aScat = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='g'), brush=pg.mkBrush(color='g'),symbol='o', size=2.5)
            self.scatters.append(aScat)
            self.pWindow.addItem(aScat)
        
        #|---- Background mode drop down box ----|
        label = QLabel('Background Scaling')
        layoutP.addWidget(label, 12,0,1,3,alignment=QtCore.Qt.AlignLeft)
        self.cbox = self.bgComboBox()
        layoutP.addWidget(self.cbox,12,3,1,3,alignment=QtCore.Qt.AlignCenter)
        
        #|---- Time label ----|
        self.time_label = QLabel('I AM A TIME')
        layoutP.addWidget(self.time_label, 12,7,1,4,alignment=QtCore.Qt.AlignRight)
                      
        #|---- Min brightness label/slider ----|
        minL = QLabel('Min Value:     ')
        layoutP.addWidget(minL, 13,0,1,9)
        self.MinSlider = QSlider()
        self.MinSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MinSlider.setMinimum(0)
        self.MinSlider.setMaximum(255)
        self.MinSlider.setValue(satStuff[0][0]['SLIVALS'][0][0])
        layoutP.addWidget(self.MinSlider, 13,3,1,9)
        self.MinSlider.valueChanged.connect(lambda x: self.s2l(x, minL, 'Min Value: '))  
        
        #|---- Min brightness label/slider ----|
        maxL = QLabel('Max Value:     ')
        layoutP.addWidget(maxL, 15,0,1,9)
        self.MaxSlider = QSlider()
        self.MaxSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MaxSlider.setMinimum(0)
        self.MaxSlider.setMaximum(255)
        self.MaxSlider.setValue(satStuff[0][0]['SLIVALS'][1][0])
        layoutP.addWidget(self.MaxSlider, 15,3,1,9)
        self.MaxSlider.valueChanged.connect(lambda x: self.s2l(x, maxL, 'Max Value: '))  
        
        #|---- EUV Dispay Mode ----|
        # If EUV switch to show log at the start. Have to do after
        # we've addded the sliders since will adjust them
        if self.satStuff[0][0]['OBSTYPE'] == 'EUV':
            self.cbox.setCurrentIndex(1)
        
        # |----- Log Fit Button ----|
        logBut = QPushButton('Log WF Fit')
        logBut.released.connect(self.LBclicked)
        layoutP.addWidget(logBut, 17, 0, 1,3)
        
        
        #|---- Save button
        saveBut = QPushButton('Save')
        saveBut.released.connect(self.SBclicked)
        layoutP.addWidget(saveBut, 17, 3, 1,3,alignment=QtCore.Qt.AlignCenter)

        #|---- Mass button ----|
        massBut = QPushButton('Mass')
        massBut.released.connect(self.MBclicked)
        layoutP.addWidget(massBut, 17, 6, 1,3,alignment=QtCore.Qt.AlignCenter)

        #|---- Exit button ----|
        exitBut = QPushButton('Exit')
        exitBut.released.connect(self.EBclicked)
        exitBut.setStyleSheet("background-color: red")
        layoutP.addWidget(exitBut, 17, 9, 1,3,alignment=QtCore.Qt.AlignCenter)
        
        #|---- Set layout ----|
        self.setLayout(layoutP)
        
        #|---- Show the background ----|
        # call so we can set the range/lims
        self.plotBackground()
        rw, cl = self.image.image.shape[:2]
        self.pWindow.setRange(xRange=[0,cl], yRange=[0,rw], padding=0)
        self.pWindow.setLimits(xMin=0,yMin=0, xMax=cl, yMax=rw)
        self.pWindow.getViewBox().suggestPadding = lambda *_: 0.0
        self.pWindow.getPlotItem().getViewBox().setAspectLocked(True)
        
        
    #|------------------------------| 
    #|----------- Layout -----------|
    #|------------------------------| 
    def bgComboBox(self):
        """
        Combo box for the background scaling type
        
        Outputs:
            cbox: the widget
     
        """
        # |----- Make Combo Box -----|
        cbox = QComboBox()
        
        # |----- Add Items -----|
        cbox.addItem('Linear')
        cbox.addItem('Log')
        cbox.addItem('SQRT')
        
        # |----- Connect Event -----|
        cbox.currentIndexChanged.connect(self.back_changed)
        return cbox
        

    #|------------------------------| 
    #|----------- Events -----------|
    #|------------------------------| 
    def s2l(self, x=None, l=None, pref=None):
        """
        Event for scaling slider changes. Changes the text label
        and updates curSet for the new min/max
        
        Inputs:
            x:      the integer slider value
            l:      the label friend for this slider
            pref:   the prefix printed in front of the index in the label
            * Parameters aren't optional but get passed through a lambda function like this
        
        Actions:
            Sets the label to pref + x
            Replots background using update min/max values
     
        """        
        # Make sure we change all tidx that match this pidx 
        if type(curSet) != type(None):
                      
            if 'Min' in pref:
                curSet[self.instTag][2][self.tslIdx] = x
                allThisTidx = self.p2t[self.pickIdx]                
                for aId in allThisTidx:
                    curSet[self.instTag][2][aId] = x
                
            elif 'Max' in pref:
                curSet[self.instTag][3][self.tslIdx] = x
                allThisTidx = self.p2t[self.pickIdx]
                for aId in allThisTidx:
                    curSet[self.instTag][3][aId] = x
                        
        l.setText(pref + str(x))
        self.plotBackground()

    def keyPressEvent(self, event):
        """
        Event for key press events. Many of these (***) will
        flag to do such for the current plot window if
        it is the active window
        
        Actions (based on key):
            return = replot (pulls out of param text box)
            q      = close a window ***
            esc    = close everything
            left   = move time slider to earlier time
            right  = move time slider to later time
            b      = switch this time to base difference ***
            r      = switch this time to running difference ***
            s      = save figs for this time ***
            l      = log current time wf/background params ***
            m      = calculate mass
            h      = show/hide wfs
            1      = switch this time to linear scaling ***
            2      = switch this time to log scaling ***
            3      = switch this time to sqrt scaling ***
            
            # Shift actions (shift + key) - do it for everyone
            B      = switch all times to base difference ***
            R      = switch all times to running difference ***
            S      = save figures for all times (this will be slowish) ***
            L      = log all times (using the paramLog/curSet) ***
            1      = switch all times to linear scaling ***
            2      = switch all times to log scaling ***
            3      = switch all times to sqrt scaling ***
            7 (&)  = propagate WF params back in time
            8 (*)  = propagate WF params forward in time
            9 (()  = propagate min/max values back in time ***
            0 ())  = propagate min/max values forward in time ***
        
        """
        #|--- Pull Params/Plot ---|
        if event.key() == QtCore.Qt.Key_Return:
            if 'mainwindow' in globals():
                for iii in range(nwfs):
                    mainwindow.updateWFpoints(wfs[iii], mainwindow.widges[iii])
                    focused_widget = mainwindow.focusWidget()
                    focused_widget.deselect()
                    mainwindow.Tsliders[0].setFocus()
        #|--- Closing Things ---|
        elif event.key() == QtCore.Qt.Key_Q: 
            self.close()
        elif event.key() == QtCore.Qt.Key_Escape:
            sys.exit()
        #|--- Saving/Logging ---|
        elif event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:  
            if 'mainwindow' in globals(): 
                if event.key() ==  QtCore.Qt.Key_L:
                    mainwindow.LBclicked(doItAll=True, singleSat=self.winidx)
                elif event.key() == QtCore.Qt.Key_S:
                    mainwindow.SBclicked(doItAll=True, singleSat=self.winidx)
                elif event.key() in [QtCore.Qt.Key_B, QtCore.Qt.Key_R]:
                    mainwindow.updateDiffMode(event.key(), doItAll=True, justSat=[self.winidx])
                elif event.key() in [33, 64, 35]:
                    if event.key() == 33: #shift 1
                        key = QtCore.Qt.Key_1
                    elif event.key() == 64: #shift 2
                        key = QtCore.Qt.Key_2
                    elif event.key() == 35: #shift 3
                        key = QtCore.Qt.Key_3
                    mainwindow.updateScaleMode(key, doItAll=True, justSat=[self.winidx])
                elif event.key() in [38, 42, 40, 41]: # shift 7, 8, 9, 0
                    mainwindow.propagateVals(event.key(), justSat=[self.winidx])
        #|--- Log ---|            
        elif event.key() == QtCore.Qt.Key_L:
            if 'mainwindow' in globals():
                mainwindow.LBclicked(singleSat=self.winidx)
        #|--- Save ---|            
        elif event.key() == QtCore.Qt.Key_S:
            if 'mainwindow' in globals():
                mainwindow.SBclicked(singleSat=self.winidx)
        #|--- Time Slider ---|
        elif event.key()== QtCore.Qt.Key_Right:
            if 'mainwindow' in globals():
                Tval = mainwindow.Tsliders[0].value()
                mainwindow.Tsliders[0].setValue(Tval+1)
        elif event.key()== QtCore.Qt.Key_Left:
            if 'mainwindow' in globals():
                Tval = mainwindow.Tsliders[0].value()
                mainwindow.Tsliders[0].setValue(Tval-1)    
        #|--- Difference mode ---|
        elif event.key() in [QtCore.Qt.Key_B, QtCore.Qt.Key_R]:
            if 'mainwindow' in globals():
                mainwindow.updateDiffMode(event.key(), justSat=[self.winidx])                                        
        #|--- Scaling mode ---|
        elif event.key() in [QtCore.Qt.Key_1, QtCore.Qt.Key_2, QtCore.Qt.Key_3]:
            if 'mainwindow' in globals():
                mainwindow.updateScaleMode(event.key(), doItAll=False, justSat=[self.winidx])
        #|--- Mass ---|
        elif event.key() == QtCore.Qt.Key_M:
            self.MBclicked()
        #|--- Show/Hide ---|
        elif event.key() == QtCore.Qt.Key_H:
            if 'mainwindow' in globals():
                for i in range(mainwindow.nTabs):
                    mainwindow.HBclicked(i)
                
    def back_changed(self,text, doItAll=False):
        """
        Event for background combo box changes
        
        Inputs:
            text:      the text box (integer) value
        
        Actions:
            Updates the index indicating background scaling type
            Resets slider min/max to best guess vals for this scaling
            Replots background
        
        """
        # Switch to the new mode
        if 'mainwindow' in globals():
            if not mainwindow.holdIt:
                # Sort out which time steps               
                if doItAll:
                    allThisTidx = range(len(self.t2p))
                else:
                    tidx = self.tslIdx
                    # Switch all tidx for the pidx of this inst
                    pidx = self.t2p[tidx]
                    allThisTidx = self.p2t[pidx]
                    
                # |--- Update log with pre-switch vals ---|    
                myDiff = curSet[self.instTag][0][self.tslIdx]
                prevScl  = curSet[self.instTag][1][self.tslIdx]
                # Log the current min/max for this diff/scl
                setLog[self.instTag][myDiff][prevScl][0][allThisTidx] = np.copy(curSet[self.instTag][2][allThisTidx])
                setLog[self.instTag][myDiff][prevScl][1][allThisTidx] = np.copy(curSet[self.instTag][3][allThisTidx])
                
                # |--- Set new diff, check existing min/max ---|    
                curSet[self.instTag][1][allThisTidx] = text
                # Pull existing min/max
                curSet[self.instTag][2][allThisTidx] = np.copy(setLog[self.instTag][myDiff][text][0][allThisTidx])
                curSet[self.instTag][3][allThisTidx] = np.copy(setLog[self.instTag][myDiff][text][1][allThisTidx])
                
                # |--- Update sliders ---|    
                minV, maxV = curSet[self.instTag][2][self.tslIdx], curSet[self.instTag][3][self.tslIdx]
                self.MinSlider.setValue(minV)  
                self.MaxSlider.setValue(maxV)  
                self.plotBackground() # this good in here? seems fine but was originally outside holdIt
                   
    def EBclicked(self):
        """
        Event for clicking the exit button
        
        Actions:
            Everything goes bye-bye
        
        """
        sys.exit()

    def SBclicked(self):
        """
        Event for clicking the save button
        
        Actions:
            Calls the save function in the parameter window but flags
            to only save results for this window
        
        """
        mainwindow.SBclicked(singleSat=self.winidx)
 
    def LBclicked(self):
        """
        Event for clicking the log wireframe button
        
        Actions:
            Calls the log WF function in the parameter window but flags
            to only log results for this window
        
        """
        mainwindow.LBclicked(singleSat=self.winidx)
        
    def MBclicked(self):
        """
        Event for clicking the mass button
        
        Does nothing yet but working on that
        
        """        
        if 'mainwindow' in globals():
            mainwindow.MBclicked()
    
    def mouse_clicked(self,event):
        """
        Event for clicking within a figure window
        
        Input:
            event: the thing triggered by clicking
        
        Actions:
            Prints the following to the terminal:
                SatName InstName pix: pixel_x pixel_y
                     Tx, Ty (arcsec):   Tx, Ty
               Proj R (Rs), PA (deg):   ProjR, PA
               Mass in pixel (1e8 g):   Mass
        
        """
        #|---- Get the event loc in pix ----|
        scene_pos = event.scenePos()
        view_pos = self.pWindow.plotItem.vb.mapSceneToView(scene_pos)
        pix = [view_pos.x(), view_pos.y()]
        
        pidx = self.t2p[self.tslIdx]
        didx = curSet[self.instTag][0][pidx]
            
        #|---- Print pix ----|
        prefA = self.satStuff[didx][pidx]['MYTAG'].replace('_',' ') + ' pix:'
        print (prefA.rjust(25), str(int(pix[0])).rjust(8), str(int(pix[1])).rjust(8))
        
        #|---- Convert to ra/dec ----| 
        skyres = self.OGims[pidx].pixel_to_world(pix[0]*u.pixel, pix[1]*u.pixel)
        Tx, Ty = skyres.Tx.to_value(), skyres.Ty.to_value()
        print ('Tx, Ty (arcsec):'.rjust(25), str(int(Tx)).rjust(8), str(int(Ty)).rjust(8))
        
        # |---- Convert to proj Rsun/PA  ----| 
        Rarc = np.sqrt(Tx**2 + Ty**2)
        Rpix = Rarc / self.satStuff[didx][pidx]['SCALE']
        # Adjust unites for HI
        if self.satStuff[didx][pidx]['OBSTYPE'] == 'HI':
            Rpix = Rpix / 3600
        RRSun = Rpix /  self.satStuff[didx][pidx]['ONERSUN']
        # PA define w/ N as 0 and E (left) as 90
        PA = (np.arctan2(-Tx,Ty) * 180 / np.pi) % 360.
        print ('Proj R (Rs), PA (deg):'.rjust(25), '{:8.2f}'.format(RRSun), '{:8.1f}'.format(PA))
        
        # |---- Get mass per pixel  ----| 
        if type(self.mIms[pidx]) != type(None):    
            px = int(pix[0])
            py = int(pix[1])
            if self.satStuff[didx][0]['OBSTYPE'] == 'COR':
                print('Mass in pixel (1e8 g):'.rjust(25), '{:8.1f}'.format(self.mIms[pidx][py,px]/1e8))
            elif self.satStuff[didx][0]['OBSTYPE'] == 'HI':
                print('Mass in pixel (1e8 g):'.rjust(25), '{:8.1f}'.format(self.mIms[pidx][py,px]/1e8))

        print ('')

    #|------------------------------| 
    #|----------- Others -----------|
    #|------------------------------| 
    def plotWFs(self, justN=0):
        """
        Function for updating the WF scatter points 
        
        Optional Inputs:
            justN: an index indicating to only update that WF
        
        Actions:
            Changes the scatter points within an image
        
        """
        # This is the slow version of pts2proj using sunpy/astropy that runs
        # way too slow to use for updating points continually via slider.
        # Keep the syntax around for comparison if needed but currently matching
        # within a pixel and running much faster
        #skyPt = SkyCoord(x=0, y=0, z=1, unit='R_sun', representation_type='cartesian', frame='heliographic_stonyhurst')
        #myPt2 = self.OGims[self.tidx].world_to_pixel(skyPt)

        #|---- Determine who to update ----|
        toDo = range(nwfs)
        if justN:
            toDo = [justN]
        
        #|---- Loop through to do cases ----|    
        for i in toDo:
            #|---- Double check should do ----|
            # Don't bother if not shown or type is none
            if wfs[i].showMe & (type(wfs[i].WFtype) != type(None)):    
                if type(curSet) != type(None):
                    didx = curSet[self.instTag][0][self.tslIdx]
                    sclidx = curSet[self.instTag][1][self.tslIdx]
                else:
                    didx, sclidx = 0, 0
                pidx = self.pickIdx
                
                #|----------------------|
                #|---- Get sat info ----|
                #|----------------------|
                pos = []
                # Location
                obs = self.satStuff[didx][pidx]['POS']
                # Scale btwn pix and arcsec
                obsScl = [self.satStuff[didx][pidx]['SCALE'], self.satStuff[didx][pidx]['SCALE']]
                if self.satStuff[didx][pidx]['OBSTYPE'] == 'HI':
                    obsScl = [self.satStuff[didx][pidx]['SCALE'] * 3600, self.satStuff[didx][pidx]['SCALE'] * 3600]
                #cent = self.satStuff[self.tidx]['SUNPIX']
                # Occulter info
                if 'OCCRARC' in self.satStuff[didx][pidx]:
                    occultR = self.satStuff[didx][pidx]['OCCRARC']
                else:
                    occultR = None
                # WCS info    
                mywcs  = self.satStuff[didx][pidx]['WCS']
                
                #|---- Set wf aesthetics ----|
                myColor =wfs[i].WFcolor
                # Check for GCS on cor1, switch if so
                if ('COR1' in self.satStuff[didx][pidx]['MYTAG']):
                    if wfs[i].WFtype in ['GCS', 'GCS*']:
                        myColor = 'cyan'
                
                # change pen wid if HI
                penwid =2
                if self.satStuff[didx][pidx]['OBSTYPE'] == 'HI':
                    penwid = 4
                
                
                #|--------------------------|
                #|---- EUV Proj2Surface ----|
                #|--------------------------|
                # For the EUV panels, check if the WF is much higher
                # than the FOV and just project it onto the surface
                # instead if it is
                flatEUV = False
                if self.satStuff[didx][pidx]['OBSTYPE'] == 'EUV':
                    pts = wfs[i].points
                    rs = np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
                    # Compare max wf radius to inst FOV
                    if np.mean(rs) > 1.5*self.satStuff[didx][pidx]['FOV']:
                        flatEUV = True
                # Downselect to fewer points for proj EUV
                toShow = range(len(wfs[i].points[:,0]))
                if flatEUV:
                    toShow = toShow[::2]
                    myColor = '#C81CDE'
                    occultR = 1. * self.satStuff[didx][pidx]['ONERSUN']
                
                #|------------------------|
                #|---- HI Flag Inside ----|
                #|------------------------|    
                # Check if the satellite is in the WF for HI bcs points
                # get weird so at least change color to warn this is the case
                if self.satStuff[didx][pidx]['OBSTYPE'] == 'HI':
                    # Get max wf R
                    myPos = self.satStuff[didx][pidx]['POS']
                    myR = myPos[2] / 7e8
                    pts = wfs[i].points
                    maxR = np.max(np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2))
                    
                    # Get AW
                    AW = None
                    if 'AW (deg)' in wfs[i].labels:
                        AWidx = np.where(wfs[i].labels == 'AW (deg)')
                        AW = wfs[i].params[AWidx][0]
                    elif 'AW_FO (deg)' in wfs[i].labels:
                        AWidx = np.where(wfs[i].labels == 'AW_FO (deg)')
                        AW = wfs[i].params[AWidx][0]
                    # add slab options
                    
                    # Rough check if inside
                    if AW:
                        if myR < maxR:
                            satLon, wflon = myPos[1], wfs[i].params[1]
                            if np.abs(satLon - wflon) < AW:
                                myColor = '#C81CDE'
                                
                #|------------------------|
                #|---- Project Points ----|
                #|------------------------| 
                allxs = []
                allys = []           
                for jj in toShow:
                    # Convert Cart to Sph
                    pt = wfs[i].points[jj,:]
                    r = np.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2)
                    lat = np.arcsin(pt[2]/r) * 180/np.pi
                    lon = np.arctan2(pt[1],pt[0]) * 180 / np.pi
                    pt = [lat, lon, r*7e8]
                    if flatEUV:
                        pt = [lat, lon, 7e8]
                    
                    # WISPR outer (at least) has issues in pts projection when CME
                    # is behind the satellite. The projection code matches IDL so unclear
                    # what the original issue is but not a porting issue. Work around
                    # by just checking if a point is behind the sat lon   
                    if 'WISPR' in self.satStuff[didx][pidx]['MYTAG']:  
                        dLon = lon - myPos[1]
                        if dLon < -180:
                            dLon +=360
                        if dLon > 0:
                            myPt = pts2proj(pt, obs, obsScl, mywcs,  occultR=occultR)
                        else:
                            myPt = []
                            
                    # Just calc all non wispr cases        
                    else:
                        myPt = pts2proj(pt, obs, obsScl, mywcs, occultR=occultR)
                    
                    
                    # If the point is in the FoV add it to draw    
                    if len(myPt) > 0:   
                        allxs.append(myPt[0][0])      
                        allys.append(myPt[0][1])       
                        #pos.append({'pos': [myPt[0][0], myPt[0][1]], 'pen':{'color':myColor, 'width':penwid}, 'brush':pg.mkBrush(myColor)})
 
                #|---- Build the points ----|
                allxs = np.array(allxs)   
                allys = np.array(allys)   
                skipit = 1
                if len(allxs) > 0:
                    if np.sqrt(np.std(allxs)**2 + np.std(allys)**2) < 40:
                        skipit = 2
                        penwid = penwid/2
                for jj in range(len(allxs))[::skipit]:
                    pos.append({'pos': [allxs[jj], allys[jj]], 'pen':{'color':myColor, 'width':penwid}, 'brush':pg.mkBrush(myColor)})
                
                #|---- Reset the scatters to new positions ----|        
                self.scatters[i].setData(pos)
 
    def plotBackground(self):
        """
        Function for updating the background image 
        
        """
        #|---- Grab data for this time/scaling ----|
        if type(curSet) != type(None):
            didx = curSet[self.instTag][0][self.tslIdx]
            sclidx = curSet[self.instTag][1][self.tslIdx]
            slMin = curSet[self.instTag][2][self.tslIdx]
            slMax = curSet[self.instTag][3][self.tslIdx]
            
        else:
            didx = 0
            sclidx = 0
            slMin = self.MinSlider.value()
            slMax = self.MaxSlider.value()
        myIm = self.myScls2[didx][self.pickIdx][sclidx]
        
        #|---- Update image ----|     
        self.image.updateImage(image=myIm, levels=(slMin, slMax))
        
        #|---- Show mass contour ----|   
        if self.nowMass: 
            bigMask = np.zeros(self.mIms[self.pickIdx].shape, dtype=float)
            for i in range(nwfs):
                if type(self.WFmasks[i]) != type(None):
                    bigMask += self.WFmasks[i]
                    seps = np.abs(wfs[i].params[1]-self.satStuff[didx][self.pickIdx]['POSLON'])
                    seps[np.where(seps > 90)] = 180 - seps[np.where(seps > 90)]
                    mySep = np.min(np.abs(seps))
                    if mySep > 80:
                        print ('!!!--- Warning PoS separation large, capping at 80 deg ---!!!')
                        mySep = 80                                            
                    # Need to convert the h to projected for elTheory
                    rpos, Bpos = wM.elTheory([wfs[i].params[0]]*np.cos(mySep*np.pi/180.), 0)
                    rsep, Bsep = wM.elTheory([wfs[i].params[0]]*np.cos(mySep*np.pi/180.), mySep)
                    sclfct = Bsep / Bpos
                    print ((self.satName + ' PoS WF' + str(i+1) + ' mass (g): ').rjust(50) + "{:.3e}".format(np.sum(self.WFmasks[i]* self.mIms[self.pickIdx])))
                    print ((self.satName + ' deProj WF' + str(i+1) + ' mass (g): ').rjust(50) + "{:.3e}".format(np.sum(self.WFmasks[i]* self.mIms[self.pickIdx]/sclfct)), ' (scale factor ', "{:.1f}".format(1/sclfct[0]), ')')
                print ('')
            self.MCimage.updateImage(image= bigMask, opacity=0.5, levels=(0,nwfs-0.5))            
        else:
            self.MCimage.updateImage(image= self.WFmasks[0], opacity=0.0, levels=(0,1))
        
        #|---- Draw stuff on top ----|     
        if self.satStuff[didx][self.pickIdx]['OBSTYPE'] != 'EUV':
            #|---- Circle at 1 Rs ----|     
            if 'SUNCIRC' in self.satStuff[didx][self.pickIdx]:
                if self.pWindow_circle:
                    self.pWindow.removeItem(self.pWindow_circle)
                self.pWindow_circle = self.pWindow.plot(self.satStuff[didx][self.pickIdx]['SUNCIRC'][0], self.satStuff[didx][self.pickIdx]['SUNCIRC'][1])
            
            #|---- Line for Solar N ----|         
            if 'SUNNORTH' in self.satStuff[didx][self.pickIdx]:
                if self.pWindow_north:
                    self.pWindow.removeItem(self.pWindow_north)
                self.pWindow_north = self.pWindow.plot(self.satStuff[didx][self.pickIdx]['SUNNORTH'][0], self.satStuff[didx][self.pickIdx]['SUNNORTH'][1], symbolSize=3, symbolBrush='w', pen=pg.mkPen(color='w', width=1))
                
        #|---- Add time labels ----|     
        self.time_label.setText(self.satStuff[didx][self.pickIdx]['DATEOBS'][:-3])
        # Make slider highlighted so key shortcuts work
        if 'mainwindow' in globals():
            tabIndex = mainwindow.tab_widget.currentIndex()
            mainwindow.Tsliders[tabIndex].setFocus()
            

# |------------------------------------------------------------|
# |------------------ Overview Window Class -------------------|
# |------------------------------------------------------------|
class OverviewWindow(QWidget):
    """
    Class for the overview window showing the relative satellite locations
    and the direction of each wireframe
    
    Inputs:
        satStuff: an array of all the satStuff dictionaries for all sats
    
    Optional Inputs:
        screenXY: size of the computer display in pixels [x,y]. 
                  used to help place windows
     
    """
    def __init__(self, satStuff, screenXY=None):
        """
        Class for the overview window showing the relative satellite locations
    
        Inputs:
            satStuff: an array of all the satStuff dictionaries for all sats
        
        Optional Inputs:
            screenXY: size of the computer display in pixels [x,y]. 
                      used to help place windows
     
        """
        super().__init__()
        
        #|---- Set up/name window ----|
        if type(screenXY) != type(None):
            # The positioning is being odd, might be bc testing with multiple
            # monitors, semi giving up on nice pos for now
            self.setGeometry( int(0.8*screenXY[0]),screenXY[1] , 400, 400) 
        self.setFixedSize(400, 400) 
        self.setWindowTitle('Polar View')
        self.satStuff = satStuff
        
        #|---- Make a layout ----|
        layoutOV =  QGridLayout()
        
        #|---- Make a plot widget ----|
        self.pWindow = pg.PlotWidget()
        self.pWindow.setMinimumSize(350, 350)
        self.pWindow.setRange(xRange=(-1.2, 1.2), yRange=(-1.2,1.2), padding=0)
        layoutOV.addWidget(self.pWindow,0,0,11,11,alignment=QtCore.Qt.AlignCenter)
        
        #|---- Create the sun and earth ----|
        twopi = np.linspace(0, 2.01*np.pi, 200)
        x_data = np.cos(twopi)
        y_data = np.sin(twopi)
        self.pWindow.plot(x_data, y_data, pen=pg.mkPen('w', width=1))
        self.pWindow.plot([0], [0], symbol='o', symbolSize=10, symbolBrush=pg.mkBrush(color='y'))
        self.pWindow.plot([0], [-1], symbol='o', symbolSize=10, symbolBrush=pg.mkBrush(color='blue'))
        
        #|---- Hide the axes ----|
        self.pWindow.hideAxis('bottom')
        self.pWindow.hideAxis('left')
        
        
        #|---- Set up scatters for sats ----|
        self.scatters = []
        self.scattersPt = []
        self.satLabs = []
        self.satStrings = []
        self.satLatStrs = []
        self.satxys = []
        self.curves = []
        self.fbis = []
        L1counter = 0
        
        # |--- Loop through the satellites ---|
        for i in range(nSats):
            #|---- Get a proj sat loc ----|
            pidx = pws[i].pickIdx
            myPos = satStuff[i][0][pidx]['POS']
            myName = satStuff[i][0][0]['OBS']
            myR = myPos[2] / 1.496e+11 
            myLon = myPos[1] * np.pi / 180.
            y = - myR * np.cos(myLon)
            x = myR * np.sin(myLon)
            
            #|---- Make a scatter and set loc ----|
            aScat = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='w'), brush=pg.mkBrush(color='w'),symbol='o', size=8)
            self.scatters.append(aScat)
            pos = []
            pos.append({'pos': [x,y]})
            self.scatters[i].setData(pos)
            self.pWindow.addItem(aScat)
            
            #|---- Add field of views ----|
            myPoint = satStuff[i][0][pidx]['POINTING'][1]
            xPt = myPoint[1] 
            yPt = -myPoint[0]
            curve1 = self.pWindow.plot([x, xPt], [y, yPt],pen=pg.mkPen('w', width=0.5))
            myPoint = satStuff[i][0][pidx]['POINTING'][2]
            xPt = myPoint[1] 
            yPt = -myPoint[0]
            curve2 = self.pWindow.plot([x, xPt], [y, yPt],pen=pg.mkPen('w', width=0.5))
            
            fbi = pg.FillBetweenItem(curve1, curve2, brush=(100, 100, 255, 75)) 
            self.curves.append([curve1, curve2])
            self.fbis.append(fbi)
            self.pWindow.addItem(fbi)
                       
            #|---- Label each sat ----|
            # Labeling sats not insts to avoid overload
            myName = satStuff[i][0][0]['SHORTNAME']
            
            if myName not in self.satStrings:
                text_item = pg.TextItem(myName, anchor=(0.5, 0.5))
                # inner cases
                if myR < 0.8:
                    xsat = myR * np.sin(myLon) + 0.15
                    ysat =  -myR * np.cos(myLon)
                # L1 cases
                elif np.abs(myLon) < np.pi / 6:
                    xsat = myR * np.sin(myLon)
                    if L1counter == 0:
                        ysat = -1.1
                    else:
                        ysat = -0.95 + L1counter * 0.1
                    L1counter +=1
                # Other cases prob ok with this?               
                else:
                    ysat = - 0.85*myR * np.cos(myLon)
                    xsat = 0.85*myR * np.sin(myLon)
                text_item.setPos(xsat, ysat)
                self.satLabs.append(text_item)
                self.pWindow.addItem(text_item)
                self.satStrings.append(myName)
                self.satLatStrs.append(myName.rjust(5)+':'+'{:.1f}'.format(myPos[0]).rjust(5))
                self.satxys.append([xsat, ysat])
        
        #|---- Set up arrows/points for the WF in ovw ----|
        self.arrows = []
        self.wfScats = []
        for i in range(nwfs):
            arrow = pg.ArrowItem(angle=-45, tipAngle=0, headLen=0, tailLen=0, tailWidth=0, pen={'color': 'w', 'width': 2}, brush='b')
            arrow.setPos(0, 0)
            self.pWindow.addItem(arrow)
            self.arrows.append(arrow)
            
            wfScat = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(color='g'),symbol='o', size=2)
            self.wfScats.append(wfScat)
            self.pWindow.addItem(wfScat)
            
        #|---- Add the lat list ----|
        self.latTit = pg.TextItem('Lat (deg)')
        self.pWindow.addItem(self.latTit) 
        self.latTit.setPos(0.75,1.22)
        self.sLSitems = []
        sortSLS = np.sort(self.satLatStrs) # alphabetize for OCD
        for jj in range(len(self.satLatStrs)):      
            self.sLSitems.append(pg.TextItem(sortSLS[jj], anchor=(1,0)))
            self.pWindow.addItem(self.sLSitems[jj]) 
            self.sLSitems[jj].setPos(1.15,1.12-0.1*jj)
        
        self.setLayout(layoutOV)
    
    def updateArrow(self, i, color='w'):
        """
        Function for updating wf longitude arrow or the
        full wf set of scatter points
        
        Inputs:
            i:      the WF index
        
        Optional Input:
            color:  the color of this WF
     
        """
        #|---- Get the WF lon ----|
        mywf = wfs[i]
        lon  = mywf.params[1]
        h    = mywf.params[0]
        
        #|--- Show arrow if close to Sun ---|
        if h < 10:
            #|---- Get arrow head loc ----|
            rlon = lon * np.pi /180.
            hL, tL = 0.1, 0.3 # head length, tail length
            aL = hL+tL
            xh = aL * np.sin(rlon)
            yh = -aL * np.cos(rlon)
            ang = -np.arctan2(yh, xh) * 180 / np.pi
        
            #|---- Update the arrow ----|
            self.arrows[i].setStyle(angle=lon-270, headWidth=0.04, headLen=hL, tailLen=tL, tailWidth=0.02, pxMode=False,  pen={'color': color, 'width': 2}, brush=color)
            tail_len = self.arrows[i].opts['tailLen']
            self.arrows[i].setPos(xh, yh)
            self.wfScats[i].setData([], [])
        
        #|--- Otherwise full wf projection ---|
        else:            
            xs = -wfs[i].points[::2,0] / 215.
            ys = wfs[i].points[::2,1] / 215.
            self.wfScats[i].setData(ys, xs)
            self.wfScats[i].setBrush(color)
            # Turn off arrow
            self.arrows[i].setStyle(angle=0, headWidth=0, headLen=0, tailLen=0, tailWidth=0., pxMode=False)
            self.arrows[i].setPos(0, 0)
    
    def updateFoV(self):
        """
        Function for updating the fields of view. It just pulls
        the values for the new time index from the satstuff 
        dictionary and updates the curves in the fill between object
        
        Inputs:
            None
        
     
        """
        for i in range(nSats):
            #|---- Get a proj sat loc ----|
            myPos = self.satStuff[i][0][pws[i].pickIdx]['POS']
            myName = self.satStuff[i][0][0]['OBS']
            myR = myPos[2] / 1.496e+11 
            myLon = myPos[1] * np.pi / 180.
            y = - myR * np.cos(myLon)
            x = myR * np.sin(myLon)
            
            #|--- Set satellite location ---|
            pos = [{'pos': [x,y]}]
            self.scatters[i].setData(pos)
            
            #|--- Set the field of view curves ---|
            myPoint = self.satStuff[i][0][pws[i].pickIdx]['POINTING'][1]
            xPt = myPoint[1] 
            yPt = -myPoint[0]
            self.curves[i][0].setData([x, xPt], [y, yPt])
            myPoint = self.satStuff[i][0][pws[i].pickIdx]['POINTING'][2]
            xPt = myPoint[1] 
            yPt = -myPoint[0]
            self.curves[i][1].setData([x, xPt], [y, yPt])
                    
    def keyPressEvent(self, event):
        """
        Event for key press events. This just tosses the event
        to the main window key press function.
        
        Actions (based on key):
            return = replot (pulls out of param text box)
            q      = close a window
            esc    = close everything
            left   = move time slider to earlier time
            right  = move time slider to later time
            b      = switch this time to base difference
            r      = switch this time to running difference
            s      = save figs for this time 
            l      = log current time wf/background params
            m      = calculate mass
            h      = show/hide wfs
            1      = switch this time to linear scaling
            2      = switch this time to log scaling
            3      = switch this time to sqrt scaling
            
            # Shift actions (shift + key) - do it for everyone
            B      = switch all times to base difference
            R      = switch all times to running difference
            S      = save figures for all times (this will be slowish)
            L      = log all times (using the paramLog/curSet)
            1      = switch all times to linear scaling
            2      = switch all times to log scaling
            3      = switch all times to sqrt scaling
            7 (&)  = propagate WF params back in time
            8 (*)  = propagate WF params forward in time
            9 (()  = propagate min/max values back in time
            0 ())  = propagate min/max values forward in time
     
        """
        if 'mainwindow' in globals():
            mainwindow.keyPressEvent(event)

           
# |------------------------------------------------------------|
# |----------------- Setup up satStuff dicts ------------------|
# |------------------------------------------------------------|
def getSatStuff(imMap):
    """
    Function to make a header like structure with keywords and 
    values but specific to what wombat desires
    
    Inputs:
        imMap:     a single observation map
           
    Outputs:
        satDict:   a dictionary with useful satellite information
                   the keys are as follows:
                        OBS:       observatory/satellite name
                        INST:      instrument name
                        MYTAG:     nice name string with obs+inst
                        OBSTYPE:   type of observation (EUV, COR, HI)
                        WAVE:      wavelength in angstroms (only for EUV)
                        SHORTNAME: shorter version of MYTAG
                        DATEOBS:   string for the date as YYYYMMDDTHH:MM:SS
                        POS:       postion of sat [lat, lon, R] in [deg, deg, m]
                        POINTING:  unit vector pointing from sat to sun center (in Stony Cart)
                        POSLON:    longitudes of the plane of sky (in equitorial plane, Stony deg)
                        SCALE:     plate scale in arcsec/pix (or deg/pix for HI)
                        CRPIX:     pixel location of the reference pixel
                        WCS:       wcs structure
                        SUNPIX:    pixel location of the center of the sun
                        ONERSUN:   one solar radius in pixels
                        FOV:       maximum radial distance of a corner in Rs
                        MASK:      array with masked pixels set to 1 (0 elsewhere)
                        OCCRPIX:   radius of occulter in pixels
                        OCCRARC:   radius of occulter in arcsecs
                        SUNCIRC:   array with the xy pixels for an outline of the sun
                        SUNNORTH:  array with xy pixels to indicate solar north direction
                        MYFITS:    the name/path of the original fits file
                        DIFFFITS:  the name/path of the file used to calc a difference img
    
    External Calls:
        fitshead2wcs from wcs_funs                       
    
    """
    #|---- Initialize dictionary ----|
    satDict = {}
    
    # |-----------------|
    # |---- Get OBS ----|
    # |-----------------|
    # Pull the hdr/map metadata
    myhdr   = imMap.meta   
    satDict['OBS'] =  myhdr['obsrvtry']
   
    # |------------------|
    # |---- Get INST ----|
    # |------------------|
    # All the satellites have different options on saving
    # instrument names vs detector names. Combine into a single
    # INST tag for wombat
    # PSP format
    if myhdr['obsrvtry'] == 'Parker Solar Probe':
        satDict['OBS'] =  myhdr['obsrvtry']
        satDict['INST'] =  myhdr['instrume'] + '_HI' + str(myhdr['detector'])
        myTag   = myhdr['obsrvtry'] + '_' + myhdr['instrume'] + '_HI' + str(myhdr['detector'])
    # SolO format
    elif myhdr['obsrvtry'] == 'Solar Orbiter':
        satDict['OBS'] =  myhdr['obsrvtry']
        satDict['INST'] = myhdr['instrume'] 
        myTag   = myhdr['obsrvtry'] + '_' + myhdr['instrume']
    elif myhdr['telescop'] == 'STEREO':
        satDict['OBS'] =  myhdr['obsrvtry'] 
        satDict['INST'] = myhdr['instrume'] + '_' + myhdr['detector']
        myTag   = myhdr['telescop'] + '_' + myhdr['instrume'] + '_' + myhdr['detector']
    elif myhdr['obsrvtry'] == 'SDO':
        satDict['OBS'] =  myhdr['obsrvtry'] 
        satDict['INST'] = myhdr['detector']
        myTag   = myhdr['obsrvtry'] + '_' + myhdr['detector']
    # Other less picky sats
    else:
        satDict['OBS'] =  myhdr['telescop']
        satDict['INST'] = myhdr['instrume'] + '_' + myhdr['detector']
        myTag   = myhdr['telescop'] + '_' + myhdr['instrume'] + '_' + myhdr['detector']
    satDict['MYTAG'] = myTag
     
    # |---------------------|
    # |---- Get OBSTYPE ----|    
    # |---------------------|
    # Flag between HI, COR, EUV
    if satDict['OBS'] in ['Parker Solar Probe', 'Solar Orbiter']:
        satDict['OBSTYPE'] = 'HI'
    elif satDict['OBS'] in ['STEREO_A', 'STEREO_B']:
        if myhdr['detector'] in ['COR1', 'COR2']:
            satDict['OBSTYPE'] = 'COR'
        elif myhdr['detector'] in ['HI1', 'HI2']:
            satDict['OBSTYPE'] = 'HI'
        else:
            satDict['OBSTYPE'] = 'EUV'
    elif satDict['OBS'] == 'SOHO':
        if myhdr['detector'] in ['C2', 'C3']:
            satDict['OBSTYPE'] = 'COR'
        else:
            satDict['OBSTYPE'] = 'EUV'
    elif satDict['OBS'] == 'SDO':
         satDict['OBSTYPE'] = 'EUV'
    
    # |------------------|
    # |---- Get WAVE ----|   
    # |------------------|
    # Add the wavelength if EUV
    if satDict['OBSTYPE'] == 'EUV':
        satDict['WAVE'] = str(myhdr['WAVELNTH'])
        satDict['MYTAG'] = satDict['MYTAG'] + '_' + satDict['WAVE']
            
    # |-----------------------|
    # |---- Get SHORTNAME ----|    
    # |-----------------------|
    shortNames = {'Parker Solar Probe':'PSP', 'Solar Orbiter':'SolO', 'STEREO_A':'STA', 'STEREO_B':'STB', 'SOHO':'SOHO', 'SDO':'SDO'}
    satDict['SHORTNAME'] = shortNames[satDict['OBS']]
    
    
    # |---------------------|
    # |---- Get DATEOBS ----|    
    # |---------------------|
    if len(myhdr['date-obs']) > 13:
        satDict['DATEOBS'] = myhdr['date-obs']
    else:
        satDict['DATEOBS'] = myhdr['date-obs'] + 'T' + myhdr['time-obs']
    satDict['DATEOBS'] = satDict['DATEOBS'].replace('/','-')
    if '.' in satDict['DATEOBS']:
        dotidx = satDict['DATEOBS'].find('.')
        satDict['DATEOBS'] = satDict['DATEOBS'][:dotidx]
     
    # |-----------------|
    # |---- Get POS ----|    
    # |-----------------|
    # Get satellite position
    obsLon = imMap.observer_coordinate.lon.degree
    obsLat = imMap.observer_coordinate.lat.degree
    obsR = imMap.observer_coordinate.radius.m
    satDict['POS'] = [obsLat, obsLon,  obsR]
    # Get sat to sun direction
    latd = obsLat * np.pi / 180.
    lond = obsLon * np.pi / 180.
    xyz = [np.cos(latd)*np.cos(lond), np.cos(latd)*np.sin(lond), np.sin(latd)]
    satDict['POINT2SUN'] = -np.array(xyz)
    pointLon = np.arctan2(satDict['POINT2SUN'][1], satDict['POINT2SUN'][0]) * 180 / np.pi
    PoSlon1 = (pointLon - 90) % 360
    PoSlon2 = (pointLon + 90) % 360
    satDict['POSLON'] = [PoSlon1, PoSlon2]
    
    # |-------------------|
    # |---- Get SCALE ----|    
    # |-------------------|
    # Plate scale in arcsec/pix for EUV/COR
    # or deg/pix for HI
    # Check to make sure same in x/y since we will assume as much
    if (imMap.scale[0].to_value() != imMap.scale[1].to_value()):
        sys.exit('xy scales not equilent. Not set up to handle this. Exiting from getSatStuff')    
    obsScl  = imMap.scale[0].to_value()
    satDict['SCALE'] = obsScl
    
    # |-------------------|
    # |---- Get CRPIX ----|    
    # |-------------------|
    # Reference pixel
    cx,cy = int(myhdr['crpix1'])-1, int(myhdr['crpix2'])-1
    satDict['CRPIX'] = [cx, cy]
    
    # |-----------------|
    # |---- Get WCS ----|    
    # |-----------------|
    myWCS = fitshead2wcs(myhdr)
    satDict['WCS'] = myWCS
    
    # |--------------------|
    # |---- Get SUNPIX ----|    
    # |--------------------|
    centS = wcs_get_pixel(myWCS, [0.,0.])
    sx, sy = centS[0], centS[1]
    satDict['SUNPIX'] = [sx, sy]
    
    # |---------------------|
    # |---- Get ONERSUN ----|    
    # |---------------------|
    # Get 1 Rs in pix
    if 'rsun' in imMap.meta:
        myRs = imMap.meta['rsun'] # in arcsec
    else:
        myDist = imMap.observer_coordinate.radius.m / 7e8
        myRs   = np.arctan2(1, myDist) * 206265
    oners = myRs/imMap.scale[0].to_value()
    if imMap.scale[0].unit == 'deg / pix':
        oners = oners / 3600
    satDict['ONERSUN'] = oners
    
    # |-----------------|
    # |---- Get FOV ----|    
    # |-----------------|
    # Get maximum radial distance of the corners
    myFOV = 0
    for i in [0,imMap.data.shape[0]-1]:
        for j in [0,imMap.data.shape[1]-1]:
            coord = wcs_get_coord(myWCS, pixels = np.array([i,j]))
            edgeR = np.sqrt(coord[0]**2 + coord[1]**2)
            thisFOV = edgeR / obsScl / oners
            if thisFOV > myFOV: myFOV = thisFOV
    satDict['FOV'] = myFOV
    
    # |---------------------------|
    # |---- Add Occulter Mask ----|    
    # |---------------------------|
    # Make mask array
    mask = np.zeros(imMap.data.shape)
    # Check that not SolO/PSP or STEREO HI
    if (satDict['OBSTYPE'] != 'HI'):   
        myOccR  = occultDict[myTag][0] # radius of the occulter in Rs
        occRpix = int(myOccR * oners)
        # Add radius of occulter in pix and arcsecs
        satDict['OCCRPIX'] = myOccR * oners
        satDict['OCCRARC'] = myOccR * oners * imMap.scale[0].to_value()
        
         # Fill in a circle around the occulter center
        for i in range(occRpix):
            j = int(np.sqrt(occRpix**2 - i**2))
            lowY = np.max([0,cy-j])
            hiY  = np.min([imMap.meta['naxis2']-2, cy+j])
            if cx+i <= imMap.meta['naxis2']-1:
                mask[cx+i, lowY:hiY+1] = 1
            if cx-i >=0:
                mask[cx-i, lowY:hiY+1] = 1    
    
        # Fill in outside FoV
        outRpix = int(occultDict[myTag][1] * oners) 
        for i in range(imMap.meta['naxis1']):
            myHdist = np.abs(cx-i)
            if myHdist >= outRpix:
                mask[i,:] = 1
            else:
                possY = int(np.sqrt(outRpix**2 - myHdist**2))
                lowY = np.max([0,cy - possY])
                hiY  = np.min([imMap.meta['naxis2'],cy + possY])
                mask[i,:lowY+1] = 1
                mask[i,hiY:] = 1
                
        # Add to dict
        satDict['MASK'] = mask
        
        # |-------------------------|
        # |---- Add Sun Outline ----|    
        # |-------------------------|
        thetas = np.linspace(0, 2.1*3.14159,100)
        xs = oners * np.cos(thetas) + sx
        ys = oners * np.sin(thetas) + sy
        satDict['SUNCIRC'] = [xs, ys]
        
        # |--------------------------|
        # |---- Add Solar N Line ----|    
        # |--------------------------|
        # Can use slow skycoord/map version bc only call once
        skyPt = SkyCoord(x=0, y=0, z=1, unit='R_sun', representation_type='cartesian', frame='heliographic_stonyhurst')
        myPt2 = imMap.world_to_pixel(skyPt)
        satDict['SUNNORTH'] = [[sx, myPt2[0].to_value()], [sy, myPt2[1].to_value()]]

    # |----------------------------|
    # |---- Add fits file/path ----|    
    # |----------------------------|
    satDict['MYFITS'] = myhdr['myFits']
    satDict['DIFFFITS'] = myhdr['diffFile']

    return satDict
    
# |------------------------------------------------------------|
# |----------------- Project points onto map ------------------|
# |------------------------------------------------------------|
def pts2proj(pts_in, obs, scale, mywcs, occultR=None):
    """
    Function to take a series of points in spherical coordinates and return the
    pixel locations on a map. This function pulls the appropriate information
    then passes it to wcs_get_pixels to perform the calculation
    
    Right now wombat only calls it using a single point at a time but it was 
    probably set up to handle an array of points but hasn't been tested for that
    in some time
    
    Inputs:
        pts_in:  a point (or list of points as npts x 3) in the form [lat, lon, r]
                 where the units are [deg, deg, x]. We advocate using Stonyhurst
                 (Earth at longitude 0) and meters for the radius but in theory
                 as long as the pts_in and obs are in the same coordinates it 
                 should work 
                 * note that pts are lat/lon order while scale is x/y
    
        obs:     an array for the observer location in the form [lat, lon, r]
                 where the units are [deg, deg, x]. We advocate using Stonyhurst
                 (Earth at longitude 0) and meters for the radius but in theory
                 as long as the pts_in and obs are in the same coordinates it 
                 should work 
    
        scale:   conversion factor ang/pix. It can be either a single value that
                 is the same for both xy or an array of [scalex, scaley]. ang should
                 be in arcsec/pix for EUV/COR or deg/pix for HI
                 * note that scale is x/y order while pts are lat/lon
    
        mywcs:   a wcs structure (converted from header prev via fitshead2wcs)

    Optional Inputs:
        occultR: radius of the inner occulter in the same angular unit as scale

           
    Outputs:
        outs:   the projected x,y pixel position(s) corresponding to pts_in
                packages as [[x1, y1], [x2, y2], ...]
    
    External calls:
        wcs_get_pixels from wcs_funcs
        
    """
   
    #|---- Useful constants ----|
    rad2arcsec = 206265
    dtor = np.pi / 180.
    
    # |----------------------------|
    # |---- Check input format ----|    
    # |----------------------------|
    # Reformat as 2D array if single pt to keep
    # the calculation consistent
    if isinstance(pts_in, list):
        pts_in = np.array(pts_in)
    
    if len(pts_in.shape) == 1:
        pts_in = np.array([pts_in])
     
    # |--------------------------------------|
    # |---- Convert to rotated Cartesian ----|    
    # |--------------------------------------|    
    
    #|---- convert pts to radians ----|
    pts_lats = pts_in[:,0]*dtor 
    pts_lons = pts_in[:,1]*dtor 
    pts_rs   = pts_in[:,2]

    #|---- convert obs to radians ----|
    obs_lat = obs[0]*dtor
    obs_lon = obs[1]*dtor
    obs_r   = obs[2]
    
    #|---- lon dif ----|
    dLon = pts_lons - obs_lon
    
    #|---- Actual conversion ----|
    # Convert from Stony heliographic (or something similar) to heliocentric cartesian
    # this is with x to right, y up, z toward obs
    x = pts_rs  * np.cos(pts_lats) * np.sin(dLon)
    y = pts_rs * (np.sin(pts_lats) * np.cos(obs_lat) - np.cos(pts_lats)*np.cos(dLon)*np.sin(obs_lat))
    z = pts_rs * (np.sin(pts_lats) * np.sin(obs_lat) + np.cos(pts_lats)*np.cos(dLon)*np.cos(obs_lat))
            
    # |----------------------------------|
    # |---- Cartesian to Proj Angles ----|    
    # |----------------------------------|
    d = np.sqrt(x**2 +  y**2 + (obs_r-z)**2)
    if mywcs['cunit'][0].lower() == 'arcsec':
        rad2unit = rad2arcsec
    elif mywcs['cunit'][0].lower() == 'deg':
        rad2unit = 180. / np.pi
    dthetax = np.arctan2(x, obs_r - z) * rad2unit 
    dthetay = np.arcsin(y/d)* rad2unit  
    
    # |---------------------------------|
    # |---- Pixels from WCS Routine ----|    
    # |---------------------------------|
    coord = wcs_get_pixel(mywcs, [dthetax, dthetay], doQuick=False)
    # Repackage
    thetax, thetay = coord[0,:], coord[1,:]
    
    # |-----------------------|
    # |---- Add Occulting ----|    
    # |-----------------------|    
    # Check if we want to throw out the points that would be behind the occulter   
    if occultR: 
        dProj = np.sqrt(dthetax**2 +  dthetay**2)
        outs = []
        for i in range(len(d)):
            if (dProj[i] > occultR) or (z[i] > 0):
                outs.append([thetax[i], thetay[i]])
        outs = np.array(outs)
    else:
        # repackage as array of [ [pixX1, pixY1], [pixX2, pixY2], ...]
        outs = np.array([thetax, thetay]).transpose()
    return outs

# |------------------------------------------------------------|
# |--------------- Set up the state variables -----------------|
# |------------------------------------------------------------|
def buildMegaVars(rD, tlabs, tmaps, satNames):
    """
    Function to reload the GUI from a save file 
    
    Inputs:
       rD:  a reload dictionary from log lines processed by
            wombatWrappper. Will be an empty structure that passes
            through fine if starting from scratch instead of reload
    
    Effects:
            Sets up the three main variables used to represent the
            current state of the plot settings and wf parameters and
            track the previous plot settings for different background
            diff/scl settings. These variables are
    
            paramLog: The wireframe parameters. This is a dictionary with keys for every possible
                      combination of wftype and wf index number (e.g. 'GCS2'), which allows for
                      duplicates of a type. The syntax is paramLog[type#][parameter_index][time_index]
                      where the parameter_index matches the order of the labels in the GUI (top to bottom)
                      and the time index corresponds to the universal time index (not pickle index). If a
                      type# has not be used or reloaded then the values will be set to None. As a user fits
                      the wf the values are stored for each type/time and can be reloaded if one switches 
                      away from one type/time then comes back to it
        
            setLog:   A log of previously used aesthetics of the background. This is a dictionary with keys 
                      corresponding to the instrument tag of each plot window which is used to track the most
                      recently used values for the min/max sliders for each combo of difference mode, scaling
                      mode, and time. The syntax is setLog[instTag][diffIndex][scaleIndx][min/max][time_idx] where 
                      diffIndex is 0 for running and 1 for base, scale is 0, 1, 2 for linear, log, sqrt, and
                      min/max is 0 for min and 1 for max. This is populated with the default values at the start 
                      and logs changes to be reloaded if this diff/scale/time is revisited
    
    
            curSet:   The current settings of the aesthetics. This is a dictionary with keys corresponding to the 
                      plot window instrument tag. The syntax is curSet[instTag][set_idx][time_idx] where set_idx
                      0 gives the difference mode, 1 gives the scaling mode, 2 gives the min slider value, 4 gives
                      the max slider value, and time_idx is universal time. This is the current state of the 
                      system and allows to cycle through different diff types and scalings across different times.
            
        
    """
    
    # Reload dict has
    # ['Params'][aWF][aTime] = [params]
    # ['Pidx'][aInst][aTime] = pickle index -> no exist now?
    # ['PlotVals'][aInst][aTime] = [scale type, diff type, min, max]
    
    # tmaps -> slider idx to pickle idx
    sliDts = [datetime.datetime.strptime(atime, "%Y-%m-%dT%H:%M") for atime in tlabs]
    sliDeltas = np.array([(atime - sliDts[0]).total_seconds() for atime in sliDts])
    
    # Repackage the reload params by slider time index
    reloadParams = {}
    nsli = len(tmaps['t2p'][0])
    
    # Allow for passing of None for rD, just make a fully blank log
    if type(rD) != type(None):
        hasRD = True
        wfs = np.array([str(key) for key in rD['Params']])
        # check if have a 2 in the tag but only using single ef
        # higher number mess ups ignored
        if len(wfs) == 1:
            if wfs[0][:-1] != '1':
                newName = wfs[0][:-1]+'1'
                rD['Params'][newName] = rD['Params'][wfs[0]]
                wfs = [newName]
    else:
        hasRD = False
        wfs = []
        rD = {}
        rD['PlotVals'] = []
    
    # Create all the possible wftype + window number tags
    allNames = {} # both a useful list of names and a dict for the param #s!
    for aWF in wf.npDict:
        for i in range(nwfs):
            allNames[aWF + str(i+1)] = wf.npDict[aWF]
    
    #|-------------------------|
    #|---- Fill in paramLog ---|
    #|-------------------------|
    # Use reload if we have it
    # otherwise fill with Nones to replace later
    allt0s = [] # all the earliest times (if we have a reload)
    paramLog = {} # indexing [WFtype#][param idx][tidx]
    noneArr = [None for i in range(nsli)]
    # Loop through every combo of WFtype + WFnum
    for aName in allNames:
        nParams = wf.npDict[aName[:-1]]
        paramLog[aName] = [np.copy(noneArr) for i in range(nParams)]
        
        if aName in wfs:
            myTimes = np.array([str(key) for key in rD['Params'][aName]])
            wfDts = [datetime.datetime.strptime(key, "%Y-%m-%dT%H:%M") for key in myTimes]
            allt0s.append(wfDts[0])
            wfDeltas = np.array([(atime - sliDts[0]).total_seconds() for atime in wfDts])
            # For each slider time find closest wf time
            for i in range(nsli):
                myDiffs = np.abs(wfDeltas - sliDeltas[i])
                myMatch = np.where(myDiffs == np.min(myDiffs))[0][0]
                #reloadParams[aName].append(rD['Params'][aName][myTimes[myMatch]])
                for ii in range(nParams):
                    paramLog[aName][ii][i] = rD['Params'][aName][myTimes[myMatch]][ii]        

    #|----------------------------------|
    #|---- Set up setLog at defaults ---|
    #|----------------------------------|
    setLog = {}
    tOnes = np.ones(nsli, dtype=int)
    for aInst in instNames:
        setLog[aInst] = [[], []]        
        if ('AIA' in aInst) or ('EUVI' in aInst):
            minVs, maxVs = [0,0,0], [191, 191, 191]
        elif ('COR' in aInst) :
            minVs, maxVs = [63,0,21], [191, 191, 191]
        elif ('C2' in aInst):
            minVs, maxVs = [0,0,21], [191,191,191]
        elif  ('C3' in aInst):
            minVs, maxVs = [37,0,37], [191,191,191]
        elif ('HI1' in aInst) or ('HI2' in aInst):
            minVs, maxVs = [63,0,21], [128,191,191]
        elif ('WISPR' in aInst) or ('Solo' in aInst):
            minVs, maxVs = [0,0,21], [128,191,191]
        else:
            sys.exit('Unknown instrument tag in reload, prob just need to add it')
      
        for didx in [0, 1]:
            setLog[aInst][didx] = [[], [], []]
            for sclidx in [0,1,2]:
                setLog[aInst][didx][sclidx] = np.copy([minVs[sclidx]*tOnes, maxVs[sclidx]*tOnes])    
    
    #|---------------------------------------------|
    #|---- Set up curSet with rD/lin as default ---|
    #|---------------------------------------------|
    curSet = {}    
    for aInst in instNames:
        curSet[aInst] = [np.zeros(nsli, dtype=int), np.zeros(nsli, dtype=int), np.copy(setLog[aInst][0][0][0]), np.copy(setLog[aInst][0][0][1])]
         
    #|---------------------------------|
    #|---- Replace with reload vals ---|
    #|---------------------------------|    
    for aInst in instNames:
        if aInst in rD['PlotVals']:
            myTimes = np.array([str(key) for key in rD['PlotVals'][aInst]])
            plotDts = [datetime.datetime.strptime(key, "%Y-%m-%dT%H:%M") for key in myTimes]
            plotDeltas = np.array([(atime - sliDts[0]).total_seconds() for atime in plotDts])
             
            # For each slider time find closest wf time
            for i in range(nsli):
                myDiffs = np.abs(plotDeltas - sliDeltas[i])
                myMatch = np.where(myDiffs == np.min(myDiffs))[0][0]
                for ii in range(4):
                    curSet[aInst][ii][i] = int(rD['PlotVals'][aInst][myTimes[myMatch]][ii])
                curSet[aInst][1][i] -= 1 
 
    
    #|---------------------------------------------|
    #|---- Set GUI at earliest time from reload ---|
    #|---------------------------------------------|
    myMatch = 0
    if hasRD:
        # Find the earliest time across all insts
        firstDiff = (np.min(allt0s) - sliDts[0]).total_seconds()
        firstdelts = np.abs(sliDeltas-firstDiff)
        myMatch = np.where(firstdelts == np.min(firstdelts))[0][0]
        mainwindow.update_tidx(myMatch+2, myId=0)
        # Set each parameter panel (time and params) 
        for i in range(nwfs):
            aWF = wfs[i]
            WFid = WFname2id[aWF[:-1]]
            mainwindow.cbs[i].setCurrentIndex(WFid)
            for j in range(len(paramLog[aWF])):
                myP = paramLog[aWF][j][myMatch]
                mainwindow.widges[i][0][j].setValue(myP)  
     
    return paramLog, setLog, curSet
        
# |------------------------------------------------------------|
# |-------------------- Setup Time Indices --------------------|
# |------------------------------------------------------------|
def sortTimeIndices(satStuff, tRes=20):
    """
    Function to take sets of disjointed times for (possibly) multiple 
    instruments and assign everyone to the time slider indices
    
    Inputs:
       satStuff: array of satStuff dictionaries for all insts
    
    Optional Inputs:
        tRes:    time slider resolution in minutes
                 defaults to 20 minutes
    
    Returns:
        nTimes:  number of time steps in the time slider
    
        tlabs:   string labels for each step in the time slider
    
        idxMaps: a dictionary with t2p and p2t for maps from t slider
                 idx to pickle idx (and oppo). each dictionary entry
                 is an array for the insts in order as in satStuff
                 the t2p maps are arrays (e.g. [0, 1, 1, 2, 3, 4, 4])
                 where the array index is t and the value is p
                 the p2t maps are dicts (e.g. {0:[0], 1:[1,2], ...})
                 where the key is the p index and the array is all the
                 matching t idxs
    """
    
    # |--------------------------------|
    # |------ Collect Everything ------|    
    # |--------------------------------|
    # Nested array of all times by instrument
    allTimes = []
    # Min/max time for each instrument
    allMins  = []
    allMaxs  = []
    
    #|--- Loop through sats ---|
    for j in range(len(satStuff)):
        aSat = satStuff[j]
        myTimes = []
        for i in range(len(aSat)):
            myTimes.append(datetime.datetime.strptime(aSat[i]['DATEOBS'], "%Y-%m-%dT%H:%M:%S"))
             
        # Check if sorted
        if not all(a <= b for a, b in pairwise(myTimes)):
            sys.exit('Exiting from sortTimeIndices, files should be provided in time order')
            
        # Track earliest/latest times
        allMins.append(myTimes[0])
        allMaxs.append(myTimes[-1])
        allTimes.append(np.array(myTimes))
    
    # |-------------------------------|
    # |------ Check Overlapping ------|    
    # |-------------------------------|    
    # Probably not entirely necessary but want to make sure aren't passed wildly 
    # separate times that will turn the time slider into chaos
    for i in range(len(allMins)):
        for j in range(len(allMins)-1):
            if allMaxs[i] < allMins[j+1]:
                sys.exit('Exiting from sortTimeIndices, observation times should be overlapping')
    
    # |--------------------------------|
    # |------ Set up slider vals ------|    
    # |--------------------------------|
    # Overall min/max and range
    totMin = np.min(allMins)
    totMax = np.max(allMaxs)
    totRng = (totMax - totMin).total_seconds() / 60.
    
    # Determine the number of slider points based on the range and resolution
    nTimes = int(totRng / tRes)+2
    
    # Make a datetime array for the slider times
    DTgeneral = []
    overMin = totMin.minute % tRes
    genDT0  = (totMin - datetime.timedelta(minutes=overMin)).replace(second=0, microsecond=0) 
    tlabs = []
    for i in range(nTimes):
        DTgeneral.append(genDT0 + datetime.timedelta(minutes=(i*tRes)))
        tstr = DTgeneral[-1].strftime("%Y-%m-%dT%H:%M")
        tlabs.append(tstr)
    DTgeneral = np.array(DTgeneral)
    tlabs = np.array(tlabs)
    
    # |-----------------------------------|
    # |------ Match obs2sli indices ------|    
    # |-----------------------------------|
    # For each slider time find the closer match in observations
    # for each instrument. Likely that multiple sli idx may be assigned
    # to the same obs, especially if time cadences dont match well
    t2ps = []
    p2ts = [] 
    p2tBFs = []
    for j in range(len(satStuff)):
        myTimes = allTimes[j]
        t2p  = np.zeros(nTimes, dtype=int)
        # Find the closest pickle time for each slider time
        for i in range(nTimes):
            myDTdiff = np.abs(myTimes-DTgeneral[i])
            mygenidx = np.where(myDTdiff == np.min(myDTdiff))[0]
            t2p[i] = mygenidx[0]
        t2ps.append(t2p)
        
        # Invert this for pickle to time slider
        p2t = {}
        for i in range(np.max(t2p)+1):
            p2t[i] = np.where(t2p == i)[0]
        p2ts.append(p2t)
        
        # Find closest tidx match to each pidx
        p2tBF = {}
        for i in range(len(myTimes)):
            theseDiffs = np.abs(myTimes[i] - DTgeneral[p2t[i]])
            minDiff = np.where(theseDiffs == np.min(theseDiffs))[0]
            p2tBF[i] = p2t[i][minDiff][0]
        p2tBFs.append(p2tBF)
        
    idxMaps = {}
    idxMaps['p2t'] = p2ts
    idxMaps['t2p'] = t2ps
    idxMaps['p2tBF'] = p2tBFs
    
    return nTimes, tlabs, idxMaps

# |------------------------------------------------------------|
# |------------------- Main Launch Function -------------------|
# |------------------------------------------------------------|
def releaseTheWombat(obsFiles, nWFs=1, overviewPlot=False, reloadDict=None, logFile=None, tRes=20, doFigLabs=True):
    """
    Main wrapper function to build and run the WOMBAT GUI

    Inputs:
        obsFiles: nested lists of maps and headers that releaseTheWombat uses to
                  set up the background images in the form [inst1, inst2, ...] 
                  where each insts is an array of [[RDmaps], [BDmaps], [hdrs]] where maps and 
                  hdrs are time series of the obs maps and their corresponding headers 
                  (e.g. [[[COR2A_RDmap1, COR2A_RDmap2, ...], [COR2A_BDmaps..] [COR2Ahdr1, 
                         COR2Ahdr2, ...]] [[C2_RDmaps], [C2_BDmaps], [C2 hdrs]]
                         [[AIA171maps], [AIA171maps] [AIA171hdrs]]])
                  *** note we will pass two sets of EUV maps that are most often the same 
                      since we default to not taking a difference but this makes it easier
                      to run consistent code across all instruments
    Optional Inputs:
        nWFs:         number of wireframes. Currently set an upper limit of 5 to keep
                      GUI from becoming overloaded
                      defaults to 1
    
        overviewPlot: flag to include the polar/top-down overview panel showing the relative
                      locations of the Sun, Earth, satellites, and wireframes
                      defaults to False
          
        reloadDict:   option to pass a reloadDictionary (from processReload in
                      wombatWrapper.py) to relaunch the GUI from a previous state
                      defaults to None (aka no reload)
    
        logFile:      name of the log file used to load a recon. Will be put into the
                      text box in the param window
                      defaults to None
        
        tRes:         time resolution (in mins) to use for the main slider. The pickled data may
                      be in higher or lower resolution but this will be mapped to the slider
                      values (potentially downselecting if data is higher res)
                      defaults to 20 mins
    
        doFigLabs:    flag to include labels when saving figs (instrument name + time) 
                      defaults to True
        

    Outputs:
        No outputs automatically generated. If the save button is hit then figures will be saved
        within wbOutputs/ as wombat_SAT+INST_YYYY-MM-DD-THHMMSS.png 

        If the log button is hit lines will be appended to the log file. The
        default log file (if none was provided) is wbOutputs/WomBlog.txt and each line contains:
            1.    time of fit  
            2.    instrument
            3.    time of observation
            4.    WF type + panel number (e.g. GCS1). Adding panel number allows multiple of same type
            5-13. WF parameter values. If a type has <9 parameters the extras are filled with None
            14.   the WOMBAT pickle
            15.   the index of the obs time in this pickle
            16.   the background difference mode (0 running diff, 1 base diff)
            17.   the scaling mode (1 linear, 2 log, 3 sqrt)
            18.   the min brightness setting (0-256)
            19.   the max brightness setting (0-256)
        
        The amount of figures saved/lines logged depends on whether one clicks a save/log button or the 
        active window when one hits the s/l short cut keys
    

    The state of the system is tracked through three key logs that are global variables
        paramLog: The wireframe parameters. This is a dictionary with keys for every possible
                  combination of wftype and wf index number (e.g. 'GCS2'), which allows for
                  duplicates of a type. The syntax is paramLog[type#][parameter_index][time_index]
                  where the parameter_index matches the order of the labels in the GUI (top to bottom)
                  and the time index corresponds to the universal time index (not pickle index). If a
                  type# has not be used or reloaded then the values will be set to None. As a user fits
                  the wf the values are stored for each type/time and can be reloaded if one switches 
                  away from one type/time then comes back to it
        
        setLog:   A log of previously used aesthetics of the background. This is a dictionary with keys 
                  corresponding to the instrument tag of each plot window which is used to track the most
                  recently used values for the min/max sliders for each combo of difference mode, scaling
                  mode, and time. The syntax is setLog[instTag][diffIndex][scaleIndx][min/max][time_idx] where 
                  diffIndex is 0 for running and 1 for base, scale is 0, 1, 2 for linear, log, sqrt, and
                  min/max is 0 for min and 1 for max. This is populated with the default values at the start 
                  and logs changes to be reloaded if this diff/scale/time is revisited
    
    
        curSet:   The current settings of the aesthetics. This is a dictionary with keys corresponding to the 
                  plot window instrument tag. The syntax is curSet[instTag][set_idx][time_idx] where set_idx
                  0 gives the difference mode, 1 gives the scaling mode, 2 gives the min slider value, 4 gives
                  the max slider value, and time_idx is universal time. This is the current state of the 
                  system and allows to cycle through different diff types and scalings across different times.
    


    WOMBAT avoids creating globals in any of the other functions but its much easier
    to maintain a limited number created here to make passing things easier
        mainWindow: the parameter window
        pws:        array of plot windows
        nSats:      number of sats/instruments = number of plot windows
        wfs:        array of wireframes in theoryland coords
        nwfs:       number of wireframes
        ovw:
        bkgpkl:
        occultDict: dictionary of (rough) inner/outer boundaries of each inst FOV
        WFname2id:  wireframe name to their index number in the WF combo boc
        paramsBuilt: 
        figLabels:
        
    """
    #|-----------------------------| 
    #|---- Dictionary Globals -----|
    #|-----------------------------|
    global occultDict, WFname2id
    # Nominal radii (in Rs) for the occulters for each instrument. Pulled from google so 
    # generally correct (hopefully) but not the most precise
    occultDict = {'STEREO_SECCHI_COR2':[3,14], 'STEREO_SECCHI_COR1':[1.5,4], 'SOHO_LASCO_C1':[1.1,3], 'SOHO_LASCO_C2':[2,6], 'SOHO_LASCO_C3':[3.7,32], 'STEREO_SECCHI_HI1':[15,80], 'STEREO_SECCHI_HI2':[80,215], 'STEREO_SECCHI_EUVI':[0,1.7],'SDO_AIA':[0,1.35]} 
    # Wireframe name to combo box index 
    WFname2id = {'GCS':1, 'Torus':2, 'Sphere':3, 'Half Sphere':4, 'Ellipse':5, 'Half Ellipse':6, 'Slab':7, 'Tube':8, 'GCS*':9}
    
    
    #|-----------------------------| 
    #|---- Other Global Setup -----|
    #|-----------------------------|
    global mainwindow, pws, nSats, wfs, nwfs, ovw, bkgpkl 
    global paramsBuilt, figLabels
    global paramLog, setLog, curSet
    paramsBuilt = False # keep from trying to build WF until param widgets are built
    figLabels = doFigLabs
    paramLog, setLog, curSet = None, None, None
    
    #|----------------------------------------| 
    #|--- Pull apart background data input ---|
    #|----------------------------------------|
    # Assign things passed into variable names
    # A bit hacky to use architecture that existed
    # before the pickle party
    WBinfo = obsFiles['WBinfo']
    proIms0 = obsFiles['proIms0']
    proIms = obsFiles['proImMaps']
    massIms = obsFiles['massIms']
    sclIms = obsFiles['scaledIms']
    satStuff = obsFiles['satStuff']
    bkgpkl = obsFiles['pickleSource']
    
    #|---- Pull sat number from obsFiles ----|
    nSats = len(WBinfo['Insts'])
    global instNames
    instNames = WBinfo['Insts']
    
    #|------------------------------| 
    #|---- Remap keys to insts -----|
    #|------------------------------|
    # Check multi time at same time
    multiTime = False
    i2inst = {}
    for i in range(nSats):
        key = WBinfo['Insts'][i]
        i2inst[i] = key
        if len(proIms[i2inst[i]]) > 1:
            multiTime = True
        proIms0[i] = proIms0.pop(key)
        proIms[i]  = proIms.pop(key)
        massIms[i] = massIms.pop(key)
        sclIms[i]  = sclIms.pop(key)
        satStuff[i] = satStuff.pop(key)
        for j in range(len(satStuff[i])):
            satStuff[i][0][j]['KEY'] = key
    # Rename this
    obsFiles = proIms
         
    #|------------------------------| 
    #|---- Initiate Wireframes -----|
    #|------------------------------|
    # To Reload
    if type(reloadDict) != type(None):
        nwfs = reloadDict['nWFs']
        if nwfs < 1:
            nwfs = 1
    # Or not to reload
    else:
        nwfs = nWFs
    # Load up noneType WFs
    wfs = [wf.wireframe(None) for i in range(nwfs)]
    
    
    #|---------------------------------| 
    #|---- Setup time slider vals -----|
    #|---------------------------------|
    if multiTime:
        nTsli, tlabs, idxMaps = sortTimeIndices([satStuff[i][0] for i in range(len(satStuff))], tRes=tRes)
    else:
        nTsli = 0
        tlabs = None
        tmaps = None
    

    #|--------------------------------| 
    #|---- Get height slider max -----|
    #|--------------------------------|
    # Optimal range of height slider depends on what observations 
    # we have. Take the max corner dist ('FOV') from each inst
    # and convert it to a nice number
    maxFoV = 0
    for i in range(nSats):
        stuff = satStuff[i][0]
        if stuff[0]['FOV'] > maxFoV: maxFoV = stuff[0]['FOV']
    # pad it a bit then round to a nice number
    maxFoV = int((1.25 * maxFoV) / 5) * 5
    # EUV only will pull 0 for this^, just set higher if < 1
    if maxFoV < 1:
        maxFoV = 1.5
    # Edit the dictionaries in wombatWF
    wf.rngDict['Height (Rs)'] = [1,maxFoV]
    if maxFoV > 25:
        wf.defDict['Height (Rs)'] = 25
    elif maxFoV < 5:
        wf.defDict['Height (Rs)'] = 1.5
            
    #|-----------------------------| 
    #|---- Launch Application -----|
    #|-----------------------------|
    app = QApplication(sys.argv)
    # Get size to use for window placement
    screen = app.primaryScreen()
    size = screen.availableGeometry()
    screenXY = [size.width(), size.height()]

    #|------------------------------| 
    #|---- Launch Plot Windows -----|
    #|------------------------------|
    pws = []
    for i in range(nSats):
        if multiTime:
            myTmap = [idxMaps['t2p'][i], idxMaps['p2t'][i], idxMaps['p2tBF'][i]]
        else:
            sing = {}
            sing[0] = [0]
            myTmap = [[0], sing, sing] 
        
        pw = FigWindow(obsFiles[i], sclIms[i], satStuff[i], massIms[i], myNum=i, tmap=myTmap, screenXY=screenXY)
        pw.show()
        pws.append(pw) 
        
    #|---------------------------------| 
    #|---- Launch Overview Window -----|
    #|---------------------------------|    
    if overviewPlot:
        ovw = OverviewWindow(satStuff, screenXY=screenXY)
        ovw.show()
    else:
        ovw = None
    
    
    #|----------------------------------| 
    #|---- Launch Parameter Window -----|
    #|----------------------------------|
    mainwindow = ParamWindow(nwfs, tlabs=tlabs)
    # check if we had a name for the output box
    if type(logFile) != type(None):
        for i in range(nwfs):
            mainwindow.textBoxes[i].setText(logFile)
        mainwindow.saveName = logFile
    # Set time slider to highlight bc reload will
    # shift focus onto text box
    mainwindow.Tsliders[0].setFocus()
    mainwindow.show()

    #|---------------------------------| 
    #|---- Set values from reload -----|
    #|---------------------------------|
    # This will pull from a real realod Dict or just set up
    # things at the default values if passed an empty one
    paramLog, setLog, curSet = buildMegaVars(reloadDict, tlabs, idxMaps, WBinfo['Insts'])
    
    # Adjust plot window parameters based on reload
    for apw in pws:
        myInst = apw.instTag
        myDif = curSet[myInst][0][apw.tslIdx]
        myscl = curSet[myInst][1][apw.tslIdx]
        myMin = curSet[myInst][2][apw.tslIdx]
        myMax = curSet[myInst][3][apw.tslIdx]
        
        mainwindow.radButs[0][myDif].setChecked(True)
        mainwindow.radButs[0][np.abs(myDif-1)].setChecked(False)
        mainwindow.holdIt = True
        apw.cbox.setCurrentIndex(myscl)   
        mainwindow.holdIt = False
        # set it slightly wrong to start to make sure labels get flagged
        apw.MinSlider.setValue(myMin+1)
        apw.MinSlider.setValue(myMin)
        apw.MaxSlider.setValue(myMax-1)
        apw.MaxSlider.setValue(myMax)

    sys.exit(app.exec_())
    
