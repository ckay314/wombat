"""
Set of functions and classes used to build and run the WOMBAT GUI

The main function is releaseTheWombat, which is likely the only function
one would need to call in an external program

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
Outputs:
    No outputs unless the save button is clicked. If clicked, it will save fits files
    for the backgrounds (processed as RD/BD) in wbFits/reloads and in wbOutputs it
    will save wombatSummaryFile.txt which can be used to reload the current setup and
    a png file for each observation panel and one for the overview panel (if present)

External Calls:
    everything from wombatWF, wombatLoadCTs, wombatMass
    fitshead2wcs, wcs_get_pixel, wcs_get_coord from wcs_funs in the prep code

"""

import sys, os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QGridLayout, QTabWidget, QSlider, QComboBox, QLineEdit, QPushButton, QRadioButton
from PyQt5 import QtCore
import pyqtgraph as pg
import datetime
from itertools import pairwise
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u


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
        # Or make the times slider for 2+ times
        if type(tlabs) != type(None):    
            self.nTsli = len(tlabs) - 1 # slider goes 0 to val so subtract 1
            self.tlabs = tlabs
        # Hide the time slider if we only have one time
        else:
            self.nTsli = 0 # random number to make it happy
            self.tlabs = ['']
        
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
        self.WFnum2type = ['None', 'GCS', 'Torus', 'Sphere', 'Half Sphere', 'Ellipse', 'Half Ellipse', 'Slab', 'Tube']
        self.WFshort = {'GCS':'GCS', 'Torus':'Tor', 'Sphere':'Sph', 'Half Sphere':'HSph', 'Ellipse':'Ell', 'Half Ellipse':'HEll', 'Slab':'Slab', 'Tube':'Tube'}
        
        # Create holder for the WF params
        self.WFparams = np.array([np.zeros(10) for i in range(nTabs)])
        
        # Holders for the param widgets so we can rm them if turn off a wf
        self.WFLays = []
        self.widges = [None for i in range(nTabs)]
        self.layouts = []
        self.cbs = []
        
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
            myTlab = 'Time selection: '+self.tlabs[0]
            Tlabel = QLabel(myTlab)
        else:
            Tlabel = QLabel('Single Time Given')
        self.Tlabel = Tlabel
        layout.addWidget(Tlabel,0,0,1,10,alignment=QtCore.Qt.AlignLeft)
        
        # |------ Time Slider ------|
        Tslider1 = QSlider()
        Tslider1.setOrientation(QtCore.Qt.Horizontal)
        # Slider doesn't like the number 1 for no apparent reason
        # so easiest just to avoid
        Tslider1.setRange(2,self.nTsli+2)
        Tslider1.valueChanged.connect(self.update_tidx)
        layout.addWidget(Tslider1, 1,0,1,11)
        self.Tslider = Tslider1


        # |------ WF Type Label ------|
        label = QLabel('Wireframe Type')
        layout.addWidget(label, 2,0,1,11,alignment=QtCore.Qt.AlignCenter)
        
        # |----- WF Drop Down Box ----|
        cbox = self.wfComboBox(i)
        self.cbs.append(cbox)
        layout.addWidget(cbox,3,0,1,11,alignment=QtCore.Qt.AlignCenter)
        
        # |------ Parameter Section Label ------|
        label = QLabel('Parameters')
        layout.addWidget(label, 5,0, 1, 5,alignment=QtCore.Qt.AlignCenter)
        
        # |----- Show/Hide WF Button ----|
        hideBut = QPushButton('Show/Hide WF')
        hideBut.released.connect(lambda: self.HBclicked(i))
        layout.addWidget(hideBut, 5, 6, 1,5)
        
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
        radBut1.toggled.connect(lambda:self.btnstate(radBut1))
        #radBut2.toggled.connect(lambda:self.btnstate(radBut2))
        self.radButs = [radBut1, radBut2]

        # |----- Background Drop Down Box ----|
        # Background mode drop down box
        label = QLabel('Background Scaling')
        layout.addWidget(label, 42,0,1,6,alignment=QtCore.Qt.AlignLeft)
        cbox = self.bgComboBox()
        layout.addWidget(cbox,42,5,1,6,alignment=QtCore.Qt.AlignCenter)
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
        cbox.addItem('|----None/Select One----|')
        cbox.addItem('GCS')
        cbox.addItem('Torus')
        cbox.addItem('Sphere')
        cbox.addItem('Half Sphere')
        cbox.addItem('Ellipse')
        cbox.addItem('Half Ellipse')
        cbox.addItem('Slab')
        cbox.addItem('Tube')
        
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
        cbox.currentIndexChanged.connect(self.back_changed)
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
                    yes it is widges not widgets 
     
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
                wBox = QLineEdit()
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
        widges[0][0].returnPressed.connect(lambda: self.b2s(widges[1][0], widges[0][0], i2f[0], myWF.ranges[0][0],nSliders, myWF, widges))     
        widges[1][1].valueChanged.connect(lambda x: self.s2b(x, widges[0][1], i2f[1], myWF.ranges[1][0], myWF, widges))  
        widges[0][1].returnPressed.connect(lambda: self.b2s(widges[1][1], widges[0][1], i2f[1], myWF.ranges[1][0],nSliders, myWF, widges))
        widges[1][2].valueChanged.connect(lambda x: self.s2b(x, widges[0][2], i2f[2], myWF.ranges[2][0], myWF, widges))  
        widges[0][2].returnPressed.connect(lambda: self.b2s(widges[1][2], widges[0][2], i2f[2], myWF.ranges[2][0],nSliders, myWF, widges))
        widges[1][3].valueChanged.connect(lambda x: self.s2b(x, widges[0][3], i2f[3], myWF.ranges[3][0], myWF, widges))  
        widges[0][3].returnPressed.connect(lambda: self.b2s(widges[1][3], widges[0][3], i2f[3], myWF.ranges[3][0],nSliders, myWF, widges))
        # |-------- Parameters 5+ -------|
        # Need to check each of the remaining bc depends on wftype
        myNP = len(myWF.labels)
        # At least 5 params
        if myNP > 4:
            widges[1][4].valueChanged.connect(lambda x: self.s2b(x, widges[0][4], i2f[4], myWF.ranges[4][0], myWF, widges))  
            widges[0][4].returnPressed.connect(lambda: self.b2s(widges[1][4], widges[0][4], i2f[4], myWF.ranges[4][0],nSliders, myWF, widges))
        # At least 6 params    
        if myNP > 5:
            widges[1][5].valueChanged.connect(lambda x: self.s2b(x, widges[0][5], i2f[5], myWF.ranges[5][0], myWF, widges))  
            widges[0][5].returnPressed.connect(lambda: self.b2s(widges[1][5], widges[0][5], i2f[5], myWF.ranges[5][0],nSliders, myWF, widges))
        # At least 7 params    
        if myNP > 6:
            widges[1][6].valueChanged.connect(lambda x: self.s2b(x, widges[0][6], i2f[6], myWF.ranges[6][0], myWF, widges))  
            widges[0][6].returnPressed.connect(lambda: self.b2s(widges[1][6], widges[0][6], i2f[6], myWF.ranges[6][0],nSliders, myWF, widges))
        # At least 8 params           
        if myNP > 7:
            widges[1][7].valueChanged.connect(lambda x: self.s2b(x, widges[0][7], i2f[7], myWF.ranges[7][0], myWF, widges))  
            widges[0][7].returnPressed.connect(lambda: self.b2s(widges[1][7], widges[0][7], i2f[7], myWF.ranges[7][0],nSliders, myWF, widges))
        # At least 9 params    
        if myNP > 8:
            widges[1][8].valueChanged.connect(lambda x: self.s2b(x, widges[0][8], i2f[8], myWF.ranges[8][0], myWF, widges))  
            widges[0][8].returnPressed.connect(lambda: self.b2s(widges[1][8], widges[0][8], i2f[8], myWF.ranges[8][0],nSliders, myWF, widges))
            
        # |---------------------------------------|
        # |------- Initiate Widget Values --------| 
        # |---------------------------------------|
        # Set things to the values the WF has
        for i in range(myNP):
            myVal = myWF.params[i]
            slidx = int((myVal - myWF.ranges[i][0])/ i2f[i])
            if slidx > nSliders -1:
                slidx = nSliders -1
                myVal = myWF.ranges[i][1]
            elif slidx < 0:
                slidx = 0
                myVal = myWF.ranges[i][0]
            widges[1][i].setValue(slidx)
            widges[0][i].setText(str(myVal))
        
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
            b      = switch to base difference
            r      = switch to running difference
            s      = save 
            m      = calculate mass
     
        """
        #|--- Pull Params/Plot ---|
        if event.key() == QtCore.Qt.Key_Return:
            for iii in range(nwfs):
                self.updateWFpoints(wfs[iii], self.widges[iii])
                focused_widget = self.focusWidget()
                focused_widget.deselect()
                self.Tslider.setFocus()
        #|--- Closing things ---|
        elif event.key() == QtCore.Qt.Key_Q: 
            self.close()    
        elif event.key() == QtCore.Qt.Key_Escape:
            sys.exit()
        #|--- Time Slider ---|
        elif event.key()== QtCore.Qt.Key_Right:
            Tval = self.Tslider.value()
            self.Tslider.setValue(Tval+1)
        elif event.key()== QtCore.Qt.Key_Left:
            Tval = self.Tslider.value()
            self.Tslider.setValue(Tval-1)            
        #|--- Difference mode ---|
        elif event.key()== QtCore.Qt.Key_B:
            self.radButs[1].setChecked(True)
        elif event.key()== QtCore.Qt.Key_R:
            self.radButs[0].setChecked(True)
        #|--- Scaling mode ---|
        elif event.key()== QtCore.Qt.Key_1:
            self.Bcbox.setCurrentIndex(0)
        elif event.key()== QtCore.Qt.Key_2:
            self.Bcbox.setCurrentIndex(1)
        elif event.key()== QtCore.Qt.Key_3:
            self.Bcbox.setCurrentIndex(2)
        #|--- Mass ---|
        elif event.key() == QtCore.Qt.Key_M:
            self.MBclicked()
        
        #|--- Saving ---|
        elif event.key() == QtCore.Qt.Key_S:
            self.SBclicked()
        
            
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
        b.setText(myStr)
        # Update the wirefram
        self.updateWFpoints(myWF, widges)

    def b2s(self,s,b, dx=None, x0=None, nSli=None, myWF=None, widges=None):
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
            * Parameters aren't optional but get passed through a lambda function like this
        
        Actions:
            Sets the corresponding box to the appropriate index based on value
            Updates the parameter in the wf structure, recalculates wf, and updates figs
     
        """
        temp = b.text()
        # Convert parameter value to slider idx 
        slidx = int((float(b.text()) - x0)/dx)
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
        
        # The above triggers s2b since the slider changes so
        # reset it to what we actual wanted instead of slider rounded val
        b.setText(temp)
        # Update the wirefram
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
        if type(wfs[idx].WFtype) == type(None):
            # Make the new WF
            myType = self.WFnum2type[a]
            wfs[idx] = wf.wireframe(myType, WFidx=idx+1)
            
            # Change the tab name to this type
            self.tab_widget.setTabText(idx,self.WFshort[myType])
            
            # Set up a new param layout
            WFLay, widges = self.WFparamLayout(wfs[idx])
            
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
            # Create a new wf object but pass it any matching
            # parameters from the previous version
            ogLabs = wfs[idx].labels
            ogParams = wfs[idx].params
            myType = self.WFnum2type[a]
            newWF = wf.wireframe(self.WFnum2type[a], WFidx=idx+1)
            newLabs = newWF.labels
            for iii in range(len(ogLabs)):
                aLab = ogLabs[iii]
                if aLab in newLabs:
                    pidx = np.where(newLabs == aLab)[0]
                    newWF.params[pidx] = ogParams[iii]
                    
            # Change the tab text        
            self.tab_widget.setTabText(idx,self.WFshort[self.WFnum2type[a]])
            
            # Update the slider layout
            thisLay = self.cleanLayout(self.WFLays[idx])
            WFLay, widges = self.WFparamLayout(newWF)
            self.layouts[idx].addLayout(WFLay, 7,0,30,11)
            self.WFLays[idx] = WFLay
            self.widges[idx] = widges
           
            # Give the structure the new wf
            wfs[idx] = newWF
        
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
            
    def back_changed(self,text):
        """
        Event for background combo box changes. The plot window combo box
        event handles most of the heavy lifting. This just passed the change
        along to each of the windows
        
        Inputs:
            text: the text box (integer) value
        
        Actions:
            Changes the background scaling for each plot window
        
        """
        for aPW in pws:
             aPW.cbox.setCurrentIndex(text)         
    
    def update_tidx(self, tval):
        """
        Event for time slider changes. It changes the time index for the 
        window and just calls the basic plot function 
        
        Inputs:
            tval: the time slider integer value
        
        Actions:
            Changes the background time step for each plot window
        
        """
        # Cannot for the life of me figure out why having tval = 1
        # makes the parameter sliders appear at 0 (values and WFs ok tho)
        # Just avoid 1 so the slider starts at 2 and shift what is passed
        for aPW in pws:
            aPW.tidx = aPW.st2obs[tval-2]
            aPW.plotBackground()    
        self.Tlabel.setText('Time selection: '+self.tlabs[tval-2])
        
    def EBclicked(self):
        """
        Event for clicking the exit button
        
        Actions:
            Everything goes bye-bye
        
        """
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

    def SBclicked(self, singleSat=None):
        """
        Event for clicking the save button. If called by the parameter
        window it will save the wf parameters/reload file and images 
        for each of the plot panels. If called by the plot panel it will
        only doing one figure
        
        Optional Inputs:
            singleSat: the index of a single plot window
        
        Actions:
            Saves reload file wombatSummaryFile.txt
            Saves a png for each plot window (only at current time index)
            Saves a png of the overview window
            Saves fits files of the processed backgrounds (all time steps)
        
        """
        
        #|------------------------------------| 
        #|-------- Save Reload File ----------|
        #|------------------------------------|
        fileName = 'wombatSummaryFile.txt'        
        outFile = open('wbOutputs/'+fileName, 'w')
        print ('Saving results in wboutputs/'+fileName)
        
        # |----- Save the wireframe parameters ----|
        for j in range(nwfs):
            aWF = wfs[j]
            # Only save if turned on
            if aWF.WFtype:
                outFile.write('WFtype'+str(j+1)+': ' + str(aWF.WFtype).replace(' ','')+'\n')
                for i in range(len(aWF.labels)):
                    thisLab = aWF.labels[i]
                    spIdx = thisLab.find(' ')
                    if spIdx > 0:
                        outStr = thisLab[:spIdx]+str(j+1)+': ' + str(aWF.params[i])
                    else:
                        outStr = thisLab+str(j+1)+': ' + str(aWF.params[i])
                    outFile.write(outStr+'\n')
                
                    
        # |----- Save the background parameters ----|
        # Check if doing single sat or all
        if type(singleSat) != type(None):
            toDo = [singleSat]
        else:
            toDo = range(nSats)
        # Look through whoever we included    
        for j in toDo:
            aPW = pws[j]
            tidx = aPW.tidx
            # Add the base background file
            outStr = 'ObsFile'+str(j+1)+': ' + aPW.satStuff[0][0]['DIFFFITS']
            outFile.write(outStr+'\n')
            # Add the names of all time steps
            for tidx in range(len(aPW.satStuff[0])):
                outStr = 'ObsFile'+str(j+1)+': ' + aPW.satStuff[0][tidx]['MYFITS']
                outFile.write(outStr+'\n')
            # Scaling parameters
            outStr = 'ObsType'+str(j+1)+': ' + aPW.satStuff[0][tidx]['MYTAG'].replace(' ','_')
            outFile.write(outStr+'\n')            
            outStr = 'Scaling'+str(j+1)+': ' +str(aPW.sclidx)
            outFile.write(outStr+'\n')
            outStr = 'MinVal'+str(j+1)+': ' +str(aPW.MinSlider.value())
            outFile.write(outStr+'\n')
            outStr = 'MaxVal'+str(j+1)+': ' +str(aPW.MaxSlider.value())
            outFile.write(outStr+'\n') 
                               
        outFile.close()

        #|------------------------------------| 
        #|---------- Save Figures ------------|
        #|------------------------------------|
        
        #|--------- Save plot windows --------|         
        for j in toDo:
            aPW = pws[j]
            tidx = aPW.tidx
            figName = 'wombat_'+ aPW.satStuff[0][tidx]['DATEOBS'].replace(':','') + '_' +  aPW.satStuff[0][tidx]['MYTAG'].replace(' ','_') +'.png'
            figGrab = aPW.pWindow.grab()
            figGrab.save('wbOutputs/'+figName)
            print ('Saving figure in wbOutputs/'+figName )
        #|------- Save overview window -------|   
        if ovw:
            figName = 'wombat_'+ pws[0].satStuff[0]['DATEOBS'].replace(':','') + '_overview.png'
            figGrab = ovw.pWindow.grab()
            figGrab.save('wbOutputs/'+figName)
            print ('Saving figure in wbOutputs/'+figName )
            
        
    def MBclicked(self):
        """
        Event for clicking the mass button
                
        """
        # Check if wireframes are turned on
        for aPW in pws:
            if not (aPW.satStuff[aPW.didx][0]['OBSTYPE'] == 'EUV'):
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
                
    
    def btnstate(self,b):
        if b.isChecked():
            setidx = 0
            oidx = 1
        else:
            setidx = 1
            oidx = 0
        for aPW in pws:            
            
            aPW.slidervals[oidx,aPW.sclidx,0] = aPW.MinSlider.value()
            aPW.slidervals[oidx,aPW.sclidx,1] = aPW.MaxSlider.value()
            
            aPW.didx = setidx
            aPW.MinSlider.setValue(aPW.slidervals[setidx,aPW.sclidx,0])  
            aPW.MaxSlider.setValue(aPW.slidervals[setidx,aPW.sclidx,1])  
            
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
        flagIt = False
        for i in range(len(widges[0])):
            if widges[0][i].text() != '':
                aWF.params[i] = float(widges[0][i].text())
            else:
                flagIt = True
        # If all set then update points and plots
        if not flagIt:
            aWF.getPoints()
            for ipw in range(nSats):
                pws[ipw].plotWFs(justN=aWF.WFidx-1)
            if ovw:
                ovw.updateArrow(aWF.WFidx-1,color=aWF.WFcolor)
    
        
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
    
        labelPW:  flag to show labels of the spacecraft/instrument name and 
                  time stamp at the bottom of the figure windows
                  defaults to True
    
        tmap:     an array that maps the slider time index to the observation index.
                  The obs aren't necessarily uniformly spaced but the t slider is
                  so a map may look like [0, 1, 1, 2, 3, 4, 4] where 10 slider indices
                  map to 5 observational indices. Also helps adjust for different
                  time resolution for different instruments.      
         
        screenXY: size of the computer display in pixels [x,y]. used to help place windows
    
     
    """
    def __init__(self, myObs, myScls, satStuff, massIms, myNum=0, labelPW=True, tmap=[0], screenXY=None, mouseEnabled=False):
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
    
            labelPW:  flag to show labels of the spacecraft/instrument name and 
                      time stamp at the bottom of the figure windows
                      defaults to True
    
            tmap:     an array that maps the slider time index to the observation index.
                      The obs aren't necessarily uniformly spaced but the t slider is
                      so a map may look like [0, 1, 1, 2, 3, 4, 4] where 10 slider indices
                      map to 5 observational indices. Also helps adjust for different
                      time resolution for different instruments.
        
            screenXY: resolution of the computer monitor. used to place windows nicer
        
            mouseEnabled: allow for scrolling and dragging of the plot data within a window.
                          Defaults to disabled because tends to cause more harm than good.
        
        External Calls:
            check4CT from wombatLoadCTs
     
        """
        super().__init__()
        
        #|---- Setup variables ----|
        self.winidx = myNum # index number for multi mode
        self.labelIt = labelPW # show labels in plot    
        self.satStuff = satStuff 
        self.satName = satStuff[0][0]['OBS'] +' '+ satStuff[0][0]['INST']
        self.OGims = myObs[0]
        self.mIms  = massIms
        self.hdrs = myObs[2]
        self.myScls2 = myScls # the scaled images
        self.tidx = 0
        self.didx = 0 # difference index
        self.sclidx = 0
        self.st2obs = tmap # slider time to obs index
        self.slidervals = np.zeros([2,3,2], dtype=int) # diff, scale time, min/max
        self.nowMass = False # show the region used to calc mass
        self.WFmasks = [np.zeros(self.mIms[0].shape, dtype=int) for i in range(nwfs)]
        
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
                    
                
                
        self.setWindowTitle(self.satName.replace('_',' '))        
        
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
        self.image = pg.ImageItem()

        #|---- Mass contour image item ----|
        self.MCimage = pg.ImageItem()
        
        #|---- Check for color table ----|
        # (from wombatLoadCTs)
        hasCT = check4CT(satStuff[0][0])
        if type(hasCT) != type(None):
             self.image.setLookupTable(hasCT)
        
        #|---- Add the image ----|
        self.pWindow.addItem(self.image)
        self.pWindow.addItem(self.MCimage)
        self.pWindow.setRange(xRange=(0,myObs[self.didx][0].data.shape[0]), yRange=(0,myObs[self.didx][0].data.shape[1]), padding=0)
        
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
        layoutP.addWidget(label, 12,0,1,5,alignment=QtCore.Qt.AlignCenter)
        self.cbox = self.bgComboBox()
        layoutP.addWidget(self.cbox,12,5,1,5,alignment=QtCore.Qt.AlignCenter)
        
        
        #|---- Fill slider defaults ----|
        for i in range(3):
            self.slidervals[:,i,0] = satStuff[self.didx][0]['SLIVALS'][0][i]
            self.slidervals[:,i,1] = satStuff[self.didx][0]['SLIVALS'][1][i]
        
        #|---- Min brightness label/slider ----|
        minL = QLabel('Min Value:     ')
        layoutP.addWidget(minL, 13,0,1,9)
        self.MinSlider = QSlider()
        self.MinSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MinSlider.setMinimum(0)
        self.MinSlider.setMaximum(255)
        self.MinSlider.setValue(satStuff[self.didx][0]['SLIVALS'][0][0])
        layoutP.addWidget(self.MinSlider, 13,3,1,9)
        self.MinSlider.valueChanged.connect(lambda x: self.s2l(x, minL, 'Min Value: '))  
        
        #|---- Min brightness label/slider ----|
        maxL = QLabel('Max Value:     ')
        layoutP.addWidget(maxL, 15,0,1,9)
        self.MaxSlider = QSlider()
        self.MaxSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MaxSlider.setMinimum(0)
        self.MaxSlider.setMaximum(255)
        self.MaxSlider.setValue(satStuff[self.didx][0]['SLIVALS'][1][0])
        layoutP.addWidget(self.MaxSlider, 15,3,1,9)
        self.MaxSlider.valueChanged.connect(lambda x: self.s2l(x, maxL, 'Max Value: '))  
        
        #|---- EUV Dispay Mode ----|
        # If EUV switch to show log at the start. Have to do after
        # we've addded the sliders since will adjust them
        if self.satStuff[self.didx][0]['OBSTYPE'] == 'EUV':
            self.cbox.setCurrentIndex(1)
        
        #|---- Save button
        saveBut = QPushButton('Save')
        saveBut.released.connect(self.SBclicked)
        layoutP.addWidget(saveBut, 17, 0, 1,3,alignment=QtCore.Qt.AlignCenter)

        #|---- Mass button ----|
        massBut = QPushButton('Mass')
        massBut.released.connect(self.MBclicked)
        layoutP.addWidget(massBut, 17, 4, 1,3,alignment=QtCore.Qt.AlignCenter)

        #|---- Exit button ----|
        exitBut = QPushButton('Exit')
        exitBut.released.connect(self.EBclicked)
        exitBut.setStyleSheet("background-color: red")
        layoutP.addWidget(exitBut, 17, 8, 1,3,alignment=QtCore.Qt.AlignCenter)
        
        #|---- Set layout ----|
        self.setLayout(layoutP)
        
        #|---- Show the background ----|
        self.plotBackground()
        
        
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
        Event for scaling slider changes.
        
        Inputs:
            x:      the integer slider value
            l:      the label friend for this slider
            pref:   the prefix printed in front of the index in the label
            * Parameters aren't optional but get passed through a lambda function like this
        
        Actions:
            Sets the label to pref + x
            Replots background using update min/max values
     
        """
        if 'Min' in pref:
            self.slidervals[self.didx,self.sclidx,0] = x
        elif 'Max' in pref:
            self.slidervals[self.didx,self.sclidx,1] = x
        l.setText(pref + str(x))
        self.plotBackground()

    def keyPressEvent(self, event):
        """
        Event for key press events. 
        
        Actions (based on key):
            return = replot (pulls out of param text box)
            q      = close a window
            esc    = close everything
            left   = move time slider to earlier time
            right  = move time slider to later time
            b      = switch to base difference
            r      = switch to running difference
            s      = save
            m      = calculate mass
     
        """
        #|--- Pull Params/Plot ---|
        if event.key() == QtCore.Qt.Key_Return:
            if 'mainwindow' in globals():
                for iii in range(nwfs):
                    mainwindow.updateWFpoints(wfs[iii], mainwindow.widges[iii])
                    focused_widget = mainwindow.focusWidget()
                    focused_widget.deselect()
                    mainwindow.Tslider.setFocus()
        #|--- Closing Things ---|
        elif event.key() == QtCore.Qt.Key_Q: 
            self.close()
        elif event.key() == QtCore.Qt.Key_Escape:
            sys.exit()
        #|--- Saving ---|
        elif event.key() == QtCore.Qt.Key_S:
            if 'mainwindow' in globals():
                mainwindow.SBclicked(singleSat=self.winidx)
        #|--- Time Slider ---|
        elif event.key()== QtCore.Qt.Key_Right:
            if 'mainwindow' in globals():
                Tval = mainwindow.Tslider.value()
                mainwindow.Tslider.setValue(Tval+1)
        elif event.key()== QtCore.Qt.Key_Left:
            if 'mainwindow' in globals():
                Tval = mainwindow.Tslider.value()
                mainwindow.Tslider.setValue(Tval-1)            
        #|--- Difference mode ---|
        elif event.key()== QtCore.Qt.Key_B:
            if 'mainwindow' in globals():
                mainwindow.radButs[1].setChecked(True)
        elif event.key()== QtCore.Qt.Key_R:
            if 'mainwindow' in globals():
                mainwindow.radButs[0].setChecked(True)
        #|--- Scaling mode ---|
        elif event.key()== QtCore.Qt.Key_1:
            if 'mainwindow' in globals():
                mainwindow.Bcbox.setCurrentIndex(0)
        elif event.key()== QtCore.Qt.Key_2:
            if 'mainwindow' in globals():
                mainwindow.Bcbox.setCurrentIndex(1)
        elif event.key()== QtCore.Qt.Key_3:
            if 'mainwindow' in globals():
                mainwindow.Bcbox.setCurrentIndex(2)
        #|--- Mass ---|
        elif event.key() == QtCore.Qt.Key_M:
            self.MBclicked()
                
    def back_changed(self,text):
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
        self.sclidx = text   
        self.MinSlider.setValue(self.slidervals[self.didx,self.sclidx,0])  
        self.MaxSlider.setValue(self.slidervals[self.didx,self.sclidx,1])  
        self.plotBackground()
                   
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
                     Tx, Ty (arcsec): Tx, Ty
               Proj R (Rs), PA (deg): ProjR, PA
        
        """
        #|---- Get the event loc in pix ----|
        scene_pos = event.scenePos()
        view_pos = self.pWindow.plotItem.vb.mapSceneToView(scene_pos)
        pix = [view_pos.x(), view_pos.y()]
        
        #|---- Print pix ----|
        prefA = self.satStuff[self.didx][self.tidx]['MYTAG'].replace('_',' ') + ' pix:'
        print (prefA.rjust(25), str(int(pix[0])).rjust(8), str(int(pix[1])).rjust(8))
        
        #|---- Convert to ra/dec ----| 
        skyres = self.OGims[self.tidx].pixel_to_world(pix[0]*u.pixel, pix[1]*u.pixel)
        Tx, Ty = skyres.Tx.to_value(), skyres.Ty.to_value()
        print ('Tx, Ty (arcsec):'.rjust(25), str(int(Tx)).rjust(8), str(int(Ty)).rjust(8))
        
        # |---- Convert to proj Rsun/PA  ----| 
        Rarc = np.sqrt(Tx**2 + Ty**2)
        Rpix = Rarc / self.satStuff[self.didx][self.tidx]['SCALE']
        # Adjust unites for HI
        if self.satStuff[self.didx][self.tidx]['OBSTYPE'] == 'HI':
            Rpix = Rpix / 3600
        RRSun = Rpix /  self.satStuff[self.didx][self.tidx]['ONERSUN']
        # PA define w/ N as 0 and E (left) as 90
        PA = (np.arctan2(-Tx,Ty) * 180 / np.pi) % 360.
        print ('Proj R (Rs), PA (deg):'.rjust(25), '{:8.2f}'.format(RRSun), '{:8.1f}'.format(PA))
        # |---- Get mass per pixel  ----| 
        if type(self.mIms) != type(None):
            px = int(pix[0])
            py = int(pix[1])
            if self.satStuff[self.didx][0]['OBSTYPE'] == 'COR':
                print('Mass in pixel (1e8 g):'.rjust(25), '{:8.1f}'.format(self.mIms[self.tidx][py,px]/1e8))
            elif self.satStuff[self.didx][0]['OBSTYPE'] == 'HI':
                print('Mass in pixel (1e8 g):'.rjust(25), '{:8.1f}'.format(self.mIms[self.tidx][py,px]/1e8))

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
                #|----------------------|
                #|---- Get sat info ----|
                #|----------------------|
                pos = []
                # Location
                obs = self.satStuff[self.didx][self.tidx]['POS']
                # Scale btwn pix and arcsec
                obsScl = [self.satStuff[self.didx][self.tidx]['SCALE'], self.satStuff[self.didx][self.tidx]['SCALE']]
                if self.satStuff[self.didx][self.tidx]['OBSTYPE'] == 'HI':
                    obsScl = [self.satStuff[self.didx][self.tidx]['SCALE'] * 3600, self.satStuff[self.didx][self.tidx]['SCALE'] * 3600]
                #cent = self.satStuff[self.tidx]['SUNPIX']
                # Occulter info
                if 'OCCRARC' in self.satStuff[self.didx][self.tidx]:
                    occultR = self.satStuff[self.didx][self.tidx]['OCCRARC']
                else:
                    occultR = None
                # WCS info    
                mywcs  = self.satStuff[self.didx][self.tidx]['WCS']
                
                #|---- Set wf aesthetics ----|
                myColor =wfs[i].WFcolor
                # change pen wid if HI
                penwid =1
                if self.satStuff[self.didx][self.tidx]['OBSTYPE'] == 'HI':
                    penwid = 4
                
                
                #|--------------------------|
                #|---- EUV Proj2Surface ----|
                #|--------------------------|
                # For the EUV panels, check if the WF is much higher
                # than the FOV and just project it onto the surface
                # instead if it is
                flatEUV = False
                if self.satStuff[self.didx][self.tidx]['OBSTYPE'] == 'EUV':
                    pts = wfs[i].points
                    rs = np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
                    # Compare max wf radius to inst FOV
                    if np.mean(rs) > 1.5*self.satStuff[self.didx][self.tidx]['FOV']:
                        flatEUV = True
                # Downselect to fewer points for proj EUV
                toShow = range(len(wfs[i].points[:,0]))
                if flatEUV:
                    toShow = toShow[::2]
                    myColor = '#C81CDE'
                    occultR = 1. * self.satStuff[self.didx][self.tidx]['ONERSUN']
                
                #|------------------------|
                #|---- HI Flag Inside ----|
                #|------------------------|    
                # Check if the satellite is in the WF for HI bcs points
                # get weird so at least change color to warn this is the case
                if self.satStuff[self.didx][self.tidx]['OBSTYPE'] == 'HI':
                    # Get max wf R
                    myPos = self.satStuff[self.didx][self.tidx]['POS']
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
                    if 'WISPR' in self.satStuff[self.didx][self.tidx]['MYTAG']:  
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
                        pos.append({'pos': [myPt[0][0], myPt[0][1]], 'pen':{'color':myColor, 'width':penwid}, 'brush':pg.mkBrush(myColor)})
                
                #|---- Reset the scatters to new positions ----|        
                self.scatters[i].setData(pos)
 
    def plotBackground(self):
        """
        Function for updating the background image 
        
        """
        #|---- Grab data for this time/scaling ----|
        myIm = self.myScls2[self.didx][self.tidx][self.sclidx]
        
        #|---- Grab min/max from slider ----|     
        slMin = self.MinSlider.value()
        slMax = self.MaxSlider.value()
        
        #|---- Update image ----|     
        self.image.updateImage(image=myIm, levels=(slMin, slMax))
        
        
        #|---- Show mass contour ----|   
        # Use fake data for now 
        #fakeIm = np.zeros(myIm.shape)
        #fakeIm[250:350] = 1
        if self.nowMass: 
            bigMask = np.zeros(myIm.shape)
            for i in range(nwfs):
                bigMask += self.WFmasks[i]
                # Get the separation 
                seps = np.abs(wfs[i].params[1]-self.satStuff[self.didx][self.tidx]['POSLON'])
                seps[np.where(seps > 90)] = 180 - seps[np.where(seps > 90)]
                mySep = np.min(np.abs(seps))
                if mySep > 80:
                    print ('!!!--- Warning PoS separation large, capping at 80 deg ---!!!')
                    mySep = 80
                # Prob need to convert the h to projected...
                rpos, Bpos = wM.elTheory([wfs[i].params[0]], 0)
                rsep, Bsep = wM.elTheory([wfs[i].params[0]], mySep)
                sclfct = Bsep / Bpos
                print ((self.satName + ' PoS WF' + str(i+1) + ' mass (g): ').rjust(50) + "{:.3e}".format(np.sum(self.WFmasks[i]* self.mIms[i])))
                print ((self.satName + ' deProj WF' + str(i+1) + ' mass (g): ').rjust(50) + "{:.3e}".format(np.sum(self.WFmasks[i]* self.mIms[i]/sclfct)), ' (scale factor ', "{:.1f}".format(1/sclfct[0]), ')')
            print ('')
            self.MCimage.updateImage(image= bigMask, opacity=0.5, levels=(0,nwfs-0.5))

            
        else:
            self.MCimage.updateImage(image= self.WFmasks[0], opacity=0.0, levels=(0,1))
        
        #|---- Draw stuff on top ----|     
        if self.satStuff[self.didx][self.tidx]['OBSTYPE'] != 'EUV':
            #|---- Circle at 1 Rs ----|     
            if 'SUNCIRC' in self.satStuff[self.didx][self.tidx]:
                self.pWindow.plot(self.satStuff[self.didx][self.tidx]['SUNCIRC'][0], self.satStuff[self.didx][self.tidx]['SUNCIRC'][1])
            
            #|---- Line for Solar N ----|         
            if 'SUNNORTH' in self.satStuff[self.didx][self.tidx]:
                self.pWindow.plot(self.satStuff[self.didx][self.tidx]['SUNNORTH'][0], self.satStuff[self.didx][self.tidx]['SUNNORTH'][1], symbolSize=3, symbolBrush='w', pen=pg.mkPen(color='w', width=1))
                
        #|---- Add text labels ----|             
        if self.labelIt:
            geom = self.pWindow.visibleRange()
            wid = geom.width()
            text_item1 = pg.TextItem(self.satStuff[self.didx][self.tidx]['OBS'] + ' ' + self.satStuff[self.didx][self.tidx]['INST'], anchor=(0, 1), fill='k')
            text_item1.setPos(0.001*wid, 0.001*wid)
            self.pWindow.addItem(text_item1)
            text_item2 = pg.TextItem(self.satStuff[self.didx][self.tidx]['DATEOBS'], anchor=(1, 1), fill='k')
            text_item2.setPos(0.999*wid, 0.001*wid)
            self.pWindow.addItem(text_item2)
        
        # Make slider highlighted so key shortcuts work
        if 'mainwindow' in globals():
            mainwindow.Tslider.setFocus()
            

# |------------------------------------------------------------|
# |------------------ Overview Window Class -------------------|
# |------------------------------------------------------------|
class OverviewWindow(QWidget):
    """
    Class for the overview window showing the relative satellite locations
    and the direction of each wireframe
    
    Inputs:
        satStuff: an array of all the satStuff dictionaries for all sats
     
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
            # The positioning is being odd, might be testing with multiple
            # monitors, semi giving up on nice pos for now
            self.setGeometry( int(0.8*screenXY[0]),screenXY[1] , 400, 400) 
        self.setFixedSize(400, 400) 
        self.setWindowTitle('Polar View')
        
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
        self.pWindow.plot(x_data, y_data, pen=pg.mkPen('w', width=3))
        self.pWindow.plot([0], [0], symbol='o', symbolSize=20, symbolBrush=pg.mkBrush(color='y'))
        self.pWindow.plot([0], [-1], symbol='o', symbolSize=15, symbolBrush=pg.mkBrush(color='blue'))
        
        #|---- Hide the axes ----|
        self.pWindow.hideAxis('bottom')
        self.pWindow.hideAxis('left')
        
        
        #|---- Set up scatters for sats ----|
        self.scatters = []
        self.satLabs = []
        self.satStrings = []
        self.satxys = []
        L1counter = 0
        for i in range(nSats):
            #|---- Get a proj sat loc ----|
            myPos = satStuff[i][pws[i].tidx]['POS']
            myName = satStuff[i][0]['OBS']
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
            
            #|---- Add to window ----|
            self.pWindow.addItem(aScat)
            
            #|---- Label each sat ----|
            # Labeling sats not insts to avoid overload
            myName = satStuff[i][0]['SHORTNAME']
            
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
                self.satxys.append([xsat, ysat])
        
        #|---- Set up arrow for the WF lons ----|
        self.arrows = []
        for i in range(nwfs):
            arrow = pg.ArrowItem(angle=-45, tipAngle=0, headLen=0, tailLen=0, tailWidth=0, pen={'color': 'w', 'width': 2}, brush='b')
            arrow.setPos(0, 0)
            self.pWindow.addItem(arrow)
            self.arrows.append(arrow)
        self.setLayout(layoutOV)
    
    def updateArrow(self, i, color='w'):
        """
        Function for updating wf longitude arrow.
        
        Inputs:
            i:      the WF index
        
        Optional Input:
            color:  the color of this WF
     
        """
        #|---- Get the WF lon ----|
        mywf = wfs[i]
        lon  = mywf.params[1]
        
        #|---- Get arrow head loc ----|
        rlon = lon * np.pi /180.
        hL, tL = 0.1, 0.3 # head length, tail length
        aL = hL+tL
        xh = aL * np.sin(rlon)
        yh = -aL * np.cos(rlon)
        ang = -np.arctan2(yh, xh) * 180 / np.pi
        
        #|---- Update the arrow ----|
        self.arrows[i].setStyle(angle=lon-270, headWidth=0.05, headLen=hL, tailLen=tL, tailWidth=0.03, pxMode=False,  pen={'color': color, 'width': 2}, brush=color)
        tail_len = self.arrows[i].opts['tailLen']
        self.arrows[i].setPos(xh, yh)
        
    def keyPressEvent(self, event):
        """
        Event for key press events. 
        
        Actions (based on key):
            return = replot (pulls out of param text box)
            q      = close a window
            esc    = close everything
            left   = move time slider to earlier time
            right  = move time slider to later time
            b      = switch to base difference
            r      = switch to running difference
            s      = save
            m      = calculate mass
     
        """
        #|--- Pull Params/Plot ---|
        if event.key() == QtCore.Qt.Key_Return:
            if 'mainwindow' in globals():
                for iii in range(nwfs):
                    mainwindow.updateWFpoints(wfs[iii], mainwindow.widges[iii])
                    focused_widget = mainwindow.focusWidget()
                    focused_widget.deselect()
                    mainwindow.Tslider.setFocus()
        #|--- Closing Things ---|
        elif event.key() == QtCore.Qt.Key_Q: 
            self.close()
        elif event.key() == QtCore.Qt.Key_Escape:
            sys.exit()
        #|--- Saving ---|
        elif event.key() == QtCore.Qt.Key_S:
            if 'mainwindow' in globals():
                mainwindow.SBclicked()
        #|--- Time Slider ---|
        elif event.key()== QtCore.Qt.Key_Right:
            if 'mainwindow' in globals():
                Tval = mainwindow.Tslider.value()
                mainwindow.Tslider.setValue(Tval+1)
        elif event.key()== QtCore.Qt.Key_Left:
            if 'mainwindow' in globals():
                Tval = mainwindow.Tslider.value()
                mainwindow.Tslider.setValue(Tval-1)            
        #|--- Difference mode ---|
        elif event.key()== QtCore.Qt.Key_B:
            if 'mainwindow' in globals():
                mainwindow.radButs[1].setChecked(True)
        elif event.key()== QtCore.Qt.Key_R:
            if 'mainwindow' in globals():
                mainwindow.radButs[0].setChecked(True)
        #|--- Scaling mode ---|
        elif event.key()== QtCore.Qt.Key_1:
            if 'mainwindow' in globals():
                mainwindow.Bcbox.setCurrentIndex(0)
        elif event.key()== QtCore.Qt.Key_2:
            if 'mainwindow' in globals():
                mainwindow.Bcbox.setCurrentIndex(1)
        elif event.key()== QtCore.Qt.Key_3:
            if 'mainwindow' in globals():
                mainwindow.Bcbox.setCurrentIndex(2) 
        #|--- Mass ---|
        elif event.key() == QtCore.Qt.Key_M:
            if 'mainwindow' in globals():
                mainwindow.MBclicked()
           


# |------------------------------------------------------------|
# |---------------- Scale the Background Imgs -----------------|
# |------------------------------------------------------------|
def makeNiceMMs(obsIn, satStuffs):
    """
    Function to convert input maps into scaled arrays with values
    between 0-255 that are ready to show as is in the plot windows.
    We process this all ahead of time so the GUI flips through 
    existing data without needing new calculations

    Inputs:
        obsIn:     The observations from a single instrument in the form
                   [[map1, map2, ...], [hdr1, hdr2, ...]]
    
        satStuffs: the header like structure created by getSatStuff.
           
    Outputs:
        allScls:   an array of three times series the data scaled using diffent
                   methods (linear, logarithmic, square root). This data is in 
                   array form, not maps.
                   e.g. [[lin1, log1, sqrt1], [lin2, log2, sqrt2], ...]
    
        satStuffs: the header like structure created by getSatStuff 
                   but with a few additional entries
    
    """
    #|-------------------------------------| 
    #|---- Configuration Dictionaries -----|
    #|-------------------------------------|
    # Dictionaries that establish the scaling of things
    # Pull the desired values for each instrument
    
    # mins/maxs on percentiles by instrument [[lower], [upper]] with [lin, log, sqrt]
    pMMs = {'AIA':[[0.001,10,1], [99,99,99]], 'SECCHI_EUVI':[[0.001,10,1], [99,99,99]], 'LASCO_C2':[[15,1,15], [97,99,97]], 'LASCO_C3':[[40,1,10], [99,99,90]], 'SECCHI_COR1':[[30,1,10], [99,99,90]], 'SECCHI_COR2':[[20,1,10], [92,99,93]], 'SECCHI_HI1':[[1,40,1], [99.9,80,99.9]], 'SECCHI_HI2':[[1,40,1],[99.9,80,99.9]], 'WISPR_HI1':[[1,40,1], [99.9,80,99.9]], 'WISPR_HI2':[[1,40,1], [99.9,80,99.9]], 'SoloHI':[[1,40,1], [99.5,80,99.5]] }
    
    # Where the background sliders start (between 0 and 255)
    sliVals = {'AIA':[[0,0,0], [191,191,191]], 'SECCHI_EUVI':[[0,32,0], [191,191,191]], 'LASCO_C2':[[0,0,21],[191,191,191]], 'LASCO_C3':[[37,0,37],[191,191,191]], 'SECCHI_COR1':[[63,0,21],[191,191,191]], 'SECCHI_COR2':[[63,0,21],[191,191,191]], 'SECCHI_HI1':[[63,0,21],[128,191,191]], 'SECCHI_HI2':[[63,0,21],[128,191,191]],  'WISPR_HI1':[[0,0,21],[128,191,191]], 'WISPR_HI2':[[0,0,21],[128,191,191]], 'SoloHI':[[0,0,21],[128,191,191]]}
    
    # Pull the configuration based on instrument
    myInst = satStuffs[0]['INST']
    myMM = pMMs[myInst]
    mySliVals = sliVals[myInst]
    
    #|--- Loop through both RD and BD ---|
    bothScls = []
    bothSatStuffs = []
    for k in range(2):
        #|-------------------------------------| 
        #|------- Pull/Clean Map data ---------|
        #|-------------------------------------|
        #|---- Make empty holder ----|
        sz = obsIn[k][0].data.shape
        allObs = np.zeros([len(satStuffs), sz[1], sz[0]])
        #|---- Fill from map data ----|
        for i in range(len(satStuffs)):
            allObs[i,:,:] = np.transpose(obsIn[k][i].data)
        
        #|---- Get overall median ----|
        imNonNaN = allObs[~np.isnan(allObs)]   
        medval  = np.median(np.abs(imNonNaN))
    
        #|---- Check if diff image ----|
        # Get the median negative value to comp to the median abs
        # value. If neg med big enough assume that is diff image
        negmed  = np.abs(np.median(imNonNaN[np.where(imNonNaN < 0)]))
        diffImg = False
        if (negmed / medval) > 0.25: # guessing at cutoff of 25%, might tune
            diffImg = True
    
        #|---- Clean out NaNs ----|
        if not diffImg:
            allObs[np.isnan(allObs)] = 0
        else:
            allObs[np.isnan(allObs)] = -9999
    
        #|---- Clean out Infs ----| 
        if not diffImg:
            allObs[np.isinf(np.abs(allObs))] = 0
            imNonNaN[np.isinf(np.abs(imNonNaN))] = 0
        else:
            allObs[np.isinf(np.abs(allObs))] = -9999
            imNonNaN[np.isinf(np.abs(imNonNaN))] = -9999
    
        #|-------------------------------------| 
        #|--------- Process the data ----------|
        #|-------------------------------------|    
        #|---- Scaled image holder ----|   
        allScls = []    
    
        #|---- Process linear imgs ----|   
        # Get vals at min/max percentile from the config dictionary
        linMin, linMax = np.percentile(imNonNaN, myMM[0][0]), np.percentile(imNonNaN, myMM[1][0])   
        # If a diff image reset min to neg val based on max   
        if diffImg:
            linMin = - 0.5*linMax
        # Calc range and scale to 0 - 255
        rng = linMax- linMin
        linIm = (allObs - linMin) * 255 / rng

        #|---- Process log imgs ----|   
        # Normalize to keep things in nice ranges
        tempIm = allObs / medval
        tempNonNan = imNonNaN / medval
        # Get min val based on config dict
        minVal = np.percentile(np.abs(tempNonNan),myMM[0][1])
        # Separate into positive and negative values
        pidx = np.where(tempIm > minVal)
        nidx = np.where(tempIm < -minVal)
        # Make new img
        logIm = np.zeros(tempIm.shape)
        # Set where abs val < min to 1
        logIm[np.where(np.abs(tempIm) < minVal)] = 1
        # Positive is just log + 1
        logIm[pidx] = np.log(tempIm[pidx] - minVal + 1)  
        # Negative is -log(abs) + 1
        logIm[nidx] = -np.log(-tempIm[nidx] - minVal + 1)  
        # Get max val from config dict and rescale
        percX = np.percentile(logIm, myMM[1][1])
        logIm = 191 * logIm / percX
    
        #|---- Process sqrt imgs ----|   
        # Normalize to keep things in nice ranges
        tempIm = allObs / medval
        # Get min val based on config dict
        minVal = np.percentile(tempNonNan,myMM[0][2])
        # Set min val to zero
        tempIm = tempIm - minVal 
        # Set all neg to zero
        tempIm[np.where(tempIm < 0)] = 0
        # Sqrt now that everyone is positive
        sqrtIm = np.sqrt(tempIm)
        # Get max val from config dict and rescale
        percX = np.percentile(sqrtIm, myMM[1][2])
        sqrtIm = 191 * sqrtIm / percX
    
    
        #|-------------------------------------| 
        #|--------- Package Results -----------|
        #|-------------------------------------|
        for i in range(len(satStuffs)):
            #|---- Add the slider init values to satStuff ----|
            satStuffs[i]['SLIVALS'] = mySliVals
    
            #|---- Package this time step ----|
            sclIms = [linIm[i], logIm[i],  sqrtIm[i]]
        
            #|---- Add a mask (if needed) ----|
            if 'MASK' in satStuffs[i]:
                midx = np.where(satStuffs[i]['MASK'] == 1)
                for k in range(3):
                    # black out all occulted
                    sclIms[k][midx] = -100. # might need to change if adjust plot ranges
                
            #|---- Append masked/scaled images to out list ----|
            allScls.append(sclIms)
        bothScls.append(allScls)
        bothSatStuffs.append(satStuffs)
    return bothScls, bothSatStuffs

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
    satDict['POINTING'] = -np.array(xyz)
    pointLon = np.arctan2(satDict['POINTING'][1], satDict['POINTING'][0]) * 180 / np.pi
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
        pts_in:  a point (or list of points, probably) in the form [lat, lon, r]
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
# |--------------- Set up GUI from reload file ----------------|
# |------------------------------------------------------------|
def reloadIt(rD):
    """
    Function to reload the GUI from a save file 
    
    Inputs:
       rD:  a reload dictionary read in from a save file in processReload
            (in wombatWrapper.py)
    
    Effects:
            Set wireframe and background parameters to the values they 
            had when the save file was generated
        
    """
    # |----------------------------------|
    # |---- Set Wireframe Parameters ----|    
    # |----------------------------------|
    
    #|---- Loop through each wireframe ----|
    for i in range(nwfs):
        ii = str(i+1)
        #|---- Determine type ----|
        if 'WFtype'+ii in rD:
            WFid = WFname2id[rD['WFtype'+ii]]
            wfs[i]  = wf.wireframe(rD['WFtype'+ii])
            mainwindow.cbs[i].setCurrentIndex(WFid)
            
            #|---- Check for params by label ----|
            for j in range(len(wfs[i].labels)):
                thisLab = wfs[i].labels[j]
                spIdx = thisLab.find(' ')
                shortStr = thisLab[:spIdx]+ii
                # If found reset the wf param value and the widges
                if shortStr in rD:
                    wfs[i].params[j] = float(rD[shortStr])
                    mainwindow.widges[i][0][j].setText(str(wfs[i].params[j]))
                # Update the slider too
                myRng = wfs[i].ranges[j]
                dx = (myRng[1] - myRng[0]) / (mainwindow.nSliders - 1)
                x0 = myRng[0]
                slidx = int((float(wfs[i].params[j]) - x0)/dx)
                mainwindow.widges[i][1][j].setValue(slidx)
            #|---- Show this wireframe ----|
            mainwindow.updateWFpoints(wfs[i], mainwindow.widges[i])

    # |-----------------------------------|
    # |---- Set Background Parameters ----|    
    # |-----------------------------------|
    for i in range(nSats):
        ii = str(i+1)
        myscl = int(rD['Scaling'+ii])
        myMin = int(rD['MinVal'+ii])
        myMax = int(rD['MaxVal'+ii])
        pws[i].cbox.setCurrentIndex(myscl)
        pws[i].MinSlider.setValue(myMin)
        pws[i].MaxSlider.setValue(myMax)
    
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
    
        tmaps:   array of indice maps between time slider and a set
                 of instrument data. 
                 e.g. for single indice maps -> [0, 1, 1, 2, 3, 4, 4]  
                 where 10 slider indices map to 5 observational indices        
    """
    
    # |--------------------------------|
    # |------ Collect Everything ------|    
    # |--------------------------------|
    # Nested array of all times by instrument
    allTimes = []
    # Min/max time for each instrument
    allMins  = []
    allMaxs  = []
    for j in range(len(satStuff)):
        aSat = satStuff[j]
        myTimes = []
        for i in range(len(aSat)):
            myTimes.append(datetime.datetime.strptime(aSat[i]['DATEOBS'], "%Y-%m-%dT%H:%M:%S"))
            
        # Check if sorted
        if not all(a <= b for a, b in pairwise(myTimes)):
            sys.exit('Exiting from sortTimeIndices, files should be provided in time order')
        
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
    tmaps = []
    for j in range(len(satStuff)):
        myTimes = allTimes[j]
        st2obs  = np.zeros(nTimes, dtype=int)
        for i in range(nTimes):
            myDTdiff = np.abs(myTimes-DTgeneral[i])
            mygenidx = np.where(myDTdiff == np.min(myDTdiff))[0]
            st2obs[i] = mygenidx[0]
        tmaps.append(st2obs)
    return nTimes, tlabs, tmaps

# |------------------------------------------------------------|
# |------------------- Main Launch Function -------------------|
# |------------------------------------------------------------|
def releaseTheWombat(obsFiles, nWFs=1, overviewPlot=False, labelPW=True, reloadDict=None):
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
        nWFs:         number of wireframes. Currently set an upper limit of 10 to keep
                      GUI from becoming overloaded
                      defaults to 1
    
        overviewPlot: flag to include the polar/top-down overview panel showing the relative
                      locations of the Sun, Earth, satellites, and wireframes
                      defaults to False
        
        labelPWs:     flag to label plot windows with the sat/inst and the date
                      string on the bottom of the panel
                      defaults to True
    
        reloadDict:   option to pass a reloadDictionary (from processReload in
                      wombatWrapper.py) to relaunch the GUI from a previous state
                      defaults to None (aka no reload)

    Outputs:
        No outputs unless the save button is clicked. If clicked, it will save fits files
        for the backgrounds (processed as RD/BD) in wbFits/reloads and in wbOutputs it
        will save wombatSummaryFile.txt which can be used to reload the current setup and
        a png file for each observation panel and one for the overview panel (if present)


    ***Avoided creating globals in any of the other functions but its much easier
    to maintain a limited number created here to make passing things easier***
        mainWindow: the parameter window
        pws:        array of plot windows
        nSats:      number of sats/instruments = number of plot windows
        wfs:        array of wireframes in theoryland coords
        nwfs:       number of wireframes
        bmodes:     array of integer background scaling modes, defaults to linear = 0
        occultDict: dictionary of (rough) inner/outer boundaries of each inst FOV
        WFname2id:  wireframe name to their index number in the WF combo boc
  
    """

    #|-----------------------------| 
    #|---- Dictionary Globals -----|
    #|-----------------------------|
    global occultDict, WFname2id
    # Nominal radii (in Rs) for the occulters for each instrument. Pulled from google so 
    # generally correct (hopefully) but not the most precise
    occultDict = {'STEREO_SECCHI_COR2':[3,14], 'STEREO_SECCHI_COR1':[1.5,4], 'SOHO_LASCO_C1':[1.1,3], 'SOHO_LASCO_C2':[2,6], 'SOHO_LASCO_C3':[3.7,32], 'STEREO_SECCHI_HI1':[15,80], 'STEREO_SECCHI_HI2':[80,215], 'STEREO_SECCHI_EUVI':[0,1.7],'SDO_AIA':[0,1.35]} 
    # Wireframe name to combo box index 
    WFname2id = {'GCS':1, 'Torus':2, 'Sphere':3, 'Half Sphere':4, 'Ellipse':5, 'Half Ellipse':6, 'Slab':7, 'Tube':8}
    
    
    #|-----------------------------| 
    #|---- Other Global Setup -----|
    #|-----------------------------|
    global mainwindow, pws, nSats, wfs, nwfs, bmodes, ovw
    
    #|---- Pull sat number from obsFiles ----|
    nSats = len(obsFiles)

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
    

    #|-----------------------------| 
    #|---- Setup observations -----|
    #|-----------------------------|
    sclIms = []    
    massIms = []   
    satStuff = []
    multiTime = False
    #|---- Loop through insts ----|
    for i in range(nSats):
        satScls = []
        someStuff = []
        someMims = []
        tNum = len(obsFiles[i][0])
        # If anyone has more than 1 then multitime mode
        if tNum > 1:
            multiTime = True

        #|---- Process each time step header ----|    
        for j in range(tNum): 
            mySatStuff = getSatStuff(obsFiles[i][0][j])
            if ('rsun' not in obsFiles[i][2][j]):
                 if 'RSUN_ARC' in obsFiles[i][2][j]:
                     obsFiles[i][2][j]['rsun'] =  obsFiles[i][2][j]['RSUN_ARC']
                 else:
                    calcRsun = mySatStuff['ONERSUN']*mySatStuff['SCALE']
                    if mySatStuff['OBSTYPE'] == 'HI':
                        calcRsun *= 3600
                    obsFiles[i][2][j]['rsun'] = calcRsun
                 
            someStuff.append(mySatStuff)
            
            if mySatStuff['OBSTYPE'] != 'EUV':
                #|---- Make mass images ----|
                # Use base difference not running
                massIm, hdrM = wM.TB2mass(obsFiles[i][1][j].data, obsFiles[i][2][j])
                #|--- Put mask on the mass im ---|
                if 'MASK' in mySatStuff:
                    massIm = (1-mySatStuff['MASK']) * massIm
            else:
                massIm = np.zeros(obsFiles[i][0][j].data.shape)
            someMims.append(np.transpose(massIm))   

        #|---- Get scaled versions of the data ----|     
        mySclIms, someStuffx2 = makeNiceMMs(obsFiles[i], someStuff) 
        
        #|---- Stuff in array ----|                 
        sclIms.append(mySclIms)
        satStuff.append(someStuffx2)
        massIms.append(someMims)
    
    #|---------------------------------| 
    #|---- Setup time slider vals -----|
    #|---------------------------------|
    if multiTime:
        nTsli, tlabs, tmaps = sortTimeIndices([satStuff[i][0] for i in range(len(satStuff))])
    else:
        nTsli = 0
        tlabs = None
    

    #|--------------------------------| 
    #|---- Get height slider max -----|
    #|--------------------------------|
    # Optimal range of height slider depends on what observations 
    # we have. Take the max corner dist ('FOV') from each inst
    # and convert it to a nice number
    maxFoV = 0
    for stuff in satStuff[0]:
        if stuff[0]['FOV'] > maxFoV: maxFoV = stuff[0]['FOV']
    # pad it a bit then round to a nice number
    maxFoV = int((1.1 * maxFoV) / 5) * 5
    # EUV only will pull 0 for this^, just set higher if < 1
    if maxFoV < 1:
        maxFoV = 1.5
    # Edit the dictionaries in wombatWF
    wf.rngDict['Height (Rs)'] = [1,maxFoV]
    if maxFoV > 20:
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
            myTmap = tmaps[i]
        else:
            myTmap = [0] 
        pw = FigWindow(obsFiles[i], sclIms[i], satStuff[i], massIms[i], myNum=i, labelPW=labelPW, tmap=myTmap, screenXY=screenXY)
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
    # Set time slider to highlight bc reload will
    # shift focus onto text box
    mainwindow.Tslider.setFocus()
    mainwindow.show()
    

    #|---------------------------------| 
    #|---- Set values from reload -----|
    #|---------------------------------|
    if type(reloadDict) != type(None):
        reloadIt(reloadDict)

    sys.exit(app.exec_())
    
