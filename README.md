#WOMBAT 
This repository contains the source code and documentation for the WOMBAT (Wireframe Object Mapped onto Background Alignment Tool) and DINGO (Density Inferred from Nice Grid Object) packages. The How2Wombat pdf provides an extensive overview of using WOMBAT and DINGO. The slides pdf gives a simplified walkthrough and we only present a basic outline in this readme.

##File Descriptions
There are 5 main WOMBAT procedures (.py files) and one for DINGO. All of these should be run by direct call at the command line, with the exception of wombatMovie as it has an extensive number of free parameters to edit in the top portion. The prepCode folder contains python programs to prepare the observations for viewing within the WOMBAT GUI. Much of this is based on the original IDL Solarsoft routines (e.g. secchi\_prep). The wombatCode folder contains python subroutines used by the six main procedures. 

##Installation
WOMBAT primarily uses common packages that most Python users likely already have installed, with PyQT being one exception. We provide a requirements.txt that can be used with pip install or wombat.yml that can be used with conda to pull the specific versions needed. Note that the post-processing routine that makes movies may have a codec issue if pip is used for ffmpeg, see the long guide for more details 

In addition to the source code, there is a Zenodo repository (https://zenodo.org/records/18841079) with some supporting files required for image processing. The two zip files should be opened and placed in the main wombat directory (i.e. wombat/spiceKernels/ and wombat/prepFiles/).

The wombat package itself does not need to be compiled or further installed before running the components.

##WOMBAT and DINGO Usage
WOMBAT is a full package for retreiving and preparing observations for use in the GUI and for post-processing results from reconstructions logged in a file. The general order of operations is
1. Pull the observations using wombatPull. This step is optional if you already have data so long as it is the correct format and located in the correct spot.
2. Process the observations into a binary save file/pickle using wombatProcess. This step is required.
3. Launch the GUI using wombatWrapper and a pickle from the processing script
Within the GUI reconstructions can be logged into a text file. This information, along with the pickle, can be used with the three other scripts to create post-processing images and movies. The available options are
- wombatPlots: basic height-time profiles and evolution of other wireframe parameters and kinematics/energetics
- wombatMovie: movies showing the wireframe reconstructions projected onto background observations
- dingo: figures showing the density within the wireframe structure in various manners

More details are available in How2Wombat.pdf and How2Wombat\_slides.pdf