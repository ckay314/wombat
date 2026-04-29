import os, sys
import numpy as np

def cleanItUp(folder):
    files = os.listdir(folder)
    # Check if name file is there
    if 'img_hdr.txt' not in files:
        sys.exit('Need img_hdr.txt file to rename fits')
    
    deets = np.genfromtxt(folder+'img_hdr.txt', dtype=str)
    files.remove('img_hdr.txt')
    
    for aF in files:
        myIdx = np.where(deets[:,0] == aF)[0][0]
        myDate = deets[myIdx, 1].replace('/','')
        myTime = deets[myIdx, 2].replace(':','')[:4]
        myInst = deets[myIdx, 3]
        myDir  = 'pullFolder/SOHO/LASCO/'+myInst+'/' 
        myName = myDate+'T'+myTime+'_'+myInst+'_'+aF
        
        os.replace(folder+aF,myDir+myName)
    
    
if __name__ == '__main__':
    folder = 'pullFolder/SOHO/LASCO/notDated/'
    # Check if given a folder
    if len(sys.argv[1:]) == 1:
        folder = sys.argv[1]
    cleanItUp(folder)
    