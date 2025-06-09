# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 16:18:53 2025

@author: oargell
"""


"""
original Script for a tracking algorithm based on mask overlap. Use the commented plt.imshow blocks to visualize each step of the tracking process
"""


# imports
import os
import numpy as np
from skimage.io import imread
from scipy.stats import mode
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.morphology import thin, dilation, opening #  binary_openin, and disk taken out
import time
import scipy.io as sio
import pickle

# Define paths 
pos = 'Pos13_1_B' # name of folder, could be made into a loop
path = '/Users\oargell\Documents\Toy_datasets\Pos13_1_B/' # absolute path to the folder with the masks
sav_path = '/Users\oargell\Desktop/' # absolute path to the folder where teh results will be saved


#Helper Functions: directly in the script for ease of checking

# 1. Binarization function
def binar(IS1):
    IS1B = IS1.copy()
    IS1B[IS1 != 0] = 1
    return IS1B

# 2. Small segmentation artifact removal funtion
def remove_artif(I2A,disk_size):  

    I2AA=np.copy(I2A) #   plt.imshow(IS2)
    # Applying logical operation and morphological opening
    I2A1 = binar(I2A);#binar(I2A) plt.imshow(I2A1)     plt.imshow(I2A)

    # Create a disk-shaped structuring element with radius 3
    selem = disk(disk_size)
    # Perform morphological opening
    I2B = opening(I2A1, selem)
       
    # Morphological dilation   plt.imshow(I2B)
    I2C = dilation(I2B, disk(disk_size))  # Adjust the disk size as needed

    I3 = I2AA * I2C # plt.imshow(I3)

    # Extract unique objects
    objs = np.unique(I3)
    objs = objs[1:len(objs)]
    
    # Initialize an image of zeros with the same size as I2A
    I4 = np.uint16(np.zeros((I3.shape[0], I3.shape[1])))
    # Mapping the original image values where they match the unique objects
    AZ=1
    for obj in objs:
        I4[I2A == obj] = AZ
        AZ=AZ+1
    
    return I4

# 3. Reindexing function
def OAM_23121_tp3(M, cel, no_obj1, A):
    tp3 = np.array(M)  # Ensure M is a numpy array
    tp3[tp3 == cel] = no_obj1 + A
    return tp3


# Load ART masks into a tensor
path_dir = [f for f in sorted(os.listdir(path)) if f.endswith('_ART_masks.tif')] # specify the mask ending, here: '_ART_masks.tif'
Masks_Tensor = [imread(os.path.join(path, f)).astype(np.uint16) for f in path_dir]# plt.imshow(Mask3[0])
    
im_no1 = 0
im_no = len(Masks_Tensor)
mm = range(im_no) # time points to track
disk_size = 6 # Defines filter threshold for excluding segmentation artifacts due to interpolation, 3 if the objects are ~ 500 pixels, 6 if object are ~ 2000 pixels

"""
Load the first mask that begins the indexing for all the cells; IS1 is updated to the most recently processed tracked mask at the end of it0
"""    
IS1 = np.copy(Masks_Tensor[im_no1]).astype('uint16') # start tracking at first time point # plt.imshow(IS1)
IS1 = remove_artif(IS1, disk_size) # remove artifacts and start tracking at first time point # plt.imshow(IS1) # # DISK SIZE WAS INITIALIZED AT THE BEGINNING 
masks = np.zeros((IS1.shape[0], IS1.shape[1], im_no)) # contains the re-labeled masks according to labels in the last tp mask
masks[:,:,im_no1] = IS1.copy() # first time point defines indexing; IS1 is first segmentation output


IblankG = np.zeros(IS1.shape, dtype="uint16") # this matrix will contain cells with segmentation gaps, IblankG is updated within the loops 
tic = time.time()

# main tracking loop cells fomr IS1 (previous segmentation) are tracked onto IS2 (Next segmentation)

for it0 in mm: # notice IS1 will be updated at each loop iteration
    print(f'it0={it0}')
    # Load the segmentation mask for the next frame that will be re-indexed at the end of the loop
    IS2 = np.copy(Masks_Tensor[it0]).astype('uint16') # plt.imshow(IS2)  
    IS2 = remove_artif(IS2, disk_size) # remove small segmentation artifacts to focus tracking on real objects 
    
    IS2C = np.copy(IS2) # plt.imshow(IS2C) # <--- a copy of IS2, gets updated in it1 loop
    IS1B = binar(IS1) # biarization of the previous frame
    
    IS3 = IS1B.astype('uint16') * IS2 # previous binary segmentation superimposed on the next segmentaion; updated in it1
    tr_cells = np.unique(IS1[IS1 != 0]) # the tracked cells present in the present mask, IS1
    
    gap_cells = np.unique(IblankG[IblankG != 0]) # the tracked cells that had a gap in their segmentation; were not detected in IS1, gets updated in it1 loop
    cells_tr = np.concatenate((tr_cells, gap_cells)) # all the cells that have been tracked up to this tp for this position

    # Allocate space for the re-indexed IS2 according to tracking
    Iblank0 = np.zeros_like(IS1)
    
    # Go to the previously tracked cells and find corresponding index in current tp being processed, IS2 becomes Iblank0
    
    if cells_tr.sum() != 0: # required in case the mask does not have anymore cells to track
        for it1 in np.sort(cells_tr): # cells are processed according to the index
            IS5 = (IS1 == it1).copy() # binary image of previous mask containing only cell # it1
            IS6A = np.uint16(thin(IS5, max_num_iter=1)) * IS3 # find the overlap area between the binary image of cell # it1 and IS3 (previous binary image superimposed on the next segmentation)

            if IS5.sum() == 0: # if the cell was missing in the original previous mask; use the previous tracked masks instead
                IS5 = (IblankG == it1).copy()
                IS6A = np.uint16(thin(IS5, max_num_iter=1)) * IS2C
                IblankG[IblankG == it1] = 0 # remove the cell from the segmentation mask, updated for next iteration 

           
            if IS6A.sum() != 0: # Find the tracked cell's corresponding index in IS2, update IS3 and IS2C 
                IS2ind = 0 if not IS6A[IS6A != 0].any() else mode(IS6A[IS6A != 0])[0]
                Iblank0[IS2 == IS2ind] = it1
                IS3[IS3 == IS2ind] = 0
                IS2C[IS2 == IS2ind] = 0

        # Define cells with segmentation gap, update IblankG, the segmentation gap mask
        seg_gap = np.setdiff1d(tr_cells, np.unique(Iblank0)) # cells in the past mask (IS1), that were not found in IS2 

        if seg_gap.size > 0:
            for itG in seg_gap:
                IblankG[IS1 == itG] = itG

        # Define cells that were not relabelled in IS2; these are the new born cells or cells entering the frame
        Iblank0B = Iblank0.copy()
        Iblank0B[Iblank0 != 0] = 1
        ISB = IS2 * np.uint16(1 - Iblank0B)
        
        # Add the new cells to Iblank0, Iblank0 becomes Iblank, the final tracked segmentation with all cells added
        newcells = np.unique(ISB[ISB != 0])
        Iblank = Iblank0.copy()
        A = 1

        if newcells.size > 0:
            for it2 in newcells:
                Iblank[IS2 == it2] = np.max(cells_tr) + A # create new index that hasn't been present in tracking
                A += 1

        masks[:, :, it0] = np.uint16(Iblank).copy() #<---convert tracked mask to uint16 and store
        IS1 = masks[:, :, it0].copy() # IS1, past mask, is updated for next iteration of it0

    else:
        masks[:, :, it0] = IS2.copy()
        IS1 = IS2.copy()

toc = time.time()
print(f'Elapsed time is {toc - tic} seconds.')


"""

 Addtional corrections and visualization to confirm the cells are OK
 
"""


"""
Vizualize all cell tracks as single heatmap "All Ob"
"""
obj = np.unique(masks)
no_obj = int(np.max(obj))
im_no = masks.shape[2]
all_ob = np.zeros((no_obj, im_no))

tic = time.time()

for ccell in range(1, no_obj + 1):
    Maa = (masks == ccell)

    for i in range(im_no):
        pix = np.sum(Maa[:, :, i])
        all_ob[ccell-1, i] = pix
        
        

plt.figure()
plt.imshow(all_ob, aspect='auto', cmap='viridis', interpolation="nearest")
plt.title("all_obj")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()

"""
Ensure all cells numbered 1-n if necessary: the tracking loop assigns a unique index to all cells, but they might not be continous 
"""

im_no = masks.shape[2]
# Find all unique non-zero cell identifiers across all time points
ccell2 = np.unique(masks[masks != 0])
# Initialize Mask2 with zeros of the same shape as masks
Mask2 = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))

# Process each unique cell ID
for itt3 in range(len(ccell2)):  # cells
    pix3 = np.where(masks == ccell2[itt3])
    Mask2[pix3] = itt3 + 1  # ID starts from 1

"""
Vizualize all cell tracks as single binary heatmap "All Ob1"
"""

# Get cell presence
Mask3 = Mask2.copy()
numbM = im_no
obj = np.unique(Mask3)
no_obj1 = int(obj.max())
A = 1

tp_im = np.zeros((no_obj1, im_no))

for cel in range(1, no_obj1+1):
    Ma = (Mask3 == cel)

    for ih in range(numbM):
        if Ma[:, :, ih].sum() != 0:
            tp_im[cel-1, ih] = 1


plt.figure()
plt.imshow(tp_im, aspect='auto', interpolation="nearest")
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()


"""
Split Interrupted tracks into tracks if necessary: use this if gaps of a couple of frames are not desired 
"""

tic = time.time()
for cel in range(1, no_obj1+1):
    print(cel)
    tp_im2 = np.diff(tp_im[cel-1, :])
    tp1 = np.where(tp_im2 == 1)[0]
    tp2 = np.where(tp_im2 == -1)[0]
    maxp = (Mask3[:, :, numbM - 1] == cel).sum()

    if len(tp1) == 1 and len(tp2) == 1 and maxp != 0:  # has one interruption
        for itx in range(tp1[0], numbM):
            tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
            Mask3[:, :, itx] = tp3.copy()
        no_obj1 += A
    
    elif len(tp1) == 1 and len(tp2) == 1 and maxp == 0:  # has one interruption
        pass
    
    elif len(tp1) == len(tp2) + 1 and maxp != 0:
        tp2 = np.append(tp2, numbM-1)

        for itb in range(1, len(tp1)):  # starts at 2 because the first cell index remains unchanged
            for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                Mask3[:, :, itx] = tp3.copy()
            no_obj1 += A
    
    elif len(tp2) == 0 or len(tp1) == 0:  # it's a normal cell, it's born and stays until the end
        pass
    
    elif len(tp1) == len(tp2):
        if tp1[0] > tp2[0]:
            tp2 = np.append(tp2, numbM-1) #check this throughly
            for itb in range(len(tp1)):
                for itx in range(tp1[itb]+1, tp2[itb + 1] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A) #+1 here
                    Mask3[:, :, itx] = tp3.copy()    
                no_obj1 += A
        elif tp1[0] < tp2[0]:
            for itb in range(1, len(tp1)): 
                for itx in range(tp1[itb] + 1, tp2[itb] + 1):  # Inclusive range
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3.copy()
                no_obj1 += A
        elif len(tp2) > 1:
            for itb in range(1, len(tp1)):
                for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3.copy()    
                no_obj1 += A
toc = time.time()
print(f'Elapsed time is {toc - tic} seconds.')


"""
Vizualize all cell tracks as single binary heatmap "tp_im"
"""
numbM = im_no
obj = np.unique(Mask3)

# Get cell presence 2
tp_im = np.zeros((int(max(obj)), im_no))

for cel in range(1, int(max(obj)) + 1):
    Ma = (Mask3 == cel)

    for ih in range(numbM):
        if Ma[:, :, ih].sum() != 0:
            tp_im[cel-1, ih] = 1


plt.figure()
plt.imshow(tp_im, aspect='auto', interpolation="nearest")
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()



"""
Exclude tracks that are not continous during the whole experiment if necessary

"""
cell_artifacts = np.zeros(tp_im.shape[0])

for it05 in range(tp_im.shape[0]):
    arti = np.where(np.diff(tp_im[it05, :]) == -1)[0]  # Find artifacts in the time series

    if arti.size > 0:
        cell_artifacts[it05] = it05 + 1  # Mark cells with artifacts

goodcells = np.setdiff1d(np.arange(1, tp_im.shape[0] + 1), cell_artifacts[cell_artifacts != 0])  # Identify good cells


im_no = Mask3.shape[2]
Mask4 = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))

for itt3 in range(goodcells.size):
    pix3 = np.where(Mask3 == goodcells[itt3])
    Mask4[pix3] = itt3 + 1  # IDs start from 1


"""
Vizualize all cell tracks as single binary heatmap "tp_im2"
"""

Mask5 = Mask4.copy()
numbM = im_no
obj = np.unique(Mask4)
no_obj1 = int(obj.max())
A = 1

tp_im2 = np.zeros((no_obj1, im_no))

for cel in range(1, no_obj1+1):
    Ma = (Mask5 == cel)

    for ih in range(numbM):
        if Ma[:, :, ih].sum() != 0:
            tp_im2[cel-1, ih] = 1

plt.figure()
plt.imshow(tp_im2, aspect='auto', interpolation="nearest")
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()


"""
Sort cell tracks according to their lentgh if necessary 
"""

cell_exists0 = np.zeros((2, tp_im2.shape[0]))
for itt2 in range(tp_im2.shape[0]):
    # Find indices of non-zero elements
    non_zero_indices = np.where(tp_im2[itt2, :] != 0)[0]
    
    # If there are non-zero elements, get first and last
    if non_zero_indices.size > 0:
        first_non_zero = non_zero_indices[0]
        last_non_zero = non_zero_indices[-1]
    else:
        first_non_zero = -1  # Or any placeholder value for rows without non-zero elements
        last_non_zero = -1   # Or any placeholder value for rows without non-zero elements
    
    cell_exists0[:, itt2] = [first_non_zero, last_non_zero]

sortOrder = sorted(range(cell_exists0.shape[1]), key=lambda i: cell_exists0[0, i])

cell_exists = cell_exists0[:, sortOrder]
art_cell_exists = cell_exists

Mask6 = np.zeros_like(Mask5)
    
for itt3 in range(len(sortOrder)):
    pix3 = np.where(Mask5 == sortOrder[itt3] + 1)  # here
    Mask6[pix3] = itt3 + 1# reassign

"""
Vizualize all cell tracks as single binary heatmap "tp_im3"
"""
Mask7 = Mask6.copy()
numbM = im_no
obj = np.unique(Mask6)
no_obj1 = int(obj.max())
A = 1

tic = time.time()
tp_im3 = np.zeros((no_obj1, im_no))
for cel in range(1, no_obj1 + 1):
    tp_im3[cel - 1, :] = ((Mask7 == cel).sum(axis=(0, 1)) != 0).astype(int)
toc = time.time()
print(f'Elapsed time is {toc - tic} seconds.')

plt.figure()
plt.imshow(tp_im3, aspect='auto', interpolation="nearest")
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()

"""
Calculate object size if necessary
"""

# 
obj = np.unique(Mask7)
no_obj = int(np.max(obj))
im_no = Mask7.shape[2]
all_ob = np.zeros((no_obj, im_no))

tic = time.time()
for ccell in range(1, no_obj + 1):
    Maa = (Mask7 == ccell)

    for i in range(im_no):
        pix = np.sum(Maa[:, :, i])
        all_ob[ccell-1, i] = pix
toc = time.time()
print(f'Elapsed time is {toc - tic} seconds.')


"""
Variables for saving with the tracked masks
"""
Final_unique_objects = np.unique(Mask7)
Final_number_objects = int(np.max(obj))
Final_number_frames = Mask7.shape[2]
Final_Object_Size = all_ob
Final_tracked_tensor = Mask7

plt.figure()
plt.imshow(Final_Object_Size , aspect='auto', cmap='viridis',interpolation="nearest")
plt.title("Cell Sizes Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")


"""
Save variables and tracks for MATLAB
"""
sio.savemat(os.path.join(sav_path, f'{pos}_ART_Tracks_MATLAB.mat'), {
    "Final_unique_objects": Final_unique_objects,
    "Final_number_objects": Final_number_objects,
    "Final_number_frames": Final_number_frames,
    "Final_Object_Size": Final_Object_Size,
    "Final_tracked_tensor": Final_tracked_tensor,
}, do_compression=True)



"""
Save tracks as tensor and other variables as pickle in Python
"""

Tracks_file = {"Final_unique_objects": Final_unique_objects,
      "Final_number_objects": Final_number_objects,
      "Final_number_frames": Final_number_frames,
      "Final_Object_Size": Final_Object_Size}

name_save=os.path.join(sav_path, f'{pos}_Tracks_vars_file.pklt')

with open(name_save, 'wb') as file:
    pickle.dump(Tracks_file, file)

name_save2=os.path.join(sav_path, f'{pos}_Tracks')
np.save(name_save2, Final_tracked_tensor)


# check tracks are saved oK by Loading data from the file
with open('/Users\oargell\Desktop/Pos13_1_B_Tracks_file.pklt', 'rb') as file:
    loaded_data = pickle.load(file)

loaded_tensor = np.load('/Users\oargell\Desktop/Pos13_1_B_Tracks.npy')
print(loaded_data)
print(loaded_tensor)


plt.figure()
plt.imshow(loaded_tensor[:,:,0] , aspect='auto', cmap='viridis',interpolation="nearest")
plt.title("Cell Sizes Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")


