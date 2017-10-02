"""
  @title: preprocess.py
  @version: 4/19/17
  @authors: Team 5
"""

# imports
import numpy as np
import pandas as pd
import dicom
import os
import cv2
import math
import time

# constants and globals
start_time = time.time()
data_dir = 'D:/stage1/stage1/' # obviously change this to the correct location
patients = os.listdir(data_dir)
labels = pd.read_csv('data/stage1_labels.csv', index_col = 0)
processed_data = []
img_dimension = 50
num_slices = 20

# function for pre-processing a single patient
def preprocess(patient):
    # reference for reading patient slices: https://www.kaggle.com/dfoozee/data-science-bowl-2017/full-preprocessing-tutorial
    patient_label = labels.get_value(patient, 'cancer')
    print("\tpatient ID: " + patient)
    print("\tpatient Label: " + str(patient_label))
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    chunk_slices = []
    slices = [cv2.resize(np.array(n.pixel_array),(img_dimension,img_dimension)) for n in slices]

    # reference for all of this chunking madness: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    # this code is pretty gross, TODO: clean this bit up
    chunk_sizes = math.floor(len(slices) / num_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        chunk_slices.append(slice_chunk)

    if len(chunk_slices) == num_slices - 1:
        chunk_slices.append(chunk_slices[-1])

    if len(chunk_slices) == num_slices - 2:
        chunk_slices.append(chunk_slices[-1])
        chunk_slices.append(chunk_slices[-1])

    if len(chunk_slices) == num_slices + 2:
        val = list(map(mean, zip(*[chunk_slices[num_slices-1],chunk_slices[num_slices],])))
        del chunk_slices[num_slices]
        chunk_slices[num_slices-1] = val
        
    if len(chunk_slices) == num_slices + 1:
        val = list(map(mean, zip(*[chunk_slices[num_slices-1],chunk_slices[num_slices],])))
        del chunk_slices[num_slices]
        chunk_slices[num_slices-1] = val

    # converting the csv label to an easier to use numpy array
    if patient_label == 1:
        # cancer
        patient_label = np.array([0,1])
    else:
        # not cancer
        patient_label = np.array([1,0])

    # return the 20 normalized slices and the corresponding patient label
    return np.array(chunk_slices), patient_label

# simple local helper mean function
def mean(n):
    return (sum(n) / len(n))

# create n-sized chunks from list l
def chunks(l, n):
    count = 0
    for i in range(0, len(l), n):
        if (count < num_slices):
            # fancy python yield statement
            yield l[i:i + n]
            count = count + 1

# iterate through all patients and preprocess their data
# add preprocessed data to processed_data list
for num, patient in enumerate(patients):
    print("Currently preprocessing patient " + str(num))
    new_time = time.time()
    try:
        patient_data, patient_label = preprocess(patient)
        processed_data.append([patient_data, patient_label])
        print("\tTime to preprocess: %s seconds." % (time.time() - new_time))
    except KeyError as e:
        print('\tError: unlabeled data, skipping patient...')

# save processed_data list into 'processed_data.npy' for cnn consumption
np.save('processed_data.npy', processed_data)
# how long dis take?
print("\nDone! Total running time: %s." % (time.time() - start_time))
