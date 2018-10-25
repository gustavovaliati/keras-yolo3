#!/usr/bin/python3

import scipy.io as sio
import numpy as np

f_original = open('bbox.txt', 'r');

tmp_array = [];

# columns
# left top right bot bbox frame
first = True;
max_frame = -1;
for line in f_original:
    if first:
        first = False;
        continue;

    data = line.rstrip().split(' ');
    data = list(map(float, data))

    # Set frames to start in 1 instead of zero.
    data[5] = data[5] + 1

    #max frame
    if (data[5] > max_frame):
        max_frame = data[5]

    tmp_array.append(data)

mapping = {}

for row in tmp_array:
    fr = row[5] - 1; #ajust index because here arrays starts in zero.
    if not fr in mapping:
        mapping[fr] = [row];
    else:
        mapping[fr] = np.concatenate((mapping[fr], [np.array(row)]), axis=0);


np_array = np.zeros(int(max_frame), dtype=object);
print(np_array)
for key in mapping:
    key = int(key)
    val = mapping[key]
    np_array[key] = mapping[key]


for i,row in enumerate(np_array):
    if isinstance( row, int ): # is zero (empty)
        # np_array[i] = np.array([0.,0.,0.,0.,0.,0.])
        np_array[i] = []
    print(i,row)

sio.savemat('grv.mat', {'detections':np_array})
