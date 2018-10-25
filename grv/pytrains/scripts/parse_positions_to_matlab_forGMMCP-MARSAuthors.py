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

    #max frame
    if (data[5] > max_frame):
        max_frame = data[5]
    '''
    parsing to fit the MARS's gmmcp implementation.
    pos 1 = x -> x1
    pos 2 = y -> y1
    pos 3 = w -> x2
    pos 4 = h -> y2
    '''
    data[2] = data[2] + data[0] - 1
    data[3] = data[3] + data[1] - 1

    tmp_array.append(data)

mapping = {}

for row in tmp_array:
    fr = row[5] - 1; #ajust index because here arrays starts in zero.
    if not fr in mapping:
        mapping[fr] = [row];
    else:
        mapping[fr] = np.concatenate((mapping[fr], [np.array(row)]), axis=0);

print(mapping[44])

np_array = np.zeros(int(max_frame), dtype=object);
print(np_array)
for key in mapping:
    key = int(key)
    val = mapping[key]
    np_array[key] = mapping[key]

print(np_array[44])

for i,row in enumerate(np_array):
    if isinstance( row, int ): # is zero (empty)
        # np_array[i] = np.array([0.,0.,0.,0.,0.,0.])
        np_array[i] = []
    print(i,row)

sio.savemat('grv-mars.mat', {'detections':np_array})
