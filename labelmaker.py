import os
from os import path





prefix = 'real'
data_dir = os.listdir('real_data')
file = open(prefix+'real_labels.txt','w')

classes = 4
samples = len(data_dir)

for i in range(classes):
    for j in range(int(samples/classes)):
        if j + 1 != classes*int(samples/classes): file.write(str(i)+'\n')
        else: file.write(str(i))
file.close()