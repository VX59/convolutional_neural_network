from general_sorting import *
import numpy as np
import random as rand
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN)

gcnns = General_Sorting(48,12)
t = gcnns.G16X8
t.preprocess(EMBOSS)
t.load_preprocessed_data()

enssemble = []

#t.unpack_folders('dataset_folder/cropped_images/','dataset_2/')
for i in range(8):
   t.rename_model('t'+str(i))
   enssemble.append(t.train(False,k=(i)))


print(enssemble)
print(np.average(enssemble))
print(np.argmax(enssemble))