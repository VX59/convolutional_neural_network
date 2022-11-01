from Sorters import Binary_Sorter,Quad_Sorter,Octal_Sorter,sixteen_2_16_Sorter,Sorter_32x16
import numpy as np
import random as rand
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN)

t = Quad_Sorter(input_size=100,dataset_start=0,name='t')

