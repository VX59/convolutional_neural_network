from Sorters import Quad_Sorter
import numpy as np
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN)

enssemble = []


t = Quad_Sorter(input_size=100,name='ha')
t.preprocess(EMBOSS)
enssemble.append(t.train(False,kfold=False))

t = Quad_Sorter(input_size=100,name='hb')
t.preprocess(EMBOSS)
enssemble.append(t.train(False,kfold=False))

t = Quad_Sorter(input_size=100,name='hc')
t.preprocess(EMBOSS)
enssemble.append(t.train(False,kfold=False))

t = Quad_Sorter(input_size=100,name='hd')
t.preprocess(EMBOSS)
enssemble.append(t.train(False,kfold=False))

print(enssemble)
print(np.average(enssemble))
print(np.argmax(enssemble))