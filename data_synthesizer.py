import os
from PIL import Image, ImageFilter
import shutil
from tqdm import tqdm
import random
target='2X2_sorter/train_synth'
working_dir = "2X2_sorter/train_data/"
SCALE = 148
tr = 0
r = 2

translations = 2
step = 2

def overwrite():
    shutil.rmtree(target+'_data/')
    os.remove(target+'_log.txt')
#overwrite()
dir_list = os.listdir(working_dir)  
dir_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))   # classes

#output_dir = target+'_data/'
log =  open(target+'_log.txt','a')

for j in tqdm (range (len(dir_list)), desc="Loading..."):    
    a = j*8
    file = dir_list[j]
    filepath = working_dir+'/'+file
    image = Image.open(filepath)
    for i in range(translations*j,step*(j+1)):
        image = image.rotate(r,translate=(tr,tr))
        image.save(target+'_data/'+str(a+i)+'.png')
        log.write(str(a+i)+'.png'+'\n')
    image = Image.open(filepath)
    for i in range(translations*j,step*(j+1)):
        image = image.rotate(-r,translate=(tr,-tr))
        image.save(target+'_data/'+str(a+i)+'.png')
        log.write(str(a+i)+'.png'+'\n')
    image = Image.open(filepath)
    for i in range(translations*j,step*(j+1)):
        image = image.rotate(-r,translate=(-tr,tr))
        image.save(target+'_data/'+str(a+i)+'.png')
        log.write(str(a+i)+'.png'+'\n')
    image = Image.open(filepath)
    for i in range(translations*j,step*(j+1)):
        image = image.rotate(r,translate=(-tr,-tr))
        image.save(target+'_data/'+str(a+i)+'.png')
        log.write(str(a+i)+'.png'+'\n')
"""
    image = Image.open(filepath)
    for i in range(8,10):
        image = image.rotate(0,translate=(tr,tr))
        image.save(target+'_data/'+str(a+i)+'.png')
        log.write(str(a+i)+'.png'+'\n')
    image = Image.open(filepath)
    for i in range(10,12):
        image = image.rotate(0,translate=(-tr,tr))
        image.save(target+'_data/'+str(a+i)+'.png')
        log.write(str(a+i)+'.png'+'\n')
    image = Image.open(filepath)
    for i in range(12,14):
        image = image.rotate(0,translate=(-tr,-tr))
        image.save(target+'_data/'+str(a+i)+'.png')
        log.write(str(a+i)+'.png'+'\n')
    image = Image.open(filepath)
    for i in range(14,16):
        image = image.rotate(0,translate=(tr,-tr))
        image.save(target+'_data/'+str(a+i)+'.png')
        log.write(str(a+i)+'.png'+'\n')
"""
print('created '+target+' set')