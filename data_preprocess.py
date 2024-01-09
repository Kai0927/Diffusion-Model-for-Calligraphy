# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:58:32 2024

@author: K1
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import PIL

filename = glob.glob("D:\\D_file\\Research assistance\\Machine learning\\Diffusion_Figure\\dragon\\*.jpg")
filename.sort(key=lambda f: int(re.sub('\D', '', f)))
#filename = ["D:/D_file/Research assistance/Machine learning/Diffusion_Figure/dragon/img1.jpg"]

step = 0
for k in range(len(filename)):
    #img = io.imread(filename[k])
    step+=1
    print(filename[k])
    img = np.asarray(PIL.Image.open(filename[k]).convert('RGB'))
    img = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(type(img), img.shape)
    
    img[img > 100] = 255
    
    img_resize = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
                    
    plt.imshow(img_resize, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
    cv2.imwrite('D:\\D_file\\Research assistance\\Machine learning\\Diffusion_Figure\\output_dragon\\' + f'output{step}.jpg', img_resize)
