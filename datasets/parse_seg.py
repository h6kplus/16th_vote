from typing import Tuple
import cv2
import json
import os
import numpy as np
from matplotlib import pyplot as plt

annot_file="data/test_dataset/train.json"
with open(annot_file,'r') as f:
    annot=json.load(f)

# print(annot)

for image in annot["images"]:
    img=cv2.imread(os.path.join("data/test_dataset/train",image["seg_name"]))
    # print(img.shape)
    # print(np.unique(img.reshape(-1,3),axis=0).shape)
    # exit()
    count=0
    print(img)
    # exit()
    for i in (range(img.shape[0])):
        for j in range(img.shape[1]):
            # print(i,j,img[i,j,:])
            found=False
            if img[i,j,0]==6:
                img[i,j,0]=255

            # if tuple(img[i,j,:]) in COLOR_TO_IDX:
            #     seg[i,j]=COLOR_TO_IDX[tuple(img[i,j,:])]
            # else:
            #     seg[i,j]=255
            #     count+=1
                
  
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()
    print(count)
    exit()

