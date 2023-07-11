import numpy as np
import scipy as sp
import scipy.ndimage
import cv2
from skimage import measure


def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array

def myborder(In):
    y0 = np.array(np.nonzero(In))[0,0]
    y1 = np.array(np.nonzero(In))[0,-1]
    x0 = np.array(np.nonzero(np.transpose(In,[1,0,2])))[0,0]
    x1 = np.array(np.nonzero(np.transpose(In,[1,0,2])))[0,-1]

    r = (y1-y0)/2
    Cy = (y0+y1)/2
    Cx = (x0 + x1)/2
    x = np.linspace(0,511,512)
    y = x
    xg, yg = np.meshgrid(x, y)
    xg = np.reshape(xg,[512,512])
    yg = np.reshape(yg,[512,512])
    output = np.power(xg-Cx,2)/pow(256,2) + np.power(yg-Cy,2)/pow(r-3,2)
    output[np.where((1 >= output))] =1
    output[np.where((1 < output))] =0
    return output



def mymask(In):
    x,y,z = In.shape 
    #border = myborder(In)
    mask =np.zeros((x,y,z))
    #print(z)
    y0 = 256
    y1 = 256
    for i in range(0,z):
        #case_pixels = np.multiply(border,In[:,:,i])
        #print(str(i))
        ret, tmp = cv2.threshold(np.uint16(In[:,:,i]), 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tmp = flood_fill(tmp,h_max=255)
        all_labels = measure.label(tmp)
        indx =np.where(all_labels != all_labels[256,256])
        tmp[indx]=0
        if np.array(np.nonzero(tmp))[0,0] < y0:
            y0 = np.array(np.nonzero(tmp))[0,0]
        if np.array(np.nonzero(tmp))[0,-1] > y1:
            y1 = np.array(np.nonzero(tmp))[0,-1]
        mask[:,:,i]=tmp/255
        
    return mask, y0, y1


def mask2D(In):
    x,y = In.shape 
    #border = myborder(In)
    mask =np.zeros((x,y))


    ret, tmp = cv2.threshold(np.uint16(In[:,:]), 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tmp = flood_fill(tmp,h_max=255)
    all_labels = measure.label(tmp)
    indx =np.where(all_labels != all_labels[128,0])
    tmp[indx]=0

    mask = tmp/255
    mask[0:200,0:250]=1
    mask[0:120,0:215]=1
    return mask
