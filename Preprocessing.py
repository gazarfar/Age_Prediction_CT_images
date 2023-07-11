#This function read the dcm files preprocess the CT data and save them as 
#tfrecors file
import nibabel as nib
import os
import numpy as np
from sklearn.decomposition import PCA
from Mymask import mymask
import random
import pysptools.noise as ns
import pandas as pd
import tensorflow as tf
import pickle as pk


def lumTrans(img):
    # normalizes the input CT scan values to 0 to 255 values
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def pc_normalize(data):
    #normalizing PC components between 0 and 1
    mymax = np.max(data)
    mymin = -np.min(data)
    data[np.where(data >= 0)] = data[np.where(data >= 0)]/mymax
    data[np.where(data < 0)] = data[np.where(data < 0)]/mymin                                            
    return data

def MNF_reduce_component_2_noise_and_invert(data):
    # Reduce the second component noise and
    # return the inverse transform
    mnf = ns.MNF()
    tdata = mnf.apply(data)
    dn = ns.SavitzkyGolay()
    tdata[:,:,1:2] = dn.denoise_bands(tdata[:,:,1:2], 3, 2)
    # inverse_transform remove the PCA rotation,
    # we obtain a whitened cube with
    # a noise reduction for the second component
    return mnf.inverse_transform(tdata)

def data_reduc(case_pixels):
    #This function cancluates the PC components and returns the 
    #composite images
    pca = PCA(n_components=3)
    MaskV,y0, y1 = mymask(case_pixels)
    case_pixels = np.multiply(MaskV[y0:y1,:,:],case_pixels[y0:y1,:])
    MaskV = MaskV[y0:y1,:,:]
    randn = random.randrange(0,y1-y0-256)
    case_pixels = case_pixels[randn:randn+256,256:512,:]
    MaskV = MaskV[randn:randn+256,256:512,:]
    tmp = np.reshape(case_pixels,[256*256,case_pixels.shape[2]])
    mymean = np.mean(tmp,axis = 1, keepdims = True)
    tmp = (tmp-mymean)
    tmp = np.reshape(tmp,[256,256,case_pixels.shape[2]])
    tmp = MNF_reduce_component_2_noise_and_invert(tmp)
    tmp = np.reshape(tmp,[256*256,case_pixels.shape[2]])
    tmp = (tmp-mymean)
    pca.fit(tmp)
    myper = np.sum(pca.explained_variance_ratio_)
    tmp = pca.transform(tmp)
    Original_PC = np.reshape(tmp,[256,256,3])
    Normalized_PC = Original_PC
    if Normalized_PC[128,128,0] <= 0:
        Normalized_PC[128,128,0] = -Normalized_PC[128,128,0] 
    if Normalized_PC[128,128,1] <= 0:
        Normalized_PC[128,128,1] = -Normalized_PC[128,128,1]  
    if Normalized_PC[128,128,2] <= 0:
        Normalized_PC[128,128,2] = -Normalized_PC[128,128,2] 
    
    Normalized_PC[:,:,0] = pc_normalize(Normalized_PC[:,:,0])
    Normalized_PC[:,:,1] = pc_normalize(Normalized_PC[:,:,1])
    Normalized_PC[:,:,2] = pc_normalize(Normalized_PC[:,:,2])
    return Normalized_PC,Original_PC, myper, pca, MaskV,case_pixels


def read_age(patient_id):
    Clinicpath = 'C:\\Users\\azarf\\Documents\\Age_prediction_Spyder\\'
    Clinicname = 'nlst_780_prsn_idc_20210527.csv'
    col_list = ["age","pid"]
    Clinicdata = pd.read_csv(Clinicpath+Clinicname, usecols=col_list)
    MyBolninx = Clinicdata.pid.isin(patient_id)
    Clinicdata = Clinicdata.to_dict('list')
    Clinicdata = pd.DataFrame(Clinicdata, index = MyBolninx)
    Clinicdata = Clinicdata.loc[True];
    age = Clinicdata['age']
    return age


def _int64_ID(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_label(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_image(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
#path to CT scans in nii.gz format
path2niigz = "C:\\Users\\azarf\\Documents\\Age_prediction_Spyder\\niigzfiles\\"

#path to save the tfrecord files
CompisiteImage_path = 'C:\\Users\\azarf\\Documents\\Age_prediction_Spyder\\Tfrecords_path\\CompositeImage.tfrecords'



# open the tfrecord file
writer = tf.io.TFRecordWriter(CompisiteImage_path)

myfiles = os.listdir(path2niigz)

for i in range(0,len(myfiles)): 
    print(i)
    patient_id = int(myfiles[i][0:6])
    case = nib.load(path2niigz + myfiles[i])
    case_pixels = case.get_fdata()
    case_pixels=np.rot90(case_pixels,axes=(0,1))
    header=case.header
    spacing=np.array([header['pixdim'][3]]+[header['pixdim'][2]]+[header['pixdim'][1]], dtype=np.float32)
    case_pixels[np.isnan(case_pixels)]=-2000
    case_pixels = lumTrans(case_pixels)
    Normalized_PC,Original_PC, myper, pca, Mask, case_pixels = data_reduc(case_pixels)
    age = read_age([int(patient_id)])
    label = (age-52.5)/(77.5-52.5)
    np.save('Original_PC_'+ myfiles[i][0:6],Original_PC)
    pk.dump(pca, open("pca_" + myfiles[i][0:6] + ".pkl","wb"))

    feature = {'label': _float_label(label),
               'NormalizeCompositeImage': _float_image(Normalized_PC.ravel()),
               'ID': _int64_ID(patient_id)}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the tfrecord file
    writer.write(example.SerializeToString())
    np.savez('CTscan_cropped'+myfiles[i][0:6]+'.npz', CTScan = case_pixels, Mask = Mask )
    print(str(patient_id) +'_' + str(age))

# close the tfrecord file
writer.close()