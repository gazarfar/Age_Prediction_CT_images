# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:45:03 2023
By running this file you can creat and saves the PC heatmap for all the images in a tfrecordfile
located at path2tfrecords
You will need to update path2tfrecords and path2model
@author: azarf
"""
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from PIL import ImageOps, Image
from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib.pyplot as plt

feature_data ={'NormalizeCompositeImage': tf.io.FixedLenFeature([256,256,3], tf.float32),
               'label': tf.io.FixedLenFeature([], tf.float32),
               'ID': tf.io.FixedLenFeature([], tf.int64)}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_data)

def normalize(input):
    case = input['NormalizeCompositeImage']
    label = input['label']
    ID = input['ID']
    case = tf.where(tf.math.is_nan(case), 0., case)
    mean, variance = tf.nn.moments(case, axes=[0,1])
    x_normed = (case - mean) / tf.sqrt(variance + 1e-8) # epsilon to avoid dividing by zero
    x_normed = tf.transpose(x_normed, perm=[2,0,1])
    y = tf.map_fn(fn=lambda chanel: tf.divide(chanel-tf.reduce_min(chanel),tf.reduce_max(chanel)-tf.reduce_min(chanel)), elems=x_normed, fn_output_signature=tf.float32)
    y = tf.transpose(y, perm=[1,2,0])
    return (y,label,ID)

def read_file(filename):
    raw_dataset  = tf.data.TFRecordDataset(filename)
    raw_dataset = raw_dataset.shuffle(len(filename))
    parsed_dataset = raw_dataset.map(_parse_function)
    parsed_dataset = parsed_dataset.map(normalize)
    return parsed_dataset

def generate_heatmap(myimage,gradModel_conv):
    with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            #inputs = tf.cast(image, tf.float32)
            input = tf.cast(myimage, tf.float32)
            [convOutputs,prediction] = gradModel_conv(input)
           
    # # use automatic differentiation to compute the gradients
    grads = tape.gradient(prediction,convOutputs)

    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (256, 256)
    heatmap = cam.numpy()
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    eps=1e-8
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = np.pad(heatmap,((0,1),(0,1)), mode = 'constant', constant_values=(0,0))
    im = Image.fromarray(np.uint8(cm.gist_earth(heatmap)*255))
    im = im.resize((w,h))
    im = ImageOps.grayscale(im)
    im = np.array(im)
    im = im.astype(np.float32)
    heatmap = im/np.max(im)
    return heatmap

def PC_Transform(X,pca,num_pc,heatmap,weight,num_pc_2_keep):
    X_score = pca.transform(X)
    X_score_weighted = X_score
    if num_pc != num_pc_2_keep: 
        X_score_weighted[:,0:num_pc_2_keep] = np.multiply(X_score[:,0:num_pc_2_keep],heatmap)
        X_score_weighted[:,num_pc_2_keep:X.shape[1]] = np.multiply(X_score[:,num_pc_2_keep:X.shape[1]],np.zeros((X.shape[0],1))+weight)
    inv_X_score = pca.inverse_transform(X_score_weighted)
    inv_X_score = np.reshape(inv_X_score,[256,256,X.shape[1]])
    return inv_X_score

slices ={}
slices['100002'] = [16,22,45,58,93,114]
slices['100216'] = [26,53,67,94,98,108]
slices['100439'] = [29,48,70,87,113,130]
slices['100803'] = [11,19,39,40,43,49]
slices['100681'] = [39,51,60,99,109,130]
slices['100913'] = [6,36,46,74,81,99]


# slices['100002'] = [4,16,22,38,45,58,69,72,89,93,102,114]
# slices['100216'] = [10,11,26,39,53,67,72,79,86,94,98,108,114,126]
# slices['100439'] = [13,29,34,48,54,65,70,75,87,97,107,113,124,130]
# slices['100803'] = [9,11,19,28,39,40,43,49]
# slices['100681'] = [16,18,19,23,39,49,51,60,79,80,99,109,117,126,130]
# slices['100913'] = [6,16,26,36,46,67,74,81,99,104]


path2model = 'C:\\Users\\azarf\\Documents\\AgePrediction_20230503\\my_model_fine2\\my_model_fine2\\'
path2tfrecords = 'C:\\Users\\azarf\\Documents\\Age_prediction_Spyder\\Tfrecords_path\\CompositeImage.tfrecords'




model = tf.keras.models.load_model(path2model)
Age_inception_resnet_v2 = model.layers[1]
gradModel_conv = Model(inputs=[Age_inception_resnet_v2.get_layer('input_1').input],outputs= [Age_inception_resnet_v2.get_layer('conv2d').output,Age_inception_resnet_v2.output])

dataset = read_file(path2tfrecords)
dataset = dataset.batch(40)

batch_count = sum(1 for _ in dataset)

age = []
CTscans_PC = []
patientid = []
scan_count = 0
for images, label, ID in dataset.take(batch_count):
    age = np.append(age,label.numpy()*25+52.5)
    scan_count = images.numpy().shape[0] + scan_count
    CTscans_PC = np.append(CTscans_PC,images.numpy())
    patientid = np.append(patientid,ID.numpy())

CTscans_PC = np.reshape(CTscans_PC,[scan_count,256,256,3])
PC_heatmap = np.zeros((scan_count,256,256))
for i in range(0,scan_count):
    tmp = CTscans_PC[i,:,:,:]
    PC_heatmap[i,:,:] = generate_heatmap(np.reshape(tmp,[1,256,256,3]),gradModel_conv)
    tmp = np.load('CTscan_cropped' + str(int(patientid[i]))+'.npz')
    CTScan = tmp['CTScan']
    Mask = tmp['Mask']
    pca = PCA(n_components = 50)
    data = np.reshape(CTScan,[256*256,CTScan.shape[2]])
    pca.fit(data)
    heat3_data = PC_Transform(data,pca,50,np.reshape(PC_heatmap[i,:,:],(256*256,1)),1,3)
    #np.savez('PC_heatmaps.npz', Score_Heatmap = PC_heatmap,Patient_Id = patientid, Age = age)


    plt.figure()
    count_sub = len(slices[str(int(patientid[i]))])+2
    mymax= np.max(CTscans_PC[i,:,:,1])
    mymin= np.min(CTscans_PC[i,:,:,1])
    plt.subplot(3,5,1)
    plt.imshow(CTscans_PC[i,:,:,1], cmap = 'gray', vmin = mymin, vmax = mymax)
    plt.title('PC_'+str(int(patientid[i])))
    mymax= np.max(PC_heatmap[i,:,:])
    mymin= np.min(PC_heatmap[i,:,:])
    plt.subplot(3,5,2)
    plt.imshow(PC_heatmap[i,:,:], cmap = 'jet', vmin = mymin, vmax = mymax)
    plt.title('PC_heat')
    for j in range(0,len(slices[str(int(patientid[i]))])): 
        mymax= np.max(CTScan[:,:,slices[str(int(patientid[i]))][j]])
        mymin= np.min(CTScan[:,:,slices[str(int(patientid[i]))][j]])
        plt.subplot(3,5,2*j+3)
        plt.imshow(CTScan[:,:,slices[str(int(patientid[i]))][j]], cmap = 'gray', vmin = mymin, vmax = mymax)
        plt.title('CT Slice' + str(slices[str(int(patientid[i]))][j]))
        mymax= np.max(heat3_data[:,:,slices[str(int(patientid[i]))][j]])
        mymin= np.min(heat3_data[:,:,slices[str(int(patientid[i]))][j]])
        plt.subplot(3,5,2*j+4)
        plt.imshow(heat3_data[:,:,slices[str(int(patientid[i]))][j]], cmap = 'jet', vmin = mymin, vmax = mymax)
        plt.title('Heatmap_Slice' + str(slices[str(int(patientid[i]))][j]))

    
#plt.imshow(heatmap, cmap = 'gray')
