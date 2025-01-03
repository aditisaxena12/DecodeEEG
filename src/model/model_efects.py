import tensorflow as tf
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Embedding, Concatenate, Dropout, Lambda, TimeDistributed, LSTM, GlobalAveragePooling2D, Reshape, DepthwiseConv2D, Permute, Activation, ConvLSTM2D
from tensorflow.keras.regularizers import l2
import numpy as np                 

#this combines all models and is able to add/remove 
def model_ccn(l2_dense=0.001, l2_conv=0.001, dropout=0.2, l2_emb=0.0001, num_sub=10, num_blocks=16540, dim=[1,26,26,17], emb_rem_index=0, feature_vector_dim=512):
    #############################Inputs#########################################
    #input_1: the electrodes data
    #dim[1], dim[2], dim[3]: number of time points, number of frequencies, and number of electrodes
    input1 = Input(shape=(dim[1], dim[2], dim[3]), name='Spectrograms') 
    
    #MB is the minibatch size
    MB =  tf.shape(input1)[0]
    
    #this converts (None, t, f, e) to (t, f, None, e), i.e. it makes spectrogram volumes (where the depth is the number of training examples), and the volumes are packed across e electrodes
    input_new = tf.transpose(input1, perm=[1, 2, 0, 3])
    #(required shape for depthwise convolution: (1, h, w, batch*numfilter))
    spectograms_input = tf.reshape(input_new, [1, dim[1], dim[2], dim[3]*MB])
    
    #input 2: is a scalar, it is a subject index
    sub_pk_input = Input(shape=[], name='Subject_pk')
    #input 3: this is a scalar/dataset_emb 
    bl_pk_input = Input(shape=[], name='Block_pk')
    
    ###########################Embeddings#######################################
    ##emb_rem_index: the embedding that are to be removed
    #1: remove subject embedding
    #2: time-frequency embedding
    #3: remove electrode embedding    
    #4: remove all the embeddings
    #5: remove all the embeddings appended at the end (i.e., subject and dataset)
    #else don't remove any embeddings
    sub_emb = None
    
    kernel_tf = None
    kernel_elec = None
    x = None
    
    #1: subject Embedding
    if emb_rem_index != 1 and emb_rem_index != 4 and emb_rem_index != 5:
      #subject dimension = 5
      sub_emb = Embedding(num_sub+1, 5, input_length = 1, name = 'Subject_Embedding')(sub_pk_input)

    #4: electrode embedding
    #Linear Combination of the channels/ subject wise to account for correcting the position of electrodes)
    #Goal is to make a (1x1xe)@e filters
    if emb_rem_index != 3 and emb_rem_index != 4:
      kernel_elec = Embedding(num_blocks+1, dim[3]*dim[3], input_length=1, name='Electrode_Embedding')(bl_pk_input)
      kernel_elec = tf.reshape(kernel_elec, [MB, 1, 1, dim[3], dim[3]], name = 'Reshape_kernel_elec')
      kernel_elec = tf.transpose(kernel_elec, perm = [1, 2, 0, 3, 4])
      kernel_elec = tf.reshape(kernel_elec, [1, 1, dim[3]*MB, dim[3]])


      #Data Preprocessing: Embedding related
      #Calculates group wise convolution using depth_wise convolution (as conv2D convolves across the entire mini batch) 
      x = None
      for i in range(dim[3]):
        #the depthwise equivalent is a sum of the filter of depth_wise kernel o/p
        kern_temp = tf.reshape(kernel_elec[:, :, :, i], [1, 1, dim[3]*MB, 1])
        y_temp = tf.nn.depthwise_conv2d(spectograms_input, filter = kern_temp, strides = [1,1,1,1], padding = 'SAME')
        #sums across the filter dimension of the tensor (this this accounts for a single filter of a group wise convolution)
        y_temp = tf.reshape(y_temp, [1, dim[1], dim[2], MB, dim[3]])
        y_temp = tf.math.reduce_sum(y_temp, axis = -1, keepdims = True)
        if x == None:
          x = y_temp
        else:
          x = tf.concat([x, y_temp], axis=-1)
    
      x = tf.reshape(x, [1, dim[1], dim[2], dim[3]*MB])
      #this is of the form (1, t, f, None), as required
    else:
      x = spectograms_input  
    
    #3: time-frequency embedding
    #this is the high dimensional embedding for accounting variabilty in recordings    
    #(3x3xe)@1: depthwise convolution
    if emb_rem_index != 2 and emb_rem_index != 4: 
      kernel_tf = Embedding(num_blocks+1, 3*3*dim[3], input_length = 1, embeddings_regularizer=l2(l2_emb), name = 'Time_Frequency_Embedding')(bl_pk_input)
      kernel_tf = tf.reshape(kernel_tf, [MB, 3, 3, dim[3], 1], name = 'Reshape_kernel_tf')
      kernel_tf = tf.transpose(kernel_tf, perm = [1, 2, 0, 3, 4])
      kernel_tf = tf.reshape(kernel_tf, [3, 3, dim[3]*MB, 1])

      x = tf.nn.depthwise_conv2d(x, filter = kernel_tf, strides = [1,1,1,1], padding = 'SAME')
      x = Activation('elu')(x)
    
    ##########################Common Network####################################
    #reshaping for final computation    
    x = tf.reshape(x, [dim[1], dim[2], MB, dim[3]])
    x = tf.transpose(x, perm = [2, 0, 1, 3], name = 'after_all_preprocess')
    
    max_filter1 = (3, 2)
    max_filter2 = (3, 2)
    max_filter3 = (2, 2)


    x = Conv2D(96, (7, 7), activation='elu', padding = 'same', kernel_regularizer=l2(l2_conv), name = 'CONV_1')(x)
    x = MaxPool2D(max_filter1, name = 'MaxPool_1')(x)

    x = Conv2D(64, (5, 5), activation='elu', padding = 'same', kernel_regularizer=l2(l2_conv), name = 'CONV_2')(x)
    x = MaxPool2D(max_filter2, name = 'MaxPool_2')(x)
    
    x = Conv2D(32, (3, 3), activation='elu', padding = 'same', kernel_regularizer=l2(l2_conv), name = 'CONV_3')(x)
    x = MaxPool2D(max_filter3, name = 'MaxPool_3')(x)
    
    x = Flatten(name = 'Flatten')(x)
    x = Dropout(dropout, name = 'Dropout')(x)
    x = Dense(40, activation ='elu', kernel_regularizer=l2(l2_dense), name = 'Dense')(x)    

    #########################Final layers#######################################
    x2 = None
    #First append the subject embedding
    if sub_emb == None:
      x2 = x
    else:
      x2 = Concatenate(name = 'Concatenate')([x, sub_emb])
    
    #Classification
    #output = Dense(1, activation ='softmax', name = 'Output')(x2)
    output = Dense(feature_vector_dim, name = 'Output')(x2)
    
    return Model(inputs={"input_1": input1, "input_2": sub_pk_input, "input_3": bl_pk_input}, outputs=output)
    
if __name__ == '__main__':
  m = model_ccn()
  m.compile()
  m.summary()
  plot_model(m, to_file='/home/aditis/decodingEEG/DecodeEEG/data/results/model_plot_efects.png', show_shapes=True, show_layer_names=True) 
