'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Nov 9, 2016
Restructured Feb 4, 2018 - get data
Restructured Sep 19, 2018 - DNN training
Restructured Oct 13, 2018 - DNN training
Restructured Feb 18, 2020 - UTI to STFT
Restructured March 2, 2020 - improvements by Laszlo Toth <tothl@inf.u-szeged.hu>
 - swish, scaling to [-1,1], network structure, SGD
Documentation May 5, 2020 - more comments added
Restructured Feb 8, 2021 - train on UltraSuite / TaL / TaL 1 & TaL80 corpora

Keras implementation of the UTI-to-STFT model of
Tamas Gabor Csapo, Csaba Zainko, Laszlo Toth, Gabor Gosztolya, Alexandra Marko,
,,Ultrasound-based Articulatory-to-Acosutic Mapping with WaveGlow Speech Synthesis'', submitted to Interspeech 2020.
-> this script is for training the STFT (Mel-Spectrogram) parameters
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
from detect_peaks import detect_peaks
import os
import os.path
import gc
import re
#import tgt
import csv
import datetime
import scipy
import pickle
import random
random.seed(17)
import skimage
from subprocess import call, check_output, run


# sample from Csaba
import WaveGlow_functions

import keras
from keras import regularizers
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, InputLayer, Dropout
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
# additional requirement: SPTK 3.8 or above in PATH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True 
set_session(tf.Session(config=config))

# defining the swish activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})


# read_ult reads in *.ult file from AAA
def read_ult(filename, NumVectors, PixPerVector):
    # read binary file
    ult_data = np.fromfile(filename, dtype='uint8')
    ult_data = np.reshape(ult_data, (-1, NumVectors, PixPerVector))
    return ult_data


def read_wav(filename):
    (Fs, x) = io_wav.read(filename)
    return (x, Fs)

def write_wav(x, Fs, filename):
    # scaled = np.int16(x / np.max(np.abs(x)) * 32767)
    io_wav.write(filename, Fs, np.int16(x))

# read_meta reads in *.txt ult metadata file from AAA
def read_param(filename):    
    NumVectors = 0
    PixPerVector = 0
    # read metadata from txt
    for line in open(filename):
        # 1st line: NumVectors=64
        if "NumVectors" in line:
            NumVectors = int(line[11:])
        # 2nd line: PixPerVector=842
        if "PixPerVector" in line:
            PixPerVector = int(line[13:])
        # 3rd line: ZeroOffset=210
        if "ZeroOffset" in line:
            ZeroOffset = int(line[11:])
        # 5th line: Angle=0,025
        if "Angle" in line:
            Angle = float(line[6:].replace(',', '.'))
        # 8th line: FramesPerSec=82,926
        # Warning: this FramesPerSec value is usually not real, use calculate_FramesPerSec function instead!
        if "FramesPerSec" in line:
            FramesPerSec = float(line[13:].replace(',', '.'))
        # 9th line: first frame
        # TimeInSecsOfFirstFrame=0.95846
        if "TimeInSecsOfFirstFrame" in line:
            TimeInSecsOfFirstFrame = float(line[23:].replace(',', '.'))
    
    return (NumVectors, PixPerVector, ZeroOffset, Angle, FramesPerSec, TimeInSecsOfFirstFrame)

def cut_and_resample_wav(filename_wav_in, Fs_target):
    filename_no_ext = filename_wav_in.replace('.wav', '')
    
    filename_param = filename_no_ext + '.param'
    filename_wav_out = filename_no_ext + '_cut_22k.wav'
    
    # resample speech using SoX
    command = 'sox ' + filename_wav_in + ' -r ' + str(Fs_target) + ' ' + \
              filename_no_ext + '_22k.wav'
    call(command, shell=True)
    
    # volume normalization using SoX
    command = 'sox --norm=-3 ' + filename_no_ext + '_22k.wav' + ' ' + \
              filename_no_ext + '_22k_volnorm.wav'
    call(command, shell=True)
    
    # cut from wav the signal the part where there are ultrasound frames
    (NumVectors, PixPerVector, ZeroOffset, Angle, FramesPerSec, TimeInSecsOfFirstFrame) = read_param(filename_param)
    (speech_wav_data, Fs_wav) = read_wav(filename_no_ext + '_22k_volnorm.wav')
    init_offset = int(TimeInSecsOfFirstFrame * Fs_wav) # initial offset in samples
    speech_wav_data = speech_wav_data[init_offset - hop_length_UTI : ]
    write_wav(speech_wav_data, Fs_wav, filename_wav_out)
    
    # remove temp files
    os.remove(filename_no_ext + '_22k.wav')
    os.remove(filename_no_ext + '_22k_volnorm.wav')
    
    print(filename_no_ext + ' - resampled, volume normalized, and cut to start with ultrasound')
    


# WaveGlow / Tacotron2 / STFT parameters
samplingFrequency = 22050
n_melspec = 80
hop_length_UTI = 270 # 12 ms, corresponding to 81.5 fps at 22050 Hz sampling
stft = WaveGlow_functions.TacotronSTFT(filter_length=1024, hop_length=hop_length_UTI, \
    win_length=1024, n_mel_channels=n_melspec, sampling_rate=samplingFrequency, \
    mel_fmin=0, mel_fmax=8000)

# parameters of ultrasound images, from .param file
framesPerSec = 81.5
n_lines = 64
n_pixels = 842

# reduce ultrasound image resolution
n_pixels_reduced = 128


# TODO: modify this according to your data path
dir_base = ''
##### training data
# - females: 01fi, 02fe, 09fe
# - males: 03mn, 04me, 05ms, 06fe, 07me, 08me, 10me
# speakers = ['01fi', '02fe', '03mn', '04me', '05ms', '06fe', '07me', '08me', '09fe', '10me']
speakers = ['02fe']

for speaker in speakers:
    
    # collect all possible ult files
    ult_files_all = []
    dir_data = dir_base + speaker + '/'
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            # collect _aud and _xaud files
            if file.endswith('aud.ult'):
                ult_files_all += [dir_data + file[:-4]]
    
    # randomize the order of files
    random.shuffle(ult_files_all)
    
    # temp: only first 10 sentence
    # ult_files_all = ult_files_all[0:10]
    
    ult_files = dict()
    ult = dict()
    melspec = dict()
    ultmel_size = dict()
    
    # train: first 80% of sentences
    ult_files['train'] = ult_files_all[0:int(0.8*len(ult_files_all))]
    # valid: next 10% of sentences
    ult_files['valid'] = ult_files_all[int(0.8*len(ult_files_all)):int(0.9*len(ult_files_all))]
    # valid: last 10% of sentences
    ult_files['test'] = ult_files_all[int(0.9*len(ult_files_all)):]
    
    # print('train files: ', ult_files['train'])
    # print('valid files: ', ult_files['valid'])
    
    for train_valid in ['train', 'valid']:
        n_max_ultrasound_frames = len(ult_files[train_valid]) * 500
        ult[train_valid] = np.empty((n_max_ultrasound_frames, n_lines, n_pixels_reduced))
        melspec[train_valid] = np.empty((n_max_ultrasound_frames, n_melspec))
        ultmel_size[train_valid] = 0
        
        # load all training/validation data
        for basefile in ult_files[train_valid]:
            try:
                ult_data = read_ult(basefile + '.ult', n_lines, n_pixels)
                
                # resample and cut if necessary
                if not os.path.isfile(basefile + '_cut_22k.wav'):
                    cut_and_resample_wav(basefile + '.wav', samplingFrequency)
                    
                # load using mel_sample
                mel_data = WaveGlow_functions.get_mel(basefile + '_cut_22k.wav', stft)
                mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes=(1, 0)))
                
            except ValueError as e:
                print("wrong psync data, check manually!", e)
            else:
                ultmel_len = np.min((len(ult_data),len(mel_data)))
                ult_data = ult_data[0:ultmel_len]
                mel_data = mel_data[0:ultmel_len]
                
                print(basefile, ult_data.shape, mel_data.shape)
                
                if ultmel_size[train_valid] + ultmel_len > n_max_ultrasound_frames:
                    print('data too large', n_max_ultrasound_frames, ultmel_size[train_valid], ultmel_len)
                    raise
                
                for i in range(ultmel_len):
                    ult[train_valid][ultmel_size[train_valid] + i] = skimage.transform.resize(ult_data[i], (n_lines, n_pixels_reduced), preserve_range=True) / 255
                    
                
                melspec[train_valid][ultmel_size[train_valid] : ultmel_size[train_valid] + ultmel_len] = mel_data
                ultmel_size[train_valid] += ultmel_len
                
                print('n_frames_all: ', ultmel_size[train_valid])


        ult[train_valid] = ult[train_valid][0 : ultmel_size[train_valid]]
        melspec[train_valid] = melspec[train_valid][0 : ultmel_size[train_valid]]

        # input: already scaled to [0,1] range
        # rescale to [-1,1]
        ult[train_valid] -= 0.5
        ult[train_valid] *= 2
        # reshape ult for CNN
        ult[train_valid] = np.reshape(ult[train_valid], (-1, n_lines, n_pixels_reduced, 1))
        
    print(ult['train'].shape)
    # target: normalization to zero mean, unit variance
    melspec_scaler = StandardScaler(with_mean=True, with_std=True)
    # melspec['train'] = melspec_scaler.fit_transform(melspec['train'].reshape(-1, 1)).ravel()
    melspec['train'] = melspec_scaler.fit_transform(melspec['train'])
    melspec['valid'] = melspec_scaler.transform(melspec['valid'])

    ### single training without cross-validation
    # convolutional model, improved version
    model=Sequential()
    # https://github.com/keras-team/keras/issues/11683
    # https://github.com/keras-team/keras/issues/10417
    model.add(InputLayer(input_shape=ult['train'].shape[1:]))
    # add input_shape to first hidden layer
    model.add(Conv2D(filters=30,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001), input_shape=ult['train'].shape[1:]))
    model.add(Dropout(0.2)) 
    model.add(Conv2D(filters=60,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None) ,kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2)) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=90,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2)) 
    model.add(Conv2D(filters=120,kernel_size=(13,13),strides=(1,1),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2)) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(1000,activation='swish', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None),kernel_regularizer=regularizers.l1(0.000005)))
    model.add(Dropout(0.2)) 
    model.add(Dense(n_melspec,activation='linear'))
    # compile model
    model.compile(SGD(lr=0.1,  momentum=0.1, nesterov=True),loss='mean_squared_error', metrics=['mean_squared_error'])
    

    print(model.summary())

    # early stopping to avoid over-training
    # csv logger
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() )
    print(current_date)
    model_name = 'models/UTI_to_STFT_CNN-improved_' + speaker + '_' + current_date
    
    # callbacks
    earlystopper = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=3, verbose=1, mode='auto')
    lrr = ReduceLROnPlateau(monitor='val_mean_squared_error', patience=2, verbose=1, factor=0.5, min_lr=0.0001) 
    logger = CSVLogger(model_name + '.csv', append=True, separator=';')
    checkp = ModelCheckpoint(model_name + '_weights_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # save model
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)

    # serialize scalers to pickle
    pickle.dump(melspec_scaler, open(model_name + '_melspec_scaler.sav', 'wb'))

    # Run training
    history = model.fit(ult['train'], melspec['train'],
                            epochs = 100, batch_size = 128, shuffle = True, verbose = 1,
                            validation_data=(ult['valid'], melspec['valid']),
                            callbacks = [earlystopper, lrr, logger, checkp])
    # here the training of the DNN is finished



