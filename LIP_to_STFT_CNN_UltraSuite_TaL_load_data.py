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
Restructured March 31, 2021 - train on UltraSuite / TaL / TaL 1 & TaL80 corpora, lip video

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
import tgt
import csv
import datetime
import scipy
import pickle
import random
random.seed(17)
import skimage
from subprocess import call, check_output, run

# for loading video
import cv2

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


def read_wav(filename):
    (Fs, x) = io_wav.read(filename)
    return (x, Fs)

def write_wav(x, Fs, filename):
    # scaled = np.int16(x / np.max(np.abs(x)) * 32767)
    io_wav.write(filename, Fs, np.int16(x))


def cut_and_resample_wav_based_on_lip(filename_wav_in, Fs_target, start_time_samples):
    filename_no_ext = filename_wav_in.replace('.wav', '')
    
    filename_wav_out = filename_no_ext + '_cut_lip_22k.wav'
    
    # resample speech using SoX
    command = 'sox ' + filename_wav_in + ' -r ' + str(Fs_target) + ' ' + \
              filename_no_ext + '_22k.wav'
    call(command, shell=True)
    
    # volume normalization using SoX
    command = 'sox --norm=-3 ' + filename_no_ext + '_22k.wav' + ' ' + \
              filename_no_ext + '_22k_volnorm.wav'
    call(command, shell=True)
    
    # cut from wav the signal the part where there are lip video frames
    (speech_wav_data, Fs_wav) = read_wav(filename_no_ext + '_22k_volnorm.wav')
    speech_wav_data = speech_wav_data[int(start_time_sec * Fs_wav) - hop_length_LIP : ]
    write_wav(speech_wav_data, Fs_wav, filename_wav_out)
    
    # remove temp files
    os.remove(filename_no_ext + '_22k.wav')
    os.remove(filename_no_ext + '_22k_volnorm.wav')
    
    print(filename_no_ext + ' - resampled, volume normalized, and cut to start with lip')
    

def get_sync_from_lip_mp4(filename):

    sync_frames = []

    # read only first 20 lip images
    lip_data_start = np.fromfile(filename,
                                 count=20 * (vid_width * vid_height + vid_offset), dtype='uint8')
    # check white box in first 20 images
    for i in range(0, 20):
        img0 = lip_data_start[(vid_width * vid_height + vid_offset) * i:
                                (vid_width * vid_height + vid_offset) * i + vid_width * vid_height]
        img0 = np.reshape(img0, (vid_height, vid_width))

        # find white box on top left of the lip image
        white_box_1 = img0[60:90:2, 85:170]
        white_box_2 = img0[61:91:2, 85:170]

        if np.min(white_box_1) == 254 and np.max(white_box_1) == 254:
            # print('white box (1) found in lipvid / frame ' + str(i + 1))
            sync_frames += [i]
        if np.min(white_box_2) == 254 and np.max(white_box_2) == 254:
            # print('white box (2) found in lipvid / frame ' + str(i + 1))
            sync_frames += [i]

    return sync_frames


# from LipReading with slight modifications
# https://github.com/hassanhub/LipReading/blob/master/codes/data_integration.py
################## VIDEO INPUT ##################
def load_video_mp4(path):
    
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    fps = cap.get(cv2.CAP_PROP_FPS)
    

    buf = np.empty((frameHeight, frameWidth, frameCount), np.dtype('float32'))
    fc = 0
    ret = True
    
    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype('float32')
        buf[:,:,fc]=frame
        fc += 1
    cap.release()
    
    # reshape that first dimension is the time
    buf = np.moveaxis(buf, -1, 0)
    
    return buf

def get_sync_from_lip_mp4(lip_data):
    sync_frames = []
    
    # check white box in first 30 images
    for i in range(0, 30):
        
        # find white box on top left of the lip image
        # works on 320x240 videos
        white_box = lip_data[i, 25:50, 47:95]
        
        # print(i, np.mean(lip_data[i]), np.mean(white_box))
        
        # most pixels are white
        if np.mean(white_box) > 190:
            sync_frames += [i]
        
    return sync_frames

def read_lipsync(filename, debug_plot = False):
    fps_ntsc = 59.94
    
    (Fs, sync_data_orig) = io_wav.read(filename)
    
    sync_data = sync_data_orig.copy()
    # check only first part
    sync_data = sync_data[0 : int(1.0 * Fs)]
    sync_data_orig = sync_data.copy()
    
    sync_threshold =  max(sync_data) * 0.6
    peakind1 = detect_peaks(sync_data, mph=sync_threshold, mpd=25, threshold=0, edge='both')
    
    peakind_lip = []
    peakind_lip_num = 0
    # typical lipsync data:
    # 4 * (12ms 1kHz signal, 4 ms silence)
    # TODO: ultrasound and lip sync can disturb each other!
    for i in range(0, len(peakind1) - 2):
        # check if there is a peak before the current peak
        # if -sync_data[peakind1[i] - (peakind1[i+1] - peakind1[i])] < 0.5 * sync_threshold:
        if (peakind1[i+2] - peakind1[i+1] < 0.0012 * Fs) and (peakind1[i+1] - peakind1[i] < 0.0012 * Fs):
            # first peak of sync_data is a lipsync peak
            if peakind_lip_num == 0:
                peakind_lip += [peakind1[i]]
                peakind_lip_num += 1
                # print('time: ' + str(peakind1[i] / Fs))
            # later lip_peaks of sync_data should be roughly after NTSC (half) frame rate
            # if np.abs((peakind1[i] - peakind1[i - 1]) - Fs / fps_ntsc) < 0.0005 * Fs / 12:
            elif np.abs((peakind1[i] - peakind_lip[peakind_lip_num - 1]) - Fs / fps_ntsc) < 0.0005 * Fs:
                peakind_lip += [peakind1[i]]
                peakind_lip_num += 1
                # print('time: ' + str(peakind1[i] / Fs))
            # else:
            

    # plot for debugging
    if debug_plot == True:
        plt.figure(figsize=(18,4))
        plt.plot(sync_data_orig)
        # plt.plot(sync_data, 'r')
        for i in range(len(peakind1)):
            plt.plot(peakind1[i], sync_data_orig[peakind1[i]], 'gx')
            # plt.plot(peakind2[i], sync_data[peakind2[i]], 'r*')
        print(peakind_lip)
        for i in range(len(peakind_lip)):
            plt.plot(peakind_lip[i], sync_data_orig[peakind_lip[i]], 'ro')
        plt.xlim(peakind_lip[0] - 0.05 * Fs, peakind_lip[-1] + 0.05 * Fs)
        plt.show()
        
    if len(peakind_lip) < 2:
        peakind_lip = []
        raise ValueError('lip pulse sync data contains wrong pulses, check it manually!')

    return peakind_lip


# WaveGlow / Tacotron2 / STFT parameters
samplingFrequency = 22050
n_melspec = 80
hop_length_LIP = 367 # 33 ms, corresponding to 60 fps at 22050 Hz sampling
stft = WaveGlow_functions.TacotronSTFT(filter_length=1024, hop_length=hop_length_LIP, \
    win_length=1024, n_mel_channels=n_melspec, sampling_rate=samplingFrequency, \
    mel_fmin=0, mel_fmax=8000)

# parameters of lip videos, from .mp4 file
framesPerSec = 60
n_width = 320
n_height = 240

# reduce lip image resolution
# you can change this to anything else, but make sure that it fits into memory
n_width_reduced = 80
n_height_reduced = 60


# TODO: modify this according to your data path
dir_base = '/shared/UltraSuite_TaL/TaL80/core/'
##### training data
# - females: 01fi, 02fe, 09fe
# - males: 03mn, 04me, 05ms, 06fe, 07me, 08me, 10me
# speakers = ['01fi', '02fe', '03mn', '04me', '05ms', '06fe', '07me', '08me', '09fe', '10me']
speakers = ['01fi']

for speaker in speakers:
    
    # collect all possible lip files
    lip_files_all = []
    dir_data = dir_base + speaker + '/'
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            # collect _aud and _xaud files
            if file.endswith('aud.mp4'):
                lip_files_all += [dir_data + file[:-4]]
    
    # randomize the order of files
    random.shuffle(lip_files_all)
    
    # temp: only first 10 sentence
    # lip_files_all = lip_files_all[0:10]
    
    lip_files = dict()
    lip = dict()
    melspec = dict()
    lip_mel_size = dict()
    
    # train: first 80% of sentences
    lip_files['train'] = lip_files_all[0:int(0.8*len(lip_files_all))]
    # valid: next 10% of sentences
    lip_files['valid'] = lip_files_all[int(0.8*len(lip_files_all)):int(0.9*len(lip_files_all))]
    # valid: last 10% of sentences
    lip_files['test'] = lip_files_all[int(0.9*len(lip_files_all)):]
    
    # print('train files: ', lip_files['train'])
    # print('valid files: ', lip_files['valid'])
    
    for train_valid in ['train', 'valid']:
        n_max_lip_frames = len(lip_files[train_valid]) * 500
        lip[train_valid] = np.empty((n_max_lip_frames, n_height_reduced, n_width_reduced))
        melspec[train_valid] = np.empty((n_max_lip_frames, n_melspec))
        lip_mel_size[train_valid] = 0
        
        # load all training/validation data
        for basefile in lip_files[train_valid]:
            
            print('starting', basefile)
            
            lip_data = load_video_mp4(basefile + '.mp4')                
            # print(lip_data.shape)
            # plt.imshow(lip_data[0])
            # plt.gray()
            # plt.show()
            
            # get sync signal from video data
            sync_frames = get_sync_from_lip_mp4(lip_data)
            # print(sync_frames)
            
            
            # get sync signal from audio data
            peakind_lip = read_lipsync(basefile + '.sync', False)
            # print(peakind_lip)
            
            if len(sync_frames) != 4 or len(peakind_lip) != 4:
                print('issue with lip sync, check manually!', len(sync_frames), len(peakind_lip), sync_frames, peakind_lip)
                raise
            
            # start video data where the sync signal starts
            lip_data = lip_data[sync_frames[0]:]
            
            # start audio data where the sync signal starts
            (Fs_sync, _) = io_wav.read(basefile + '.sync')
            start_time_sec = peakind_lip[0] / Fs_sync
            
            # resample and cut if necessary
            if not os.path.isfile(basefile + '_cut_lip_22k.wav'):
                cut_and_resample_wav_based_on_lip(basefile + '.wav', samplingFrequency, start_time_sec)
                
            # load using mel_sample
            mel_data = WaveGlow_functions.get_mel(basefile + '_cut_lip_22k.wav', stft)
            mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes=(1, 0)))
            
            
            lip_mel_len = np.min((len(lip_data),len(mel_data)))
            lip_data = lip_data[0:lip_mel_len]
            mel_data = mel_data[0:lip_mel_len]
            
            print(basefile, lip_data.shape, mel_data.shape)
            
            if lip_mel_size[train_valid] + lip_mel_len > n_max_lip_frames:
                print('data too large', n_max_lip_frames, lip_mel_size[train_valid], lip_mel_len)
                raise
            
            for i in range(lip_mel_len):
                lip[train_valid][lip_mel_size[train_valid] + i] = skimage.transform.resize(lip_data[i], (n_height_reduced, n_width_reduced), preserve_range=True) / 255
            
            
            melspec[train_valid][lip_mel_size[train_valid] : lip_mel_size[train_valid] + lip_mel_len] = mel_data
            lip_mel_size[train_valid] += lip_mel_len
            
            print('n_frames_all: ', lip_mel_size[train_valid])
            
        lip[train_valid] = lip[train_valid][0 : lip_mel_size[train_valid]]
        melspec[train_valid] = melspec[train_valid][0 : lip_mel_size[train_valid]]

        # input: already scaled to [0,1] range
        # rescale to [-1,1]
        lip[train_valid] -= 0.5
        lip[train_valid] *= 2
        # reshape lip for CNN
        lip[train_valid] = np.reshape(lip[train_valid], (-1, n_height_reduced, n_width_reduced, 1))
        
    
    # now both the lip video, and the audio / spectral data are loaded
    
    # TODO: do the training itself
