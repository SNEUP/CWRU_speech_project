#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:27:15 2024

@author: pprakash
"""

import numpy as np
import mat73
import pandas as pd

from scipy.signal import filtfilt,butter,hilbert
import scipy.io as sio
from utils import *
import nltk
# from nltk.stem import WordNetLemmatizer
# import collections
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC,SVC

# from multiprocessing import Pool
# from scipy import stats

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from multiprocessing import Process, Queue
import multiprocessing
# from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.cross_decomposition import PLSSVD,PLSRegression
import pickle
import os
import sys
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# Download the CMU Pronouncing Dictionary
nltk.download('cmudict')
# Create a sentence to phoneme dictionary

words_to_phonemes = nltk.corpus.cmudict.dict()
#%%
band='spike_band_power' ## this could be high gamma band or spike band

date = 'Jan05_2024'  # here we have Jan05_2024,
data_path = f'../Speech Data/Analysis/2000Hz/{band}/{date}_all_blocks_withAIP.mat'

try:
    raw_data_jan = pd.DataFrame(sio.loadmat(data_path)['all_data'])
except:
    raw_data_jan = pd.DataFrame(mat73.loadmat(data_path)['all_data'])
date = 'Dec06_2023'  # here we have Jan05_2024, Dec06_2023 data
data_path = f'../Speech Data/Analysis/2000Hz/{band}/{date}_all_blocks_withAIP.mat'

try:
    raw_data_dec = pd.DataFrame(sio.loadmat(data_path)['all_data'])
except:
    raw_data_dec = pd.DataFrame(mat73.loadmat(data_path)['all_data'])
    
#%%
def data_cleaning_bygrid(data: pd.DataFrame, channel_cleaning_threshold=1, trial_cleaning_threshold=1.5):
    print("Cleaning data for both IFG and AIP")
    reject_trials_all = []
    clean_data_all = []
    chan2keep = []
    reject_channels_all = []
    for i in range(2):
        if i == 0:
            print("Cleaning AIP data")
            currgrid = "AIP_"
            channel_std = data[11].to_list()[0]
            
        elif i == 1 :
            print("Cleaning IFG data")
            currgrid = "IFG_"
            channel_std = data[10].to_list()[0]
        store_channame =[currgrid+str(chan+1) for chan in range(65)]
        all_data = data[i].to_list()
        ## channel cleaning
        channel_std_mean = channel_std.mean()
        reject_channels = np.where(channel_std > channel_cleaning_threshold * channel_std_mean)[0]
        channel_clean_data = [np.delete(a, reject_channels, axis=0) for a in all_data]
        
        clean_data = [np.delete(a,reject_channels,axis=0) for a in raw_data[i].to_list()]
        
        store_channame = np.delete(store_channame,reject_channels,axis=0)
        reject_channels_all.append(reject_channels)
        # numchans_ingrid = clean_data[0].shape[0]
        
        
        chan2keep.extend(store_channame)
        clean_data_all.append(clean_data)
        
        
        ## trial cleaning
        trial_mean = np.array([t.mean(axis=0).mean() for t in channel_clean_data])
        
        reject_trials = np.where(trial_mean > trial_cleaning_threshold * trial_mean.std())[0]
        # print(clean_data[]
        # for rej_idx in reject_trials:
        #     del clean_data[rej_idx]
        data[i] = clean_data
        print(reject_trials[0])
        # reject_trials = np.array([i for i in (reject_trials)])
        if i ==0:
            reject_trials_all = reject_trials
        else:
            reject_trials_all = np.concatenate((reject_trials_all,reject_trials))
            # reject_trials_all.extend(reject_trials)
    
    data = data.drop(data.index[np.unique(reject_trials_all)])
    return data, reject_channels_all, reject_trials_all, chan2keep, trial_mean

def plot_word_center(grid_data,voice_on,voice_off,word_on,channames,back=1,forward=4,bin_size=0.05):

    processed_data_word_on = [trial[:,int(word_on[i]*fs-back*fs) : int(word_on[i]*fs+forward*fs)] for i,trial in enumerate(grid_data)]
    n_channels = processed_data_word_on[0].shape[0]
    binned_data_word_on = [trial.reshape(n_channels,int(bin_size*fs),-1,order='F').mean(axis=1) for trial in processed_data_word_on]
    binned_data_word_on_averaged = np.mean(binned_data_word_on,axis=0)
    n_bins = binned_data_word_on_averaged.shape[-1]
    
    # ave_word_on = back
    ave_word_on =0
    ave_word_on_voice_on_diff = (voice_on-word_on).mean()
    # ave_phrase_on=-(word_on-phrase_on).mean()+ave_voice_on
    # std_phrase_on=(word_on-phrase_on).std()
    
    # ave_word_on=(voice_on-word_on).mean() + ave_voice_on
    # std_word_on=(voice_on-word_on).std()
    
    # ave_voice_off=(voice_on-voice_off).mean()
    # std_voice_off=(voice_off-voice_on).std()
    
    fig2,ax=plt.subplots(8,8,figsize=(18,10))
    for i in range(n_channels):
        ax[i//8,i%8].plot(np.linspace(-back,forward,n_bins),binned_data_word_on_averaged[i])
        ax[i//8,i%8].set_yticks([])
        # if i//8!=7:   
        #     ax[i//8,i%8].set_xticks([])
        # else:
        #     ax[i//8,i%8].set_xticks(np.linspace(0,(back+forward)*adjusted_fs,5), [f"{i:0.1f}" for i in np.linspace(-back,forward,5)])
        #     ax[i//8,i%8].set_xlabel("time (s)")
        
        # ax[i//8,i%8].axvspan(ave_voice_off * adjusted_fs - std_voice_off * adjusted_fs, ave_voice_off * adjusted_fs + std_voice_off * adjusted_fs, color='m', alpha=0.2)
        #ax[i//8,i%8].axvspan(ave_word_on * adjusted_fs - std_word_on * adjusted_fs, ave_word_on * adjusted_fs + std_word_on * adjusted_fs, color='g', alpha=0.2)
        # ax[i//8,i%8].axvline(ave_word_on * adjusted_fs, color='g', linestyle='--', linewidth=2)
        ax[i//8,i%8].axvline(ave_word_on_voice_on_diff, color='r', linestyle='--', linewidth=2)
        # ax[i//8,i%8].axvline(ave_word_on , color='g', linestyle='--', linewidth=2)
        ax[i//8,i%8].axvline(ave_word_on, color='g', linestyle='--', linewidth=2)
        ax[i//8,i%8].set_title(channames[i], y=1.0, pad=-14)
    plt.suptitle("aligned to word onset: binned",fontsize=20)
    plt.show()

def plot_voice_center(grid_data,voice_on,voice_off,word_on,channames,back = 4, forward = 1, bin_size=0.05):
    processed_data_voice_on = [trial[:,int(voice_on[i]*fs-back*fs) : int(voice_on[i]*fs+forward*fs)] for i,trial in enumerate(grid_data)]
    n_channels = processed_data_voice_on[0].shape[0]
    print(n_channels)
    binned_data_voice_on = [trial.reshape(n_channels,int(bin_size*fs),-1,order='F').mean(axis=1) for trial in processed_data_voice_on]
    binned_data_voice_on_averaged = np.mean(binned_data_voice_on,axis=0)
    n_bins = binned_data_voice_on_averaged.shape[-1]

    # ave_word_on = back
    # ave_word_on =0
    ave_voice_on = 0
    ave_word_on_voice_on_diff = (voice_on-word_on).mean()
    # ave_phrase_on=-(word_on-phrase_on).mean()+ave_voice_on
    # std_phrase_on=(word_on-phrase_on).std()

    # ave_word_on=(voice_on-word_on).mean() + ave_voice_on
    # std_word_on=(voice_on-word_on).std()

    fig2,ax=plt.subplots(8,8,figsize=(18,10))
    for i in range(n_channels):
        ax[i//8,i%8].plot(np.linspace(-back,forward,n_bins),binned_data_voice_on_averaged[i])
        ax[i//8,i%8].set_yticks([])
        # if i//8!=7:   
        #     ax[i//8,i%8].set_xticks([])
        # else:
        #     ax[i//8,i%8].set_xticks(np.linspace(0,(back+forward)*adjusted_fs,5), [f"{i:0.1f}" for i in np.linspace(-back,forward,5)])
        #     ax[i//8,i%8].set_xlabel("time (s)")
        
        # ax[i//8,i%8].axvspan(ave_voice_off * adjusted_fs - std_voice_off * adjusted_fs, ave_voice_off * adjusted_fs + std_voice_off * adjusted_fs, color='m', alpha=0.2)
        #ax[i//8,i%8].axvspan(ave_word_on * adjusted_fs - std_word_on * adjusted_fs, ave_word_on * adjusted_fs + std_word_on * adjusted_fs, color='g', alpha=0.2)
        # ax[i//8,i%8].axvline(ave_word_on * adjusted_fs, color='g', linestyle='--', linewidth=2)
        ax[i//8,i%8].axvline(ave_voice_on, color='r', linestyle='--', linewidth=2)
        # ax[i//8,i%8].axvline(ave_word_on , color='g', linestyle='--', linewidth=2)
        ax[i//8,i%8].axvline(-ave_word_on_voice_on_diff, color='g', linestyle='--', linewidth=2)
        ax[i//8,i%8].set_title(channames[i], y=1.0, pad=-14)

    plt.suptitle("aligned to voice onset: binned",fontsize=20)
    plt.show()
                      

#%%
day = "Jan05_2024" # options : "Dec06_2023", "Jan05_2024","All"
# concatenate data
# raw_data = pd.concat([raw_data_jan,raw_data_dec])
if day == 'Jan05_2024':
    raw_data = raw_data_jan.copy()
elif day== 'Dec06_2023':
    raw_data = raw_data_dec.copy()
elif day == "All":
    print("Combining across datasets")
    raw_data = pd.concat([raw_data_jan,raw_data_dec])
else:
    print("Incorrect selection, try again")
## throw away the bad trials: only used in Jan data
## hard coded here
#bad_trials=[27,30,34,50,53,55,56,58,60,62,65,66,67,68,69,83,85,94]
#raw_data=raw_data.drop(raw_data.index[bad_trials])
# all_data = raw_data[1].to_list()
# ## channel cleaning
# channel_std = raw_data[10].to_list()[0]
# # channel_std_std = channel_std.std()
# reject_channels = np.where(channel_std>channel_std.mean())[0]
# channel_clean_data = [np.delete(a,reject_channels,axis=0) for a in all_data]
# ### trial cleaning
# trial_std = np.array([t.mean(axis=0).std() for t in channel_clean_data])
# reject_trials = np.where(trial_std>1.5*trial_std.mean())[0]
# a bit too complicate but i don't know any better method so far

# raw_data=raw_data.drop(raw_data.index[reject_trials])
# clean_data=[np.delete(a,reject_channels,axis=0) for a in raw_data[0].to_list()]
# print(f"rejected_channels: {reject_channels} \n")
# print(f"rejected_trials: {reject_trials}")
clean_data, reject_channels_all, reject_trials_all, chankept, _= data_cleaning_bygrid(raw_data)

# fig1,ax=plt.subplots(2,1,figsize=(7,3))
# ax[0].plot(channel_std)
# ax[1].plot(trial_std)
# plt.show()
# %%
# process the labels
correct_label = clean_data[12].to_numpy()
answered_label = clean_data[13].to_numpy()
answered_words = clean_data[8].to_numpy()
#answered_semantic_label=raw_data[12].to_numpy()
correct_label = np.array([a[0] for a in correct_label])
answered_label = np.array([a[0] for a in answered_label])
#answered_semantic_label=np.array([a[0] if a!=np.NaN else np.NaN for a in answered_semantic_label])
# lemmatizer = WordNetLemmatizer()
# answered_lexical_item = np.array([lemmatizer.lemmatize(a[0][0][0], pos='v') for a in answered_words])
# first_phonemes = np.array([words_to_phonemes[a][0][0] for a in answered_lexical_item])

# process the time stamps
phrase_on = clean_data[2].to_numpy()
word_on = clean_data[4].to_numpy()
voice_on = clean_data[6].to_numpy()
voice_off = clean_data[7].to_numpy()

# phrase_on = np.array([a[0] for a in phrase_on])
# word_on = np.array([a[0] for a in word_on])
# voice_on = np.array([a[0] for a in voice_on])
# voice_off = np.array([a[0] for a in voice_off])

# recording frequency rate
fs = 2000
# print(collections.Counter(answered_lexical_item))
#print(collections.Counter(answered_semantic_label))
# print(collections.Counter(answered_label))
#%% Centered around Word onset
# define the decoding time range and bin the data
# aligning to the voice onset
AIP_data = clean_data[0].tolist()
IFG_data = clean_data[1].tolist()
chankept_AIP = [s for s in chankept if "AIP" in s]
chankept_IFG = [s for s in chankept if "IFG" in s]
plot_word_center(IFG_data,voice_on,voice_off,word_on,chankept_IFG)
plot_word_center(AIP_data,voice_on,voice_off,word_on,chankept_AIP)

plot_voice_center(IFG_data,voice_on,voice_off,word_on,chankept_IFG)
plot_voice_center(AIP_data,voice_on,voice_off,word_on,chankept_AIP)


#%%
now = datetime.now()
datefoldername= now.strftime("%m-%d-%y")
savefilepath = "Figures/AIP_analysis/%s/"%(day) + datefoldername +"/"
if not os.path.exists(savefilepath):
    os.makedirs(savefilepath)
savefigname = "TrialAveraged_AIP_%s"%(day)
 
pp = PdfPages(savefilepath + savefigname)
figs = [plt.figure(n) for n in plt.get_fignums()]
for fig in (figs): ## will open an empty extra figure :(
    fig.savefig(pp,format='pdf')
pp.close()
plt.close('all')