# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:01:00 2015

@author: xiispace
"""
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import wave
import struct
import os
import numpy
import csv
import sys

def read_wav(wav_file):
    ''' 读取wav音频文件'''
    w = wave.open(wav_file)
    n = 60*10000
    if w.getnframes() < n*2:
        raise ValueError(u'音频太短了')
    frames = w.readframes(n)
    wav_data1 = struct.unpack('%dh' % n, frames)
    frames = w.readframes(n)
    wav_data2 = struct.unpack('%dh' % n, frames)
    return wav_data1, wav_data2
    
def compute_chunk_features(mp3_file):
    '''返回特征向量'''
    #把mp3转换成单声道，10khz的wav
    mpg123_command = 'D:\Code\Python\音乐图谱\mpg123\mpg123.exe -w "%s" -r 10000 -m "%s"'
    out_file = 'temp.wav'
    cmd = mpg123_command % (out_file, mp3_file)
    temp = subprocess.call(cmd) #音频格式转换
    
    wav_data1, wav_data2 = read_wav(out_file)
    
    return features(wav_data1), features(wav_data2)

def moments(x):
    mean = x.mean() #均值
    std = x.var()**0.5 #标准差
    skewness = ((x - mean)**3).mean() / std**3 #偏态
    kurtosis = ((x - mean)**4).mean() / std**4 #峰态
    return [mean, std, skewness, kurtosis]

def fftfeatures(wavdata):
    f = numpy.fft.fft(wavdata)
    f = f[2:(f.size / 2 + 1)]
    f = abs(f)
    total_power = f.sum()
    f = numpy.array_split(f, 10)
    return [e.sum() / total_power for e in f]

def features(x):
	'''特征值提取'''
    x = numpy.array(x)
    f = []
    
    xs = x
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))
    
    xs = x.reshape(-1, 10).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))
    
    xs = x.reshape(-1, 100).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))
    
    xs = x.reshape(-1, 1000).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))
    
    f.extend(fftfeatures(x))
    return f

# names:
 
# amp1mean
# amp1std
# amp1skew
# amp1kurt
# amp1dmean
# amp1dstd
# amp1dskew
# amp1dkurt
# amp10mean
# amp10std
# amp10skew
# amp10kurt
# amp10dmean
# amp10dstd
# amp10dskew
# amp10dkurt
# amp100mean
# amp100std
# amp100skew
# amp100kurt
# amp100dmean
# amp100dstd
# amp100dskew
# amp100dkurt
# amp1000mean
# amp1000std
# amp1000skew
# amp1000kurt
# amp1000dmean
# amp1000dstd
# amp1000dskew
# amp1000dkurt
# power1
# power2
# power3
# power4
# power5
# power6
# power7
# power8
# power9
# power10
 
#最优特征
# amp10mean
# amp10std
# amp10skew
# amp10dstd
# amp10dskew
# amp10dkurt
# amp100mean
# amp100std
# amp100dstd
# amp1000mean
# power2
# power3
# power4
# power5
# ower6
# power7
# power8
# power9
 
#Main Script
MusicPath='D:\Code\Python\音乐图谱\music'
data=[]
labels=[]
#降维，取前两个最优特征做为横纵坐标
for path, dirs, files in os.walk(MusicPath):
    for f in files:
        if not f.endswith('.mp3'):
            continue
        print(f)
        if f.find('G.E.M.邓紫棋')!=-1:
            labels.append(1)
        elif f.find('李健')!=-1:
            labels.append(2)
        elif f.find('王菲')!=-1:
            labels.append(3)
        elif f.find('李玉刚')!= -1:
            labels.append(4)
        elif f.find('莫文蔚') != -1:
            labels.append(5)
        mp3_file = os.path.join(path, f)
        
        tail, track = os.path.split(mp3_file)
        tail, dir1 = os.path.split(tail)
        tail, dir2 = os.path.split(tail)
        
        try:
            feature_vec1, feature_vec2 = compute_chunk_features(mp3_file)
            mainIndex = [8,9]
            mainF_vec1 = [feature_vec1[x] for x in mainIndex]
            mainF_vec2 = [feature_vec2[x] for x in mainIndex]
            data.append(mainF_vec1)
#            data.append(mainF_vec2)
        except:   
            print('error')
            continue

data = numpy.array(data)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data[:, 0], data[:, 1], 15.0*numpy.array(labels),
           15.0*numpy.array(labels))
plt.show()
