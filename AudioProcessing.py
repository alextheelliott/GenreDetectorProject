import librosa
import numpy as np
import pandas as pd
import os
import csv

import matplotlib.pyplot as plt
import librosa.display

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Program Started...")
print("Starting First Init")

audio_root = os.getcwd()+'\\songs\\'
output_root = os.getcwd()+'\\output\\'
songs = os.listdir(audio_root)

model = keras.models.load_model('model65.h5')
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

data_t = pd.read_csv('data.csv')
data_t = data_t.drop(['filename'],axis=1)
scaler = StandardScaler()
scaler.fit_transform(np.array(data_t.iloc[:, :-1], dtype = float))

print("First Init Completed")
for song in songs:
    print(song + " initializing")
    audio_data = audio_root + song
    x , sr = librosa.load(audio_data, sr=44100)

    output_values = []

    ## Amplitude ##

#    plt.figure(figsize=(14, 5))
#    librosa.display.waveplot(x, sr=sr)
    output_values.append(x)

    ## Spectrogram ##

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
#    plt.figure(figsize=(14, 5))
#    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
#    plt.colorbar()
    output_values.append(Xdb)

    ## Chroma feature ##

    hop_length = (sr//10)
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
#    plt.figure(figsize=(15, 5))
#    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
    output_values.append(chromagram)

    #plt.show()

####Get genre

    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' lmaoextracolumnheader'

    f = open('profiles\\'+song.replace('.wav', '')+'.csv', 'w', newline='')

    y, sr = librosa.load(audio_data, mono=True, duration=40)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y, hop_length=(sr//10))
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'filename {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    to_append += ' lmaoextracolumn'
    with f:
        writer = csv.writer(f)
        writer.writerow(header.split())
        writer.writerow(to_append.split())

    data = pd.read_csv('profiles\\'+song.replace('.wav', '')+'.csv')
    data = data.drop(['filename'],axis=1)

    genre_list = genres
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(genre_list)

    X_sca = scaler.transform(np.array(data.iloc[:, :-1], dtype = float))
    print(X_sca)

#    X_train, X_test, y_train, y_test = train_test_split(X_sca, np.array([y_enc, y_enc]), test_size=1)

#    print(X_train)
    results = model.predict(X_sca)
    print(results)

####take every 4410th value of output_values[0] and every output_values[2] value.

    ## Output File ##
    f = open(output_root + song.split('.')[0] + '.txt', 'w')
    f.write(str(results).replace('\n', '').replace('[[', '').replace(']]', '').replace(' ', ',').replace(',,', ',') + '\n')
    for i in range(0, len(output_values[0]), 4410):
        f.write(str(x[i]) + ',')
        for j in range(0, 12):
            try:
                f.write(str(chromagram[j][i//4410]) + ',')
            except:
                f.write('0.0,')
        f.write('\n')
    f.close()

    print(song + " completed")