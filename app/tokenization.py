import nltk
from nltk import tokenize

import tensorflow_text as tf_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

from datetime import datetime
import numpy as np
from unidecode import unidecode
import re
import pandas as pd
import glob
import os


def split_path(img_path):
	fname = os.path.basename(img_path)
	path = os.path.dirname(img_path)
	return path, fname

def pad_seq_tokenizier(sentences):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(sentences)
	sequences = tokenizer.texts_to_sequences(sentences)
	sequences = np.array(sequences)
	sequences = pad_sequences(
		sequences, maxlen=50, dtype='int32',
		padding='pre', truncating='pre', value=0.0
	)
	return sequences

df = pd.read_csv('/home/marten/Desktop/workdir/bkp_gan/app2/helpers/mean_sea_level.csv')
img_paths = list(sorted(glob.glob('/home/marten/Desktop/workdir/gans/data/*/*/*.npy')))

for img_path in img_paths:

	path, fname = split_path(img_path)
	datestr = fname.split('.')[0]

	df_dates = pd.read_csv(img_path.replace('.npy','.csv'))['dates'].to_list()
	date_i, date_f = df_dates[0], df_dates[-1]
	sentences = df[(df['dates'] >= date_i)&(df['dates']<=date_f)]['text_mean_sea_level'].to_list()
	sequences = pad_seq_tokenizier(sentences)

	file_out = f"{path}/msl_{fname.replace('.npy','.arr.npy')}"; print(file_out)

	np.save(file_out, sequences)
