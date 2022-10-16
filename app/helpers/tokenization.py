import nltk
from nltk import tokenize

import tensorflow_text as tf_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

import numpy as np
from unidecode import unidecode
import re


def normalize_sentence(sentence):
	norm_sentence = sentence.lower()
	norm_sentence = re.sub(r'[^\w\s]', '', norm_sentence)
	norm_sentence = norm_sentence.strip()
	norm_sentence = unidecode(norm_sentence)
	norm_sentence = ' '.join(norm_sentence.split())
	return norm_sentence

def filter_data(text_data, return_token=False, language='portuguese'):
	stop_words = nltk.corpus.stopwords.words(language)

	tokenizer = tf_text.WhitespaceTokenizer() #tf_text.UnicodeScriptTokenizer()
	tokens = tokenizer.tokenize(text_data)

	pos_phrase = list()
	for token in tokens.to_list():
		new_tokens = list()
		for word in token:
			word_norm = normalize_sentence(word.decode())

			if word_norm not in stop_words:
				new_tokens.append(word_norm)
		pos_phrase.append(' '.join(new_tokens))

	return pos_phrase

text_data = [
	'ola, amooo.r vc dúrmiu bíen?', 'ségunda fŕáse q eu inventíu, tá bme?'
]
pos = filter_data(text_data, return_token=False)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(pos)
vocab_size = len(tokenizer.word_index)+1

sequences = np.array(tokenizer.texts_to_sequences(pos))
sequences = pad_sequences(
	sequences, maxlen=50, dtype='int32',
    padding='pre', truncating='pre', value=0.0
)


print(pos)
print(tokenizer.word_index)
print(vocab_size)
print(sequences.shape)