from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import sequence

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def is_number(query):
    return bool(num_regex.match(query))


lemmatizer = WordNetLemmatizer()
stopwords_ = stopwords.words('english')


def clean_text(line):
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stopwords_]
    text_stem = [lemmatizer.lemmatize(w) for w in text_rmstop]
    return text_stem


def single_input(text, vocab, maxlen):
    text = clean_text(text)
    indices = []
    for word in text:
        if is_number(word):
            indices.append(vocab['<num>'])
        elif word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(vocab['<unk>'])

    indices = sequence.pad_sequences([indices], maxlen=maxlen)
    return indices


def data_set_input(file, vocab, maxlen):
    inputs = open(file).readlines()
    output = [single_input(i, vocab, maxlen) for i in inputs]
    return output

