from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
stopwords_ = stopwords.words('english')


def clean_text(line):
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stopwords_]
    text_stem = [lemmatizer.lemmatize(w) for w in text_rmstop]
    return text_stem


def preprocess_train(domain):
    file_in = open(f'../datasets/{domain}/train.txt', 'r')
    file_out = open(f'../preprocessed_data/{domain}/train.txt', 'w')
    for line in file_in:
        preprocessed_text = clean_text(line)
        if len(preprocessed_text) > 0:
            file_out.write(' '.join(preprocessed_text) + '\n')


def preprocess_test(domain):
    file_in1 = open(f'../datasets/{domain}/test.txt')
    file_in2 = open(f'../datasets/{domain}/test_label.txt')
    file_out1 = open(f'../preprocessed_data/{domain}/test.txt')
    file_out2 = open(f'../preprocessed_data/{domain}/test_label.txt')

    for line, label in zip(file_in1, file_in2):
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
            continue
        preprocessed_text = clean_text(line)
        if len(preprocessed_text) > 0:
            file_out1.write(' '.join(preprocessed_text) + '\n')
            file_out2.write(label + '\n')


def preprocess(domain):
    if domain in ['restaurant', 'beer']:
        print('\t' + domain + ' train set ...')
        preprocess_train(domain)
        print('\t' + domain + ' test set ...')
        preprocess_test(domain)


# print('Preprocessing raw data...')
# preprocess('restaurant')
# preprocess('beer')
