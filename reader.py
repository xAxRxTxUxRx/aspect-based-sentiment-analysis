import operator
import re

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def is_number(query):
    return bool(num_regex.match(query))


def create_vocab(domain, vocab_size):
    assert domain in ['restaurant', 'beer']
    source = f'../preprocessed_data/{domain}/train.txt'

    total_words, unique_words = 0, 0
    word_freqs = {}

    f = open(source, 'r')
    for line in f:
        words = line.split()
        for word in words:
            if not is_number(word):
                try:
                    word_freqs[word] += 1
                except KeyError:
                    word_freqs[word] = 1
                    unique_words += 1
                total_words += 1

    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('  keep the top %i words' % vocab_size)

    vocab_file = open(f'../preprocessed_data/{domain}/vocab', 'w')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word + '\t' + str(0) + '\n')
            continue
        vocab_file.write(word + '\t' + str(word_freqs[word]) + '\n')
    vocab_file.close()
    return vocab


def read_dataset(domain, phase, vocab):
    assert domain in ['restaurant', 'beer']
    assert phase in ['train', 'test']

    source = '../preprocessed_data/' + domain + '/' + phase + '.txt'
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    f = open(source, 'r')
    for line in f:
        words = line.strip().split()

        indices = []
        for word in words:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)

    print('   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, maxlen_x


def get_data(domain, vocab_size=0):
    print('Reading data from', domain)
    print(' Creating vocab ...')
    vocab = create_vocab(domain, vocab_size)
    print(' Reading dataset ...')
    print('  train set')
    data_train, maxlen_1 = read_dataset(domain, 'train', vocab)
    print('  test set')
    data_test, maxlen_2 = read_dataset(domain, 'test', vocab)

    maxlen = max(maxlen_1, maxlen_2)
    return vocab, data_train, data_test, maxlen


if __name__ == "__main__":
    vocab, train_x, test_x, maxlen = get_data('restaurant', vocab_size=9000)
    print(len(vocab))
    print(len(train_x))
    print(len(test_x))
    print(maxlen)
