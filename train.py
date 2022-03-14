import tensorflow as tf
import numpy as np
from time import time
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

args = {
    'batch_size': 64,
    'aspect_size': 14,
    'domain': 'beer',
    'algorithm': 'rmsprop',
    'epochs': 15,
    'emb_dim': 200,
    'neg_size': 20,
    'ortho_reg': 0.1,
    'vocab_size': 10000,
    'seed': 1234
}
args['emb_path'] = f'../preprocessed_data/{args["domain"]}/w2v_embedding'
args['output_dir'] = f'../output/{args["domain"]}'
msg = '-' * 15
np.random.seed(args['seed'])

print('\n' + msg + 'Args' + msg)
for arg in args:
    print(f'{arg}: {args[arg]}')

#### Prepare data
print('\n' + msg + 'Prepare data' + msg)
import reader
from tensorflow.keras.preprocessing import sequence

vocab, train_x, test_x, maxlen = reader.get_data(args['domain'], args['vocab_size'])
train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)


class PosGenerator:

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __call__(self):
        n_batch = len(self.data) / self.batch_size
        batch_count = 0
        np.random.shuffle(self.data)
        while True:
            if batch_count == n_batch:
                batch_count = 0
                np.random.shuffle(self.data)
            batch = self.data[batch_count * self.batch_size:(batch_count + 1) * self.batch_size]
            batch_count += 1
            yield tuple(batch)


class NegGenerator:
    def __init__(self, data, negative_factor, batch_size):
        self.data_len = data.shape[0]
        self.dim = data.shape[1]
        self.neg_size = negative_factor
        self.batch_size = batch_size
        self.data = data

    def __call__(self):
        while True:
            idx = np.random.choice(self.data_len, self.batch_size * self.neg_size)
            samples = self.data[idx].reshape(self.batch_size, self.neg_size, self.dim)
            yield tuple(samples)


#### Построение модели
print('\n' + msg + 'Build model' + msg)
from model import build_model

model = build_model(args, maxlen, vocab)

from optimizers import get_optimizer

optimizer = get_optimizer(args['algorithm'])


def max_margin_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])

#### Тренировка модели
print('\n' + msg + 'Train model' + msg)
from tqdm import tqdm

from Word2VecReader import W2VEmbReader
from tensorflow.data import Dataset

emb_reader = W2VEmbReader(args['emb_path'])

min_loss = float('inf')
for epoch in range(args['epochs']):
    t0 = time()
    loss, max_margin_loss = 0., 0.
    pos_gen = iter(Dataset.from_generator(PosGenerator(train_x, args['batch_size']), output_types=(tuple([tf.int32]*(args['batch_size'])))).repeat(1))
    neg_gen = iter(Dataset.from_generator(NegGenerator(train_x, args['neg_size'], args['batch_size']), output_types=(tuple([tf.int32]*(args['batch_size'])))).repeat(1))
    batches_per_epoch = len(train_x) / args['batch_size']

    # Тренировка модели на батчах
    for b in tqdm(range(int(batches_per_epoch)), position=0):
        pos_input = tf.stack(pos_gen.get_next())
        neg_input = tf.stack(neg_gen.get_next())
        batch_loss, batch_max_margin_loss = model.train_on_batch(x=[pos_input, neg_input],
                                                                 y=np.ones((args['batch_size'], 1)))
        loss += batch_loss / batches_per_epoch
        max_margin_loss += batch_max_margin_loss / batches_per_epoch

    tr_time = time() - t0

    if loss < min_loss:
        min_loss = loss

        # Сохранения модели
        model_source = args['output_dir'] + '/model_params'
        model.save_weights(model_source)

        # Извлечение аспектов
        aspect_file = open(args['output_dir'] + '/aspect.log', 'w')

        vectors = model.get_layer('aspect_emb').weights[0].numpy()
        n_clusters = vectors.shape[0]
        for i in range(n_clusters):
            vector = vectors[i, :]
            top_n_words = emb_reader.get_model().wv.similar_by_vector(vector, topn=50, restrict_vocab=None)
            print(f'\nAspect {i}:')
            for word in top_n_words:
                print(word[0] + ':' + str(round(word[1], 3)), end=' | ')
            print()
            aspect_file.write(f'Aspect {i}:\n')
            aspect_file.write(' '.join([x[0] for x in top_n_words]) + '\n\n')

    print(f'Epoch {epoch}, train: {tr_time}')
    print(f'Total loss: {loss}, max_margin_loss: {max_margin_loss}, ortho_reg: {loss - max_margin_loss}')
