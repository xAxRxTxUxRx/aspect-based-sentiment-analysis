import tensorflow as tf
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

args = {
    'batch_size': 64,
    'aspect_size': 14,
    'domain': 'restaurant',
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

#### Подготовка модели
print('\n' + msg + 'Get data' + msg)
import reader
from tensorflow.keras.preprocessing import sequence

vocab, train_x, test_x, maxlen = reader.get_data(args['domain'], args['vocab_size'])
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

#### Построение такой же модели
print('\n' + msg + 'Build model' + msg)
from model import build_model

model = build_model(args, maxlen, vocab)
model.load_weights(args['output_dir'] + '/model_params')

from optimizers import get_optimizer

optimizer = get_optimizer(args['algorithm'])


def max_margin_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])

#### Оценка
from sklearn.metrics import classification_report
from tensorflow.keras import Model
from sklearn.metrics import accuracy_score


def evaluation(true, predict, domain):
    true_label = []
    predict_label = []

    if domain == 'restaurant':

        for line in predict:
            predict_label.append(line.strip())

        for line in true:
            true_label.append(line.strip())

        accuracy = accuracy_score(true_label, predict_label)

        print(f'Accuracy: {accuracy}')
        print(classification_report(true_label, predict_label,
                                    labels=['Food', 'Staff', 'Ambience', 'Price', 'Miscellaneous'],
                                    digits=3))
        return accuracy

    else:
        for line in predict:
            label = line.strip()
            if label == 'smell' or label == 'taste':
                label = 'taste+smell'
            predict_label.append(label)

        for line in true:
            label = line.strip()
            if label == 'smell' or label == 'taste':
                label = 'taste+smell'
            true_label.append(label)

        accuracy = accuracy_score(true_label, predict_label)

        print(f'Accuracy: {accuracy}')
        print(classification_report(true_label, predict_label,
                                    labels=['feel', 'taste+smell', 'look', 'overall', 'None'], digits=3))
        return accuracy


def prediction(test_labels, aspect_probs, cluster_map, domain):
    label_ids = np.argsort(aspect_probs, axis=1)[:, -1]
    predict_labels = [cluster_map[label_id] for label_id in label_ids]
    return evaluation(open(test_labels), predict_labels, domain)


test_fn = Model(inputs=model.get_layer('sentence_input').input, outputs=model.get_layer('p_t').output)
aspect_probs = test_fn.predict(test_x)

cluster_map = {0: 'Food', 1: 'Food', 2: 'Ambience', 3: 'Ambience',
                          4: 'Miscellaneous', 5: 'Food', 6: 'Food', 7: 'Miscellaneous', 8: 'Staff',
                          9: 'Price', 10: 'Food', 11: 'Staff',
                          12: 'Miscellaneous', 13: 'Ambience'}

print(f'--- Results on {args["domain"]} domain ---')
test_labels = f'../preprocessed_data/{args["domain"]}/test_label.txt'
prediction(test_labels, aspect_probs, cluster_map, domain=args['domain'])
