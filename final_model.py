import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from model import build_model
from optimizers import get_optimizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import reader


class ABAE_model:
    def __init__(self, domain, cluster_map):
        self.args = {
            'batch_size': 64,
            'aspect_size': 14,
            'domain': domain,
            'algorithm': 'rmsprop',
            'epochs': 15,
            'emb_dim': 200,
            'neg_size': 20,
            'ortho_reg': 0.1,
            'vocab_size': 10000,
            'seed': 1234,
            'maxlen': 156
        }
        self.args['emb_path'] = f'../preprocessed_data/{self.args["domain"]}/w2v_embedding'
        self.args['output_dir'] = f'../output/{self.args["domain"]}'
        np.random.seed(self.args['seed'])

        self.vocab = reader.create_vocab(self.args['domain'], self.args['vocab_size'])
        self.cluster_map = cluster_map

    def get_model(self):
        model = build_model(args=self.args, maxlen=self.args['maxlen'], vocab=self.vocab)
        model.load_weights(self.args['output_dir'] + '/model_params')

        optimizer = get_optimizer(self.args['algorithm'])

        def max_margin_loss(y_true, y_pred):
            return tf.reduce_mean(y_pred)

        model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])
        self.predict_fn = Model(inputs=[model.get_layer('sentence_input').input],
                                outputs=[model.get_layer('p_t').output])
        self.predict_fn.compile()
        return self.predict_fn

    def predict(self, aspect_probs):
        max_prob = 0.
        max_prob_index = 0
        for i in range(len(aspect_probs)):
            prob = float(aspect_probs[i])
            if prob > max_prob:
                max_prob = float(prob)
                max_prob_index = i
        return self.cluster_map[max_prob_index], round(max_prob, 3)


class Sentiment_Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

        self.model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    def predict(self, review):
        tokens = self.tokenizer.encode(review, return_tensors='pt')
        result = self.model(tokens)
        return int(torch.argmax(result.logits))+1
