from gensim.models import Word2Vec
import gensim.downloader


def main(domain):
    train_data = open(f'../preprocessed_data/{domain}/train.txt').readlines()
    train_data = [line.split() for line in train_data]
    model_file = f'../preprocessed_data/{domain}/w2v_embedding'

    model = Word2Vec(train_data, vector_size=200, window=5, min_count=8, workers=4, sg=1)
    print(model.wv.most_similar('smee', topn=10))
    model.save(model_file)


if __name__ == '__main__':
    main('beer')
