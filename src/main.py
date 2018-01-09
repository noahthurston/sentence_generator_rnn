import utils
import train_theano

vocab_size = 8000
hidden_dim = 100
num_epochs = 25
learning_rate = 0.020
sentence_corpus = "../data/trump_all_tweets.csv"
saved_model = "../data/trump_all_tweets-100-8000-50_epochs.npz"

def train():
    x_train, y_train, index_to_word, word_to_index = utils.load_data(sentence_corpus, vocab_size=vocab_size)
    train_theano.train_model(x_train, y_train, training_data_name=sentence_corpus[:-4], load_model_file=saved_model)

def generate():
    x_train, y_train, index_to_word, word_to_index = utils.load_data(sentence_corpus)
    train_theano.generate_examples(saved_model, index_to_word, word_to_index, vocab_size, hidden_dim, 20, 6)

# train()
# generate()
