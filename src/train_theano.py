import sys
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano

def train_with_sgd(model, x_train, y_train, training_data_name="training_data_name", learning_rate=0.005, nepoch=50, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        #optionally evaluate loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(x_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # Saving model parameters
            save_model_parameters_theano("../data/%s-%d-%d-%s.npz" % (training_data_name, model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(x_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

# trains the model
def train_model(x_train, y_train, training_data_name="training_data_name", load_model_file="", num_epochs=50, learning_rate=0.010, hidden_dim=100, vocab_size=8000):
    model = RNNTheano(vocab_size, hidden_dim=hidden_dim)
    t1 = time.time()
    model.sgd_step(x_train[10], y_train[10], learning_rate)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

    if load_model_file != "":
        print "loading model: %s" % load_model_file
        load_model_parameters_theano(load_model_file, model)

    train_with_sgd(model, x_train, y_train, nepoch=num_epochs, learning_rate=learning_rate, training_data_name=training_data_name)


def generate_sentence(model, index_to_word, word_to_index,
                      unknown_token = "unknown_token",
                      sentence_start_token = "sentence_start",
                      sentence_end_token = "sentence_end"):

    # start with start token
    new_sentence = [word_to_index[sentence_start_token]]
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)

    sentence_str = []
    for i in new_sentence[1:-1]:
        sentence_str.append(index_to_word[i])

    return sentence_str


# generates multiple sentences
def generate_examples(model_name, index_to_word, word_to_index, vocab_size=8000, hidden_dim=100, num_sentences=10, sentences_min_length=4):

    model = RNNTheano(vocab_size, hidden_dim)
    load_model_parameters_theano(model_name, model)
    sentences = []

    for i in range(num_sentences):
        sent = []
        while len(sent) < sentences_min_length:
            sent = generate_sentence(model, index_to_word, word_to_index)
        print " ".join(sent)

    while len(sentences) < num_sentences:
        sent = generate_sentence(model, index_to_word, word_to_index)
        if len(sent) >= sentences_min_length: sentences.append(sent)

