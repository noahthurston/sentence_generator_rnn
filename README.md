# sentence_generator_rnn

Sentence generator created using a recurrent neural network from WildML's RNN tutorial: 
www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/

Preprocessing is done with NLTK and is optimized for tweets, so that hashtags and handles are preserved and web links are not included as part of the vocabulary. 

Included is a .csv with about 21,000 Donald Trump tweets downloaded from trumptwitterarchive.com that can be used as training data. Also included is a 100-node model that was trained with these tweets for 50 epochs with a learning rate starting at 0.020. 
