from keras.utils import Sequence
import numpy as np

# modified from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, sentences_indexed, seq_len, n_words, batch_size=32, shuffle=True):
        'Initialization'
        self.sentences_indexed = sentences_indexed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_words = n_words
        self.seq_len = seq_len
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.sentences_indexed) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of sentences
        sentences_indexed_temp = [self.sentences_indexed[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(sentences_indexed_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sentences_indexed))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sentences_indexed_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((self.batch_size, self.seq_len - 1, self.n_words), dtype=np.bool)
        y = np.zeros((self.batch_size, self.n_words), dtype=np.bool)

        # Generate data
        for i, sentence in enumerate(sentences_indexed_temp):
            # Generate X
            for t, w in enumerate(sentence[:-1]):
                X[i, t, w] = 1

            y[i, sentence[-1]] = 1

        return X, y