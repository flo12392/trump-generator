import keras
import numpy as np
from .utils import sample

class ExamplesGeneratorCallback(keras.callbacks.Callback):

    def __init__(self, sentences_indexed, index_word, examples_file_loc, seq_len, n_words):
        self.sentences_indexed = sentences_indexed
        self.index_word = index_word
        self.examples_file_loc = examples_file_loc
        self.seq_len = seq_len
        self.n_words = n_words
        open(self.examples_file_loc, 'w').close() # clear the file

    def on_epoch_end(self, epoch, logs={}):

        examples_file = open(self.examples_file_loc, "a")
        # Function invoked at end of each epoch. Prints generated text.
        examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

        # Randomly pick a seed sequence
        seed = self.sentences_indexed[np.random.randint(len(self.sentences_indexed))]

        for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
            examples_file.write('\n----- Diversity:' + str(diversity) + '\n')
            examples_file.write(
                '----- Generating with seed:\n"' + ' '.join([self.index_word[x] for x in seed]) + '"\n')

            sentence = seed.copy()
            full_sentence = sentence.copy()

            for i in range(200):
                sentence = sentence[1:]
                x_pred = np.zeros((1, self.seq_len - 1, self.n_words), dtype=np.bool)
                for t, w in enumerate(sentence):
                    x_pred[0, t, w] = 1

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                sentence.append(next_index)
                full_sentence.append(next_index)

            full_sentence = [self.index_word[x] for x in full_sentence]
            examples_file.write(' '.join(full_sentence))
        examples_file.write('\n' + '=' * 80 + '\n\n')
        examples_file.close()