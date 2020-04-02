from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import time

# decide maximum number of words to load in dataset
max_features = 20000

# decide maximum words in a sentence
maxlen = 50
batch_size = 32

# load the imdb dataset
print("Loading data")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# pad the sequence
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# show a data sample
print("Sample of x_train array = ", x_train[0])
print("Sample of y_train array = ", y_train[0])

# get vocabs to convert words to numbers
imdb_vocab = imdb.get_word_index()

# create a small vocabulary
small_vocab = {key: value for key, value in imdb_vocab.items() if value < 20}
print("Vocabulary = ", small_vocab)

# function to get sentence forom integer array
# rever look up words in vocabulary


def get_original_text(int_arr):
    word_to_id = {k: (v+3) for k, v in imdb_vocab.items()}
    # add three new tokesn into the word to id dict
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    # reverse the word to id dict
    # key is now id, value is word
    id_to_word = {value: key for key, value in word_to_id.items()}
    return " ".join(id_to_word[id] for id in int_arr)


# define sentiment
sentiment_labels = ["Negative", "Positve"]

print("------------------------------")
print("Some Sentence and Sentiment Samples")
for i in range(5):
    print("Training Sentence = ", get_original_text(x_train[i]))
    print("Sentiment = ", sentiment_labels[y_train[i]])
    print("------------------------------")

# build the model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(64, dropout=0.5))
model.add(Dense(1, activation="sigmoid"))

# try using different optimizers and different optimizer configs
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# train the model
model.fit(x_train, y_train, batch_size=batch_size,
          epochs=2, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print("Test score: ", score)
print("Test accuracy ", acc)

# save model via time stamp
model.save("./experiments/imdb_nlp_{0}.h5".format(int(time.time())))
