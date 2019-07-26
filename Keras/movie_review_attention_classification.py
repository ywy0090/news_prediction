import tensorflow as tf
from keras import backend as K
from keras_preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import LSTM, Embedding, Dense, Reshape, Flatten, TimeDistributed, Activation, RepeatVector, Permute, \
    multiply, Lambda, Input
from attention_decoder import AttentionDecoder

vocab_size = 10000

pad_id = 0
start_id = 1
oov_id = 2
index_offset = 2
max_len = 200
rnn_cell_size = 128
embedding_dim = 128


def attention_model(n_timesteps_in, n_features):
    model = Sequential()
    embedding_layer = Embedding(vocab_size,
                                embedding_dim,
                                input_length=max_len)
    att_layer = AttentionDecoder(32, n_features)
    model.add(embedding_layer)
    model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
    model.add(att_layer)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def attention_keras_model(n_timesteps_in, n_features):
    lstm_unit = 150
    sequence_input = Input(shape=(max_len,))
    embedding_layer = Embedding(vocab_size,
                                embedding_dim,
                                input_length=max_len)(sequence_input)
    LSTM_layer = LSTM(lstm_unit, input_shape=(n_timesteps_in, n_features), return_sequences=True)(embedding_layer)
    # Attention layer
    attention = TimeDistributed(Dense(1, activation='tanh'))(LSTM_layer)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(lstm_unit)(attention)
    attention = Permute([2, 1])(attention)
    #
    sent_representation = multiply([LSTM_layer, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-1))(sent_representation)
    output = Dense(1, activation='sigmoid')(sent_representation)
    model = Model(inputs=sequence_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size, start_char=start_id,
                                                                            oov_char=oov_id, index_from=index_offset)
    word2idx = tf.keras.datasets.imdb.get_word_index()
    idx2word = {v + index_offset: k for k, v in word2idx.items()}
    idx2word[pad_id] = '<PAD>'
    idx2word[start_id] = '<START>'
    idx2word[oov_id] = '<OOV>'
    x_train = sequence.pad_sequences(x_train,
                                     maxlen=max_len,
                                     truncating='post',
                                     padding='post',
                                     value=pad_id)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len,
                                    truncating='post',
                                    padding='post',
                                    value=pad_id)
    # att_model = attention_model(n_timesteps_in=rnn_cell_size, n_features=embedding_dim)
    att_model = attention_keras_model(n_timesteps_in=rnn_cell_size, n_features=embedding_dim)
    att_model.fit(x_train,
                  y_train,
                  epochs=10,
                  batch_size=200,
                  validation_split=.3,
                  verbose=1)
    att_res = att_model.evaluate(x_test, y_test)
    print(att_res)
