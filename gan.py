from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from utils import read_notes_from_files,get_input_output,create_midi


class GAN():
  def __init__(self, rows):
    self.sequence_length = rows
    self.seq_shape = (self.sequence_length, 1)
    self.noise_dim = 1000
    self.disc_loss = []
    self.gen_loss =[]
    self.notes = read_notes_from_files()
    self.vocab = len(set(self.notes))
    print(self.vocab)

    optimizer = Adam(0.0002, 0.5)

    self.discriminator = self.build_discriminator()
    self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    self.generator = self.build_generator()
    z = Input(shape=(self.noise_dim,))
    generated_seq = self.generator(z)

    self.discriminator.trainable = False
    validity = self.discriminator(generated_seq)
    self.combined = Model(z, validity)
    self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

  def build_discriminator(self):

    model = Sequential()
    model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    seq = Input(shape=self.seq_shape)
    validity = model(seq)

    return Model(seq, validity)

  def build_generator(self):
    model = Sequential()
    model.add(Dense(256, input_dim=self.noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
    model.add(Reshape(self.seq_shape))
    model.summary()
        
    noise = Input(shape=(self.noise_dim,))
    seq = model(noise)

    return Model(noise, seq)

  def train(self, epochs, batch_size=128, sample_interval=50):
    notes = read_notes_from_files()
    n_vocab = len(set(notes))
    X_train, y_train = get_input_output(notes,n_vocab)
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
      idx = np.random.randint(0, X_train.shape[0], batch_size)
      real_seqs = X_train[idx]
      noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
      gen_seqs = self.generator.predict(noise)

      d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
      d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

      noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
      g_loss = self.combined.train_on_batch(noise, real)

      if epoch % sample_interval == 0:
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        self.disc_loss.append(d_loss[0])
        self.gen_loss.append(g_loss)

    self.generate(notes)
    #self.plot_loss()

  def generate(self, input_notes):
    notes = input_notes
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    noise = np.random.normal(0, 1, (1, self.noise_dim))
    predictions = self.generator.predict(noise)

    pred_notes = [x*float(self.vocab/2)+float(self.vocab/2) for x in predictions[0]]
    pred_notes = [int_to_note[int(x)] for x in pred_notes]

    create_midi(pred_notes, 'predicted')
    
gan = GAN(rows=100)
gan.train(epochs=800, batch_size=32, sample_interval=1)
