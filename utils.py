# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 20:41:41 2020

@author: ACER
"""
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

def read_notes_from_files():
  notes = []
  for file in glob.glob("Pokemon MIDIs/*.mid"):
    song = converter.parse(file)
    notes_in_song = None
    try: 
            s2 = instrument.partitionByInstrument(song)
            notes_in_song = s2.parts[0].recurse() 
    except: 
            notes_in_song = song.flat.notes
    for element in notes_in_song:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
  return notes


def get_input_output(notes,n_vocab):
  sequence_length = 100
  pitchnames = sorted(set(item for item in notes))
  pitch_to_int = dict((note, number) for number, note in enumerate(pitchnames))
  disc_input = []
  gen_output = []
  for i in range(0,len(notes)-sequence_length):
    input_sequence = notes[i:i+sequence_length]
    out_sequence = notes[i+sequence_length]
    disc_input.append([pitch_to_int[pitch] for pitch in input_sequence])
    gen_output.append(pitch_to_int[out_sequence])
  
  n_inputs = len(disc_input)
  disc_input = np.reshape(disc_input,(n_inputs, sequence_length, 1))
  disc_input = (disc_input - float(n_vocab)/2) / (float(n_vocab)/2)
  gen_output = to_categorical(gen_output)

  return (disc_input, gen_output)

def create_midi(prediction_output, filename):
    offset = 0
    output_notes = []
    for item in prediction_output:
        pattern = item[0]

        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))