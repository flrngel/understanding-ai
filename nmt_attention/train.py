import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import tflearn
from util import *

batch_size = 32
pad_length = 60
embed_vector_size = 100
rnn_hidden_size = 100

# init
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(0.1, global_step, 500, 0.95, staircase=True)

# Load classes
src_table = tf.contrib.lookup.index_table_from_file('./iwslt15/vocab.en', default_value=0)
tgt_table = tf.contrib.lookup.index_table_from_file('./iwslt15/vocab.vi', default_value=0)

#src_table_size = src_table.size()
#tgt_table_size = tgt_table.size()
src_table_size = 17191
tgt_table_size = 7709
src_eos_id = tf.cast(src_table.lookup(tf.constant('</s>')), tf.int64)
tgt_eos_id = tf.cast(tgt_table.lookup(tf.constant('</s>')), tf.int64)

# Read data
src_dataset = tf.contrib.data.TextLineDataset(['./iwslt15/train.en'])
tgt_dataset = tf.contrib.data.TextLineDataset(['./iwslt15/train.vi'])

# Convert data to word indices
src_dataset = src_dataset.map(lambda string: tf.string_split([string]).values)
src_dataset = src_dataset.map(lambda words: (words, tf.size(words)))
src_dataset = src_dataset.map(lambda words, size: (src_table.lookup(words), size))

tgt_dataset = tgt_dataset.map(lambda string: tf.string_split([string]).values)
tgt_dataset = tgt_dataset.map(lambda words: (words, tf.size(words)))
tgt_dataset = tgt_dataset.map(lambda words, size: (tgt_table.lookup(words), size))

# zip data
dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))

# batch
batched_dataset = dataset.padded_batch(batch_size,
    padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])),(tf.TensorShape([None]), tf.TensorShape([]))),
    padding_values=((src_eos_id, 0), (tgt_eos_id, 0)))
batched_iterator = batched_dataset.make_initializable_iterator()
((source, source_lengths), (target, target_lengths)) = batched_iterator.get_next()

# Load embedding
src_embed = tf.Variable(tf.random_normal([100000, embed_vector_size], stddev=0.1))
tgt_embed = tf.Variable(tf.random_normal([100000, embed_vector_size], stddev=0.1))

src_lookup = tf.nn.embedding_lookup(src_embed, source)
tgt_lookup = tf.nn.embedding_lookup(tgt_embed, target)

# Projection Layer
projection_layer = layers_core.Dense(tgt_table_size)

# Model
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, src_lookup, sequence_length=source_lengths, dtype=tf.float32)

decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size)
decode_helper = tf.contrib.seq2seq.TrainingHelper(tgt_lookup, target_lengths)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, decode_helper, encoder_state, output_layer=projection_layer)
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output

# Loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
opt = optimizer.minimize(loss, global_step=global_step)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  sess.run(batched_iterator.initializer)

  for i in range(100000):
    try:
      print(sess.run([opt, loss, global_step]))
    except tf.errors.OutOfRangeError:
      sess.run(batched_iterator.initializer)
