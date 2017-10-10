import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import tflearn
from util import *
from bleu import compute_bleu
from tensorflow.contrib.seq2seq import tile_batch
from tensorflow.contrib.rnn import MultiRNNCell

mode = 'train'

beam_width = 10
batch_size = 32
embed_vector_size = 100
rnn_hidden_size = 128
rnn_layer_depth = 1

# init
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(0.01, global_step, 500, 0.95, staircase=True)

# Load classes
src_table = tf.contrib.lookup.index_table_from_file('./iwslt15/vocab.en', default_value=0)
tgt_table = tf.contrib.lookup.index_table_from_file('./iwslt15/vocab.vi', default_value=0)

#src_table_size = src_table.size()
#tgt_table_size = tgt_table.size()
src_table_size = 17191
tgt_table_size = 7709
src_eos_id = tf.cast(src_table.lookup(tf.constant('</s>')), tf.int64)
tgt_eos_id = tf.cast(tgt_table.lookup(tf.constant('</s>')), tf.int64)
tgt_sos_id = tf.cast(tgt_table.lookup(tf.constant('<s>')), tf.int64)

# file placeholder
src_files = tf.placeholder(tf.string, shape=[None])
tgt_files = tf.placeholder(tf.string, shape=[None])

# Read data
src_dataset = tf.contrib.data.TextLineDataset(src_files)
tgt_dataset = tf.contrib.data.TextLineDataset(tgt_files)

# Convert data to word indices
src_dataset = src_dataset.map(lambda string: tf.concat([['<s>'], tf.string_split([string]).values, ['</s>']], 0))
src_dataset = src_dataset.map(lambda words: (words, tf.size(words)))
src_dataset = src_dataset.map(lambda words, size: (src_table.lookup(words), size))

tgt_dataset = tgt_dataset.map(lambda string: tf.concat([['<s>'], tf.string_split([string]).values, ['</s>']], 0))
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

# Load embedding (dic limits to 100000)
src_embed = tf.Variable(tf.random_normal([100000, embed_vector_size], stddev=0.1))
tgt_embed = tf.Variable(tf.random_normal([100000, embed_vector_size], stddev=0.1))

src_lookup = tf.nn.embedding_lookup(src_embed, source)
tgt_lookup = tf.nn.embedding_lookup(tgt_embed, target)

# Projection Layer
projection_layer = layers_core.Dense(tgt_table_size)

# Model
encoder_cells = MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size) for i in range(rnn_layer_depth)])
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cells, src_lookup, sequence_length=source_lengths, dtype=tf.float32)

# batch_size tensor
b_size = tf.size(source_lengths)

## use basic cell
decoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size) for i in range(rnn_layer_depth)]

if mode == 'train':
  ## Attention
  attention_mechanism = tf.contrib.seq2seq.LuongAttention(rnn_hidden_size, encoder_outputs, memory_sequence_length=source_lengths)
  decoder_cells[-1] = tf.contrib.seq2seq.AttentionWrapper(decoder_cells[-1], attention_mechanism, attention_layer_size=rnn_hidden_size)
  decoder_cell = MultiRNNCell(decoder_cells)
  decoder_state = [state for state in encoder_state]
  decoder_state[-1] = decoder_cells[-1].zero_state(batch_size=b_size, dtype=tf.float32).clone(cell_state=encoder_state[-1])
  decoder_state = tuple(decoder_state)

  decode_helper = tf.contrib.seq2seq.TrainingHelper(tgt_lookup, target_lengths)
  decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, decode_helper, decoder_state, output_layer=projection_layer)
  outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
  logits = outputs.rnn_output

  # Train Loss
  weights = tf.to_float(tf.concat([tf.not_equal(target[:, :-1], tgt_eos_id), tf.expand_dims(tf.fill([b_size], tf.constant(True)), 1)], 1))
  loss = tf.contrib.seq2seq.sequence_loss(logits, target, weights=weights)

  # Optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate)
  opt = optimizer.minimize(loss, global_step=global_step)

  # Summary
  tf.summary.scalar('train_loss', loss)

elif mode == 'inference':
  # tile batch for beam search decoder
  encoder_outputs = tile_batch(encoder_outputs, beam_width)
  encoder_state = tile_batch(encoder_state, beam_width)
  source_lengths = tile_batch(source_lengths, beam_width)

  attention_mechanism = tf.contrib.seq2seq.LuongAttention(rnn_hidden_size, encoder_outputs, memory_sequence_length=source_lengths)
  decoder_cells[-1] = tf.contrib.seq2seq.AttentionWrapper(decoder_cells[-1], attention_mechanism, attention_layer_size=rnn_hidden_size)
  decoder_cell = MultiRNNCell(decoder_cells)
  decoder_state = [state for state in encoder_state]
  decoder_state[-1] = decoder_cells[-1].zero_state(batch_size=b_size * beam_width, dtype=tf.float32).clone(cell_state=encoder_state[-1])
  decoder_state = tuple(decoder_state)

  infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, tgt_embed, tf.fill([b_size * beam_width], tf.to_int32(tgt_sos_id)), tf.to_int32(tgt_eos_id),
      decoder_state, beam_width, output_layer=projection_layer)
  infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder, maximum_iterations=tf.round(tf.reduce_max(source_lengths) * 2))
  infer_result = infer_outputs.predicted_ids

# Training Section
saver = tf.train.Saver()
merged = tf.summary.merge_all()

with tf.Session() as sess:
  sm = tf.summary.FileWriter('./logs', sess.graph)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())

  if mode == 'train':
    epoch = 0
    sess.run(batched_iterator.initializer, feed_dict={src_files: ['./iwslt15/train.en'], tgt_files: ['./iwslt15/train.vi']})
    for i in range(100000):
      try:
        _, _loss, _global_step = sess.run([opt, loss, global_step])
        if global_step % 100 == 0:
          summary = sess.run(merged)
          sm.add_summary(summary, i)
        print([epoch, _loss, _global_step])
      except tf.errors.OutOfRangeError:
        save_path = saver.save(sess, "./logs/model.ckpt")
        sess.run(batched_iterator.initializer, feed_dict={src_files: ['./iwslt15/train.en'], tgt_files: ['./iwslt15/train.vi']})
        epoch += 1
        if epoch == 1:
          break
  elif mode == 'inference':
    saver.restore('./logs/model.ckpt')
    sess.run(batched_iterator.initializer, feed_dict={src_files: ['./iwslt15/tst2012.en'], tgt_files: ['./iwslt15/tst2012.vi']})
    _result, _target = sess.run([infer_result, target])
    print((_result[0][:,0], _target[0]))
