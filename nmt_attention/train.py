from pprint import pprint
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import tflearn
from util import *
from bleu import compute_bleu
from tensorflow.contrib.seq2seq import tile_batch
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.util import nest
from model import Model

mode = 'train'

beam_width = 10
batch_size = 32
embed_vector_size = 100
rnn_hidden_size = 128
rnn_layer_depth = 4

def get_cell():
  return tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size)

def wrap_attention(encoder_outputs, source_lengths, cell=None):
  if cell is None:
    cell = get_cell()
  attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_hidden_size, encoder_outputs, memory_sequence_length=source_lengths, normalize=True)
  attention = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=rnn_hidden_size)
  return attention

def wrap_rnn_layer(decoder_cells):
  return MultiRNNCell(decoder_cells)

# Model
train_graph = tf.Graph()
eval_graph = tf.Graph()

with train_graph.as_default():
  train_model = Model()
  batched_iterator, src_files, tgt_files = train_model.init()
  train_model.encoder_model()
  opt, loss, global_step = train_model.decode_model()

  init1 = tf.global_variables_initializer()
  init2 = tf.tables_initializer()
  saver = tf.train.Saver()

  for i in tf.get_collection("trainable_variables"):
    print(i)
    if i.name == 'decoder/multi_rnn_cell/cell_1/attention_wrapper/attention_layer/kernel:0':
      ssibal = tf.reduce_mean(i)

with eval_graph.as_default():
  eval_model = Model()
  eval_batched_iterator, eval_src_files, eval_tgt_files = eval_model.init()
  eval_model.encoder_model()
  infer_result, infer_target, eval_global_step, infer_tgt = eval_model.decode_model(train_mode=False)
  eval_saver = tf.train.Saver()
  eval_init1 = tf.tables_initializer()

  for i in tf.get_collection("trainable_variables"):
    print(i)
    if i.name == 'decoder/multi_rnn_cell/cell_1/attention_wrapper/attention_layer/kernel:0':
      ssibal2 = tf.reduce_mean(i)

# Training Section
#merged = tf.summary.merge_all()

sess = tf.Session(graph=train_graph)
eval_sess = tf.Session(graph=eval_graph)

#sm = tf.summary.FileWriter('./logs', sess.graph)

sess.run([init1, init2])
eval_sess.run([eval_init1])

if mode == 'train':
  epoch = 0
  sess.run(batched_iterator.initializer, feed_dict={src_files: ['./iwslt15/train.en'], tgt_files: ['./iwslt15/train.vi']})
  for i in range(1000000):
    try:
      _, _loss, _global_step = sess.run([opt, loss, global_step])
      print([epoch, _loss, _global_step])
      #print(sess.run(logits)[0])
      if _global_step % 30 == 1:
        #summary = sess.run(merged)
        #sm.add_summary(summary, i)
        save_path = saver.save(sess, "./logs/model.ckpt")
        eval_saver.restore(eval_sess, './logs/model.ckpt')
        #print((sess.run(ssibal), eval_sess.run(ssibal2)))
        eval_sess.run(eval_batched_iterator.initializer, feed_dict={eval_src_files: ['./iwslt15/tst2012.en'], eval_tgt_files: ['./iwslt15/tst2012.vi']})
        _result = eval_sess.run(infer_result)
        #_result, _target = eval_sess.run([infer_result, infer_target])
        print((_result[0][:,0], infer_tgt[0]))
      if i % 30 == 0:
        pass
        #_result, _target = sess.run([infer_result, target])
        #print((_result[1], _target[1]))
        #print(sess.run(tf.fill([b_size], tf.to_int32(tgt_sos_id))))
        #print(sess.run(tf.ones([b_size], dtype=tf.int32) * tf.to_int32(tgt_sos_id)))
    except tf.errors.OutOfRangeError:
      save_path = saver.save(sess, "./logs/model.ckpt")
      sess.run(batched_iterator.initializer, feed_dict={src_files: ['./iwslt15/train.en'], tgt_files: ['./iwslt15/train.vi']})
      epoch += 1
      if epoch == 12:
        break
elif mode == 'inference':
  #sm = tf.summary.FileWriter('./test_logs', sess.graph)
  sess.run(batched_iterator.initializer, feed_dict={src_files: ['./iwslt15/tst2012.en'], tgt_files: ['./iwslt15/train.vi']})
  _result, _target = sess.run([infer_result, target])
  print((_result[1], _target[1]))
  saver.restore(sess, './logs/model.ckpt')
  sess.run(batched_iterator.initializer, feed_dict={src_files: ['./iwslt15/tst2012.en'], tgt_files: ['./iwslt15/train.vi']})
  _result, _target = sess.run([infer_result, target])
  print((_result[1][:,0], _target[1]))
