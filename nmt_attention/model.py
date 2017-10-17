import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq import tile_batch
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.util import nest

class Model(object):
  def __init__(self):
    self.batch_size = 32
    self.beam_width = 1
    self.rnn_layer_depth = 2
    self.rnn_hidden_size = 128
    self.embed_vector_size = 100

  def init(self):
    # init
    self.global_step = global_step = tf.Variable(0, trainable=False, name='global_step')
    self.learning_rate = learning_rate = tf.train.exponential_decay(0.0001, global_step, 500, 0.95, staircase=True)

    # Load classes
    src_table = tf.contrib.lookup.index_table_from_file('./iwslt15/vocab.en', default_value=0)
    tgt_table = tf.contrib.lookup.index_table_from_file('./iwslt15/vocab.vi', default_value=0)

    #src_table_size = src_table.size()
    #tgt_table_size = tgt_table.size()
    src_table_size = 17191
    tgt_table_size = 7709
    src_eos_id = tf.cast(src_table.lookup(tf.constant('</s>')), tf.int64)
    self.tgt_eos_id = tgt_eos_id = tf.cast(tgt_table.lookup(tf.constant('</s>')), tf.int64)
    self.tgt_sos_id = tgt_sos_id = tf.cast(tgt_table.lookup(tf.constant('<s>')), tf.int64)

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
    batched_dataset = dataset.padded_batch(self.batch_size,
        padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])),(tf.TensorShape([None]), tf.TensorShape([]))),
        padding_values=((src_eos_id, 0), (tgt_eos_id, 0)))
    batched_iterator = batched_dataset.make_initializable_iterator()
    ((source, source_lengths), (target, target_lengths)) = batched_iterator.get_next()

    self.target = target
    self.target_lengths = target_lengths
    self.source_lengths = source_lengths

    # Load embedding (dic limits to 100000)
    src_embed = tf.Variable(tf.random_normal([100000, self.embed_vector_size], stddev=0.1))
    self.tgt_embed = tgt_embed = tf.Variable(tf.random_normal([100000, self.embed_vector_size], stddev=0.1))

    self.src_lookup = src_lookup = tf.nn.embedding_lookup(src_embed, source)
    self.tgt_lookup = tgt_lookup = tf.nn.embedding_lookup(tgt_embed, target)

    # Projection Layer
    self.projection_layer = projection_layer = layers_core.Dense(tgt_table_size)

    return batched_iterator, src_files, tgt_files

  def get_cell(self):
    return tf.nn.rnn_cell.BasicLSTMCell(self.rnn_hidden_size)

  def wrap_multi_rnn(self, rnn_layer):
    return MultiRNNCell(rnn_layer)

  def wrap_attention(self, encoder_outputs, source_lengths):
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        self.rnn_hidden_size, encoder_outputs, source_lengths)
    attention = tf.contrib.seq2seq.AttentionWrapper(
        self.get_cell(), attention_mechanism, attention_layer_size=self.rnn_hidden_size)
    return attention

  def encoder_model(self):
    with tf.variable_scope('encoder'):
      encoder_cells = self.wrap_multi_rnn([self.get_cell() for i in range(self.rnn_layer_depth)])
      self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
          encoder_cells, self.src_lookup, sequence_length=self.source_lengths, dtype=tf.float32)

  def decode_model(self, train_mode=True):
    encoder_outputs = self.encoder_outputs
    source_lengths = self.source_lengths
    encoder_state = self.encoder_state
    global_step = self.global_step
    projection_layer = self.projection_layer
    tgt_sos_id = self.tgt_sos_id
    tgt_eos_id = self.tgt_eos_id
    target = self.target
    target_lookup = self.tgt_lookup
    target_lengths = self.target_lengths
    learning_rate = self.learning_rate
    target_embed = self.tgt_embed

    if train_mode == True:
      reuse = False
    else:
      reuse = False

    # Prepare
    b_size = tf.size(source_lengths)
    if train_mode == False:
      b_size_t = tf.to_int32(b_size * self.beam_width)

      encoder_outputs = tile_batch(encoder_outputs, self.beam_width) #tile_batch(encoder_outputs, beam_width)
      encoder_state = tile_batch(encoder_state, self.beam_width)
      source_lengths = tile_batch(source_lengths, self.beam_width)
    else:
      b_size_t = b_size

    with tf.variable_scope('decoder'):
      # Create decoder_cell
      rnn_layer = [self.get_cell() for i in range(self.rnn_layer_depth)]
      
      # Create attention cell (top of rnn layer)
      rnn_layer[-1] = self.wrap_attention(encoder_outputs, source_lengths)
      multi_rnn = self.wrap_multi_rnn(rnn_layer)

      # sync cell state with encoder

      decoder_state = [enc for enc in encoder_state]
      decoder_state[-1] = rnn_layer[-1].zero_state(batch_size=b_size_t, dtype=tf.float32).clone(
          cell_state=encoder_state[-1])
      decoder_state = tuple(decoder_state)

      if train_mode == True:
        # define decoder
        decode_helper = tf.contrib.seq2seq.TrainingHelper(
            target_lookup, target_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            multi_rnn, decode_helper, decoder_state, output_layer=projection_layer)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        # Train Loss
        weights = tf.to_float(tf.concat([
          tf.expand_dims(tf.fill([b_size], tf.constant(True)), 1), tf.not_equal(target[:, :-1], tgt_eos_id)]
          , 1))
        loss = tf.contrib.seq2seq.sequence_loss(logits, target, weights=weights)

        # Optimize
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        opt = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
        return opt, loss, global_step

      else:
        infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            multi_rnn, target_embed, tf.fill([b_size], tf.to_int32(tgt_sos_id)), tf.to_int32(tgt_eos_id),
            decoder_state, self.beam_width, output_layer=projection_layer)
        infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            infer_decoder, maximum_iterations=tf.round(tf.reduce_max(source_lengths) * 2))
        infer_result = infer_outputs.predicted_ids
        return infer_result, target, global_step
