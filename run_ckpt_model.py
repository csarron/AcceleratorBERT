import argparse
import os
import modeling
import tensorflow as tf
import numpy as np
import logging
logger = logging.getLogger('eet')

logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: %(message)s",
                        "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)

def init_ckpt(init_checkpoint=None, use_tpu=False):
  tvars = tf.trainable_variables()
  initialized_variable_names = {}
  scaffold_fn = None
  if init_checkpoint:
    (assignment_map, initialized_variable_names
    ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

  logger.info("**** Trainable Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    logger.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)

def create_model(bert_config, input_ids, segment_ids,input_mask, num_labels):
  """Creates a classification model."""
  # input_mask = tf.cast(tf.cast(input_ids, tf.bool), tf.float32)
  model = modeling.BertModel(
      config=bert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())
  logits = tf.matmul(output_layer, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)
  probabilities = tf.nn.softmax(logits, axis=-1, name="prob")
  return probabilities


def main(args):
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
    tf.gfile.MakeDirs(args.output_dir)
    saved_model_path = os.path.join(args.output_dir, args.output_name)
    input_ids=[101, 2054, 2154, 2001, 1996, 2208, 2209, 2006, 1029, 102, 1996, 2208, 2001, 2209, 2006, 2337, 1021, 1010, 2355, 1010, 2012, 11902, 1005, 1055, 3346, 1999, 1996, 2624, 3799, 3016, 2181, 2012, 4203, 10254, 1010, 2662, 1012, 102, 0, 0]
    segment_ids=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    input_ids = input_ids[:args.max_seq_length] + [0]*(args.max_seq_length-len(input_ids))
    segment_ids = segment_ids[:args.max_seq_length]+ [0]*(args.max_seq_length-len(segment_ids))
    export_all = True if args.mode == 'all' else False
    with tf.Session() as sess:
      input_ids_ph = tf.placeholder(shape=[1, args.max_seq_length],
                                  dtype=tf.int32, name='input_ids')
      segment_ids_ph = tf.placeholder(shape=[1, args.max_seq_length],
                                  dtype=tf.int32, name='segment_ids')
      input_mask_ph = tf.placeholder(shape=[1, args.max_seq_length],
                                  dtype=tf.float32, name='input_mask')
      prob = create_model(bert_config, input_ids_ph, segment_ids_ph, input_mask_ph, args.num_labels)
      # if args.quantize:
      #   g = tf.get_default_graph()
      #   tf.contrib.quantize.create_eval_graph(input_graph=g)
      #   eval_graph_file = saved_model_path + "_eval.pb"
      #   with open(eval_graph_file, 'w') as f:
      #     f.write(str(g.as_graph_def()))
      init_ckpt(args.ckpt_file)
      sess.run(tf.initializers.global_variables())

      if export_all or args.mode == 'ckpt':
        logger.info("saving model checkpoints...")
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.save(sess, saved_model_path)
        input_ids = np.reshape(input_ids, [1, args.max_seq_length])
        segment_ids = np.reshape(segment_ids, [1, args.max_seq_length])
        input_mask = input_ids.astype(np.bool).astype(np.float32)
        model_prob = sess.run(prob, {
          input_ids_ph: input_ids, 
          segment_ids_ph: segment_ids,
          input_mask_ph: input_mask,
        })
        logger.info('model_outputs: \n{}\n'.format(model_prob))

      if export_all or args.mode == 'lite':
        tflite_file = os.path.join(args.output_dir, '{}.tflite'.format(args.output_name))
        logger.info("exporting tflite model...")
        converter = tf.lite.TFLiteConverter.from_session(sess, [input_ids_ph, segment_ids_ph, input_mask_ph],
                                                          [prob])
        if args.quantize:
          converter.optimizations = [tf.lite.Optimize.DEFAULT]
          # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
          # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
          # converter.inference_input_type = tf.uint8
          # converter.inference_output_type = tf.uint8

          def representative_dataset_gen():
            for _ in range(10):
              # Get sample input data as a numpy array in a method of your choosing.
              yield [np.zeros([1, args.max_seq_length], dtype=np.int32), 
                     np.zeros([1, args.max_seq_length], dtype=np.int32),
                     np.ones([1, args.max_seq_length], dtype=np.float32),
                     ]
          converter.representative_dataset = representative_dataset_gen
            # converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
        else:
          converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        with open(tflite_file, "wb") as f:
            f.write(tflite_model)
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--bert_config_file', type=str, default=None)
    parser.add_argument('-cf', '--ckpt_file', type=str, default=None)
    parser.add_argument("-od", "--output_dir", type=str, default=None)
    parser.add_argument("-on", "--output_name", type=str, default=None)
    parser.add_argument("-m", "--mode", type=str, choices=("lite", "ckpt", "all"),
                        default='ckpt')
    parser.add_argument("-n", "--num_labels", type=int, default=2)
    parser.add_argument("-l", "--max_seq_length", type=int, default=40)
    parser.add_argument("-q", "--quantize", action='store_true', help="quantize the tflite model")
    main(parser.parse_args())
