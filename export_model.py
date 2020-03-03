import argparse
import os
import modeling
import tensorflow as tf

def create_model(bert_config, input_ids, segment_ids,input_mask, num_labels):
  """Creates a classification model."""
  input_mask = tf.cast(tf.cast(input_ids, tf.bool), tf.float32)
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

    with tf.Session() as sess:
      input_ids_ph = tf.placeholder(shape=[1, args.max_seq_length],
                                  dtype=tf.int32, name='input_ids')
      segment_ids_ph = tf.placeholder(shape=[1, args.max_seq_length],
                                  dtype=tf.int32, name='segment_ids')
    #   input_mask_ph = tf.placeholder(shape=[1, args.max_seq_length],
    #                               dtype=tf.int32, name='input_mask')
      prob = create_model(bert_config, input_ids_ph, segment_ids_ph, None, args.num_labels)
      sess.run(tf.initializers.global_variables())
      saver = tf.train.Saver(var_list=tf.global_variables())
      print("saving model checkpoints...")
      saved_model_path = os.path.join(args.output_dir, 'model')
      saver.save(sess, saved_model_path)

      print("exporting saved model...")
      save_dir = os.path.join(args.output_dir, 'saved_model')
      if tf.gfile.Exists(save_dir):
        tf.gfile.DeleteRecursively(save_dir)
      tf.saved_model.simple_save(sess, save_dir,
                                inputs={'input_id': input_ids_ph,
                                'segment_ids': segment_ids_ph,
                                # 'input_mask': input_mask_ph,
                                },
                                outputs={'prob': prob})

      tflite_file = os.path.join(args.output_dir, 'model.tflite')
      print("exporting tflite model...")
      converter = tf.lite.TFLiteConverter.from_session(sess, [input_ids_ph, segment_ids_ph],
                                                        [prob])
      if args.quantize:
          converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

      converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
      tflite_model = converter.convert()
      with open(tflite_file, "wb") as f:
          f.write(tflite_model)
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--bert_config_file', type=str, default=None)
    parser.add_argument("-o", "--output_dir", type=str, default='bert_classifier')
    parser.add_argument("-n", "--num_labels", type=int, default=2)
    parser.add_argument("-l", "--max_seq_length", type=int, default=None)
    parser.add_argument("-q", "--quantize", action='store_true', help="quantize the tflite model")
    main(parser.parse_args())
