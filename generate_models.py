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

# def main(args):
#     bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
def generate_model(hidden_size, num_hidden_layers, num_attention_layers, intermediate_size, output_dir, max_seq_length, num_labels, quantize):
    tf.reset_default_graph()
    bert_config = modeling.BertConfig(30522, hidden_size, num_hidden_layers, num_attention_layers, intermediate_size)

    # print(bert_config.to_json_string())


    tf.gfile.MakeDirs(output_dir)

    with tf.Session() as sess:
      input_ids_ph = tf.placeholder(shape=[1, max_seq_length],
                                  dtype=tf.int32, name='input_ids')
      segment_ids_ph = tf.placeholder(shape=[1, max_seq_length],
                                  dtype=tf.int32, name='segment_ids')
    #   input_mask_ph = tf.placeholder(shape=[1, max_seq_length],
    #                               dtype=tf.int32, name='input_mask')
      prob = create_model(bert_config, input_ids_ph, segment_ids_ph, None, num_labels)
      sess.run(tf.initializers.global_variables())
      saver = tf.train.Saver(var_list=tf.global_variables())
      print("saving model checkpoints...")
      saved_model_path = os.path.join(output_dir, 'model')
      saver.save(sess, saved_model_path)

      print("exporting saved model...")
      save_dir = os.path.join(output_dir, 'saved_model')
      if tf.gfile.Exists(save_dir):
        tf.gfile.DeleteRecursively(save_dir)
      tf.saved_model.simple_save(sess, save_dir,
                                inputs={'input_id': input_ids_ph,
                                'segment_ids': segment_ids_ph,
                                # 'input_mask': input_mask_ph,
                                },
                                outputs={'prob': prob})

      tflite_file = os.path.join(output_dir, 'model.tflite')
      print("exporting tflite model...")
      converter = tf.lite.TFLiteConverter.from_session(sess, [input_ids_ph, segment_ids_ph],
                                                        [prob])
      if quantize:
          converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

      converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
      tflite_model = converter.convert()
      with open(tflite_file, "wb") as f:
          f.write(tflite_model)

def convert_to_ir(model_meta_path):

  command ='python -m mo_tf --output prob --disable_nhwc_to_nchw --input input_ids{i32},segment_ids{i32}' \
    + ' --input_meta_graph "' + model_meta_path + '"' \
    + ' --output_dir "' + os.path.dirname(model_meta_path) + '"'
  print(command)
  os.system(command)

def main():
  hidden_size_arr = [256, 512, 768]
  num_hidden_layers_arr = [3, 4, 6, 8, 9, 12]
  num_attention_layers_arr = [4, 6, 8, 12, 16]
  intermediate_size_arr = [3072] # [768, 1024, 1536, 2048, 3072]
  experiments_dir = 'experiments'

  model_count = 0
  for hidden_size in hidden_size_arr:
    for num_hidden_layers in num_hidden_layers_arr:
      for num_attention_layers in num_attention_layers_arr:
        for intermediate_size in intermediate_size_arr:
          output_dir = experiments_dir + '/hidden_size_' + str(hidden_size) \
            + '_num_hidden_layers_' + str(num_hidden_layers) \
            + '_num_attention_layers_' + str(num_attention_layers) \
            + '_intermediate_size_' + str(intermediate_size)

          try:
            generate_model(hidden_size, num_hidden_layers, num_attention_layers, \
              intermediate_size, output_dir, 384, 2, True)
          except ValueError as ve:
            print(ve)

          convert_to_ir(output_dir + '/model.meta')

          model_count+=1

  print('Model count: ', model_count)

if __name__ == '__main__':
  main()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--bert_config_file', type=str, default=None)
    # parser.add_argument("-o", "--output_dir", type=str, default='bert_classifier')
    # parser.add_argument("-n", "--num_labels", type=int, default=2)
    # parser.add_argument("-l", "--max_seq_length", type=int, default=None)
    # parser.add_argument("-q", "--quantize", action='store_true', help="quantize the tflite model")
    # main(parser.parse_args())