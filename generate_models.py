import argparse
import os
import modeling
import tensorflow as tf
from multiprocessing import Pool, Queue, Process, Manager
import signal
import time
from accelerator_util import *

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
def generate_model(output_dir,
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_seq_length = 384,
                num_labels = 2,
                quantize = True):
    tf.reset_default_graph()
    bert_config = modeling.BertConfig(vocab_size=vocab_size,
                                      hidden_size=hidden_size,
                                      num_hidden_layers=num_hidden_layers,
                                      num_attention_heads=num_attention_heads,
                                      intermediate_size=intermediate_size)

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

      # tflite_file = os.path.join(output_dir, 'model.tflite')
      # print("exporting tflite model...")
      # converter = tf.lite.TFLiteConverter.from_session(sess, [input_ids_ph, segment_ids_ph],
      #                                                   [prob])
      # if quantize:
      #     converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

      # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
      # tflite_model = converter.convert()
      # with open(tflite_file, "wb") as f:
      #     f.write(tflite_model)

def worker(worker_id, queue):
  model_info = queue.get()
  model_dir_path = generated_models_dir + '/' + model_info['model_name']
  make_dir(model_dir_path)
  print(model_dir_path)

  try:
    generate_model(model_dir_path,
                  model_info['vocab_size'],
                  model_info['hidden_size'],
                  model_info['num_hidden_layers'],
                  model_info['num_attention_heads'],
                  model_info['intermediate_size'],
                  model_info['input_size'])
  except ValueError as ve:
    print('Error generating model: {}'.format(ve))

  convert_to_ir(generated_models_dir + '/' + model_info['model_meta_path'])

def generate_models_info():
  l_vocab_size = [30522] #[1000, 30522]
  l_hidden_size = [768]
  l_num_hidden_layers = [9] # 12 doesn't work
  l_num_attention_heads = [8, 9, 12, 16, 24, 32] #[1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32]
  l_intermediate_size = [1024, 3072] # [1024, 3072]
  l_input_size = [100, 200, 300, 400, 500] #[100, 200, 300, 384, 400, 500]

  return generate_models_info_base(l_vocab_size,
                          l_hidden_size,
                          l_num_hidden_layers,
                          l_num_attention_heads,
                          l_intermediate_size,
                          l_input_size)

generated_models_dir = 'generated_models'
num_workers = 6

def main():
  make_dir(generated_models_dir)
  manager = Manager()
  queue = manager.Queue()
  models_info = generate_models_info()
  for model_info in models_info:
    queue.put(model_info)
  pool = Pool(num_workers, init_worker)

  while not queue.empty():
    qsize = queue.qsize()
    workers = []

    for worker_id in range(num_workers if num_workers > qsize else qsize):
      workers.append(pool.apply_async(worker, (worker_id, queue,)))

    try:
      [worker.get() for worker in workers]
    except:
      print('Interrupted')
      pool.terminate()
      pool.join()

if __name__ == '__main__':
  main()