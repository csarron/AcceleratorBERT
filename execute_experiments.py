#!/usr/bin/env python
import re
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import time
import numpy as np
import modeling
import threading
import signal
import tensorflow as tf
import logging
logger = logging.getLogger('eet')

from openvino.inference_engine import IENetwork, IECore

logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: %(message)s",
                        "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.propagate = False

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


def generate_bert_model(bert_config, max_seq_length, model_name, model_dir):
    num_labels = 2
    ckpt_file = None

    tf.gfile.MakeDirs(model_dir)
    saved_model_path = os.path.join(model_dir, model_name)
    input_ids=[101, 2054, 2154, 2001, 1996, 2208, 2209, 2006, 1029, 102, 1996, 2208, 2001, 2209, 2006, 2337, 1021, 1010, 2355, 1010, 2012, 11902, 1005, 1055, 3346, 1999, 1996, 2624, 3799, 3016, 2181, 2012, 4203, 10254, 1010, 2662, 1012, 102, 0, 0]
    segment_ids=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    input_ids = input_ids[:max_seq_length] + [0]*(max_seq_length-len(input_ids))
    segment_ids = segment_ids[:max_seq_length]+ [0]*(max_seq_length-len(segment_ids))

    with tf.Session() as sess:
        input_ids_ph = tf.placeholder(shape=[1, max_seq_length], dtype=tf.int32, name='input_ids')
        segment_ids_ph = tf.placeholder(shape=[1, max_seq_length], dtype=tf.int32, name='segment_ids')
        input_mask_ph = tf.placeholder(shape=[1, max_seq_length], dtype=tf.float32, name='input_mask')
        prob = create_model(bert_config, input_ids_ph, segment_ids_ph, input_mask_ph, num_labels)
        init_ckpt(ckpt_file)
        sess.run(tf.initializers.global_variables())

        logger.info("saving model checkpoints...")
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.save(sess, saved_model_path)
        input_ids = np.reshape(input_ids, [1, max_seq_length])
        segment_ids = np.reshape(segment_ids, [1, max_seq_length])
        input_mask = input_ids.astype(np.bool).astype(np.float32)
        model_prob = sess.run(prob, {
          input_ids_ph: input_ids,
          segment_ids_ph: segment_ids,
          input_mask_ph: input_mask,
        })
        # logger.info('model_outputs: \n{}\n'.format(model_prob))

def get_model_list():
    model_list = []

    for max_seq_length in [128]:
        for num_hidden_layers in [2, 4, 6]:
            for hidden_size in [128, 144, 160, 192, 208, 224, 240]:
                for num_attention_heads in [2, 4, 8, 16]:
        # for num_hidden_layers in [2]:
        #     for hidden_size in [128, 144]:
        #         for num_attention_heads in [16]:
                    bert_config = modeling.BertConfig(
                        vocab_size = 30522,
                        hidden_size = hidden_size,
                        num_hidden_layers = num_hidden_layers,
                        num_attention_heads = num_attention_heads,
                        intermediate_size = 3072,
                        hidden_act = "gelu",
                        hidden_dropout_prob = 0.1,
                        attention_probs_dropout_prob = 0.1,
                        max_position_embeddings = 512,
                        type_vocab_size = 16,
                        initializer_range = 0.02)

                    model_name = "L-{}_H-{}_A-{}".format(num_hidden_layers, hidden_size, num_attention_heads)
                    model_dir = os.path.join(general_dir, model_name)
                    model_meta = os.path.join(model_dir, model_name + '.meta')
                    model_xml = os.path.join(model_dir, model_name + '.xml')

                    model_list.append((bert_config, max_seq_length, model_name, model_dir, model_meta, model_xml))

    return model_list

running = True

def generate_model(model_entry):

    if running == False:
        print('----- Exiting')
        return

    logger.info('----- Generating model ' + model_entry[2])

    generate_bert_model(bert_config = model_entry[0], max_seq_length = model_entry[1],
        model_name = model_entry[2], model_dir = model_entry[3])

    openvino_command = 'python -m mo_tf --output prob --disable_nhwc_to_nchw --progress --input input_ids{{i32}},segment_ids{{i32}},input_mask{{f32}} --input_meta_graph {} --output_dir {}'.format(model_entry[4], model_entry[3])

    logger.info('----- Executing OpenVINO command: \n' + openvino_command)
    os.system(openvino_command)

def generate_models(model_list):
    thread_count = 6

    threads = []
    i = 0
    while i < len(model_list):
        threads.append(threading.Thread(target = generate_model, args = (model_list[i],)))

        if (i > 0 and i % thread_count == 0) or (i == len(model_list) - 1):
            for x in threads:
                x.start()
            for x in threads:
                x.join()
            threads.clear()

        i += 1

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    running = False
    sys.exit(0)

def run_ncs(xml_path, max_seq_length, iteration_count):
    # log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    # args = build_argparser().parse_args()
    # model_xml = args.model
    model_bin = os.path.splitext(xml_path)[0] + ".bin"
    input_ids=[101, 2054, 2154, 2001, 1996, 2208, 2209, 2006, 1029, 102, 1996, 2208, 2001, 2209, 2006, 2337, 1021, 1010, 2355, 1010, 2012, 11902, 1005, 1055, 3346, 1999, 1996, 2624, 3799, 3016, 2181, 2012, 4203, 10254, 1010, 2662, 1012, 102, 0, 0]
    segment_ids=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    input_ids = input_ids[:max_seq_length] + [0]*(max_seq_length-len(input_ids))
    segment_ids = segment_ids[:max_seq_length]+ [0]*(max_seq_length-len(segment_ids))
    logger.info("input_ids_len={}, segment_ids_len={}".format(len(input_ids), len(segment_ids)))
    logger.info("Creating Inference Engine")
    ie = IECore()

    ie.set_config({'VPU_HW_STAGES_OPTIMIZATION': 'NO'}, "MYRIAD")
    logger.info("Loading network files:\n\t{}\n\t{}".format(xml_path, model_bin))
    net = IENetwork(model=xml_path, weights=model_bin)

    logger.info("Preparing input blobs")
    # input_blob = next(iter(net.inputs))
    # print('net.input_blob:', input_blob)
    logger.info('net.inputs: {}'.format(net.inputs))
    for k, v in net.inputs.items():
        logger.info('net input: {}, shape={}'.format(k, v.shape))

    logger.info('net.outputs: {}'.format(net.outputs))
    for k, v in net.outputs.items():
        logger.info('net output: {}, shape={}'.format(k, v.shape))

    logger.info("Loading model to the plugin")
    # mapped_device = _DEVICES.get(device_name, device_name)
    # correct_device = None
    # for device in ie.available_devices:
    #     if re.match(mapped_device, device):
    #         correct_device = device
    #         full_name = ie.get_metric(device, "FULL_DEVICE_NAME")
    #         logger.info("Device: {}, {}".format(correct_device, full_name))
    #         break
    # exec_net = ie.load_network(network=net, device_name=correct_device)
    exec_net = ie.load_network(network=net, device_name="MYRIAD")

    # res = exec_net.infer(inputs={input_blob: np.zeros([1, args.size])})
    input_ids = np.reshape(input_ids, [1, max_seq_length])
    segment_ids = np.reshape(segment_ids, [1, max_seq_length])
    input_mask = input_ids.astype(np.bool).astype(np.float32)

    logger.info("Waiting 5 sec")
    time.sleep(5)
    logger.info("############ Starting inference in synchronous mode")
    infer_times = []
    for iteration in range(iteration_count):
        start = time.perf_counter()
        # # res = exec_net.infer(inputs={input_blob: np.zeros([1, args.size])})
        # input_ids = np.reshape(input_ids, [1, max_seq_length])
        # segment_ids = np.reshape(segment_ids, [1, max_seq_length])
        # input_mask = input_ids.astype(np.bool).astype(np.float32)

        logger.info('############ starting inference iter {}'.format(iteration+1))
        res = exec_net.infer(inputs={
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'input_mask': input_mask,
        })
        inference_time = time.perf_counter() - start
        infer_times.append(inference_time * 1000)
        logger.info('############ latency iter {}: {:.1f}ms'.format(iteration+1, inference_time * 1000))

        # if iteration % 2 == 0:
        #     logger.info("Waiting 10 sec")
        #     time.sleep(10)


    # logger.info("############ Processing output blob")
    # # logger.info("results b: {}".format(res))
    # for k, v in res.items():
    #     logger.info('net results: {}, shape={}'.format(k, v.shape))
    # logger.info('prob: {}, shape={}'.format(res['prob'], res['prob'].shape))
    logger.info('############ model: {}, input size={}, latency avg={:.1f} ms, std={:.3f} ms'.format(
        xml_path, max_seq_length, np.mean(infer_times), np.std(infer_times)))

general_dir = 'data/custom_bert'

def run_models(model_list):
    model_list = get_model_list()

    for _, max_seq_length, _, _, _, model_xml in model_list:
        logger.info(model_xml)
        run_ncs(model_xml, max_seq_length, iteration_count = 10)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    model_list = get_model_list()

    # generate_models(model_list)
    run_models(model_list)

if __name__ == '__main__':
    sys.exit(main() or 0)