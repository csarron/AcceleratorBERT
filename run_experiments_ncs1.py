from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
# import cv2
import numpy as np
from accelerator_util import *

import time
import logging
logger = logging.getLogger('eet')

logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: %(message)s",
                        "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.propagate = False
from openvino.inference_engine import IENetwork, IECore

def run_model(model_info, results_file):
    # log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    model_xml_path_abs = generated_models_dir + '/' + model_info['model_xml_path']

    if not os.path.exists(model_xml_path_abs):
      logger.error("File {} doesn't exist".format(model_xml_path_abs))
      return

    model_bin_path_abs = replace_ext(model_xml_path_abs, "bin")
    ie = IECore()
    logger.info("Loading network files:\n\t{}\n\t{}".format(model_xml_path_abs, model_bin_path_abs))
    net = IENetwork(model=model_xml_path_abs, weights=model_bin_path_abs)

    input_blob = next(iter(net.inputs))
    # logger.info('net.inputs: {}'.format(net.inputs))
    # for k, v in net.inputs.items():
    #     logger.info('net input: {}, shape={}'.format(k, v.shape))

    # logger.info('net.outputs: {}'.format(net.outputs))
    # for k, v in net.outputs.items():
    #     logger.info('net output: {}, shape={}'.format(k, v.shape))
    # # out_blob = next(iter(net.outputs))
    # # net.batch_size = len(args.input)

    exec_net = ie.load_network(network=net, device_name='MYRIAD')
    # res = exec_net.infer(inputs={input_blob: np.zeros([1, input_size])}) // No need, the latency is pretty precise

    start = time.perf_counter()
    res = exec_net.infer(inputs={input_blob: np.zeros([1, model_info['input_size']])})
    inference_time = time.perf_counter() - start
    perf_counts = exec_net.requests[0].get_perf_counts()

    # perf_counts_log_path = 'experiments_input_size_logs/input_size_' + str(input_size).zfill(3) + '_' + get_timestamp() + '.ncs1.log'
    # write_ncs_perf_counts(perf_counts, perf_counts_log_path)
    results_file.write(model_info['results_file_row'])
    results_file.write(str(get_total_perf_time(perf_counts)) + '\n')

    # logger.info("Processing output blob")
    # logger.info("results b: {}".format(res))
    # for k, v in res.items():
    #     logger.info('net results: {}, shape={}, value={}'.format(k, v.shape, v))

    # experiments_log_file.write(str(hidden_size) + '\t' + str(num_hidden_layers) +'\t' + str(num_attention_layers) +'\t' + str(intermediate_size) + '\t' + ('%.2f' % (inference_time * 1000)) + '\n')

    # with open('perf_counts_ncs1/' + model_dir_name + '_' + get_timestamp() + '.log', 'w') as perf_counts_file:
    #   log_perf_counts(perf_counts, perf_counts_file)

def generate_models_info():
  l_vocab_size = [1000, 30522]
  l_hidden_size = [768]
  l_num_hidden_layers = [9] # 12 doesn't work
  l_num_attention_heads = [1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32]
  l_intermediate_size = [1024] #[1024, 3072]
  l_input_size = [384]

  return generate_models_info_base(l_vocab_size,
                          l_hidden_size,
                          l_num_hidden_layers,
                          l_num_attention_heads,
                          l_intermediate_size,
                          l_input_size)

generated_models_dir = 'generated_models_3'

def main():
  results_file_path = 'experiments_10Mar/experiment_' + get_timestamp() + '.ncs1.results.log'
  make_parent_dir(results_file_path)
  # print(results_file_path)

  models_info = generate_models_info()
  # for model_info in models_info:
  #   print(model_info['model_name'])

  with open(results_file_path, 'w') as results_file:
    # write_ncs_perf_header(results_file)
    results_file.write(models_info[0]['results_file_header'] + 'total_time_us\n')
    for model_info in models_info:
      run_model(model_info, results_file)

if __name__ == '__main__':
    sys.exit(main() or 0)
