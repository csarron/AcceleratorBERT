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

# def build_argparser():
#     parser = ArgumentParser(add_help=False)
#     args = parser.add_argument_group('Options')
#     args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
#     args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
#                       type=str)
#     args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
#                       required=False,
#                       type=str, nargs="+")
#     args.add_argument("-l", "--cpu_extension",
#                       help="Optional. Required for CPU custom layers. "
#                            "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
#                            " kernels implementations.", type=str, default=None)
#     args.add_argument("-d", "--device",
#                       help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
#                            "acceptable. The sample will look for a suitable plugin for device specified. Default "
#                            "value is CPU",
#                       default="MYRIAD", type=str)
#     args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
#     args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
#     args.add_argument("-s", "--size", help="input size", default=384, type=int)

#     return parser

# def log_perf_counts(perf_counts, perf_counts_file):
#     perf_counts_file.write('name\tlayer_type\texec_type\tstatus\treal_time_us\n')

#     for layer, stats in perf_counts.items():
#         # log_content = "{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'],
#         #     stats['exec_type'], stats['status'], stats['real_time'])
#         log_content = "{}\t{}\t{}\t{}\t{}\n".format(layer, stats['layer_type'],
#             stats['exec_type'], stats['status'], stats['real_time'])
#         # print(log_content)
#         perf_counts_file.write(log_content)

# def run_model(hidden_size, num_hidden_layers, num_attention_layers, intermediate_size, experiments_log_file):
#     # log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

#     model_dir_name = 'hidden_size_' + str(hidden_size).zfill(4) \
#         + '_num_hidden_layers_' + str(num_hidden_layers).zfill(2) \
#         + '_num_attention_layers_' + str(num_attention_layers).zfill(2) \
#         + '_intermediate_size_' + str(intermediate_size).zfill(4)

#     model_dir = 'experiments/' + model_dir_name

#     # args = build_argparser().parse_args()
#     model_xml = model_dir + '/model.xml'
#     model_bin = os.path.splitext(model_xml)[0] + ".bin"

#     if not os.path.exists(model_xml):
#       return

#     logger.info("Creating Inference Engine")
#     ie = IECore()
#     logger.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
#     net = IENetwork(model=model_xml, weights=model_bin)

#     input_blob = next(iter(net.inputs))
#     # print('net.input_blob:', input_blob)
#     logger.info('net.inputs: {}'.format(net.inputs))
#     for k, v in net.inputs.items():
#         logger.info('net input: {}, shape={}'.format(k, v.shape))

#     logger.info('net.outputs: {}'.format(net.outputs))
#     for k, v in net.outputs.items():
#         logger.info('net output: {}, shape={}'.format(k, v.shape))
#     # out_blob = next(iter(net.outputs))
#     # net.batch_size = len(args.input)


#   # for i in range(0, 5000):
#   # print('%.1fms' % (inference_time * 1000))

#     exec_net = ie.load_network(network=net, device_name='MYRIAD.1.3-ma2450')
#     res = exec_net.infer(inputs={input_blob: np.zeros([1, 384])})

#     start = time.perf_counter()
#     res = exec_net.infer(inputs={input_blob: np.zeros([1, 384])})
#     inference_time = time.perf_counter() - start
#     perf_counts = exec_net.requests[0].get_perf_counts()

#     # logger.info("Processing output blob")
#     # logger.info("results b: {}".format(res))
#     # for k, v in res.items():
#     #     logger.info('net results: {}, shape={}, value={}'.format(k, v.shape, v))

#     experiments_log_file.write(str(hidden_size) + '\t' + str(num_hidden_layers) +'\t' + str(num_attention_layers) +'\t' + str(intermediate_size) + '\t' + ('%.2f' % (inference_time * 1000)) + '\n')

#     with open('perf_counts_ncs1/' + model_dir_name + '_' + get_timestamp() + '.log', 'w') as perf_counts_file:
#       log_perf_counts(perf_counts, perf_counts_file)

# def main_1():
#   hidden_size_arr = [768] #[256, 512, 768]
#   num_hidden_layers_arr = [3, 4, 6, 8, 9]#, 12]
#   num_attention_layers_arr = [4, 6, 8, 12, 16]
#   intermediate_size_arr = [2048, 3072] # [768, 1024, 1536, 2048, 3072]

#   experiments_log_path = 'experiment_' + get_timestamp() + '.ncs1.log'

#   with open(experiments_log_path, 'w') as experiments_log_file:
#     experiments_log_file.write('hidden_size' + '\t' + 'num_hidden_layers' +'\t' + 'num_attention_layers' +'\t' + 'intermediate_size' + '\t' + 'inference_time_ms' + '\n')

#     for hidden_size in hidden_size_arr:
#       for num_hidden_layers in num_hidden_layers_arr:
#         for num_attention_layers in num_attention_layers_arr:
#           for intermediate_size in intermediate_size_arr:
#             try:
#               run_model(hidden_size, num_hidden_layers, num_attention_layers, intermediate_size, experiments_log_file)
#             except ValueError as ve:
#               print(ve)
#             except Exception as e:
#               print(e)


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
    # res = exec_net.infer(inputs={input_blob: np.zeros([1, input_size])})

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


generated_models_dir = 'generated_models'

def main():
  results_file_path = 'experiments_10Mar/experiment_' + get_timestamp() + '.ncs1.results.log'
  make_parent_dir(results_file_path)
  # print(results_file_path)

  models_info = generate_models_info()
  # for model_info in models_info:
  #   print(model_info['model_name'])

  with open(results_file_path, 'w') as results_file:
    # write_ncs_perf_header(results_file)
    results_file.write(models_info[0]['results_file_header'] + '\n')
    for model_info in models_info:
      run_model(model_info, results_file)

if __name__ == '__main__':
    sys.exit(main() or 0)
