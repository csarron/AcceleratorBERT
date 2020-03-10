# from __future__ import print_function
import sys
import os
# from argparse import ArgumentParser, SUPPRESS
# import cv2
# import numpy as np
# import logging as log
# from time import time
# from openvino.inference_engine import IENetwork, IEPlugin
from datetime import datetime

def generate_models_info():
  l_vocab_size = [30522] #30522
  l_hidden_size = [768]
  l_num_hidden_layers = [12]
  l_num_attention_heads = [1]#, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32]
  l_intermediate_size = [3072] #3072
  l_input_size = [384]

  return generate_models_info_base(l_vocab_size,
                          l_hidden_size,
                          l_num_hidden_layers,
                          l_num_attention_heads,
                          l_intermediate_size,
                          l_input_size)

def generate_models_info_base(l_vocab_size,
                          l_hidden_size,
                          l_num_hidden_layers,
                          l_num_attention_heads,
                          l_intermediate_size,
                          l_input_size):

  models_info = []

  for vocab_size in l_vocab_size:
    for hidden_size in l_hidden_size:
      for num_hidden_layers in l_num_hidden_layers:
        for num_attention_heads in l_num_attention_heads:
          for intermediate_size in l_intermediate_size:
            for input_size in l_input_size:
              model_info = {'vocab_size': vocab_size,
                                'hidden_size': hidden_size,
                                'num_hidden_layers': num_hidden_layers,
                                'num_attention_heads': num_attention_heads,
                                'intermediate_size': intermediate_size,
                                'input_size': input_size,
                                }
              model_name = ''
              results_file_header = ''
              results_file_row = ''
              for key, value in model_info.items():
                model_name += key + '_' + str(value).zfill(4) + '_'
                results_file_header += key + '\t'
                results_file_row += str(value) + '\t'
              model_info['model_name'] = model_name
              model_info['model_meta_path'] = model_name + '/model.meta'
              model_info['model_xml_path'] = model_name + '/model.xml'
              model_info['results_file_header'] = results_file_header
              model_info['results_file_row'] = results_file_row
              models_info.append(model_info)

  return models_info

def make_dir(dir_path):
  os.makedirs(dir_path, exist_ok=True)

def make_parent_dir(dir_path):
  make_dir(os.path.dirname(dir_path))

def get_path_without_ext(file_path):
  return os.path.splitext(file_path)[0]

def replace_ext(file_path, new_ext):
  return get_path_without_ext(file_path) + '.' + new_ext

def get_timestamp():
  return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

# def write_ncs_perf_header(perf_counts_file):
#     perf_counts_file.write('name\tlayer_type\texec_type\tstatus\treal_time_us\n')

# def write_ncs_perf_results(perf_counts, perf_counts_file):
#     for layer, stats in perf_counts.items():
#         perf_counts_file.write("{}\t{}\t{}\t{}\t{}\n".format(
#           layer,
#           stats['layer_type'],
#           stats['exec_type'],
#           stats['status'],
#           stats['real_time']))

def write_ncs_perf_counts(perf_counts, file_name):
  os.makedirs(os.path.dirname(file_name), exist_ok=True)

  with open(file_name, 'w') as perf_counts_file:
    perf_counts_file.write('name\tlayer_type\texec_type\tstatus\treal_time_us\n')
    for layer, stats in perf_counts.items():
        perf_counts_file.write("{}\t{}\t{}\t{}\t{}\n".format(
          layer,
          stats['layer_type'],
          stats['exec_type'],
          stats['status'],
          stats['real_time']))

def get_total_perf_time(perf_counts):
  total_execution_time = 0
  for layer, stats in perf_counts.items():
      total_execution_time += stats['real_time']
  return total_execution_time

# def write_ncs_total_execution_time(perf_counts_file, total_execution_time):
#   perf_counts_file.write('name\tlayer_type\texec_type\tstatus\treal_time_us\n')