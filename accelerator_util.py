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