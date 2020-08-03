#!/usr/bin/env python
import re
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import time
import numpy as np
import threading
import signal
import json
from datetime import datetime
from datetime import timedelta
import tensorflow as tf
import logging
log = logging.getLogger('eet')
import cv2

from openvino.inference_engine import IENetwork, IECore#, IEPlugin

log.setLevel(logging.INFO)
fmt = logging.Formatter("%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: %(message)s",
                        "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
log.addHandler(handler)
log.propagate = False

xml_path = 'faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.xml'
# xml_path = 'uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12_S-384.xml'
bin_path = os.path.splitext(xml_path)[0] + ".bin"

ie = IECore()
# network = IENetwork(model=xml_path, weights=bin_path)
network = ie.read_network(model=xml_path, weights=bin_path)
network.add_outputs("detection_output")

input_blob = next(iter(network.inputs))
out_blob = next(iter(network.outputs))
network.batch_size = 1

# n, c,
h, w = network.inputs[input_blob].shape
images = np.ndarray(shape=(h, w))

executable_network = ie.load_network(network=network, device_name='MYRIAD')

start = time.perf_counter()
res = executable_network.infer(inputs={input_blob: images})
inference_time = time.perf_counter() - start

log.info('Latency {:.1f} ms'.format(inference_time * 1000))

# supported_layers = plugin.get_supported_layers(network)
# not_supported_layers = [l for l in network.layers.keys() if l not in supported_layers]
# if len(not_supported_layers) != 0:
#   log.error("Not Supported Layers : "+str(not_supported_layers))

# def image_to_tensor(image,channels,h,w,info=""):
#   print(image[0])
#   image_tensor=np.zeros(shape=(1,channels,h,w),dtype=np.float32)
#   if image.shape[:-1]!=(h,w):
#     log.warning("Image {} is resized from {} to {}".format(info, image.shape[:-1],(h,w)))
#   image=cv2.resize(image,(w,h))
#   image = image.transpose((2, 0, 1))
#   image_tensor[0]=image

#   return image_tensor

# image_np = cv2.imread('car_1.bmp')
# im = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
# image_org=image_np.copy()
# i0=image_to_tensor(im,3,h,w)
# res = executable_network.infer(inputs={input_wrapper: images})

# # network=IENetwork(model=xml_path, weights=bin_path)

# input_wrapper = next(iter(network.inputs))
# # n, c, h, w = network.inputs[input_wrapper].shape
# out_wrapper = next(iter(network.outputs))

# # plugin=IEPlugin(device="MYRIAD")
# # plugin=IEPlugin(device="CPU")
# # plugin.add_cpu_extension(CPU_EXTPATH)

# supported_layers = plugin.get_supported_layers(network)
# not_supported_layers = [l for l in network.layers.keys() if l not in supported_layers]
# if len(not_supported_layers) != 0:
#   log.error("Not Supported Layers : "+str(not_supported_layers))

# # Execution_Network=plugin.load(network=network)
# # Execution_Network = ie.load_network(network=network, device_name="MYRIAD")
# log.info("Network Loaded")

# # Inference :
# # image_np = cv2.imread('car_1.bmp')
# # im = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
# # image_org=image_np.copy()
# #used to resize image with good dimensions
# # i0=image_to_tensor(im,c,h,w)
# i0 = None
# res=Execution_Network.infer(inputs={input_wrapper: i0})

# def image_to_tensor(image,channels,h,w,info=""):
#   print(image[0])
#   image_tensor=np.zeros(shape=(1,channels,h,w),dtype=np.float32)
#   if image.shape[:-1]!=(h,w):
#     log.warning("Image {} is resized from {} to {}".format(info, image.shape[:-1],(h,w)))
#   image=cv2.resize(image,(w,h))
#   image = image.transpose((2, 0, 1))
#   image_tensor[0]=image

#   return image_tensor



# def run_ncs():
#     num_hidden_layers = 12
#     hidden_size = 768
#     num_attention_heads = 12
#     max_seq_length = 128
#     xml_path = 'faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.xml'
#     # xml_path = 'uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12_S-128.xml'

#     model_bin = os.path.splitext(xml_path)[0] + ".bin"
#     input_ids=[101, 2054, 2154, 2001, 1996, 2208, 2209, 2006, 1029, 102, 1996, 2208, 2001, 2209, 2006, 2337, 1021, 1010, 2355, 1010, 2012, 11902, 1005, 1055, 3346, 1999, 1996, 2624, 3799, 3016, 2181, 2012, 4203, 10254, 1010, 2662, 1012, 102, 0, 0]
#     segment_ids=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
#     input_ids = input_ids[:max_seq_length] + [0]*(max_seq_length-len(input_ids))
#     segment_ids = segment_ids[:max_seq_length]+ [0]*(max_seq_length-len(segment_ids))
#     logger.info("input_ids_len={}, segment_ids_len={}".format(len(input_ids), len(segment_ids)))
#     logger.info("Creating Inference Engine")
#     ie = IECore()

#     # ie.set_config({'VPU_HW_STAGES_OPTIMIZATION': 'NO'}, "MYRIAD")
#     logger.info("Loading network files:\n\t{}\n\t{}".format(xml_path, model_bin))
#     net = IENetwork(model=xml_path, weights=model_bin)

#     logger.info("Preparing input blobs")
#     # input_blob = next(iter(net.inputs))
#     # print('net.input_blob:', input_blob)
#     logger.info('net.inputs: {}'.format(net.inputs))
#     for k, v in net.inputs.items():
#         logger.info('net input: {}, shape={}'.format(k, v.shape))

#     logger.info('net.outputs: {}'.format(net.outputs))
#     for k, v in net.outputs.items():
#         logger.info('net output: {}, shape={}'.format(k, v.shape))

#     logger.info("Loading model to the plugin")
#     exec_net = ie.load_network(network=net, device_name="MYRIAD")

#     # res = exec_net.infer(inputs={input_blob: np.zeros([1, args.size])})
#     input_ids = np.reshape(input_ids, [1, max_seq_length])
#     segment_ids = np.reshape(segment_ids, [1, max_seq_length])
#     input_mask = input_ids.astype(np.bool).astype(np.float32)

#     logger.info('############ starting inference iter {}'.format(iteration))
#     start = time.perf_counter()
#     res = exec_net.infer(inputs={
#         'input_ids': input_ids,
#         'segment_ids': segment_ids,
#         'input_mask': input_mask,
#     })
#     inference_time = time.perf_counter() - start
#     logger.info('############ {:.1f} ms'.format(inference_time * 1000))

# def main():
#     run_ncs()

# if __name__ == '__main__':
#     sys.exit(main() or 0)