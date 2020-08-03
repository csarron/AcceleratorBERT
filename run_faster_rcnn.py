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

# To convert the tf model to OpenVINO, go to faster_rcnn_resnet101_coco_2018_01_28 directory and execute:
# python -m mo_tf --input_model frozen_inference_graph.pb --output_dir . --tensorflow_use_custom_operations_config "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\extensions\front\tf\faster_rcnn_support.json" --tensorflow_object_detection_api_pipeline_config ./pipeline.config --reverse_input_channels --data_type FP16 --input_shape [1,600,1024,3]

xml_path = 'faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.xml'
bin_path = os.path.splitext(xml_path)[0] + ".bin"

ie = IECore()
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