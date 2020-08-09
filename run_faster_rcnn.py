#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import logging
log = logging.getLogger('eet')
from openvino.inference_engine import IENetwork, IECore#, IEPlugin
from argparse import ArgumentParser, SUPPRESS

log.setLevel(logging.INFO)
fmt = logging.Formatter("%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: %(message)s",
                        "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
log.addHandler(handler)
log.propagate = False

# To convert the tf model to OpenVINO, go to faster_rcnn_resnet101_coco_2018_01_28 directory and execute:
# python -m mo_tf --input_model frozen_inference_graph.pb --output_dir . --tensorflow_use_custom_operations_config "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\extensions\front\tf\faster_rcnn_support.json" --tensorflow_object_detection_api_pipeline_config ./pipeline.config --reverse_input_channels --data_type FP16 --input_shape [1,600,1024,3]

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    return parser

def main():
    args = build_argparser().parse_args()
    xml_path = args.model
    bin_path = os.path.splitext(xml_path)[0] + ".bin"

    ie = IECore()
    network = ie.read_network(model=xml_path, weights=bin_path)
    # network.add_outputs("detection_output")

    input_blob = next(iter(network.inputs))
    out_blob = next(iter(network.outputs))
    network.batch_size = 1

    # n, c, h, w = network.inputs[input_blob].shape
    # images = np.ndarray(shape=(h, w))
    images = np.ndarray(shape=network.inputs[input_blob].shape)

    executable_network = ie.load_network(network=network, device_name='MYRIAD')

    start = time.perf_counter()
    res = executable_network.infer(inputs={input_blob: images})
    inference_time = time.perf_counter() - start

    log.info('Latency {:.1f} ms'.format(inference_time * 1000))

if __name__ == '__main__':
    sys.exit(main() or 0)