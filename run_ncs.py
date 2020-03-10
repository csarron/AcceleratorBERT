#!/usr/bin/env python
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import time
import numpy as np
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


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=False,
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="MYRIAD", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
    args.add_argument("-s", "--max_seq_length", help="input size", default=40, type=int)
    args.add_argument('-c', '--count', type=int, default=5,
        help='Number of times to run inference')
    return parser


def main():
    # log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    input_ids=[101, 2054, 2154, 2001, 1996, 2208, 2209, 2006, 1029, 102, 1996, 2208, 2001, 2209, 2006, 2337, 1021, 1010, 2355, 1010, 2012, 11902, 1005, 1055, 3346, 1999, 1996, 2624, 3799, 3016, 2181, 2012, 4203, 10254, 1010, 2662, 1012, 102, 0, 0]
    segment_ids=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    input_ids = input_ids[:args.max_seq_length] + [0]*(args.max_seq_length-len(input_ids))
    segment_ids = segment_ids[:args.max_seq_length]+ [0]*(args.max_seq_length-len(segment_ids))
    logger.info("input_ids_len={}, segment_ids_len={}".format(len(input_ids), len(segment_ids)))

    # Plugin initialization for specified device and load extensions library if specified
    logger.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    logger.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    logger.info("Preparing input blobs")
    # input_blob = next(iter(net.inputs))
    # print('net.input_blob:', input_blob)
    logger.info('net.inputs: {}'.format(net.inputs))
    for k, v in net.inputs.items():
        logger.info('net input: {}, shape={}'.format(k, v.shape))
    
    logger.info('net.outputs: {}'.format(net.outputs))
    for k, v in net.outputs.items():
        logger.info('net output: {}, shape={}'.format(k, v.shape))
    # out_blob = next(iter(net.outputs))
    # net.batch_size = len(args.input)

    # # Read and pre-process input images
    # n, c, h, w = net.inputs[input_blob].shape
    # images = np.ndarray(shape=(n, c, h, w))
    # for i in range(n):
    #     image = cv2.imread(args.input[i])
    #     if image.shape[:-1] != (h, w):
    #         log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
    #         image = cv2.resize(image, (w, h))
    #     image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    #     images[i] = image
    # log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    logger.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    logger.info("Starting inference in synchronous mode")
    infer_times = []
    for iteration in range(args.count):
        start = time.perf_counter()
        # res = exec_net.infer(inputs={input_blob: np.zeros([1, args.size])})
        input_ids = np.reshape(input_ids, [1, args.max_seq_length])
        segment_ids = np.reshape(segment_ids, [1, args.max_seq_length])
        input_mask = input_ids.astype(np.bool).astype(np.float32)
        res = exec_net.infer(inputs={
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'input_mask': input_mask,
        })
        inference_time = time.perf_counter() - start
        infer_times.append(inference_time * 1000)
        logger.info('latency iter{} {:.1f}ms'.format(iteration+1, inference_time * 1000))


    # Processing output blob
    logger.info("Processing output blob")
    # logger.info("results b: {}".format(res))
    for k, v in res.items():
        logger.info('net results: {}, shape={}'.format(k, v.shape))
    logger.info('prob: {}, shape={}'.format(res['prob'], res['prob'].shape))
    logger.info('latency avg={:.1f} ms, std={:.3f} ms'.format(np.mean(infer_times), np.std(infer_times)))

if __name__ == '__main__':
    sys.exit(main() or 0)
