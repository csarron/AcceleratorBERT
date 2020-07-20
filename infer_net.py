# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import logging
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

import nets_factory
from tx2_power_logger import PowerLogger
from tx2_power_logger import getNodes

slim = tf.contrib.slim

logger = logging.getLogger('eet')

logger.setLevel(logging.INFO)
fmt = logging.Formatter(
    "%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: %(message)s",
    "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


def main(args):
    if not args.output_file:
        raise ValueError(
            'You must supply the path to save to with --output_file')
    if args.cpu:
        device = 'CPU'
        print("using CPU for model prediction ")
    else:
        device = 'GPU'
        print("using GPU, Available: ",
              len(tf.config.experimental.list_physical_devices('GPU')))
    fields = ['model', 'latency'] + ['energy_{}'.format(n[0])
                                     for n in getNodes()]
    with open(args.output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model_name in ["squeezenet", "mobilenet_v1_1.0",
                           "mobilenet_v2_1.0", "resnet_v1_50",
                           "inception_v3", "alexnet_v2"]:
            tf.reset_default_graph()
            depth_multiplier_dict = {}
            with tf.device('/{}:*'.format(device)):
                sess = tf.Session()
                if model_name.startswith('mobilenet'):
                    name_parts = model_name.split('_')
                    depth = float(name_parts[-1])
                    depth_multiplier_dict = {'depth_multiplier': depth}
                    model_name = '_'.join(name_parts[:-1])
                logger.info("model={}, depth_multiplier={}".format(model_name, depth_multiplier_dict))
                network_fn = nets_factory.get_network_fn(
                    model_name, num_classes=
                    (args.num_classes - args.labels_offset), is_training=False)
                image_size = args.image_size or network_fn.default_image_size
                placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                             shape=[1, image_size, image_size,
                                                    3])

                logits, _ = network_fn(placeholder, **depth_multiplier_dict)
                prob = tf.nn.softmax(logits, name='output')
                sess.run(tf.initializers.global_variables())
                inputs = np.random.rand(1, image_size, image_size, 3)
                # warm_up 2 rounds
                for _ in range(2):
                    model_prob = sess.run(prob, {
                        placeholder: inputs,
                    })
                logger.info(
                    'start benchmarking, model_prob: \n{}\n'.format(model_prob))
                infer_times = []
                infer_energy = defaultdict(list)
                for iteration in range(args.iterations):
                    pl = PowerLogger(interval=0.05, nodes=getNodes())
                    pl.start()
                    pl.recordEvent('run model!')
                    start = time.perf_counter()
                    passes = 10
                    for i in range(passes):
                        model_prob = sess.run(prob, {
                            placeholder: inputs,
                        })
                    inference_time = (time.perf_counter() - start) / passes
                    pl.stop()
                    iteration_energy = pl.getEnergyTraces()
                    for energy_name, energy_val in iteration_energy.items():
                        infer_energy[energy_name].append(energy_val / passes)
                        logger.info('{}_energy iter{} {:.3f} J'.format(
                            energy_name, iteration + 1, + energy_val / passes))
                    infer_times.append(inference_time * 1000)
                    logger.info('latency iter{} {:.3f}ms'.format(
                        iteration + 1, inference_time * 1000))
                latency_avg = np.mean(infer_times)
                latency_std = np.std(infer_times)
                logger.info('{}: latency_avg={:.3f} ms, std={:.3f} ms'.format(
                    device, latency_avg, latency_std))
                record = {'model': model_name,
                          'latency': latency_avg}
                for k, v in infer_energy.items():
                    logger.info(
                        '{}: {}_energy_avg={:.3f} J, std={:.3f} J'.format(
                            device, k, np.mean(v), np.std(v)))
                    record['energy_{}'.format(k)] = np.mean(v)
                # results.append(record)
                writer.writerow(record)
                f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str,
                        help="network type")
    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="network type")
    parser.add_argument("-s", "--image_size", type=int, default=224,
                        help="input image size")
    parser.add_argument("-n", "--num_classes", type=int, default=1001,
                        help="number of classification classes")
    parser.add_argument("-l", "--labels_offset", type=int, default=0,
                        help="An offset for the labels in the dataset. This flag is primarily used "
                             "to  evaluate the VGG and ResNet architectures which do not use a"
                             " background class for the ImageNet dataset.")
    parser.add_argument("-cpu", "--cpu", action='store_true',
                        help="run on cpu")
    parser.add_argument("-i", "--iterations", type=int, default=10)

    main(parser.parse_args())
