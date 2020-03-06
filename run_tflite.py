import argparse
import time
import tflite_runtime.interpreter as tflite
import platform
import numpy as np

# see https://www.tensorflow.org/lite/guide/python

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file, run_on_tpu=False):
  model_file, *device = model_file.split('@')
  delegates = [tflite.load_delegate(EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})] if run_on_tpu else None
  return tflite.Interpreter(model_path=model_file,
                            experimental_delegates=delegates)

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=False, default='model.tflite',
       help='File path of .tflite file.')
  parser.add_argument("-l", "--max_seq_length", type=int, default=384)
  parser.add_argument('-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  parser.add_argument("-tpu", "--tpu", action='store_true', help="run on tpu")
  args = parser.parse_args()

  interpreter = make_interpreter(args.model, args.tpu)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_shape = input_details[0]['shape']
  input_type = input_details[0]['dtype']

  print(input_details)
  # Example output: [{'name': 'input_1', 'index': 0, 'shape': array([  1, 384]), 'dtype': <class 'numpy.int32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}}]
  print('running on {}'.format('edge tpu' if args.tpu else 'host machine'))
  input_content = np.zeros([1, args.max_seq_length])
  input_data = np.array(input_content, dtype=input_type)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  for _ in range(args.count):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print('%.1fms' % (inference_time * 1000))

  print(output_data)

if __name__ == '__main__':
  main()