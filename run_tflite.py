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

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(model_path=model_file,
      # To run on host, comment the line below
      experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})]
  )

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=False, default='model.tflite',
       help='File path of .tflite file.')
  parser.add_argument("-l", "--max_seq_length", type=int, default=384)

  args = parser.parse_args()

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_shape = input_details[0]['shape']
  input_type = input_details[0]['dtype']

  print(input_details)
  # Example output: [{'name': 'input_1', 'index': 0, 'shape': array([  1, 384]), 'dtype': <class 'numpy.int32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}}]

  input_content = np.zeros([1, args.max_seq_length])
  input_data = np.array(input_content, dtype=input_type)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data)

if __name__ == '__main__':
  main()