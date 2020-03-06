import argparse
import time
import tflite_runtime.interpreter as tflite #
import platform
import numpy as np
from datetime import datetime

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(model_path=model_file,
      experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})]
  )

def get_timestamp():
  return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

def run_model(hidden_size, num_hidden_layers, num_attention_layers, intermediate_size, experiments_log_file):

  model_dir = 'experiments/hidden_size_' + str(hidden_size) \
    + '_num_hidden_layers_' + str(num_hidden_layers) \
    + '_num_attention_layers_' + str(num_attention_layers) \
    + '_intermediate_size_' + str(intermediate_size)

  tflite_model_path = model_dir + '/model.tflite'

  interpreter = make_interpreter(tflite_model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_shape = input_details[0]['shape']
  input_type = input_details[0]['dtype']

  input_content = np.zeros([1, 384])
  input_data = np.array(input_content, dtype=input_type)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  start = time.perf_counter()
  # for i in range(0, 5000):
  interpreter.invoke()
  inference_time = time.perf_counter() - start
  # print('%.1fms' % (inference_time * 1000))

  experiments_log_file.write(str(hidden_size) + '\t' + str(num_hidden_layers) +'\t' + str(num_attention_layers) +'\t' + str(intermediate_size) + '\t' + str((inference_time * 1000)) + '\n')

def main():
  hidden_size_arr = [256]#, 512, 768]
  num_hidden_layers_arr = [3]#, 4, 6, 8, 9, 12]
  num_attention_layers_arr = [4]#, 6, 8, 12, 16]
  intermediate_size_arr = [3072] # [768, 1024, 1536, 2048, 3072]

  experiments_log_path = 'experiment_' + get_timestamp() + '.edgetpu.log'

  with open(experiments_log_path, 'w') as experiments_log_file:
    experiments_log_file.write('hidden_size' + '\t' + 'num_hidden_layers' +'\t' + 'num_attention_layers' +'\t' + 'intermediate_size' + '\t' + 'inference_time_ms' + '\n')

    for hidden_size in hidden_size_arr:
      for num_hidden_layers in num_hidden_layers_arr:
        for num_attention_layers in num_attention_layers_arr:
          for intermediate_size in intermediate_size_arr:
            try:
              run_model(hidden_size, num_hidden_layers, num_attention_layers, intermediate_size, experiments_log_file)
            except ValueError as ve:
              print(ve)

if __name__ == '__main__':
  main()