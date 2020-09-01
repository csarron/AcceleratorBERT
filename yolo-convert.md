## Convert yolov5 to OpenVINO format (Windows 10)

Install a working PyTorch version

```
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Clone the yolov5 repository

```
git clone https://github.com/ultralytics/yolov5
```

Add the repository to python path

```
set PYTHONPATH=%PYTHONPATH%;c:\Users\alexa\Documents\VQA\yolov5
```

Change the onnx export code https://github.com/ultralytics/yolov5/blob/master/models/export.py#L59 from `opset_version=12` to `opset_version=10`

```
torch.onnx.export(model, img, f, verbose=False, opset_version=10, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])
```

Download a pretrained checkpoint (e.g.: `yolov5x.pt`) from here https://github.com/ultralytics/yolov5/releases/tag/v3.0

Export a yolov5 model to onnx format

```
c:\Users\alexa\Documents\VQA\yolov5 (master -> origin)
(pyenvtorch) λ python models/export.py --weights yolov5x.pt --img 640 --batch 1
```

Activate the OpenVINO environment

```
"c:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
```

Convert the onnx model to OpenVINO IR format

```
c:\Users\alexa\Documents\VQA\yolov5 (master -> origin)
python -m mo_onnx --input_model yolov5x.onnx
```

Run the resulted model on NCS2:

```
c:\Users\alexa\Documents\BERT\AcceleratorBERT (master -> origin)
λ python run_faster_rcnn.py --model C:\Users\alexa\Documents\VQA\yolov5\yolov5x.xml
```

Additional info: https://github.com/ultralytics/yolov5/issues/251

## Convert yolov3 to OpenVINO format (Windows 10)

Clone the tensorflow yolov3 repository:

```
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
```

Download `coco.names` from https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

Download `yolov3.weights` and `yolov3-tiny.weights` from https://drive.google.com/drive/folders/1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0

Export yolov3 frozen tensorflow protobuf model:

```
c:\Users\alexa\Documents\VQA\tensorflow-yolo-v3 (master -> origin)
python convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
```

Export yolov3-tiny frozen tensorflow protobuf model:

```
c:\Users\alexa\Documents\VQA\tensorflow-yolo-v3 (master -> origin)
python convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny
```

Activate the OpenVINO environment

```
"c:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
```

Convert yolov3 tensorflow model to OpenVINO format:

```
c:\Users\alexa\Documents\VQA\tensorflow-yolo-v3 (master -> origin)
λ python -m mo_tf --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\extensions\front\tf\yolo_v3.json" --batch 1
```

Convert yolov3--tiny tensorflow model to OpenVINO format:

```
c:\Users\alexa\Documents\VQA\tensorflow-yolo-v3 (master -> origin)
λ python -m mo_tf --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\extensions\front\tf\yolo_v3_tiny.json" --batch 1
```

Run the resulted model on NCS2:

```
c:\Users\alexa\Documents\BERT\AcceleratorBERT (master -> origin)
λ python run_faster_rcnn.py --model C:\Users\alexa\Documents\VQA\tensorflow-yolo-v3\frozen_darknet_yolov3_model.xml
```

Additional info: https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html