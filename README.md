# Usage

````python
python export_model.py -c bert_config.json -o data/bert_base -l 384

python3 mo_tf.py --input_meta_graph data/bert_base/model.meta \
--output prob \
--disable_nhwc_to_nchw \
--input input_ids{i32},segment_ids{i32}

python run_ncs.py -m model.xml

python run_tflite.py -m data/bert_base/model.tflite

````
