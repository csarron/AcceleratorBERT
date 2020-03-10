# Usage

````bash
python export_model.py -c bert_config.json -o data/bert_base -l 384

python3 mo_tf.py --input_meta_graph data/bert_base/model.meta \
--output prob \
--disable_nhwc_to_nchw \
--input input_ids{i32},segment_ids{i32}

python run_ncs.py -m model.xml

python run_tflite.py -m data/bert_base/model.tflite

# export bert_base for different input size
cd /opt/intel/openvino/deployment_tools/model_optimizer/
src_dir="this source code path"
for size in `seq 510 -20 10` ; do
  echo "running ${size}..."
  python3 "${src_dir}/run_ckpt_model.py" -c "${src_dir}/bert_config.json" -l ${size} 2>&1 | tee "${src_dir}/data/bert_base/tf_seq${size}.log"

  python3 mo_tf.py --input_meta_graph "data/bert_base/model_seq${size}.meta" \
  --output prob \
  --disable_nhwc_to_nchw \
  --input input_ids{i32},segment_ids{i32},input_mask{f32} \
  --progress --output_dir data/bert_base_ncs

  python3 "${src_dir}/run_ncs.py" -m data/bert_base_ncs/model_seq${size}.xml -s ${size} 2>&1 | tee "data/bert_base_ncs_seq${size}.log"
  rm -rf "data/bert_base/model_seq*"
  echo `pwd`;
done

# only run the converted models
# -d MYRIAD.26.2-ma2450 for ncs1
# -d MYRIAD.26.3-ma2480 for ncs2
for size in `seq 510 -20 10` ; do
  echo "running ${size}..."
  python3 "${src_dir}/run_ncs.py" -m data/bert_base_ncs/model_seq${size}.xml -s ${size} 2>&1 | tee "data/bert_base_ncs_seq${size}.log"
  echo `pwd`;
done

````
