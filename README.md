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
  python3 "${src_dir}/run_ckpt_model.py" -c "${src_dir}/bert_config.json" -l ${size} -od data/bert_base -on model_seq${size} 2>&1 | tee data/tf_seq${size}.log

  python3 mo_tf.py --input_meta_graph "data/bert_base/model_seq${size}.meta" \
  --output prob \
  --disable_nhwc_to_nchw \
  --input input_ids{i32},segment_ids{i32},input_mask{f32} \
  --progress --output_dir data/bert_ncs

  for d in ncs1 ncs2; do
    echo "running ${m} on ${d}..."
    python3 "${src_dir}/run_ncs.py" -m data/bert_ncs/model_seq${size}.xml -s ${size} -d ${d} 2>&1 | tee data/${d}_seq${size}.log
  done
  rm -rf "data/bert_base/model_seq*"
  echo `pwd`;
done


# for bert 24 models
for a in 2 4 8 12; do
  h=$((a*64))
  for l in $(seq 2 2 12); do
    for size in 100 ; do
      m=l${l}_h${h}_s${size}
      echo "running m=${m}..."

      python3 "${src_dir}/run_ckpt_model.py" -c data/bert_24/uncased_L-${l}_H-${h}_A-${a}/bert_config.json -cf data/bert_24/uncased_L-${l}_H-${h}_A-${a}/bert_model.ckpt -od data/bert_24 -on model_${m} -l ${size} 2>&1 | tee data/bert_24/tf_${m}.log

      python3 mo_tf.py --input_meta_graph data/bert_24/model_${m}.meta \
      --output prob \
      --disable_nhwc_to_nchw \
      --input input_ids{i32},segment_ids{i32},input_mask{f32} \
      --progress --output_dir data/bert_24_ncs

      python3 "${src_dir}/run_ncs.py" -m data/bert_24_ncs/model_${m}.xml -s ${size} 2>&1 | tee data/bert_24_ncs_${m}.log

    done
  done
done


for a in 2 4 8 12; do
  h=$((a*64))
  for l in $(seq 2 2 12); do
    for size in 100 ; do
      m=l${l}_h${h}_s${size}
      for d in ncs1 ncs2; do
        echo "running ${m} on ${d}..."
        python3 "${src_dir}/run_ncs.py" -m data/bert_24_ncs/model_${m}.xml -s ${size} -d ${d} 2>&1 | tee data/bert_${d}_${m}.log
      done
    done
  done
done
````


train models
````bash
2-512, 4-128
4-512, 8-128
6-512,10-256,12-128
4-768,10-512

# for bert 24 models
for a in 2 4 8 12; do
  h=$((a*64))
  for l in $(seq 2 2 12); do
      m=l${l}_h${h}
      echo "running m=${m}..."
      BERT_BASE_DIR=data/init/uncased_L-${l}_H-${h}_A-${a}

      python run_classifier.py \
        --task_name=SST \
        --do_train=true \
        --do_eval=true \
        --data_dir=data/glue_data/SST-2 \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=32 \
        --learning_rate=2e-5 \
        --num_train_epochs=3.0 \
        --output_dir=data/sst_${m} 2>&1 | tee data/sst-${m}.log
  done
done

````