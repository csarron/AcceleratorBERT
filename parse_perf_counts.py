import sys
import csv
import os
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from argparse import ArgumentParser, SUPPRESS

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('--experiment_dir', type=str, required=True)
    return parser.parse_args()

def get_model_list():
    model_list = []

    for max_seq_length in [128]:
        for num_hidden_layers in [2, 4, 6, 8, 10, 12]:
            for num_attention_heads in [2, 4, 8, 12]:
                hidden_size = num_attention_heads * 64
                model_list.append((num_hidden_layers, hidden_size, num_attention_heads, max_seq_length))

    return model_list

def breakdown_by_layer(L, H, A, S, layer_idx, rows):
    output_trans = 0
    attn = 0
    input_trans = 0
    ff1 = 0
    ff2 = 0

    for index, row in rows.iterrows():
        curr_L = row['L']
        curr_H = row['H']
        curr_A = row['A']
        curr_S = row['S']
        curr_layer_name = row['LayerName']
        curr_real_time_us = row['RealTime(us)']
        layer_root_name = 'bert/encoder/layer_' + str(layer_idx)

        if L == curr_L and H == curr_H and A == curr_A and S == curr_S:
            if curr_layer_name.startswith(layer_root_name + '/attention/output'):
                output_trans += curr_real_time_us

            if curr_layer_name.startswith(layer_root_name + '/attention/self/key') \
                or curr_layer_name.startswith(layer_root_name + '/attention/self/query') \
                or curr_layer_name.startswith(layer_root_name + '/attention/self/sub') \
                or curr_layer_name.startswith(layer_root_name + '/attention/self/transpose') \
                or curr_layer_name.startswith(layer_root_name + '/attention/self/value'):
                input_trans += curr_real_time_us
            elif curr_layer_name.startswith(layer_root_name + '/attention/self'):
                attn += curr_real_time_us

            if curr_layer_name.startswith(layer_root_name + '/intermediate'):
                ff1 += curr_real_time_us

            if curr_layer_name.startswith(layer_root_name + '/output'):
                ff2 += curr_real_time_us

    total_time_us = output_trans + attn + input_trans + ff1 + ff2

    return total_time_us, output_trans, attn, input_trans, ff1, ff2


def main():
    rows = pd.read_csv(os.path.join(args.experiment_dir, 'perf_counts.tsv'), sep='\t', header=0)

    output_file = open(os.path.join(args.experiment_dir, 'breakdown.tsv'), 'w+')
    output_file.write('L\tH\tA\tS\tlayer_idx\ttotal_time_us\toutput_trans\toutput_trans_%\tattn\tattn_%\tinput_trans\tinput_trans_%\tff1\tff1_%\tff2\tff2_%\n')
    model_list = get_model_list()

    for model_info in model_list:
        L = model_info[0]
        H = model_info[1]
        A = model_info[2]
        S = model_info[3]

        for layer_idx in range(L):
            total_time_us, output_trans, attn, input_trans, ff1, ff2 = breakdown_by_layer(L, H, A, S, layer_idx, rows)

            combined_row = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(L, H, A, S, layer_idx, total_time_us, output_trans, output_trans/total_time_us, attn, attn/total_time_us, input_trans, input_trans/total_time_us, ff1, ff1/total_time_us, ff2, ff2/total_time_us)
            output_file.write(combined_row)

if __name__ == '__main__':
    args = build_argparser()
    sys.exit(main() or 0)