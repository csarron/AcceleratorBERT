import sys
import csv
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from argparse import ArgumentParser, SUPPRESS

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('--power_measurement_tsv_file', type=str, required=True)
    args.add_argument('--latency_measurement_tsv_file', type=str, required=True)
    return parser.parse_args()

def get_latency_datetime(datetime_str):
    return datetime.strptime(datetime_str, "%d-%m-%Y_%H-%M-%S")

def get_power_datetime(date_str, time_str):
    return datetime.strptime(date_str + ' ' + time_str, "%Y-%m-%d %H:%M:%S")

def get_timestamp():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

def main():
    latency_rows = pd.read_csv(args.latency_measurement_tsv_file, sep='\t', header=0)
    power_rows = pd.read_csv(args.power_measurement_tsv_file, sep='\t', header=0)

    output_file = open('experiments/latency_power_' + get_timestamp() + '.tsv', 'w+')
    output_file.write('L\tH\tA\tS\tLatency(ms)\tPower(W)\tEnergy(J)\tLatency_BeginTimestamp\tLatency_EndTimestamp\tPower_BeginTimestamp\tPower_EndTimestamp\n')

    for latency_index, latency_row in latency_rows.iterrows():
        latency_begin_timestamp = get_latency_datetime(latency_row['BeginTimestamp'])
        latency_end_timestamp = get_latency_datetime(latency_row['EndTimestamp'])

        # print(latency_begin_timestamp)
        power_values = []
        power_begin_timestamp = None
        power_end_timestamp = None

        for power_index, power_row in power_rows.iterrows():
            power_timestamp = get_power_datetime(power_row['Date'], power_row['Time'])

            if power_timestamp >= (latency_begin_timestamp + timedelta(0, 1)) and (power_timestamp <= (latency_end_timestamp - timedelta(0, 1))):
                if power_begin_timestamp is None:
                    power_begin_timestamp = power_timestamp

                power_end_timestamp = power_timestamp

                voltage = float(power_row['Voltage(V)'])
                current = float(power_row['Current(A)'])
                power = voltage * current
                power_values.append(power)
                # print('voltage: {}, current: {}, power: {}, time: {}'.format(voltage, current, power, power_timestamp))

        mean_power = np.mean(power_values)
        energy = float(latency_row['Latency(ms)'])/1000.0 * mean_power

        combined_row = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            latency_row['L'], latency_row['H'], latency_row['A'], latency_row['S'],
            latency_row['Latency(ms)'], mean_power, energy, latency_row['BeginTimestamp'], latency_row['EndTimestamp'], power_begin_timestamp, power_end_timestamp)

        output_file.write(combined_row)

    output_file.close()

if __name__ == '__main__':
    args = build_argparser()
    sys.exit(main() or 0)