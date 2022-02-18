#!/usr/bin/env python3

import argparse
import json
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--min', type=float, help='minimum WER')
    parser.add_argument('-x', '--max', type=float, help='minimum WER')
    parser.add_argument('source_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    print(args)

    if args.min is None and args.max is None:
        raise ValueError('Either --min or --max must be specified.')
    if args.min is not None and args.max is not None and args.min > args.max:
        raise ValueError('--min must be less than or equal to --max.')

    wav_table = {}
    with open(os.path.join(args.source_dir, 'wav.scp'), 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    spk_table = {}
    with open(os.path.join(args.source_dir, 'utt2spk'), 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            spk_table[arr[0]] = arr[1]

    wer_table = {}
    with open(os.path.join(args.source_dir, 'utt2wer'), 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wer_table[arr[0]] = float(arr[1])

    lines = []
    with open(os.path.join(args.source_dir, 'text'), 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            txt = arr[1] if len(arr) > 1 else ''

            assert key in wav_table
            assert key in spk_table
            assert key in wer_table
            line = dict(key=key, wav=wav_table[key], spk=spk_table[key], wer=wer_table[key], txt=txt)
            lines.append(line)

    lines.sort(key=lambda x: x['wer'])
    new_lines = []
    for line in lines:
        if (args.min is not None and line['wer'] < args.min) or (args.max is not None and line['wer'] > args.max):
            continue
        new_lines.append(line)

    new_lines.sort(key=lambda x: x['key'])

    if False:
        with open(args.output_file, 'w', encoding='utf8') as fout:
            for line in new_lines:
                del line['wer']
                json_line = json.dumps(line, ensure_ascii=False)
                fout.write(json_line + '\n')
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'wav.scp'), 'w', encoding='utf8') as fout_wav, \
             open(os.path.join(args.output_dir, 'text'), 'w', encoding='utf8') as fout_txt, \
             open(os.path.join(args.output_dir, 'utt2spk'), 'w', encoding='utf8') as fout_spk, \
             open(os.path.join(args.output_dir, 'utt2wer'), 'w', encoding='utf8') as fout_wer:
            for line in new_lines:
                fout_wav.write('{key} {wav}\n'.format(**line))
                fout_txt.write('{key} {txt}\n'.format(**line))
                fout_spk.write('{key} {spk}\n'.format(**line))
                fout_wer.write('{key} {wer}\n'.format(**line))

    print("Filtered {:.0f}% {} utterances into {} utterances, from {} into {}".format(len(new_lines) / len(lines) * 100, len(lines), len(new_lines), args.source_dir, args.output_dir))
