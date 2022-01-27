#!/usr/bin/env python3

import argparse
import json
import os
import shutil

from pydub import AudioSegment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--min_dur', default=5.0, type=float, help='minimum duration')
    parser.add_argument('source_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

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

    dur_table = {}
    with open(os.path.join(args.source_dir, 'utt2dur'), 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            dur_table[arr[0]] = float(arr[1])

    lines = []
    with open(os.path.join(args.source_dir, 'text'), 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            txt = arr[1] if len(arr) > 1 else ''

            assert key in wav_table
            assert key in spk_table
            assert key in dur_table
            line = dict(key=key, wav=wav_table[key], spk=spk_table[key], dur=dur_table[key], txt=txt)
            lines.append(line)

    lines.sort(key=lambda x: x['dur'])
    new_lines = []
    short_lines = []
    for line in lines:
        if line['dur'] < args.min_dur:
            short_lines.append(line)
            if sum(x['dur'] for x in short_lines) > args.min_dur:
                segment = AudioSegment.from_file(short_lines[0]['wav'])
                for f in short_lines[1:]:
                    segment = segment.append(AudioSegment.silent(duration=500, frame_rate=segment.frame_rate), crossfade=0)
                    segment = segment.append(AudioSegment.from_file(f['wav']), crossfade=0)
                new_wav_name = line['wav'][:-4] + '_combined.wav'
                segment.export(new_wav_name, format='wav')
                line = dict(
                    key=line['key']+'-combined',
                    wav=new_wav_name,
                    txt=' '.join(x['txt'] for x in short_lines),
                    dur=segment.duration_seconds,
                    spk='--'.join(x['spk'] for x in short_lines),
                )
                short_lines = []
            else:
                continue
        new_lines.append(line)

    new_lines.sort(key=lambda x: x['key'])

    if False:
        with open(args.output_file, 'w', encoding='utf8') as fout:
            for line in new_lines:
                del line['dur']
                json_line = json.dumps(line, ensure_ascii=False)
                fout.write(json_line + '\n')
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'wav.scp'), 'w', encoding='utf8') as fout_wav, \
             open(os.path.join(args.output_dir, 'text'), 'w', encoding='utf8') as fout_txt, \
             open(os.path.join(args.output_dir, 'utt2spk'), 'w', encoding='utf8') as fout_spk, \
             open(os.path.join(args.output_dir, 'utt2dur'), 'w', encoding='utf8') as fout_dur:
            for line in new_lines:
                fout_wav.write('{key} {wav}\n'.format(**line))
                fout_txt.write('{key} {txt}\n'.format(**line))
                fout_spk.write('{key} {spk}\n'.format(**line))
                fout_dur.write('{key} {dur}\n'.format(**line))

    print("Combined {} utterances into {} utterances".format(len(lines), len(new_lines)))
