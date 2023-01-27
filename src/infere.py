#!/usr/bin/env/python3
"""
Script for performing inference with the StutterFormer model.

To run this script, do the following:
> python infere.py --path={path_of_the_entry_file} --experiment={experiment_name} --seed={experiment_seed}

Author
 * Jordi Capdevila Mas
"""

import argparse

import torchaudio
from speechbrain.pretrained import SepformerSeparation as separator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str,
                        help='Path of the audio file')
    parser.add_argument('experiment', type=str,
                        help='Name of the experiment of the model')
    parser.add_argument('seed', type=int,
                        help='Seed of the experiment of the model')

    args = parser.parse_args()

    waveform, sr = torchaudio.load(args.path)
    model = separator.from_hparams(
        source="../out/" + args.experiment + "/" + str(args.seed) + "/save/best",
        savedir="../pretrained_models/tiny-stutterformer"
    )

    sources = model.separate_file(path=args.path)
    torchaudio.save("enhanced.flac", sources[:, :, 0].detach().cpu(), 16000)
