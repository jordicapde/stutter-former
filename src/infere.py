#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on WHAM! and WHAMR!
datasets. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-wham.yaml --data_folder /your_path/wham_original
> python train.py hparams/sepformer-whamr.yaml --data_folder /your_path/whamr

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures.

Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import argparse

import torchaudio
from speechbrain.pretrained import SepformerSeparation as separator

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    #
    # # Required positional argument
    # parser.add_argument('path', type=str,
    #                     help='Path of the audio file')
    # parser.add_argument('experiment', type=str,
    #                     help='Name of the experiment of the model')
    # parser.add_argument('seed', type=int,
    #                     help='Seed of the experiment of the model')
    #
    # args = parser.parse_args()
    args = {
        'experiment': 'tiny-stutterformer',
        'seed': '2345',
        'path': '../data/LibriStutter/LibriStutter Audio/289/121665/289-121665-0033.flac',
    }

    # waveform, sr = torchaudio.load(args.path)
    model = separator.from_hparams(
        # source="../out/" + args.experiment + "/" + str(args.seed) + "/save/best",
        source="../out/" + args['experiment'] + "/" + str(args['seed']) + "/save/best",
        savedir="../pretrained_models/tiny-stutterformer"
    )

    # sources = model.separate_file(path=args.path)
    sources = model.separate_file(path=args['path'])
    torchaudio.save("enhanced.flac", sources[:, :, 0].detach().cpu(), 16000)
