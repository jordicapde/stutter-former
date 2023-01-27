#!/usr/bin/env/python3
"""
Script for training the StutterFormer model on LibriStutter and LibriSpeech datasets.

To run this script, do the following:
> python train.py hparams/{hyperparams_file}.yaml

Author
 * Jordi Capdevila Mas
"""

import logging
import sys

from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

import speechbrain as sb
from stutterformer import StutterFormer


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Provide audio pipelines
    @sb.utils.data_pipeline.takes("file_path_stutter")
    @sb.utils.data_pipeline.provides("stutter_sig")
    def audio_pipeline_stutter(file_path_stutter):
        stutter_sig = sb.dataio.dataio.read_audio(file_path_stutter)
        return stutter_sig

    @sb.utils.data_pipeline.takes("file_path_speech")
    @sb.utils.data_pipeline.provides("speech_sig")
    def audio_pipeline_speech(file_path_speech):
        speech_sig = sb.dataio.dataio.read_audio(file_path_speech)
        return speech_sig

    # 2. Define datasets
    datasets = {}
    data_info = {
        "train": hparams["train_data"],
        "valid": hparams["valid_data"],
        "test": hparams["test_data"]
    }

    for dataset, dataset_path in data_info.items():
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=dataset_path,
            dynamic_items=[audio_pipeline_stutter, audio_pipeline_speech],
            replacements={"data_root": hparams["data_folder"]},
            output_keys=["id", "stutter_sig", "speech_sig"]
        )

    return datasets


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    args = sys.argv[1:]
    hparams_file, run_opts, overrides = sb.parse_arguments(args)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation
    from prepare_data import prepare_libristutter_librispeech_csv
    run_on_main(
        prepare_libristutter_librispeech_csv,
        kwargs={
            "datapath": hparams["data_folder"],
            "datafile": hparams["data_file"],
            "savepath": hparams["save_folder"],
            "skip_prep": hparams["skip_prep"]
        },
    )

    # Create dataset objects
    datasets = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = StutterFormer(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        # Training
        separator.fit(
            epoch_counter=separator.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts_valid"],
        )

    # Evaluation
    separator.evaluate(
        test_set=datasets["test"],
        max_key="pesq",
        test_loader_kwargs=hparams["dataloader_opts"]
    )
    separator.save_results(datasets["test"])
