"""
Author
 * Cem Subakan 2020

The .csv preperation functions for WSJ0-Mix.
"""

import csv

import numpy as np
import pandas as pd
import os


def prepare_librispeech_csv(
    datapath, datafile, savepath, skip_prep=False
):
    """
    Prepares the csv files for wham or whamr dataset

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        skip_prep (bool): If True, skip data preparation
    """

    if skip_prep:
        return

    create_librispeech_csv(
        datapath,
        datafile,
        savepath
    )


def create_librispeech_csv(
    datapath,
    datafile,
    savepath,
    set_types=["train", "valid", "test"]
):
    """
    This function creates the csv files to get the speechbrain data loaders for the whamr dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
        fs (int) : the sampling rate
        version (str) : min or max
        savename (str) : the prefix to use for the .csv files
        set_types (list) : the sets to create
    """

    csv_input = pd.read_csv(datafile)
    csv_input['file_path_stutter'] = csv_input.apply(
        lambda row: os.path.join(datapath, row["file_path_stutter"]),
        axis=1)
    csv_input['file_path_speech'] = csv_input.apply(
        lambda row: os.path.join(datapath, row["file_path_speech"]),
        axis=1)

    input_split = np.split(csv_input, [int(.7 * len(csv_input)), int(.85 * len(csv_input))])

    for (set_type, split) in zip(set_types, input_split):
        csv_columns = [
            "ID",
            "file_path_stutter",
            "file_path_speech"
        ]

        with open(
            os.path.join(savepath, set_type + ".csv"), "w", newline=""
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for index, row in split.iterrows():

                row = {
                    "ID": index,
                    "file_path_stutter": row["file_path_stutter"],
                    "file_path_speech": row["file_path_speech"]
                }
                writer.writerow(row)
