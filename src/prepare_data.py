"""
The .csv preparation functions.

Author
 * Jordi Capdevila Mas
"""

import csv

import numpy as np
import pandas as pd
import os


def prepare_libristutter_librispeech_csv(
    datapath, datafile, savepath, skip_prep=False, set_types=["train", "valid", "test"]
):
    """
    Prepares the csv files for the experiment

    Arguments:
    ----------
        datapath (str) : path for the LibriStutter and LibriSpeech datasets.
        datafile (str) : filename of the .csv resume file
        savepath (str) : path where we save the csv file.
        skip_prep (bool): If True, skip data preparation
    """

    if skip_prep:
        return

    csv_resume = pd.read_csv(datafile)
    csv_resume['file_path_stutter'] = csv_resume.apply(
        lambda row: os.path.join(datapath, row["file_path_stutter"]),
        axis=1)
    csv_resume['file_path_speech'] = csv_resume.apply(
        lambda row: os.path.join(datapath, row["file_path_speech"]),
        axis=1)

    input_split = np.split(csv_resume, [int(.7 * len(csv_resume)), int(.85 * len(csv_resume))])

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
