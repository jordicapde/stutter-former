import csv

import librosa
import pandas as pd
import speechbrain as sb
import torch
import torchaudio
from pydub import AudioSegment

from classes.librispeech import LibriSpeech
from classes.libristutter import LibriStutter


def read_dataset():
    included_files = None  # ["8975-270782-0113", "78-368-0001"]
    # stutter = LibriStutter("../data/Toy LibriStutter", transcriptions=True, included_files=included_files)
    # speech = LibriSpeech("../data/Toy LibriSpeech", transcriptions=True, included_files=included_files)
    # FEINA
    stutter = LibriStutter("C:/Users/jordi.capdevila/Downloads/_xxx/TFM datasets/LibriStutter/Adaptat", transcriptions=True)
    speech = LibriSpeech("C:/Users/jordi.capdevila/Downloads/_xxx/TFM datasets/LibriSpeech/Adaptat", transcriptions=True)
    # CASA
    # stutter = LibriStutter("C:/Users/Jordi/Desktop/Jordi/2022/TFM/datasets/LibriStutter/Adaptat", transcriptions=True)
    # speech = LibriSpeech("C:/Users/Jordi/Desktop/Jordi/2022/TFM/datasets/LibriSpeech/Adaptat", transcriptions=True)
    return stutter, speech


def print_resume(stutter, speech):
    stutter.print_resume()
    speech.print_resume()

    print()
    print(f'Clean entries:')
    for e in stutter.entries:
        if e.stutter_count() == 0:
            print(e.file)

    print()
    print(f'Unaligned entries:')
    for e_stutter in stutter.entries:
        e_speech = speech.get_entry(e_stutter.file)
        #stutter_set = set([e.utterance for e in e_stutter.transcription])
        #speech_set = set([e.utterance for e in e_speech.transcription])

        stutter_set = set([
            (e_stutter.transcription[i].utterance, e_stutter.transcription[i + 1].utterance)
            for i in range(len(e_stutter.transcription) - 1)
        ])
        speech_set = set([
            (e_speech.transcription[i].utterance, e_speech.transcription[i + 1].utterance)
            for i in range(len(e_speech.transcription) - 1)
        ])
        similar_set = stutter_set.intersection(speech_set)

        if (len(similar_set) / len(stutter_set)) < 0.2:
            print(e_stutter.file)
            print(len(similar_set) / len(stutter_set))
            print(e_stutter.clean_transcription())
            print(e_speech.clean_transcription())


def save_resume(stutter, speech):
    header = [
        'speaker', 'speaker_gender', 'chapter', 'file',
        'audio_length_stutter', 'audio_length_speech', 'audio_length_difference',
        'transcription_length', 'transcription_similarity',
        'stutter_count', 'interjection_count', 'repetition_sound_count',
        'repetition_word_count', 'repetition_phrase_count', 'prolongation_count',
        'file_path_stutter', 'file_path_speech'
    ]
    data = []
    for e_stutter in stutter.entries:
        e_speech = speech.get_entry(e_stutter.file)
        stutter_set = set([
            (e_stutter.transcription[i].utterance, e_stutter.transcription[i + 1].utterance)
            for i in range(len(e_stutter.transcription) - 1)
        ])
        speech_set = set([
            (e_speech.transcription[i].utterance, e_speech.transcription[i + 1].utterance)
            for i in range(len(e_speech.transcription) - 1)
        ])
        similar_set = stutter_set.intersection(speech_set)

        data += [
            [e_stutter.speaker, e_stutter.chapter, e_stutter.file,
             e_stutter.audio_length, e_speech.audio_length, e_stutter.audio_length - e_speech.audio_length,
             e_speech.transcription_length, len(similar_set) / len(speech_set),
             e_stutter.stutter_count(), e_stutter.interjection_count(), e_stutter.repetition_sound_count(),
             e_stutter.repetition_word_count(), e_stutter.repetition_phrase_count(), e_stutter.prolongation_count(),
             e_stutter.file_path, e_speech.file_path]]

    with open('dataset_final.csv', 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(header)
        writer.writerows(data)


def change_resume():
    csv_input = pd.read_csv('dataset_final.csv')
    csv_input['file_path_speech'] = csv_input.apply(
        lambda row: "LibriSpeech/" + str(row["speaker"]) + "/" + str(row["chapter"]) + "/" + row["file"] + ".flac",
        axis=1)
    csv_input.to_csv('dataset_final_2.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)


def first_operations_datasets():
    resume = pd.read_csv("../data/_complete.csv")
    for index, row in resume.iterrows():
        FOLDER = str(row['speaker']) + '/' + str(row['chapter']) + '/'
        FILE = row['file']
        PATH_SPEECH = 'C:/Users/Jordi/Desktop/Jordi/2022/TFM/datasets/LibriSpeech/Final/'
        PATH_STUTTER = 'C:/Users/Jordi/Desktop/Jordi/2022/TFM/datasets/LibriStutter/Final/'

        FILE_SPEECH = PATH_SPEECH + FOLDER + FILE + '.flac'
        FILE_STUTTER = PATH_STUTTER + 'LibriStutter Audio/' + FOLDER + FILE + '.flac'
        FILE_TRANS = PATH_STUTTER + 'LibriStutter Annotations/' + FOLDER + FILE + '.csv'

        speech = AudioSegment.from_file(FILE_SPEECH, 'flac')
        stutter = AudioSegment.from_file(FILE_STUTTER, 'wav')

        stutters = []
        with open(FILE_TRANS) as trans:
            trans_reader = csv.reader(trans, delimiter=',')

            for trans_row in trans_reader:
                if trans_row[3] != '0':
                    begin, end = float(trans_row[1]), float(trans_row[2])
                    begin = int(begin * 1000)
                    end = int(end * 1000)
                    diff = end - begin

                    copy = speech[:begin] + AudioSegment.silent(duration=diff) + speech[begin:]
                    speech = copy

        speech_duration = int(speech.duration_seconds * 1000)
        stutter_duration = int(stutter.duration_seconds * 1000)
        diff = stutter_duration - speech_duration

        if diff > 0:
            copy = speech + AudioSegment.silent(duration=diff + 10)
            speech = copy

        speech = speech[:stutter_duration - 2]
        stutter = stutter[:stutter_duration - 2]

        speech.export(FILE_SPEECH, format="flac")
        stutter.export(FILE_STUTTER, format="flac",
                       parameters=["-ar", "16000", "-sample_fmt", "s16"]
                       )

def cut_same_length():
    # resume = pd.read_csv("../data/similarity67_1000.csv")
    # for index, row in resume.iterrows():
    #     PATH_SPEECH = '../data/' + row['file_path_speech']
    #     PATH_STUTTER = '../data/' + row['file_path_stutter']
    #
    #     speech = AudioSegment.from_file(PATH_SPEECH, 'flac')
    #     stutter = AudioSegment.from_file(PATH_STUTTER, 'flac')
    #
    #     speech_duration = int(speech.duration_seconds * 1000)
    #     stutter_duration = int(stutter.duration_seconds * 1000)
    #     diff = stutter_duration - speech_duration
    #
    #     if diff != 0:
    #         print('++++++++++++++++++++++++++')
    #         print(row['file'])
    #         print('Speech:' + str(speech_duration))
    #         print('Stutter:' + str(stutter_duration))
    #
    #         duration = min(speech_duration, stutter_duration)
    #         speech = speech[:duration]
    #         stutter = stutter[:duration]

    # speech.export(PATH_SPEECH, format="flac")
    # stutter.export(PATH_STUTTER, format="flac")

    resume = pd.read_csv("../data/similarity67_1000.csv")
    for index, row in resume.iterrows():
        PATH_SPEECH = '../data/' + row['file_path_speech']
        PATH_STUTTER = '../data/' + row['file_path_stutter']

        speech = sb.dataio.dataio.read_audio(PATH_SPEECH)
        stutter = sb.dataio.dataio.read_audio(PATH_STUTTER)

        speech_duration = speech.size()[0]
        stutter_duration = stutter.size()[0]
        diff = stutter_duration - speech_duration

        if diff != 0:
            print('++++++++++++++++++++++++++')
            print(row['file'])
            print('Speech:' + str(speech_duration))
            print('Stutter:' + str(stutter_duration))

            duration = min(speech_duration, stutter_duration)
            speech = speech[:duration]
            stutter = stutter[:duration]


if __name__ == '__main__':
    EXPERIMENT = "tiny-stutterformer"
    SEED = "2345"

    data = pd.read_csv("../out/" + EXPERIMENT + "/" + SEED + "/save/test.csv")
    entry = data.sample()
    print("jskfdsl")
