import csv

from .dataset import Dataset
from .dataset_entry import TranscriptionEntry


class LibriStutter(Dataset):
    def __init__(self, path, transcriptions=False, included_files=None):
        super().__init__(path)
        self.path_audio = self.path + "/LibriStutter Audio"
        self.path_transcriptions = self.path + "/LibriStutter Annotations"
        self._parse_dataset(self.path_audio, included_files)

        if transcriptions:
            self.__parse_transcriptions()

    def __parse_transcriptions(self):
        for entry in self.entries:
            with open(self.path_transcriptions + "/" + entry.speaker + "/" + entry.chapter + "/" + entry.file + ".csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                transcription = []

                for row in csv_reader:
                    transcription += [TranscriptionEntry(row[0].upper(), float(row[1]), float(row[2]), int(row[3]))]

            entry.set_transcription(transcription)

    def print_resume(self):
        print('-------------------')
        print('LIBRISTUTTER DATASET')
        super().print_resume()
        print()
        print(f'Total interjection: {sum([e.interjection_count() for e in self.entries])}')
        print(f'Total sound repetitions: {sum([e.repetition_sound_count() for e in self.entries])}')
        print(f'Total word repetitions: {sum([e.repetition_word_count() for e in self.entries])}')
        print(f'Total phrase repetitions: {sum([e.repetition_phrase_count() for e in self.entries])}')
        print(f'Total prolongations: {sum([e.prolongation_count() for e in self.entries])}')
        print('-------------------')
