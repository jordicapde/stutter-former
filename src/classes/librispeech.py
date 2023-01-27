from .dataset import Dataset
from .dataset_entry import TranscriptionEntry


class LibriSpeech(Dataset):
    def __init__(self, path, transcriptions=False, included_files=None):
        super().__init__(path)
        self._parse_dataset(self.path, included_files)

        if transcriptions:
            self.__parse_transcriptions(included_files)

    def __parse_transcriptions(self, included_files=None):
        for chapter in self.chapters:
            with open(self.path + "/" + chapter.speaker + "/" + chapter.id_ + "/" + chapter.speaker + "-" + chapter.id_ + ".trans.txt") as file:
                for line in file.readlines():
                    file = line.split()[0]
                    if not included_files or file in included_files:
                        transcription = []

                        for word in line.split()[1:]:
                            transcription += [TranscriptionEntry(word.upper())]

                        self.get_entry(file).set_transcription(transcription)

    def print_resume(self):
        print('-------------------')
        print('LIBRISPEECH DATASET')
        super().print_resume()
        print('-------------------')
