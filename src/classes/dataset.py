import os

from .dataset_entry import Speaker, Chapter, Entry


class Dataset:
    def __init__(self, path):
        self.path = path
        self.speakers = []
        self.chapters = []
        self.entries = []

    def _parse_dataset(self, path, included_files=None):
        if included_files:
            included_speakers = [f.split('-')[0] for f in included_files]
            included_chapters = [f.split('-')[1] for f in included_files]
            included_files = [f + '.flac' for f in included_files]

        speakers = sorted(os.listdir(path))
        if included_files:
            speakers = list(filter(lambda s: s in included_speakers, speakers))

        self.speakers = [Speaker(speaker) for speaker in speakers]

        for speaker in speakers:
            chapters = sorted(os.listdir(path + "/" + speaker))
            if included_files:
                chapters = list(filter(lambda c: c in included_chapters, chapters))

            self.chapters += [Chapter(chapter, speaker) for chapter in chapters]

            for chapter in chapters:
                files = sorted(os.listdir(path + "/" + speaker + "/" + chapter))
                files = list(filter(lambda f: f.endswith(".flac"), files))  # Filter only audio files
                if included_files:
                    files = list(filter(lambda f: f in included_files, files))

                self.entries += [Entry(speaker, chapter, file, path + "/" + speaker + "/" + chapter) for file in files]

    def get_entry(self, file):
        return next(filter(lambda e: e.file == file, self.entries), None)

    def print_resume(self):
        print(f'Speakers: {len(self.speakers)}')
        print(f'Chapters: {len(self.chapters)}')
        print(f'Entries: {len(self.entries)}')
        print()
        print(f'Total words: {sum([e.transcription_length for e in self.entries])}')
        print(f'Total duration: {sum([e.audio_length for e in self.entries])} sec')
