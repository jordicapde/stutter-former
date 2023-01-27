import librosa as librosa

from collections import Counter


class Speaker:
    def __init__(self, id_):
        self.id_ = id_


class Chapter:
    def __init__(self, id_, speaker):
        self.id_ = id_
        self.speaker = speaker


class Entry:
    def __init__(self, speaker, chapter, file, path):
        self.speaker = speaker
        self.chapter = chapter
        self.file = file.split(".")[0]
        self.file_path = path + "/" + file

        print(f'File {self.file}')
        self.audio_length = 0
        # y, sr = librosa.load(self.file_path)
        # self.audio_length = librosa.get_duration(y=y, sr=sr)

        self.transcription = []
        self.transcription_length = 0
        self.utterance_counter = None

    def set_transcription(self, transcription):
        self.utterance_counter = Counter([t.utterance_type for t in transcription])
        self.transcription = transcription
        self.transcription_length = self.utterance_counter[TranscriptionEntry.CLEAN]

    def clean_transcription(self):
        return " ".join([t.utterance for t in self.transcription if t.utterance_type == TranscriptionEntry.CLEAN])

    def stutter_count(self):
        return self.interjection_count() + self.repetition_count() + self.prolongation_count()

    def interjection_count(self):
        return self.utterance_counter[TranscriptionEntry.INTERJ] if self.utterance_counter else 0

    def repetition_count(self):
        return self.utterance_counter[TranscriptionEntry.REP_SOUND] + \
               self.utterance_counter[TranscriptionEntry.REP_WORD] + \
               self.utterance_counter[TranscriptionEntry.REP_PHRASE] if self.utterance_counter else 0

    def repetition_sound_count(self):
        return self.utterance_counter[TranscriptionEntry.REP_SOUND] if self.utterance_counter else 0

    def repetition_word_count(self):
        return self.utterance_counter[TranscriptionEntry.REP_WORD] if self.utterance_counter else 0

    def repetition_phrase_count(self):
        return self.utterance_counter[TranscriptionEntry.REP_PHRASE] if self.utterance_counter else 0

    def prolongation_count(self):
        return self.utterance_counter[TranscriptionEntry.PROLONGATION] if self.utterance_counter else 0


class TranscriptionEntry:
    CLEAN = 0
    INTERJ = 1  # interjection
    REP_SOUND = 2  # sound repetition
    REP_WORD = 3  # word repetition
    REP_PHRASE = 4  # phrase repetition
    PROLONGATION = 5  # prolongation

    def __init__(self, utterance, begin=0.0, end=0.0, utterance_type=0):
        self.utterance = utterance
        self.begin = begin
        self.end = end
        self.utterance_type = utterance_type

    def is_clean(self):
        return self.utterance_type == self.CLEAN

    def has_stutter(self):
        return not self.is_clean()

    def has_interjection(self):
        return self.utterance_type == self.INTERJ

    def has_repetition(self):
        return self.utterance_type in [self.REP_SOUND, self.REP_WORD, self.REP_PHRASE]

    def has_prolongation(self):
        return self.utterance_type == self.PROLONGATION
