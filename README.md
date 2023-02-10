# StutterFormer: eliminating stuttering disfluences with AI
The interaction between humans and machines has experienced a change in recent years thanks to the possibility of using voice as a way of communication. But current speech recognition models are not as reliable when it comes to recognize non-normative speech, such as stuttering speech.

This Master's Thesis proposes the StutterFormer architecture, an artificial intelligence model that aims to be able to receive a speech sample with stuttering disfluencies, and return it with the disfluencies attenuated or eliminated. In this way, it would be applicable as a process prior to speech recognition, for example, or simply to improve the fluency and intelligibility of a speech sample.

The model has been trained in a supervised way as a speech enhancement task, with the LibriStutter and LibriSpeech datasets, which offer the same samples with and without disfluencies. Even so, the model has not been able to eliminate these disfluencies as expected.

## Contribution
The goal of the model is to learn how to apply the necessary changes to the voice samples in the places where disfluencies occur. Specifically, it must learn to approximate these moments of disfluency towards moments of silence. The evaluation of the model is done by comparing the estimated output with the expected baseline.

### Datasets
LibriStutter is the input data source for the model. LibriStutter is a set of speech samples with disfluencies, with the particularity that these have been generated synthetically and also labeled automatically.

*LibriStutter (https://arxiv.org/abs/2009.11394v1)*
```bibtex
@misc{Kourkounakis2020,
      title={FluentNet: End-to-End Detection of Speech Disfluency with Deep Learning}, 
      author={Tedd Kourkounakis and Amirhossein Hajavi and Ali Etemad},
      year={2020},
      doi={10.48550/arxiv.2009.11394}
}
```
On the other hand, LibriSpeech is the output data source of the model. LibriSpeech is an open set of English speech samples. All the samples correspond to book readings and have a total volume of 1000 hours.

*LibriSpeech (https://doi.org/10.1109/ICASSP.2015.7178964)*
```bibtex
@misc{Panayotov2015,
      title={Librispeech: An ASR corpus based on public domain audio books}, 
      author={Vassil Panayotov and Guoguo Chen and Daniel Povey and Sanjeev Khudanpur},
      year={2015},
      doi={10.1109/ICASSP.2015.7178964}
}
```
The existence of these two datasets gives the option of having the same speech samples with and without disfluencies, this was not offered in any other way. This offers the possibility to implement the task with a supervised-training approach.

### Model
The proposed model is based on the Sep-Former architecture, a state-of-the-art model in terms of separation and speech enhancement. More specifically, it has been based on the work of the same authors in which they implement RE-SepFormer, an alternative configuration of the model that requires less complexity and reduces the necessary computational resources.

*RE-SepFormer (https://arxiv.org/abs/2206.09507)*
```bibtex
@misc{subakan2022resourceefficient,
      title={Resource-Efficient Separation Transformer}, 
      author={Cem Subakan and Mirco Ravanelli and Samuele Cornell and Frédéric Lepoutre and François Grondin},
      year={2022},
      eprint={2206.09507},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

### Implementation
The implementation is based on the SpeechBrain framework (https://speechbrain.github.io/):
```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

## Results
You can see the results and listen to the outputs of the model in the following Colab file:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jordicapde/stutter-former/blob/master/notebooks/5_results.ipynb)


In summary:
* The model is not changing only the intervals where disfluencies occur, but the entire signal.
* In the waveform representation, it can be seen that the model amplifies or reduces the amplitude of the signal in all its points.
* In the spectrograms, it can be seen how the output given by the model can be similar to the original spectrogram distribution, but all the frequency spectrum has been corrupted with noise.
* Listening to the model estimations, the given output corresponds to the audio input, but it has been corrupted with noise that distorts the voice. On some occasions, the volume of the voice is also lower than the original. As for the disfluencies, they remain intact in the same way as in the input sample.