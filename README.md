# StutterFormer: eliminating stuttering disfluences with AI
The interaction between humans and machines has experienced a change in recent years thanks to the possibility of using voice as a way of communication. But current speech recognition models are not as reliable when it comes to recognize non-normative speech, such as stuttering speech.

This Master's Thesis proposes the StutterFormer architecture, an artificial intelligence model that aims to be able to receive a speech sample with stuttering disfluencies, and return it with the disfluencies attenuated or eliminated. In this way, it would be applicable as a process prior to speech recognition, for example, or simply to improve the fluency and intelligibility of a speech sample.

The model has been trained in a supervised way as a speech enhancement task, with the LibriStutter and LibriSpeech datasets, which offer the same samples with and without disfluencies. Even so, the model has not been able to eliminate these disfluencies as expected.

## Resources
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

The model is based on RE-SepFormer (https://arxiv.org/abs/2206.09507):
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
