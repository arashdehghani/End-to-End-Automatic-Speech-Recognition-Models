# End-to-End-Automatic-Speech-Recognition
End-to-end ASR system implemented in Tensorflow. Using Timit dataset.

### Bidirectional RNN
```
$ ./utils/timit_preprocess.sh <timit_directory> <path_to_save_mfcc_feature>
$ ./main/train_BiRNN.py <path_to_save_mfcc_feature>
```

#### Accuracy (about 20 epochs)
testing PER: 0.28 

### TODO list:
* DeepSpeech model

