# End-to-End-Automatic-Speech-Recognition
End-to-end ASR system implemented in Tensorflow. (Now only support TIMIT dataset)

## Install and Usage
#### Install dependency
```
$ pip3 install -r requirements.txt
```

#### Preprocess TIMIT dataset
```bash
$ ./utils/timit_preprocess.sh <timit_directory> <path_to_save_mfcc_feature>
```

#### Train bidirectional RNN
```bash
$ ./main/train_BiRNN.py <mfcc_path_you_just_saved>
```

#### Result (about 20 epochs)  
testing PER: 0.28 

## TODO list:
* DeepSpeech model

