# End-to-End-Automatic-Speech-Recognition
End-to-end ASR system implemented in Tensorflow. (Now only support TIMIT dataset)

### Install dependency
```
$ pip3 install -r requirements.txt
```

### Preprocess timit dataset
Before runing the script, install *parallel* and *sox* first.  
```bash
$ ./src/utils/timit_preprocess.sh <timit_directory> <path_to_save_mfcc_feature>
```

### Train models

* Bidirectional RNN
```bash
$ ./src/main/train_BiRNN.py <mfcc_path_you_just_saved>
``` 

* DeepSpeech2
```bash
$ ./src/main/train_deepspeech2.py <mfcc_path_you_just_saved>
```

### Result  
* Bidirectional RNN  
Testing PER: 0.28 (about 20 epochs, can be better if you train the model with more epochs)  

* DeepSpeech2 (with small revise)  
Testing PER: 0.24 (about 25 epochs)

