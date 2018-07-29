# Reference: https://github.com/XenderLiu/Listen-Attend-and-Spell-Pytorch/blob/master/util/timit_preprocess.sh

if [ $# -ne 2 ]; then
    echo "Usage: timit_preprocess.sh <timit_directory> <output_fileName>"
    exit 1
fi

echo 'Turn NIST into RIFF...'
echo ' '

# Turn NIST wav file into RIFF
find $1 -name '*.wav' | parallel -P20 sox {} '{.}_riff.wav'
# Extract mfcc features
python3 ./src/utils/timit_preprocess.py $1 $2
