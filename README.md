# poison-regularizer

We hope to find a regularizer to help mitigate posioned attacks on current SOTA language vision models. 

# Setup

Clone the Repository

```
git clone git@github.com:onlypham/poison-regularizer.git
cd poison-regularizer
```

Create Conda Environment

```
source ../conda/bin/activate
conda create --name poison-regularizer python=3.10
conda activate poison-regularizer
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Installing Packages

```
pip install wandb
pip install tensorflow
pip install tqdm
pip install ftfy
pip install regex
pip install pandas
pip install nltk
pip install kaggle
export PYTHONPATH="$PYTHONPATH:$PWD/pkgs/openai"
```

Download [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download)

```
# place in corresponding directories
data/GCC/Train_GCC-training.tsv
data/GCC/Validation_GCC-1.1.0-Validation.tsv
```

Create Cut Version of Conceptual Captions

```
# keep first 10,000 lines
python3 utils/cut.py
```

Download CC Images

```
python3 utils/gather_cc.py data/GCC/Train_GCC-training.tsv
python3 utils/gather_cc.py data/GCC/Validation_GCC-1.1.0-Validation.tsv
# cut version
python3 utils/gather_cc.py data/GCC/Train_GCC-training-cut.tsv
python3 utils/gather_cc.py data/GCC/Validation_GCC-1.1.0-Validation-cut.tsv
```

Run Model

```
python3 src/main.py --name 100K_regularization --model_name RN50 \
  --train_data data/GCC/Train_GCC-training-cut_output.csv \
  --validation_data data/GCC/Validation_GCC-1.1.0-Validation-cut_output.csv \
  --eval_data_type CIFAR10 \
  --eval_test_data_dir data/CIFAR10/test \
  --eval_train_data_dir data/CIFAR10/train \
  --image_key filepath --caption_key title \
  --device gpu --wandb --num_workers 8 \
  --cylambda1 0.25 --cylambda2 0.25 \
  --pretrained --distributed

python3 src/main.py --name 100K_regularization_poison --model_name RN50 \
  --train_data data/GCC/Train_GCC-training-cut_output_poison.csv \
  --validation_data data/GCC/Validation_GCC-1.1.0-Validation-cut_output.csv \
  --eval_data_type CIFAR10 \
  --eval_test_data_dir data/CIFAR10/test \
  --eval_train_data_dir data/CIFAR10/train \
  --image_key filepath --caption_key title \
  --device gpu --wandb --num_workers 8 \
  --cylambda1 0.25 --cylambda2 0.25 \
  --pretrained --distributed

python3 src/main.py --name 100K_no_regularization --model_name RN50 \
  --train_data data/GCC/Train_GCC-training-cut_output.csv \
  --validation_data data/GCC/Validation_GCC-1.1.0-Validation-cut_output.csv \
  --eval_data_type CIFAR10 \
  --eval_test_data_dir data/CIFAR10/test \
  --eval_train_data_dir data/CIFAR10/train \
  --image_key filepath --caption_key title \
  --device gpu --wandb --num_workers 8 \
  --pretrained --distributed
```

This repository contains the official PyTorch implementation of the following paper:

**CyCLIP: Cyclic Contrastive Language-Image Pretraining**<br>
Shashank Goel (UCLA), Hritik Bansal (UCLA), Sumit Bhatia (MDSR Lab, Adobe Systems), Ryan A. Rossi (Adobe Research), Vishwa Vinay (Adobe Research), Aditya Grover (UCLA)<br>
[https://arxiv.org/abs/2205.14459](https://arxiv.org/abs/2205.14459)

Changes: self.root, self.images[idx] -> self.images[idx] (data.py)