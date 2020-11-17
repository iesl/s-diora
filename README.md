## S-DIORA

This is the official repo for our EMNLP 2020 paper Unsupervised Parsing with S-DIORA: Single Tree Encoding for Deep Inside-Outside Recursive Autoencoders, which presents an improved variant of DIORA that encodes a single tree rather than a softly-weighted mixture of trees by employing a hard argmax operation and a beam at each cell in the chart. If you use this code for research, please cite our paper as follows:

```
@inproceedings{drozdov2020diora,
  title={Unsupervised Parsing with S-DIORA: Single Tree Encoding for Deep Inside-Outside Recursive Autoencoders},
  author={Drozdov, Andrew and Rongali, Subendhu and Chen, Yi-Pei and O{'}Gorman, Tim and Iyyer, Mohit and McCallum, Andrew},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020},
}
```

The paper is available online: https://www.aclweb.org/anthology/2020.emnlp-main.392/

For questions/concerns/bugs please contact adrozdov at cs.umass.edu.

## Setup

```
conda create -n s-diora python=3.6 -y
source activate s-diora
pip install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Data

For NLI data, concatenate the SNLI and MultiNLI datasets and randomly select a number of sentence equal to the size of the PTB training data.

The format of input data is shown in `sample_data/example.jsonl`.

For the pre-trained DIORA model, use the mlp-softmax version available here: https://github.com/iesl/diora

## Experiments

The three main experiments are:

- `S-DIORA_{None}`: Initialize S-DIORA and evaluate immediately.
- `S-DIORA_{NLI}`: Fine-tune S-DIORA on external data.
- `S-DIORA_{PTB}`: Fine-tune S-DIORA on in-domain data.

To run these experiments in this code base use the following commands:

```
# S-DIORA_{None}
# No training required.

# S-DIORA_{NLI}
python main.py --cuda \
    --experiment_name s_diora_ft_nli  \
    --default_experiment_directory ~/tmp/diora/log \
    --batch_size 24 \
    --validation_batch_size 32 \
    --load_model_path ~/checkpoints/mlp-softmax/model.pt \
    --lr 0.001 \
    --train_data_type wsj_emnlp \
    --train_filter_length 30 \
    --train_path ~/data/allnli_downsampled.jsonl \
    --validation_data_type wsj_emnlp \
    --validation_path ~/data/ptb_dev.jsonl \
    --validation_filter_length 40 \
    --elmo_cache_dir ~/data/elmo \
    --emb elmo \
    --eval_after 1 \
    --eval_every_epoch 1 \
    --eval_every_batch -1 \
    --log_every_batch 100 \
    --max_epoch 5 \
    --max_step -1 \
    --opt adam \
    --save_after 0 \
    --model_config '{"topk-mlp": {"normalize": "unit", "K": 3, "outside": true, "size": 400}}' \
    --eval_config '{"unlabeled_binary_parsing": {"name": "eval-k2", "cky_mode": "diora", "K": 2, "enabled": true, "outside": false, "ground_truth": "~/data/ptb-dev-diora.parse"}}' \
    --loss_config '{"greedy_reconstruct_loss": {"reconstruct_loss": "reconstruct_softmax_v2_loss", "train_tree": true}}' \
    --loss_config '{"reconstruct_softmax_v2": {"skip": true, "path": "./resource/nli_top_10k.txt"}}'

# S-DIORA_{PTB}
python main.py --cuda \
    --experiment_name s_diora_ft_ptb  \
    --default_experiment_directory ~/tmp/diora/log \
    --batch_size 24 \
    --validation_batch_size 32 \
    --load_model_path ~/checkpoints/mlp-softmax/model.pt \
    --lr 0.002 \
    --train_data_type wsj_emnlp \
    --train_filter_length 30 \
    --train_path ~/data/ptb_train.jsonl \
    --validation_data_type wsj_emnlp \
    --validation_path ~/data/ptb_dev.jsonl \
    --validation_filter_length 40 \
    --elmo_cache_dir ~/data/elmo \
    --emb elmo \
    --eval_after 1 \
    --eval_every_epoch 1 \
    --eval_every_batch -1 \
    --log_every_batch 100 \
    --max_epoch 5 \
    --max_step -1 \
    --opt adam \
    --save_after 0 \
    --model_config '{"topk-mlp": {"normalize": "unit", "K": 3, "outside": true, "size": 400}}' \
    --eval_config '{"unlabeled_binary_parsing": {"name": "eval-k2", "cky_mode": "diora", "K": 2, "enabled": true, "outside": false, "ground_truth": "~/data/ptb-dev-diora.parse"}}' \
    --loss_config '{"greedy_reconstruct_loss": {"reconstruct_loss": "reconstruct_softmax_v2_loss", "train_tree": true}}' \
    --loss_config '{"reconstruct_softmax_v2": {"skip": true, "path": "./resource/ptb_top_10k.txt"}}'

```

For evaluation, set the `--eval_only_mode` flag and add `"write": true` to the eval_config.

## License

Copyright 2020, University of Massachusetts Amherst

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
