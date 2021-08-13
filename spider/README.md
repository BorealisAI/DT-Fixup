# Spider

The DT-Fixup initialization part is located in `semparser/nn/optimizers.py`

To replicate our results:

### Replicate the Environment

```
conda env create -f env.yml -n dtfixup
conda activate dtfixup
```

### Prepare the Dataset

```
python -m spacy download en_core_web_sm
python prepare_data.py
```

Download pretrained roberta-large from HuggingFace to the folder `./roberta-large`.

### Training

```
python -m semparser.run --config_path config.yml --commit 0 --do_preprocess --do_training
```

### Inference on the Dev Set

```
python -m semparser.run --config_path tmp/dtfixup/config.yml --commit 0 --do_evaluation \
  EXP_RUNNER/evaluator/for_inference:True PREPROCESSOR/dev_data_path:spider/dev.bin
```

