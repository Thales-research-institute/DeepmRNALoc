# DeepmRNALoc: A novel predictor of eukaryotic mRNA subcellular localization based on deep learning
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)



## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data
The fasta data and  can be download here.
- [BaiduPan](https://pan.baidu.com/s/1_sJ97N2te9CJxlbYz8DzfA), password: gv9y 


## Reproducibility

To easily reproduce the results you can follow the next steps:
1. Initialize the docker image using: `make init`.
2. Download the datasets using: `make dataset`.
3. Run each script in `scripts/` using `make run_module module="bash ETTh1.sh"` for each script.
4. Alternatively, run all the scripts at once:
```
for file in `ls scripts`; do make run_module module="bash scripts/$script"; done
```

## Usage
Commands for training and testing the model with *ProbSparse* self-attention on Dataset ETTh1, ETTh2 and ETTm1 respectively:

```bash
# ETTh1
python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h

# ETTh2
python -u main_informer.py --model informer --data ETTh2 --attn prob --freq h

# ETTm1
python -u main_informer.py --model informer --data ETTm1 --attn prob --freq t
```

More parameter information please refer to `main_informer.py`.

We provide a more detailed and complete command description for training and testing the model:

```python
python -u main_informer.py --model <model> --data <data>
--root_path <root_path> --data_path <data_path> --features <features>
--target <target> --freq <freq> --checkpoints <checkpoints>
--seq_len <seq_len> --label_len <label_len> --pred_len <pred_len>
--enc_in <enc_in> --dec_in <dec_in> --c_out <c_out> --d_model <d_model>
--n_heads <n_heads> --e_layers <e_layers> --d_layers <d_layers>
--s_layers <s_layers> --d_ff <d_ff> --factor <factor> --padding <padding>
--distil --dropout <dropout> --attn <attn> --embed <embed> --activation <activation>
--output_attention --do_predict --mix --cols <cols> --itr <itr>
--num_workers <num_workers> --train_epochs <train_epochs>
--batch_size <batch_size> --patience <patience> --des <des>
--learning_rate <learning_rate> --loss <loss> --lradj <lradj>
--use_amp --inverse --use_gpu <use_gpu> --gpu <gpu> --use_multi_gpu --devices <devices>
```

The detailed descriptions about the arguments are as following:

| Parameter name | Description of parameter |
| --- | --- |
| model | The model of experiment. This can be set to `FCN`, '' |
| data           | The dataset name                                             |
| checkpoints      | The path of model checkpoints    |
| train      | whether the model will be trained                  |
| train_epochs     | Train epochs (defaults to 8)                 |
| batch_size      | The batch size of training input data (defaults to 4)                  |
| patience      | Early stopping patience                            |
| learning_rate      | Optimizer learning rate (defaults to 0.001)                  |
| loss      | Loss function (defaults to sparse_categorical_crossentropy)                  |


## <span id="resultslink">Results</span>
The five-fold cross-validation accuracy of DeepmRNALoc in the cytoplasm, endoplasmic reticulum, extracellular region, mitochondria and nucleus were 0.895, 0.594, 0.308, 0.944 and 0.865, respectively.

## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the following paper:<br/>
Paper is waiting for being published!

## Contact
If you have any questions, feel free to contact Thales research institute through Email (Thales_research@163.com) or Github issues. Pull requests are highly welcomed!

## Acknowledgments
Thanks for the computing infrastructure provided by Thales research institute.
At the same time, thank you all for your attention to this work!

