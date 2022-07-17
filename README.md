# DeepmRNALoc: A novel predictor of eukaryotic mRNA subcellular localization based on deep learning
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)


## Requirements

- Python == 3.6.13
- opencv-python == 4.5.1.48
- tensorflow-gpu == 1.14.0
- scikit-learn == 0.24.0
- pandas ==  1.1.5
- pandas == 1.1.5
- numpy == 1.19.5
- matplotlib == 3.3.4
- h5py == 2.10.0

Dependencies can be installed using the following command:
```bash
conda create -n DeepmRNALoc python=3.6.13
conda activate DeepmRNALoc

pip install -r requirements.txt
```
- CUDA == 10.0 (This is just a suggestion to make sure your program works properly)
- how to install CUDA and cuDNN:
```
conda install cudatoolkit=10.0   
conda install cudnn=7.6.5
```

## Usage For AI researchers

You can also run it from the command line

feature extract:
```
    cd ./DeepmRNALoc
    python extract_feature.py
```
Tips: It might take a long time.

train and test:
```
    python main.py --model [modelname, default = FCN] --train
```
only test:
```
    python main.py --model [modelname, default = FCN]
```
Tips: Please check the root path before run the main.py .

inference:
```
    python inference.py
```

For more parameter information, please refer to `main.py`.

## Usage For biomedical researchers

We deployed a trained model on a dedicated server, which is publicly available at http://www.peng-lab.org:8080/mRNA/, to make it easy for biomedical researcher users to utilize DeepmRNALoc in their research activity. 

Users can upload their mRNA sequences to the server, and then they can quickly obtain the predicted results of the mRNA subcellular localization.

## <span id="resultslink">Results</span>
DeepmRNALoc's five-fold cross-validation accuracies were 0.895, 0.594, 0.308, 0.944, and 0.865 in the cytoplasm, endoplasmic reticulum, extracellular region, mitochondria, and nucleus, respectively.

## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the Github:<br/>
https://github.com/Thales-research-institute/DeepmRNALoc<br/>

Papers involving DeepmRNALocd have been submitted to an academic journal.

## Contact
If you have any questions, please feel free to contact Shihang Wang (Email: wshshou@163.com) or Zhehan Shen (Email: zshen1224@163.com). 
Pull requests are highly welcomed!

## Acknowledgments
Thanks to Thales Institute and Shanghai Ocean University for providing computing infrastructure.<br/>
Thank you all for your attention to this work.

