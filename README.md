# DeepmRNALoc: A novel predictor of eukaryotic mRNA subcellular localization based on deep learning
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)


## Requirements

- Python == 3.6.13
- opencv-python == 4.5.1.48
- tensorflow-gpu == 1.14.0
- scikit-learn == 0.24.0
- pandas ==  1.1.5
- Keras == 2.3.1
- pandas == 1.1.5
- numpy == 1.19.5
- matplotlib == 3.3.4
- h5py == 2.10.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data
The fasta data can be downloaded here.
- [BaiduPan](https://pan.baidu.com/s/1_sJ97N2te9CJxlbYz8DzfA), password: gv9y 

## Usage
To make our model as user-friendly as possible, a web site called DeepmRNALoc was developed.<br/>
DeepmRNALoc can be publicly accessed by http://97.74.87.26:8080/mRNA/. 

You can also run it from the command line

feature extract:
```
    cd ./DeepmRNALoc
    python extract_feature.py
```
Tips: It might take a long time.

train and test:
```
    python mRNALoc/main.py --model [modelname] --train
```
only test:
```
    python mRNALoc/main.py --model [modelname]
```
Tips: Please check the root path before run the main.py .

inference:
```
    python inference.py
```

For more parameter information, please refer to `main.py`.

## <span id="resultslink">Results</span>
The five-fold cross-validation accuracy of DeepmRNALoc in the cytoplasm, endoplasmic reticulum, extracellular region, mitochondria and nucleus were 0.895, 0.594, 0.308, 0.944 and 0.865, respectively.

## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the Github:<br/>
https://github.com/Thales-research-institute/DeepmRNALoc<br/>

We are waiting for the paper involving DeepmRNALoc to be published.

## Contact
If you have any questions, feel free to contact Thales research institute through Email (Thales_research@163.com) or Github issues. Pull requests are highly welcomed!

## Acknowledgments
Thanks to Thales Institute and Shanghai Ocean University for providing computing infrastructure.<br/>
At the same time, thank you all for your attention to this work!

