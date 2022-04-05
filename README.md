# DeepmRNALoc: A novel predictor of eukaryotic mRNA subcellular localization based on deep learning
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)


## Requirements

- Python 3.6
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
The fasta data and  can be download here.
- [BaiduPan](https://pan.baidu.com/s/1_sJ97N2te9CJxlbYz8DzfA), password: gv9y 

## Usage
To make our model as user-friendly as possible, a web site called DeepmRNALoc was developed.<br/>
DeepmRNALoc can be publicly accessed by http://DeepmRNALoc.html. 

You can also run it from the command line

```bash
python -u main.py --model DeepmRNALoc --fa test.fasta
```

More parameter information please refer to `main.py`.

## <span id="resultslink">Results</span>
The five-fold cross-validation accuracy of DeepmRNALoc in the cytoplasm, endoplasmic reticulum, extracellular region, mitochondria and nucleus were 0.895, 0.594, 0.308, 0.944 and 0.865, respectively.

## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the Github:<br/>
https://github.com/Thales-research-institute/DeepmRNALoc<br/>

Paper is waiting for being published!

## Contact
If you have any questions, feel free to contact Thales research institute through Email (Thales_research@163.com) or Github issues. Pull requests are highly welcomed!

## Acknowledgments
Thanks for the computing infrastructure provided by Thales research institute and Shanghai Ocean University.<br/>
At the same time, thank you all for your attention to this work!

