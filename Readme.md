## Install
```bash
$ git clone git@github.com:yeyeck/classic_nets.git
$ cd classic_nets
$ pip install -r requirements.txt
```
## Dataset
Since the limit of hardware, the nets are trained on Caltech256. It can be download [here](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) or in [Kaggle](https://www.kaggle.com/caikunye/caltech256) which has been split into transet and testset. The directory as follow:
```bash
|--data_root
    |--train
        |--category1
            --**.jpg
        |--category2
            --**.jpg
        |...
    |--val
        |--category1
            --**.jpg
        |--category2
            --**.jpg
        |...
``` 
## Traning
```bash
python train.py --data /caltech256 --hyper ./config/hyper.yaml --class-num 256 --batch-size 256 --name exp
```
