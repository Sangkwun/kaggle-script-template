# ML pipline

## How to run

```
$ python -m molecule.main split_fold --n_fold 5

$ python -m molecule.main train --model_configs lgb_base.json

```

## Config system



## Docker 

```
# build docker file
$ docker build -f Dockerfile . -t kaggle

# start bash
$ docker run -it -v $(pwd)/../input:/input -v $(pwd):/molecule kaggle bash
```