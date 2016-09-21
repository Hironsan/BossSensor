# BossSensor
Hide screen when boss is approaching.

## Demo

## Requirements

* WebCamera
* python3.5
* OSX
* virtualenv

## Usage
First, Train boss image.

```
$ python boss_train.py
```


Second, start BossSensor. 

```
$ python camera_reader.py
```

## Install
install OpenCV

install virtualenv

```
virtualenv venv --python=python3.5
source venv/bin/activate
pip install -r requirements.txt
```


## Licence

[MIT](https://github.com/Hironsan/BossSensor/blob/master/LICENSE)

## Author

[Hironsan](https://github.com/Hironsan)