# ScanPay: Food Detection Using Faster R-CNN

ScanPay is a food detection platform based on deep learning.

### API

URL: `http://127.0.0.1:5000`

METHOD: POST

PARAMS:
- image: file

RESPONSE (FAIL):
```
{
    "status": "fail",
    "data": {
        "image": null,
        "labels": null,
        "names": null,
        "prices": null
    }
}
```
RESPONSE (SUCCESS):
```
{
    "status": "success",
    "data": {
        "image": path,
        "labels": [label],
        "names": [name],
        "prices": [price]
    }
}
```

### Web App

To download the repository:

`git clone https://github.com/leoofficial/food-detection.git`

Then you need to install the basic dependencies to run the project on your system:

`
pip install -r requirements.txt
`

Then to run the Flask app:

`python -m flask run`

![](./docs/截屏2020-03-16上午2.03.43.png)
![](./docs/截屏2020-03-16上午2.03.57.png)

### Mobile App

[Mobile app developed by Jerrick](https://github.com/JerrickCai/foodRecognitionApp)

### Datasets

[UEC FOOD 100](http://foodcam.mobi/dataset100.html)
