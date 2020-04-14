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

### Database

URI: `mongodb+srv://jla597:jla597@scanpay-5whgk.mongodb.net/test?retryWrites=true&w=majority`

SCHEMA:
```
{
    _id: ObjectId,
    label: Int32,
    name: String,
    price: Double
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

### Experiment / Report

[Google Docs](https://docs.google.com/document/d/1Xrz1bQRj7HlFYtqOIWdIlWOK060So0lePB57yRNKV8s)


### Datasets

[Preprocessed Dataset](https://drive.google.com/file/d/101hzCMlGhmegbX-8san278gAVEds_Qit/view?usp=sharing)


### Papers

[R-CNN](https://arxiv.org/pdf/1311.2524.pdf) (Related)

[Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf) (Related)

[Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)


### Tutorials

[Lecture on Detection](https://www.youtube.com/watch?v=nDPWywWRIRo)

[Finetune a Pre-trained Model](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)