# object_prediction_use_tflite

:warning: **I don't do that process with float type model: Be very careful here!**


| function        | parameter                                                                                                                                                                                              | usage                                                                                                                                                         |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| set_interpreter | image_path=str()<br> MODEL_PATH=str()                                                                                                                                                                  | Help you to set interpreter in memory and change image to tflite format, it also return original image let you can do something with original image.          |
| predict         | output_details=list() <br>A list of output details. <br> interpreter = tflite interpreter object                                                                                                              | Help you to get predict information like predict score and predict boxes or something you need. In this repo I only return what I need like scores and boxes. |
| get_predictBox  | boxes=list()<br> A list with detect boxes.<br> scores=list()<br> A list with scores data.<br> img=cv2 image format<br> A cv2 image object.<br> classes=str()<br> A classes show on detect boxes | This function just help you to draw some boxes on your specify image.                                                                                         |


## deploy


### open into this dir and install requirements:

```

pip install -r requirements.txt

```

<br>

### you can use this module in your python interpreter like:

```python
>>> import cv2
>>> from utils import detect
1.15.0
True
"""
some log from GPU detail
"""
>>> MODEL_PATH = r"AI-Aquaculturing/ForEdgetpuModels/edgetpu_koifish_1000000/koifish_detect-100w.tflite"
>>> img_path = r'AI-Aquaculturing/101.png'
>>> interpreter, output_details, img=detect.set_interpreter(img_path, MODEL_PATH) # set your interpreter to your memory
set time is coast:0.08032512664794922
>>> boxes, scores =detect.predict(output_details, interpreter) # start get your detect information
>>> img_withBox, Population =detect.get_predictBox(boxes, scores, img) # draw detect box by your detect information
>>> cv2.imwrite("output.jpg", img_withBox) # save your image with detect boxes
True
>>> print("Population:{}".format(Population)) # print your Population
Population:5
>>>
```
