# object_prediction_use_tflite

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
