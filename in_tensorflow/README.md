## 转换成.tflite
### 方法:
1.先用[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)把计算图和参数固化到.pb文件  
2.再用Tensorflow Lite转换器将.pb转成.tflite，用法在[这里](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/python_api.md).  
转换模型有多种方式，这里使用tf.lite.TFLiteConverter.from_frozen_graph从固化的模型转     

具体方法见项目下/in_tensorflow/test/freeze_tflite.py文件  
  
参考:https://www.tensorflow.org/lite/guide/get_started#3_use_the_tensorflow_lite_model_for_inference_in_a_mobile_app  
  
### 问题:
1. 许多Tensorflow的操作(opt)还不支持转换，例如tf.layers.batch_normalization
  
  
## .tflite的用法
1. 参考官方教程https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android  
2. 我的做的[安卓项目](https://github.com/yemiekai/Vedio_Voice)含人脸对比功能


