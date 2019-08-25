## 在Android设备上使用TensorFlow有两种方法  
### 一. 通过[TensorFlow-Android](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android)   
这里也有两种方法： 
#### 1.通过.so动态链接库  
这需要编译，详情看上面的链接。   
#### 2. 通过引入Java JAR包，在AndroidStudio里添加library引用   
在Android项目的`build.gradle`加入下面代码即可：
```
allprojects {
    repositories {
        jcenter()
    }
}

dependencies {
    compile 'org.tensorflow:tensorflow-android:+'
}
```
`org.tensorflow:tensorflow-android:+`的`+`表示加载最新的版本，可以向下面这样指定版本：
```
dependencies {
    implementation 'org.tensorflow:tensorflow-android:1.13.1'
}
```
**注意**：这里的版本必须高于等于训练时用的版本，否则在Android下`new TensorFlowInferenceInterface(AssetManager assetManager, String model)`时会加载模型失败。    
所有版本可以在这里找到：https://jcenter.bintray.com/org/tensorflow/  
### 二.使用TensorFlow Lite
要将TensorFlow模型转换成.tflite
#### 方法:
1.先用[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)把计算图和参数固化到.pb文件。  
2.再用Tensorflow Lite转换器将.pb转成.tflite，用法在[这里](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/python_api.md)。 
转换模型有多种方式，这里使用tf.lite.TFLiteConverter.from_frozen_graph从固化的模型转。      

具体方法见项目下/in_tensorflow/test/freeze_tflite.py文件。  
  
参考:https://www.tensorflow.org/lite/guide/get_started#3_use_the_tensorflow_lite_model_for_inference_in_a_mobile_app  
  

#### .tflite的用法
1. 参考官方教程https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android  
2. 我的做的[安卓项目](https://github.com/yemiekai/Vedio_Voice)含人脸对比功能。

## 问题:
### 1. 许多Tensorflow的操作(opt)还不支持转换。
### 2. 使用BN层(tf.layers.batch_normalization)的模型无法转换。
这个问题在[这里](https://blog.csdn.net/zaf0516/article/details/89958962)找到答案：是模型参数`is_training`要直接赋值为`True`or`False`，不能用`tf.placeholder`在`sess.run`时用`feed_dict`里指定。  
### 3. 使用.pb和.tflite运行的结果不一样。
先用`tensorflow.python.tools.freeze_graph.freeze_graph`把模型固化成.pb文件，再使用`tf.lite.TFLiteConverter.from_frozen_graph`将固化的模型转成.tflite。结果发现同一张图片，加载.pb运行的结果和加载.tflite运行的结果不一样。  
经过测试发现，加载.tflite运行的结果，看起来像网络未训练过一样，可能是转换时参数问题？有待研究原因。
  
  
