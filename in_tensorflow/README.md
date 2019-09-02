# 在Android设备上使用TensorFlow有两种方法
  
## 一. 使用[TensorFlow-Android](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android)   
步骤如下
### 步骤1. 先得到TensorFlow的jar包
有2种方法得到jar包, 我用的是方法1 
##### （方法1）直接从[JCenter](https://bintray.com/google/tensorflow/tensorflow)引用TensorFlow AAR包(在AndroidStudio里添加library引用) 
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
“`org.tensorflow:tensorflow-android:+`”的“`+`”表示加载最新的版本，可以像下面这样指定版本：
```
dependencies {
    implementation 'org.tensorflow:tensorflow-android:1.13.1'
}
```
**注意**：这里的版本必须高于等于训练时用的版本，否则在Android下`new TensorFlowInferenceInterface(AssetManager assetManager, String model)`时会加载模型失败。    
所有已发布的版本可以在这里找到：https://jcenter.bintray.com/org/tensorflow/    
##### （方法2）自己编译 
详情看上面[链接](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android)里官方的说明。   
  
### 步骤2. 将python下保存的模型文件(ckpt)固化成.pb文件    
使用代码[DeepLearningCode/in_tensorflow/test/freeze_tflite.py](https://github.com/yemiekai/DeepLearningCode/blob/master/in_tensorflow/test/freeze_tflite.py)里`freeze_graph_and_parameter()`函数
  
### 步骤3. 将.pb文件放在Android项目的assets文件夹下  
   
### 步骤4. 在Android中加载模型
示例代码如下：
```
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class Demo{
    private static final String MODEL_FILE  = "file:///android_asset/freezed_model.pb";
    private TensorFlowInferenceInterface inferenceInterface;
    
    private boolean loadModel(Activity activity) {
        try {
            inferenceInterface = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_FILE);
            Log.d("Demo", "load model success");
        }catch(Exception e){
            Log.d("Demo", "load model failed");
            return false;
        }
        return true;
    }
}
```
### 步骤5. 使用模型进行推理
主要使用`TensorFlowInferenceInterface`类的`feed()`方法输入数据，`run()`方法运行模型，`fetch()`方法获取结果。具体使用方法请看[官方Demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)或者我的另外一个[项目](https://github.com/yemiekai/Vedio_Voice)中[InsightFace.java](https://github.com/yemiekai/Vedio_Voice/blob/master/app/src/main/java/com/yemiekai/vedio_voice/tflite/InsightFace.java)的代码。
## 二. 使用[TensorFlow Lite](https://www.tensorflow.org/lite/guide/get_started#3_use_the_tensorflow_lite_model_for_inference_in_a_mobile_app)
### 步骤1. 将python下保存的模型文件(ckpt)固化成.pb文件    
使用代码[DeepLearningCode/in_tensorflow/test/freeze_tflite.py](https://github.com/yemiekai/DeepLearningCode/blob/master/in_tensorflow/test/freeze_tflite.py)里`freeze_graph_and_parameter()`函数
  
### 步骤2. 使用TensorFlow Lite[转换器](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/python_api.md)将.pb文件转成.tflite文件    
使用代码[DeepLearningCode/in_tensorflow/test/freeze_tflite.py](https://github.com/yemiekai/DeepLearningCode/blob/master/in_tensorflow/test/freeze_tflite.py)里`convert_tflite()`函数
  
### 步骤3. 将.tflite文件放在Android项目的assets文件夹下   
### 步骤4. 在Android中加载并使用模型
参考官方教程https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android    
我做的[安卓项目](https://github.com/yemiekai/Vedio_Voice)含人脸对比功能，不过是使用.pb文件的，用.tflite还有点问题。

## 问题:
### 1. 许多Tensorflow的操作(opt)还不支持转换。
### 2. 使用BN层(tf.layers.batch_normalization)的模型无法转换。
这个问题在[这里](https://blog.csdn.net/zaf0516/article/details/89958962)找到答案：是模型参数`is_training`要直接赋值为`True`or`False`，不能用`tf.placeholder`在`sess.run`时用`feed_dict`里指定。  
### 3. 使用.pb和.tflite运行的结果不一样。
先用`tensorflow.python.tools.freeze_graph.freeze_graph`把模型固化成.pb文件，再使用`tf.lite.TFLiteConverter.from_frozen_graph`将固化的模型转成.tflite。结果发现同一张图片，加载.pb运行的结果和加载.tflite运行的结果不一样。  
经过测试发现，加载.tflite运行的结果，看起来像网络未训练过一样，可能是转换时参数问题？有待研究原因。
  
  
