import torch
print("PyTorch 版本:", torch.__version__)         # 输出安装的 PyTorch 版本
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("PyTorch 编译时 CUDA 版本:", torch.version.cuda)
    print("当前 GPU:", torch.cuda.get_device_name(0))
import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 版本:", torch.version.cuda)
print("cuDNN 版本:", torch.backends.cudnn.version())
import tensorflow as tf
print("TensorFlow 版本:", tf.__version__)         # 输出安装的 TF 版本
print("是否支持 CUDA:", tf.test.is_built_with_cuda())
print("GPU 是否可用:", tf.config.list_physical_devices('GPU'))