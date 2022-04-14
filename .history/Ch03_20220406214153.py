import numpy as np
from sklearn.datasets import fetch_openml
print("数据集加载中，请稍后……")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print("准备完毕")
X, y = mnist["data"], mnist["target"] #X里面是数据，y里面是标签
y = y.astype(np.uint8)#将标签从文本格式转换为数字格式
X.shape()
#数据类型
type(mnist)
#%%展示图片
import matplotlib
import matplotlib.pyplot as plt
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
i
# %%
