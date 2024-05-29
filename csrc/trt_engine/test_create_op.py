#%%
import paddle
from trt_engine_ops import *
# %%

img = paddle.randn([1, 3, 224, 224])
a = trtengine_op(img, "./sample_engine.trt", "input")
# %%
# %%
