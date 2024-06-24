import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# 加载 TensorRT 引擎
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 准备输入数据
def preprocess_input(input_data):
    # 这里假设 input_data 是一个 numpy 数组
    # 你可以根据实际需求进行预处理
    return np.asarray(input_data, dtype=np.float32)

# 执行推理
def infer(engine, input_data):
    context = engine.create_execution_context()

    # 分配输入和输出的内存
    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)
    input_size = trt.volume(input_shape) * engine.max_batch_size
    output_size = trt.volume(output_shape) * engine.max_batch_size

    # 分配主机和设备内存
    d_input = cuda.mem_alloc(input_size * np.dtype(np.float32).itemsize)
    d_output = cuda.mem_alloc(output_size * np.dtype(np.float32).itemsize)

    # 将输入数据复制到设备
    cuda.memcpy_htod(d_input, input_data)

    # 执行推理
    context.execute_v2(bindings=[int(d_input), int(d_output)])

    # 从设备复制输出数据
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, d_output)

    return output_data

if __name__ == "__main__":
    engine_file_path = './sample_engine.trt'
    input_data = np.ones(1, 3, 224, 224).astype(np.float32)

    # 加载 TensorRT 引擎
    engine = load_engine(engine_file_path)

    # 执行推理
    output_data = infer(engine, input_data)

    print("推理结果:", output_data)