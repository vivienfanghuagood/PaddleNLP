import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Helper function to create random weights
def create_weights(shape):
    return np.random.rand(*shape).astype(np.float32)

# Initialize TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Create a TensorRT builder, network and config
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.max_workspace_size = 1 << 20  # 1 MiB

# Define input tensor with dynamic shape
input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(-1, 64))

# Create random weights for MatMul and Add layers
weight = create_weights((64, 64))
bias = create_weights((1, 64))  # Adjust shape to be broadcastable

# Add MatMul layer
weight_tensor = network.add_constant(weight.shape, trt.Weights(weight)).get_output(0)
matmul_layer = network.add_matrix_multiply(input_tensor, trt.MatrixOperation.NONE, weight_tensor, trt.MatrixOperation.NONE)

# Add Add layer
bias_tensor = network.add_constant(bias.shape, trt.Weights(bias)).get_output(0)
add_layer = network.add_elementwise(matmul_layer.get_output(0), bias_tensor, trt.ElementWiseOperation.SUM)

# Add ReLU layer
relu_layer = network.add_activation(add_layer.get_output(0), trt.ActivationType.RELU)

# Mark the output of ReLU layer as the output of the network
network.mark_output(tensor=relu_layer.get_output(0))

# Create optimization profile for dynamic input shapes
profile = builder.create_optimization_profile()
min_shape = (1, 64)
opt_shape = (4, 64)
max_shape = (8, 64)
profile.set_shape("input", min_shape, opt_shape, max_shape)
config.add_optimization_profile(profile)

# Build the engine
engine = builder.build_engine(network, config)

# Function to perform inference
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return the host outputs.
    return [out.host for out in outputs]

# Allocate buffers and create a CUDA stream.
input_shape = (1, 64)
input_data = np.random.rand(*input_shape).astype(np.float32)

output_shape = (input_shape[0], 64)

d_input = cuda.mem_alloc(input_data.nbytes)
# import pdb;pdb.set_trace()
d_output = cuda.mem_alloc(int(np.prod(output_shape) * input_data.itemsize))

bindings = [int(d_input), int(d_output)]

# Create a stream
stream = cuda.Stream()

# Create execution context
context = engine.create_execution_context()
context.set_binding_shape(0, input_shape)

# Transfer input data to device
cuda.memcpy_htod(d_input, input_data)

# Perform inference
outputs = np.empty(output_shape, dtype=np.float32)
do_inference(context, bindings, [d_input], [outputs], stream)

# Print the output
print("Inference output:")
print(outputs)
