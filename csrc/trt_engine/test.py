import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(batch_size):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         builder.create_builder_config() as config:
    
        input_shape = (-1, 3, -1, -1)
        input_tensor = network.add_input(name="input", dtype=trt.float32, shape=input_shape)
        
        # 卷积层参数
        num_output_maps = 32
        kernel_shape = (3, 3)
        input_channels = 3
        
        # 生成随机权重
        kernel_size = (num_output_maps, input_channels, kernel_shape[0], kernel_shape[1])
        weights = np.random.normal(loc=0.0, scale=0.01, size=kernel_size).astype(np.float32)
        
        # 添加卷积层
        conv_layer = network.add_convolution_nd(input_tensor, num_output_maps=num_output_maps, kernel_shape=kernel_shape, kernel=trt.Weights(weights))
        conv_layer.stride_nd = (1, 1)
        conv_layer.padding_nd = (1, 1)
        
        activation_layer = network.add_activation(conv_layer.get_output(0), type=trt.ActivationType.RELU)
        network.mark_output(activation_layer.get_output(0))
        
        profile = builder.create_optimization_profile()
        profile.set_shape("input", min=(1, 3,  224, 224), opt=(batch_size, 3,  224, 224), max=(batch_size * 2, 3,  224, 224))
        config.add_optimization_profile(profile)
        
        return builder.build_engine(network, config)

batch_size = 8
engine = build_engine(batch_size)

with open("sample_engine.trt", "wb") as f:
    f.write(engine.serialize())