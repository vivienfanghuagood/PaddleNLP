#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA_ERROR(val) { check_cuda_error((val), #val, __FILE__, __LINE__); }
void check_cuda_error(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void conv2d_nchw_kernel(float *input, float *output, float *kernel, int B, int C, int H, int W, int K, int R, int S) {
    int W_out = W - S + 1;
    int H_out = H - R + 1;
    int out_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_index < B * C * H_out * W_out) {
        int w_out = out_index % W_out;
        int h_index = out_index / W_out;
        int h_out = h_index % H_out;
        int c_index = h_index / H_out;
        int c = c_index % C;
        int b = c_index / C;

        float pixel_value = 0.0;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < S; ++j) {
                int h_in = h_out + i;
                int w_in = w_out + j;
                pixel_value += input[b * (C * H * W) + c * (H * W) + h_in * W + w_in] * kernel[c * (R * S) + i * S + j];
            }
        }

        output[out_index] = pixel_value;
    }
}

__global__ void conv2d_nhwc_kernel(float *input, float *output, float *kernel, int B, int H, int W, int C, int R, int S) {
    int W_out = W - S + 1;
    int H_out = H - R + 1;
    int out_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_index < B * H_out * W_out * C) {
        int c = out_index % C;
        int w_index = out_index / C;
        int w_out = w_index % W_out;
        int h_index = w_index / W_out;
        int h_out = h_index % H_out;
        int b = h_index / H_out;

        float pixel_value = 0.0;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < S; ++j) {
                int h_in = h_out + i;
                int w_in = w_out + j;
                pixel_value += input[b * (H * W * C) + h_in * (W * C) + w_in * C + c] * kernel[c * (R * S) + i * S + j];
            }
        }
        
        output[out_index] = pixel_value;
    }
}

void initialize_data(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    const int B = 32, H = 512, W = 512, C = 3, R = 3, S = 3;
    const int W_out = W - S + 1;
    const int H_out = H - R + 1;
    const size_t input_size = B * H * W * C * sizeof(float);
    const size_t output_size = B * H_out * W_out * C * sizeof(float);
    const size_t kernel_size = C * R * S * sizeof(float);

    float *input, *output, *kernel;
    cudaMallocManaged(&input, input_size);
    cudaMallocManaged(&output, output_size);
    cudaMallocManaged(&kernel, kernel_size);

    initialize_data(input, B * H * W * C);
    initialize_data(kernel, C * R * S);

    int threadsPerBlock = 256;
    int blocksPerGridNCHW = (B * C * H_out * W_out + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridNHWC = (B * H_out * W_out * C + threadsPerBlock - 1) / threadsPerBlock;

    for(int i=0; i< 5; ++i)
        conv2d_nchw_kernel<<<blocksPerGridNCHW, threadsPerBlock>>>(input, output, kernel, B, C, H, W, C, R, S);
    cudaDeviceSynchronize();
    auto startNCHW = std::chrono::high_resolution_clock::now();
    for(int i=0; i< 100; ++i)
        conv2d_nchw_kernel<<<blocksPerGridNCHW, threadsPerBlock>>>(input, output, kernel, B, C, H, W, C, R, S);
    cudaDeviceSynchronize();
    auto endNCHW = std::chrono::high_resolution_clock::now();

    for(int i=0; i< 5; ++i)
        conv2d_nhwc_kernel<<<blocksPerGridNHWC, threadsPerBlock>>>(input, output, kernel, B, H, W, C, R, S);
    cudaDeviceSynchronize();
    auto startNHWC = std::chrono::high_resolution_clock::now();
    for(int i=0; i< 100; ++i)
        conv2d_nhwc_kernel<<<blocksPerGridNHWC, threadsPerBlock>>>(input, output, kernel, B, H, W, C, R, S);
    cudaDeviceSynchronize();
    auto endNHWC = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsedNCHW = endNCHW - startNCHW;
    std::chrono::duration<double, std::milli> elapsedNHWC = endNHWC - startNHWC;

    std::cout << "NCHW kernel execution time: " << elapsedNCHW.count() << " ms\n";
    std::cout << "NHWC kernel execution time: " << elapsedNHWC.count() << " ms\n";

    cudaFree(input);
    cudaFree(output);
    cudaFree(kernel);

    return 0;
}