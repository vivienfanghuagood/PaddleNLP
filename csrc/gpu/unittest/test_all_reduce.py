import paddle

from paddlenlp_ops import trt_reduce
import paddle.distributed as dist

from paddlenlp.trl import llm_utils

class CustomAllReduceTest(unittest.TestCase):
    def test_custom_allreduce():
        dist.init_parallel_env()
        input_tensor = paddle.ones([1, 512], "float16") / 2
        input_tensor_copy = paddle.ones([1, 512], "float16") / 2

        dist.all_reduce(input_tensor_copy)
        print("nccl all reduce: ", input_tensor_copy)

        tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()
        out = trt_reduce(input_tensor, tensor_parallel_rank, tensor_parallel_degree)
        print("custom all reduce: ", out)
        np.testing.assert_array_equal(input_tensor_copy, out, err_msg="trt_reduce get different result") 

if __name__ == "__main__":
    unittest.main()