# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import site
import sys
import unittest

import numpy as np
from utils import check_output, check_output_allclose

import paddle
from paddle import static
from paddle.utils.cpp_extension.extension_utils import run_cmd
from paddle.vision.transforms import Compose, Normalize


def custom_relu_dynamic(func, device, dtype, np_x, use_func=True):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype)
    t.stop_gradient = False
    t.retain_grads()

    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.retain_grads()
    out.stop_gradient = False

    out.backward()

    if t.grad is None:
        return out.numpy(), t.grad
    else:
        return out.numpy(), t.grad.numpy()


def custom_relu_static(
    func, device, dtype, np_x, use_func=True, test_infer=False
):
    paddle.enable_static()
    paddle.set_device(device)

    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = func(x) if use_func else paddle.nn.functional.relu(x)
            static.append_backward(out)

            exe = static.Executor()
            exe.run(static.default_startup_program())
            # in static graph mode, x data has been covered by out
            out_v = exe.run(
                static.default_main_program(),
                feed={'X': np_x},
                fetch_list=[out.name],
            )

    paddle.disable_static()
    return out_v


def custom_relu_static_inference(func, device, np_data, np_label, path_prefix):
    paddle.set_device(device)

    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            # simple module
            data = static.data(
                name='data', shape=[None, 1, 28, 28], dtype='float32'
            )
            label = static.data(name='label', shape=[None, 1], dtype='int64')

            hidden = static.nn.fc(data, size=128)
            hidden = func(hidden)
            hidden = static.nn.fc(hidden, size=128)
            predict = static.nn.fc(hidden, size=10, activation='softmax')
            loss = paddle.nn.functional.cross_entropy(input=hidden, label=label)
            avg_loss = paddle.mean(loss)

            opt = paddle.optimizer.SGD(learning_rate=0.1)
            opt.minimize(avg_loss)

            # run start up model
            exe = static.Executor()
            exe.run(static.default_startup_program())

            # train
            for _ in range(4):
                exe.run(
                    static.default_main_program(),
                    feed={'data': np_data, 'label': np_label},
                    fetch_list=[avg_loss],
                )

            # save inference model
            static.save_inference_model(path_prefix, [data], [predict], exe)

            # get train predict value
            predict_v = exe.run(
                static.default_main_program(),
                feed={'data': np_data, 'label': np_label},
                fetch_list=[predict],
            )

    return predict_v


def custom_relu_double_grad_dynamic(func, device, dtype, np_x, use_func=True):

    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)
    t.retain_grads()

    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.retain_grads()
    dx = paddle.grad(
        outputs=out,
        inputs=t,
        grad_outputs=paddle.ones_like(t),
        create_graph=True,
        retain_graph=True,
    )

    ddout = paddle.grad(
        outputs=dx[0],
        inputs=out.grad,
        grad_outputs=paddle.ones_like(t),
        create_graph=False,
    )

    assert ddout[0].numpy() is not None
    return dx[0].numpy(), ddout[0].numpy()


class TestNewCustomOpXpuSetUpInstall(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = 'cd {} && {} custom_relu_xpu_setup.py install'.format(
            cur_dir, sys.executable
        )
        run_cmd(cmd)

        site_dir = site.getsitepackages()[0]
        custom_egg_path = [
            x
            for x in os.listdir(site_dir)
            if 'custom_relu_xpu_module_setup' in x
        ]
        assert len(custom_egg_path) == 1, "Matched egg number is %d." % len(
            custom_egg_path
        )
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

        # usage: import the package directly
        import custom_relu_xpu_module_setup

        self.custom_op = custom_relu_xpu_module_setup.custom_relu

        self.dtypes = ['float32']
        self.device = 'xpu'

        # config seed
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def test_static(self):
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out = custom_relu_static(self.custom_op, self.device, dtype, x)
            pd_out = custom_relu_static(
                self.custom_op, self.device, dtype, x, False
            )
            check_output(out, pd_out, "out")

    def test_dynamic(self):
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out, x_grad = custom_relu_dynamic(
                self.custom_op, self.device, dtype, x
            )
            pd_out, pd_x_grad = custom_relu_dynamic(
                self.custom_op, self.device, dtype, x, False
            )
            check_output(out, pd_out, "out")
            check_output(x_grad, pd_x_grad, "x_grad")

    def test_static_save_and_load_inference_model(self):
        paddle.enable_static()
        np_data = np.random.random((1, 1, 28, 28)).astype("float32")
        np_label = np.random.random((1, 1)).astype("int64")
        path_prefix = "self.custom_op_inference/custom_relu"

        predict = custom_relu_static_inference(
            self.custom_op, self.device, np_data, np_label, path_prefix
        )
        # load inference model
        with static.scope_guard(static.Scope()):
            exe = static.Executor()
            [
                inference_program,
                feed_target_names,
                fetch_targets,
            ] = static.load_inference_model(path_prefix, exe)
            predict_infer = exe.run(
                inference_program,
                feed={feed_target_names[0]: np_data},
                fetch_list=fetch_targets,
            )
            check_output(predict, predict_infer, "predict")
        paddle.disable_static()

    def test_static_save_and_run_inference_predictor(self):
        paddle.enable_static()
        np_data = np.random.random((1, 1, 28, 28)).astype("float32")
        np_label = np.random.random((1, 1)).astype("int64")
        path_prefix = "self.custom_op_inference/custom_relu"
        from paddle.inference import Config, create_predictor

        predict = custom_relu_static_inference(
            self.custom_op, self.device, np_data, np_label, path_prefix
        )
        # load inference model
        config = Config(path_prefix + ".pdmodel", path_prefix + ".pdiparams")
        predictor = create_predictor(config)
        input_tensor = predictor.get_input_handle(
            predictor.get_input_names()[0]
        )
        input_tensor.reshape(np_data.shape)
        input_tensor.copy_from_cpu(np_data.copy())
        predictor.run()
        output_tensor = predictor.get_output_handle(
            predictor.get_output_names()[0]
        )
        predict_infer = output_tensor.copy_to_cpu()
        predict = np.array(predict).flatten()
        predict_infer = np.array(predict_infer).flatten()
        check_output_allclose(predict, predict_infer, "predict")
        paddle.disable_static()

    def test_func_double_grad_dynamic(self):
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out, dx_grad = custom_relu_double_grad_dynamic(
                self.custom_op, self.device, dtype, x
            )
            pd_out, pd_dx_grad = custom_relu_double_grad_dynamic(
                self.custom_op, self.device, dtype, x, False
            )
            check_output(out, pd_out, "out")
            check_output(dx_grad, pd_dx_grad, "dx_grad")

    def test_with_dataloader(self):
        paddle.disable_static()
        paddle.set_device(self.device)
        # data loader
        transform = Compose(
            [Normalize(mean=[127.5], std=[127.5], data_format='CHW')]
        )
        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', transform=transform
        )
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        for batch_id, (image, _) in enumerate(train_loader()):
            out = self.custom_op(image)
            pd_out = paddle.nn.functional.relu(image)
            check_output_allclose(out, pd_out, "out", atol=1e-2)

            if batch_id == 5:
                break
        paddle.enable_static()


if __name__ == '__main__':
    # compile, install the custom op egg into site-packages under background
    # Currently custom XPU op does not support Windows
    if os.name == 'nt':
        sys.exit()
    unittest.main()