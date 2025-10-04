import os
import tvm
import os.path
from typing import List
import numpy as np
import torch.nn as nn
import copy
import torch
import onnx
import json
from tvm import relay

from .type_map import TorchTypeDict


class TorchModel:
    INPUT_NAME_TEMPLATE = "ModelInput_{index}"

    OUTPUT_NAME_TEMPLATE = "ModelOutput_{index}"

    def __init__(
            self,
            torch_model: nn.Module,
            batch_size: int,
            input_sizes: List[List[int]],
            input_types: List[str],
            output_num: int,
            work_dir,
            model_name: str,
            target_device
    ):
        self.target_device = target_device
        self.device = self.target_device.torch_target
        self.batch_size = batch_size
        # torch.save(torch_model, "/tmp/tmp.tar")
        self.torch_model = copy.deepcopy(torch_model)
        self.torch_model.load_state_dict(torch_model.state_dict())
        self.torch_model.eval().to(self.device)
        self.input_sizes = input_sizes
        self.input_num = len(self.input_sizes)
        self.input_types = input_types
        self.output_num = output_num
        self.fp = getattr(self.torch_model, "fp", torch.float32)
        self.dummy_input = tuple([
            torch.ones(
                [batch_size] + input_size,
                requires_grad=False,
                dtype=self.fp,
                device=self.target_device.torch_target
            )
            for input_size, input_type in zip(input_sizes, input_types)
        ])
        self.model_name = model_name
        self.work_dir = work_dir
        if not os.path.isdir(self.work_dir):
            os.mkdir(self.work_dir)

        self.tar_path = os.path.join(self.work_dir, f'{model_name}.tar')
        self.libpath = os.path.join(self.work_dir, f'{model_name}.so')
        self.graph_json_path = os.path.join(self.work_dir, f'{model_name}.json')
        self.param_path = os.path.join(self.work_dir, f'{model_name}.params')
        self.onnx_path = os.path.join(self.work_dir, f"{self.model_name}.onnx")

        self.input_names = [self.INPUT_NAME_TEMPLATE.format(index=i) for i in range(self.input_num)]
        self.output_names = [self.OUTPUT_NAME_TEMPLATE.format(index=i) for i in range(self.output_num)]

        self.onnx_model = self.torch2onnx(False, self.onnx_path, dynamic_axes=False)
        # self.script_model = self.torch2script()

    def torch2onnx(self, do_constant_folding, onnx_path, dynamic_axes):
        self.torch_model.eval()

        # Export the model
        if dynamic_axes:
            torch.onnx.export(
                self.torch_model,  # model being run
                self.dummy_input,  # model input (or a tuple for multiple inputs)
                onnx_path,  # where to save the model
                export_params=True,  # store the trained parameter weights inside the model file
                # opset_version=10,  # the ONNX version to export the model to
                do_constant_folding=do_constant_folding,  # whether to execute constant folding for optimization
                input_names=self.input_names,  # the model's input names
                output_names=self.output_names,  # the model's output names
                dynamic_axes={
                    'ModelInput_0': {0: 'batch_size'},  # variable length axes
                    'ModelOutput_0': {0: 'batch_size'}}
            )
        else:
            torch.onnx.export(
                self.torch_model,  # model being run
                self.dummy_input,  # model input (or a tuple for multiple inputs)
                onnx_path,  # where to save the model
                export_params=True,  # store the trained parameter weights inside the model file
                # opset_version=10,  # the ONNX version to export the model to
                do_constant_folding=do_constant_folding,  # whether to execute constant folding for optimization
                input_names=self.input_names,  # the model's input names
                output_names=self.output_names,  # the model's output names
            )
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        return onnx_model

    #
    def torch2script(self):
        scripted_model = torch.jit.trace(self.torch_model, self.dummy_input).eval()
        return scripted_model

    def forward(self, input_lists: List[torch.Tensor]):
        self.torch_model.eval().to(self.device)
        input_lists = [d.to(self.device) for d in input_lists]
        predicts = self.torch_model(*input_lists)

        if isinstance(predicts, torch.Tensor):
            predicts = [predicts]
        elif isinstance(predicts, list):
            pass
        else:
            raise NotImplementedError
        predicts = [d.detach().to('cpu') for d in predicts]
        return predicts


# def tvm_inference(self, input_lists: List[np.array], dev) -> List[torch.Tensor]:
#     lib = load_module(self.libpath)
#     # 创建 module 对象
#     # loaded_json = open(self.graph_json_path).read()
#     # loaded_params = bytearray(open(self.param_path, "rb").read())
#     # module = graph_runtime.create(loaded_json, lib, dev)
#     module = graph_executor.GraphModule(lib["default"](dev))
#     # module.load_params(loaded_params)
#
#     # set input data
#     for name, x, x_type in zip(self.input_names, input_lists, self.input_types):
#         module.set_input(name, tvm.nd.array(x.astype(x_type)))
#
#     module.run()
#     all_outs = []
#     for i in range(self.output_num):
#         out = module.get_output(0)
#         all_outs.append(torch.from_numpy(out.asnumpy()))
#     return all_outs
#

#
# def torch_inference(self, input_lists: List[np.array], device) -> List[torch.Tensor]:
#     input_tensor = [
#         torch.from_numpy(x).to(device).to(TorchTypeDict[input_type])
#         for x, input_type in zip(input_lists, self.input_types)
#     ]
#     torch_model = self.torch_model.eval().to(device)
#     predicts = torch_model(*input_tensor)
#
#     if isinstance(predicts, torch.Tensor):
#         predicts = [predicts]
#     elif isinstance(predicts, list):
#         pass
#     else:
#         raise NotImplementedError
#     predicts = [d.detach().to('cpu') for d in predicts]
#     return predicts


class CompiledModel:
    def __init__(self, ori_model: TorchModel, compiled_model):
        self.ori_model = ori_model
        self.fp = ori_model.fp
        self.compiled_model = compiled_model

        self.input_types = self.ori_model.input_types
        self.input_names = self.ori_model.input_names
        self.output_num = self.ori_model.output_num
        self.libpath = self.ori_model.libpath
        self.graph_json_path = self.ori_model.graph_json_path
        self.param_path = self.ori_model.param_path

    def inference(self, input_lists: List[np.array]) -> List[torch.Tensor]:
        raise NotImplementedError


class TorchCompiledModel(CompiledModel):
    def __init__(self, ori_model: TorchModel, compiled_model, device):
        super().__init__(ori_model, compiled_model)
        assert isinstance(compiled_model, nn.Module)
        self.device = device

    def forward(self, input_lists: List[torch.tensor]) -> List[torch.Tensor]:
        input_tensor = [
            x.to(self.device).to(self.fp)
            for x, input_type in zip(input_lists, self.input_types)
        ]
        self.compiled_model = self.compiled_model.eval().to(self.device)
        predicts = self.compiled_model(*input_tensor)

        if isinstance(predicts, torch.Tensor):
            predicts = [predicts]
        elif isinstance(predicts, list):
            pass
        else:
            raise NotImplementedError
        predicts = [d.detach().to('cpu') for d in predicts]
        return predicts


class TVMCompiledModel(CompiledModel):
    def __init__(self, ori_model: TorchModel, compiled_model):
        super().__init__(ori_model, compiled_model)

    def forward(self, input_lists: List[torch.Tensor]) -> List[torch.Tensor]:

        # set input data
        for name, t_x, x_type in zip(self.input_names, input_lists, self.input_types):
            x = t_x.detach().cpu().numpy()
            self.compiled_model.set_input(
                name, tvm.nd.array(
                    x,
                    device=self.ori_model.target_device.tvm_dev))
        self.compiled_model.run()

        all_outs = []
        for i in range(self.output_num):
            out = self.compiled_model.get_output(i)
            all_outs.append(torch.from_numpy(out.asnumpy()))
        return all_outs

    @property
    def graph(self):
        try:
            with open(self.graph_json_path, "r") as f:
                return json.load(f)
        except:
            raise NotImplementedError

    @property
    def parameters(self):
        loaded_params = bytearray(open(self.param_path, "rb").read())
        p = relay.load_param_dict(loaded_params)
        p_dict = {k: p[k].asnumpy() for k in p.keys()}
        return p_dict


class OnnxCompiledModel(CompiledModel):
    def __init__(self, ori_model: TorchModel, compiled_model):
        super().__init__(ori_model, compiled_model)


    def forward(self, input_lists: List[torch.Tensor]) -> List[torch.Tensor]:
        input_dict = {}
        for name, x, x_type in zip(self.input_names, input_lists, self.input_types):
            input_dict[name] = x.detach().cpu().numpy()

        outputs = self.compiled_model.run(None, input_dict)
        outputs = [torch.from_numpy(d).cpu() for d in outputs]
        return outputs


class TensorRTCompiledModel(CompiledModel):
    def __init__(self, ori_model: TorchModel, compiled_model, device):
        super().__init__(ori_model, compiled_model)
        self.device = device


    def forward(self, input_lists: List[torch.Tensor]) -> List[torch.Tensor]:
        input_tensor = [
            x.to(self.device).to(self.fp)
            for x, input_type in zip(input_lists, self.input_types)
        ]
        self.compiled_model = self.compiled_model.eval().to(self.device)
        predicts = self.compiled_model(*input_tensor)

        if isinstance(predicts, torch.Tensor):
            predicts = [predicts]
        elif isinstance(predicts, list):
            pass
        else:
            raise NotImplementedError
        predicts = [d.detach().to('cpu') for d in predicts]
        return predicts


class IREECompiledModel(CompiledModel):
    def __init__(self, ori_model: TorchModel, compiled_model, device):
        super().__init__(ori_model, compiled_model)
        self.device = device

    def forward(self, input_lists: List[torch.Tensor]) -> List[torch.Tensor]:
        input_tensor = [
            x.to(self.device).to(self.fp)
            for x, input_type in zip(input_lists, self.input_types)
        ]
        # input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        # result = vm_module.main(input)
        # print(result.to_host())

        self.compiled_model = self.compiled_model.eval().to(self.device)
        predicts = self.compiled_model(*input_tensor)

        if isinstance(predicts, torch.Tensor):
            predicts = [predicts]
        elif isinstance(predicts, list):
            pass
        else:
            raise NotImplementedError
        predicts = [d.detach().to('cpu') for d in predicts]
        return predicts