import io
import numpy as np
import torch.onnx
from model import U2NET

torch_model = U2NET(3,1)
model_path = "./saved_models/latest.pth"
batch_size = 1

torch_model.load_state_dict(torch.load(model_path))
torch_model.eval()

x = torch.randn(batch_size, 3, 320, 320, requires_grad=True)
torch_out = torch_model(x)

torch.onnx.export(torch_model, x, "latest.onnx", export_params=True, opset_version=16, do_constant_folding=True, input_names = ['input'], output_names = ['output'], dynamic_axes = {'input' : {0: 'batch_size'}, 'output': {0: 'batch_size'}})