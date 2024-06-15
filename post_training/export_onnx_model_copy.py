import os
import torch
import onnx
from main_training import AnimalClassifier

# Load the model
hparams = {
    'learning_rate': 0.0005,
}

def export_as_onnx(model_path, input_shape, onnx_filename):
    model = AnimalClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    dummy_input = torch.randn(input_shape)
    onnx_path = os.path.join(os.path.dirname(model_path), onnx_filename)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

