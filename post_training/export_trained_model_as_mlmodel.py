import torch
import coremltools as ct
import numpy as np
import pytorch_lightning as pl

from training.net_architecture import DnCNN
from training.utils import HParams
# Calea către checkpoint
model_path = "/home/smbanaru/Desktop/DisNet/saved_models/DnCNN_SIDD_small_50x50_experiment_1.ckpt"

lightning_model = DnCNN.load_from_checkpoint(model_path)
state_dict = lightning_model.state_dict()

# hparams = HParams(num_layers=17, learning_rate=0.001)
hparams = {"num_layers": 17, "learning_rate": 0.001}
model = DnCNN(hparams=hparams)  
model.load_state_dict(state_dict)
model.eval() 
model._trainer = pl.Trainer()


dummy_input = torch.randn(1, 3, 50, 50)
traced_model = torch.jit.trace(model, dummy_input)
example_inputs = (torch.rand(1, 3, 50, 50),)
exported_program = torch.export.export(model, example_inputs)

model_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=dummy_input.shape)],
)
model_from_export = ct.convert(exported_program)

model_from_trace.save(model_path.split(".")[0] + "_traced" + ".mlpackage")
model_from_export.save(model_path.split(".")[0] + "_exported" + ".mlpackage")

# mlmodel = ct.models.MLModel(model_path.split(".")[0] + ".mlmodel")
# optimized_model = ct.models.neural_network.convert_neural_network_weights_to_float16(mlmodel)
# optimized_model.save(model_path.split(".")[0] + "_float16" + ".mlmodel")

# input_data = (np.random.rand(1, 3, 50, 50)).astype(np.uint8)

# original_output = mlmodel.predict({"input": input_data})
# optimized_output = optimized_model.predict({"input": input_data})

# print("MAE between float32 and float16 mlmodel:", np.abs(original_output - optimized_output))
