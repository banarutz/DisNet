import socket
from net_architecture import DnCNN, EncoderDecoderDenoising

def is_port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False  
        except socket.error:
            return True  
        
def model_picker(cfg):
    if cfg.training.net_type == "EDD" and cfg.data.resume:
        print(f"Resuming training from checkpoint: {cfg.data.resume}. Used architecture: {cfg.training.net_type}.")
        model = EncoderDecoderDenoising.load_from_checkpoint(cfg.data.resume, strict=True)
    elif cfg.training.net_type == "EDD" and not cfg.data.resume:
        print(f"Starting training from scratch. Used architecture: {cfg.training.net_type}.")
        model = EncoderDecoderDenoising(cfg.model)
    elif cfg.training.net_type == "DnCNN" and cfg.data.resume:
        print(f"Resuming training from checkpoint: {cfg.data.resume}. Used architecture: {cfg.training.net_type}.")
        model = DnCNN.load_from_checkpoint(cfg.data.resume, strict=True)
    elif cfg.training.net_type == "DnCNN" and not cfg.data.resume:
        print(f"Starting training from scratch. Used architecture: {cfg.training.net_type}.")
        model = DnCNN(cfg.model)
    
    return model
        
class HParams:
    def __init__(self, num_layers, learning_rate):
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
    