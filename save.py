import os
import torch

def save_model(model, model_type, model_path=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    if model_path is None:
        model_path = "models/model_{}".format(model_type)
    print('Saving models to {}'.format(model_path))
    torch.save(model.state_dict(), model_path)


def load_model(model,model_path):
    print('Loading models from {}'.format(model_path))
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))