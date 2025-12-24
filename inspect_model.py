import torch
import os

model_path = os.path.join('Sign Language Recognition', 'CNN_model_number_SIBI.pth')
try:
    state_dict = torch.load(model_path, map_location='cpu')
    # Print keys to find the final layer
    for key in state_dict.keys():
        if 'linearLayers' in key and 'weight' in key:
            print(f"{key}: {state_dict[key].shape}")
except Exception as e:
    print(e)
