import torch
import pickle

# Method 1: Load the complete model (less version compatible)
loaded_model = torch.load('/Users/mac/Desktop/Bio_meta_tran/Mollytest/puffmodel/model_architecture.pt')
print(loaded_model.eval())  # Set the model to evaluation mode

# Method 2: Load with state dict (more version compatible)
# First, initialize your model with the same architecture
# Assuming you have your model class defined as 'ModelClass'
# model = ModelClass()  # Replace ModelClass with your actual model class
# model.load_state_dict(torch.load('/Users/mac/Desktop/Bio_meta_tran/Mollytest/puffmodel/model_state_dict.pt'))
# model.eval()

# Optional: Load model config if saved
try:
    with open('/Users/mac/Desktop/Bio_meta_tran/Mollytest/puffmodel/model_config.pkl', 'rb') as f:
        config = pickle.load(f)
except FileNotFoundError:
    print("No config file found")