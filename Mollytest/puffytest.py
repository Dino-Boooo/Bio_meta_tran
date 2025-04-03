import torch
import pickle
import os

# Load the model
model = torch.load('model/model.pth')
model.eval()

# Create directory if it doesn't exist
# os.makedirs('saved_model', exist_ok=True)

# Save the model architecture
torch.save(model, '/Users/mac/Desktop/Bio_meta_tran/Mollytest/puffmodel/model_architecture.pt')

# Save just the state dict (recommended for version compatibility)
torch.save(model.state_dict(), '/Users/mac/Desktop/Bio_meta_tran/Mollytest/puffmodel/model_state_dict.pt')

# Optional: Save model configuration if it's a transformer model
if hasattr(model, 'config'):
    with open('/Users/mac/Desktop/Bio_meta_tran/Mollytest/puffmodel/model_config.pkl', 'wb') as f:
        pickle.dump(model.config, f)

print("Model saved successfully!")