import h5py
import json

with h5py.File('C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/model2_train/Result Train/fruit_model.h5', 'r') as f:
    config = f.attrs.get('model_config')
    if config:
        print(json.loads(config))
    else:
        print("No model configuration found.")
