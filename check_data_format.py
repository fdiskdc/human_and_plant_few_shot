"""
Check the actual format of multirm data
"""
import numpy as np

# Check a single file
data = np.load('npy/multirm/51split/split/train_in.npy', allow_pickle=True)

print(f"Total samples: {len(data)}")
print(f"First sample type: {type(data[0])}")
print(f"First sample shape: {data[0].shape if hasattr(data[0], 'shape') else 'N/A'}")

# Check first few samples
for i in range(min(5, len(data))):
    if isinstance(data[i], np.ndarray):
        if data[i].dtype.type == np.bytes_ or data[i].dtype.kind == 'S':
            # Bytes array, need to decode
            seq = data[i].tobytes().decode('ascii', errors='ignore')
            print(f"Sample {i}: bytes, decoded length = {len(seq)}")
        else:
            print(f"Sample {i}: shape={data[i].shape}, dtype={data[i].dtype}")
    else:
        print(f"Sample {i}: type={type(data[i])}")