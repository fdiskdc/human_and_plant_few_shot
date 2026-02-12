import pandas as pd
import numpy as np
import os

# H5文件路径
h5_file = 'npy/multirm/MultiRM_data.h5'

# 输出目录
output_dir = 'npy/multirm'
os.makedirs(output_dir, exist_ok=True)

# 所有key
keys = ['/test_in_nucleo', '/test_out', '/train_in_nucleo', '/train_out', '/valid_in_nucleo', '/valid_out']

for key in keys:
    print(f"Processing {key}...")

    # 使用pandas读取H5数据
    data = pd.read_hdf(h5_file, key=key).values

    print(f"  Shape: {data.shape}, dtype: {data.dtype}")

    # 确定输出文件名
    output_file = os.path.join(output_dir, key.lstrip('/') + '.npy')

    if key.endswith('_in_nucleo'):
        # 字符数组，保存为s1类型
        np.save(output_file, np.array(data, dtype='|S1'))
        print(f"  Saved as s1 to {output_file}")
    elif key.endswith('_out'):
        # 数字数组，保存为int类型
        np.save(output_file, np.array(data, dtype=int))
        print(f"  Saved as int to {output_file}")

print("\nAll files saved successfully!")
