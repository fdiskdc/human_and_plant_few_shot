"""
ipynb/view_npz.py - 查看 .npz 文件(脚本版) / View .npz files (script version)

查看 .npz 文件(脚本版)
View .npz files (script version)

功能模块 / Modules:
- 命令行版 .npz 查看器
- (详见源代码 / see source code)

输入 / Inputs:
- 命令行参数(超参、数据路径等)/ CLI args (hyperparams, data paths, etc.)
- 配置文件(json/yaml) / config files (json/yaml)
- 上一阶段产物(.npy/.pt/.json)/ prior-stage outputs

输出 / Outputs:
- 产物文件(.pt/.json/.npy/.png/.pdf)/ output files
- 标准输出 / 日志 / stdout / logs

数据流 / Data Flow:
1. 加载数据/配置 / Load data/config
2. 主循环(训练/推理/分析)/ Main loop
3. 保存/导出 / Save/export

相关文件 / Related Files:
- 调用 / Calls: view_npz.py、prepare_umap_*.py
- 被调用 / Called by: view_npz.py、prepare_umap_*.py 相关的训练/推理/分析脚本

使用示例 / Usage Example:
    python ipynb/view_npz.py

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""

import os
import numpy as np


def view_npz_files(data_dir='data'):
    """
    读取指定目录下的所有 npz 文件并打印每个 key 的数组 shape
    """
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

    for npz_file in sorted(npz_files):
        npz_path = os.path.join(data_dir, npz_file)
        print(f'\n{"=" * 60}')
        print(f'File: {npz_file}')
        print(f'{"=" * 60}')

        data = np.load(npz_path)

        for key in sorted(data.keys()):
            arr = data[key]
            print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')

        data.close()


if __name__ == '__main__':
    view_npz_files()
