import os
import re

BASE = "/home/dc/vscode/vscode20260406/rgcnformer_sum"
HDR = "/home/dc/vscode/vscode20260406/rgcnformer_sum/.omo/hdr"
JOBS = [
    (f"{BASE}/model/EvoRMD/Script/dataset.py",         f"{HDR}/dataset.txt",       "simple"),
    (f"{BASE}/model/EvoRMD/Script/utils.py",           f"{HDR}/utils.txt",         "simple"),
    (f"{BASE}/model/EvoRMD/Script/main.py",            f"{HDR}/main.txt",          "strip_main"),
    (f"{BASE}/model/EvoRMD/Script/preprocess_data.py", f"{HDR}/preprocess.txt",    "simple"),
    (f"{BASE}/model/EvoRMD/Script/embedding.py",       f"{HDR}/embedding.txt",     "strip_embed"),
    (f"{BASE}/model/EvoRMD/Script/downsampling.py",    f"{HDR}/downsampling.txt",  "simple"),
    (f"{BASE}/model/EvoRMD/Script/train_val_test.py",  f"{HDR}/tvt.txt",           "simple"),
    (f"{BASE}/model/EvoRMD/Script/model.py",           f"{HDR}/model.txt",         "strip_docstring"),
    (f"{BASE}/plot_zero_fewshot_analysis.R",           f"{HDR}/pzfa.txt",          "r_shebang"),
    (f"{BASE}/plot_zero_fewshot_export.R",             f"{HDR}/pzfe.txt",          "r_shebang"),
    (f"{BASE}/ipynb/plot_flops_auc.R",                 f"{HDR}/pfa.txt",           "r_simple"),
    (f"{BASE}/ipynb/plot_ablation_scatter.R",          f"{HDR}/pas.txt",           "r_simple"),
    (f"{BASE}/ipynb/plot_ablation_grouped.R",          f"{HDR}/pag.txt",           "r_simple"),
    (f"{BASE}/ipynb/plot_ablation_grouped_v2.R",       f"{HDR}/pag2.txt",          "r_simple"),
    (f"{BASE}/ipynb/para_comp/generate_plot.R",        f"{HDR}/pcomp.txt",         "r_shebang"),
    (f"{BASE}/ipynb/len_comp/auc_stability_comparison.R", f"{HDR}/aucst.txt",      "r_simple"),
]
def strip_leading_blank_lines(lines):
    i = 0
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    return lines[i:]


def strip_main_comment(lines):
    if lines and lines[0].strip() == '# main.py':
        return strip_leading_blank_lines(lines[1:])
    return lines


def strip_embed_comment(lines):
    out = list(lines)
    for _ in range(2):
        if out and (out[0].lstrip().startswith('#') or out[0].strip() == ''):
            out = out[1:]
        else:
            break
    return strip_leading_blank_lines(out)
def strip_chinese_class_docstring(content):
    tq = chr(34)*3
    needle = "PyTorch Dataset"
    lines = content.splitlines(keepends=True)
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == tq and i+1 < len(lines) and needle in (lines[i+1] if i+1 < len(lines) else tq):
            new_lines.append(line)
            i += 1
            while i < len(lines):
                if lines[i].strip() == tq:
                    new_lines.append(lines[i])
                    i += 1
                    break
                i += 1
            continue
        new_lines.append(line)
        i += 1
    return chr(34).join(new_lines)
def process_file(path, header_path, mode):
    with open(header_path, 'r', encoding='utf-8') as f:
        header = f.read()
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if mode == 'simple':
        lines = content.splitlines(keepends=True)
        body = ''.join(strip_leading_blank_lines(lines))
    elif mode == 'strip_main':
        lines = content.splitlines(keepends=True)
        body = ''.join(strip_main_comment(lines))
    elif mode == 'strip_embed':
        lines = content.splitlines(keepends=True)
        body = ''.join(strip_embed_comment(lines))
    elif mode == 'strip_docstring':
        body = strip_chinese_class_docstring(content)
    elif mode == 'r_shebang':
        lines = content.splitlines(keepends=True)
        if lines and lines[0].startswith('#!'):
            keep = [lines[0]]
            rest = lines[1:]
            i = 0
            while i < len(rest) and (rest[i].strip() == '' or rest[i].lstrip().startswith('#')):
                i += 1
            body = ''.join(keep + rest[i:])
        else:
            body = content
    elif mode == 'r_simple':
        lines = content.splitlines(keepends=True)
        i = 0
        while i < len(lines) and (lines[i].strip() == '' or lines[i].lstrip().startswith('#')):
            i += 1
        body = ''.join(lines[i:])
    else:
        body = content
    if not body.startswith(chr(10)):
        sep = chr(10)
    else:
        sep = ''
    new_content = header + sep + body
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return len(new_content)

for path, header_path, mode in JOBS:
    n = process_file(path, header_path, mode)
    print('OK', path, n)
