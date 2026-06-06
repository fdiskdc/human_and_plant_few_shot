# 双语 docstring 编写指南 / Bilingual Docstring Guide

> 本文档定义 RGCNFormer 项目中 Python 函数级双语 docstring 的标准格式。
> This document defines the standard format for function-level bilingual docstrings in the RGCNFormer project.

## 1. 原则 / Principles

- **不修改 / Do NOT modify** 已有文件级头注释（header docstring / module docstring）
- **不修改 / Do NOT modify** 任何功能性代码
- 中英双语并列在同一 docstring 内（不创建独立页面）
- 使用 **Google 风格** docstring（不适用 NumPy 风格）

- Do NOT modify existing file-level header comments
- Do NOT modify any functional code
- Chinese and English are side-by-side inside the same docstring
- Use **Google style** docstrings (not NumPy)

## 2. 完整模板 / Full Template

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    [中文简述一句话] / [English one-line description]

    [中文详细描述函数功能、算法、注意点] /
    [Detailed description of function purpose, algorithm, caveats]

    Args / 参数:
        param1 (type): [中文描述] / [English description]
        param2 (type, optional): [中文描述] / [English description]. Defaults to default.

    Returns / 返回:
        type: [中文描述] / [English description]

    Raises / 异常:
        ValueError: [中文描述] / [English description]
        RuntimeError: [中文描述] / [English description]

    Calls / 调用:
        - other_function(): [中文描述] / [English description]
        - AnotherClass.method(): [中文描述] / [English description]

    Called by / 被调用:
        - parent_function(): [中文描述] / [English description]

    Example / 示例:
        >>> result = function_name(1, "test")
        >>> print(result)
        42

    Note / 备注:
        [任何额外说明] / [Any additional notes]
    """
    pass
```

## 3. 类模板 / Class Template

```python
class MyClass:
    """
    [中文简述] / [English brief description]

    [中文详细说明] / [Detailed description]

    Attributes / 属性:
        attr1 (type): [中文描述] / [English description]
        attr2 (type): [中文描述] / [English description]

    Example / 示例:
        >>> obj = MyClass(attr1=1, attr2="test")
        >>> result = obj.method(42)
    """

    def __init__(self, attr1: int, attr2: str):
        """
        [中文简述] / [English brief].

        Args / 参数:
            attr1 (int): [中文] / [English]
            attr2 (str): [中文] / [English]
        """
        self.attr1 = attr1
        self.attr2 = attr2
```

## 4. 简化模板（短函数） / Short Template (Short Functions)

```python
def simple_func(x: int) -> bool:
    """[中文简述] / [English brief].

    Args / 参数:
        x (int): [中文输入] / [English input].

    Returns / 返回:
        bool: [中文结果] / [English result].
    """
    return x > 0
```

## 5. 方法 docstring / Method Docstring

```python
class Trainer:
    def train_epoch(self, dataloader) -> float:
        """
        [中文] 训练一个 epoch / Train one epoch.

        Args / 参数:
            dataloader (DataLoader): [中文] 数据加载器 / [English] data loader.

        Returns / 返回:
            float: [中文] 平均损失 / [English] average loss.
        """
        pass
```

## 6. 注意事项 / Cautions

1. **不要删除已有 docstring**：如果函数已有简短英文 docstring，**扩展**为双语，不要删除。
2. **保持代码风格一致**：缩进 4 空格，行宽不超过 100 字符（中文可稍宽）。
3. **避免行内 docstring**（如 `"""short"""` 单行），除非是 1 行的简单函数。
4. **示例代码**：使用 `>>>` doctest 风格（仅在示例可执行时使用）。
5. **中英文之间用空格分隔**：例如 `"中文描述 / English description"`，便于阅读。

1. **Do NOT delete existing docstrings** — extend short English ones to bilingual.
2. **Match the code style**: 4-space indent, line width ≤ 100 chars (Chinese may be slightly wider).
3. **Avoid one-line docstrings** unless the function is trivially short.
4. **Use `>>>` doctest style** for runnable examples.
5. **Separate Chinese and English with `/` and a space**: e.g. `"中文 / English"`.

## 7. 必填字段 / Required Fields

| 字段 / Field | 必填 / Required | 适用 / Applies to |
|---|---|---|
| 一句话简述 / One-liner | ✅ | 所有函数 / all functions |
| Args / 参数 | ✅ | 有参数时 / when there are args |
| Returns / 返回 | ✅ | 有返回值时 / when returning |
| Raises / 异常 | ⭕ (推荐 / recommended) | 可能抛异常时 / when raising |
| Calls / 调用 | ⭕ (推荐) | 调用其他函数时 / when calling others |
| Example / 示例 | ❌ (可选) | 复杂函数 / complex functions |

## 8. 反例 / Bad Examples

❌ **不要做 / Do NOT do**:

```python
# 1. 删除已有 docstring
def f(x):
    """existing docstring"""  # ← 保留并扩展，不要删除
    return x

# 2. 修改代码逻辑
def f(x):
    """新增 docstring"""
    return x + 1  # ← 不要改

# 3. 不分中英文
def f(x):
    """add one to x"""  # ← 缺少中文
    return x + 1
```

## 9. 完整示例 / Complete Example

```python
def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    计算分类 F1 分数 / Calculate classification F1 score.

    支持二分类、多分类、多标签三种模式，由 `average` 决定聚合方式。
    Supports binary, multi-class, and multi-label modes, aggregated via `average`.

    Args / 参数:
        y_true (np.ndarray): [中文] 真实标签 (N,) 或 (N, C) / [English] true labels.
        y_pred (np.ndarray): [中文] 预测标签或概率 / [English] predicted labels or probs.
        average (str, optional): [中文] 聚合方式 ('micro'/'macro'/'samples') / [English]
            aggregation strategy. Defaults to 'macro'.

    Returns / 返回:
        float: [中文] F1 分数 / [English] F1 score in [0, 1].

    Raises / 异常:
        ValueError: [中文] 当 y_true 与 y_pred 形状不匹配 / [English] shape mismatch.

    Calls / 调用:
        - sklearn.metrics.f1_score(): [中文] 底层 F1 计算 / [English] underlying F1 calc.

    Called by / 被调用:
        - train_human_mrmodn.evaluate(): [中文] 训练评估 / [English] training eval.
    """
    from sklearn.metrics import f1_score
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return f1_score(y_true, y_pred, average=average)
```

## 10. 参考 / References

- Google Python Style Guide: https://google.github.io/styleguide/pyguide.html#383-functions-and-methods
- PEP 257 (Docstring Conventions): https://peps.python.org/pep-0257/
