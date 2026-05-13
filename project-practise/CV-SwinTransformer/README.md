# CV-SwinTransformer

这个目录目前是一个**偏模块级**的练习项目，重点不是完整训练流程，而是实现和验证 Swin Transformer 中的关键计算模块。

## 当前内容

| 文件 | 作用 |
| --- | --- |
| `attention.py` | 实现窗口划分、窗口还原、窗口注意力和 `SwinBlock` |
| `test.py` | 用随机张量快速验证 `SwinBlock` 的输出 shape |

## 当前完成度

目前更像是一个“局部机制验证”目录，而不是完整的图像分类工程。

也就是说，这里重点练习的是：
- `window_partition`
- `window_reverse`
- `WindowAttention`
- `shifted window` 对应的 attention mask

## 运行方式

在当前目录下可以直接运行：

```powershell
python .\test.py
```

它会构造一个随机输入，检查 `SwinBlock` 的输出维度是否符合预期。

## 适合学习的点

这个目录适合用来理解：
- 局部窗口注意力为什么能降低计算量
- shift 之后为什么需要 attention mask
- 从 `(B, H*W, C)` 到窗口表示再还原回去的张量变形过程

## 后续可以继续补的方向

如果你后续想把这里扩展成更完整的项目，可以考虑继续增加：
- patch embedding
- patch merging
- 多 stage 的 Swin 主干
- 分类头
- 训练脚本与数据加载器

