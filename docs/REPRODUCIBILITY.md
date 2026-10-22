# 实验复现说明

本研究比较 MNIST 全连接和卷积降噪自编码器。正式结果固定为 19 种配置 × 3 个种子（42、2026、3407），共 57 次训练；另有 3 次验证集选权重预实验。测试包含 57 个模型及两个非学习参照在 7 个噪声水平下的 413 组评价。

## 环境与依赖

实际正式运行环境：Python 3.13.15、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、NumPy 2.1.3、scikit-image 0.25.2、SciPy 1.16.3、Matplotlib 3.10.0、PyYAML 6.0.3，Tesla T4。CUDA 12.8 指 PyTorch 构建版本。正式日志没有记录 pandas 和 tqdm 的精确版本；兼容范围见 requirements-runtime.txt。

本地功能检查环境名为 mnist_py310：Python 3.10.22、torch 2.10.0+cpu、torchvision 0.25.0+cpu、NumPy 2.2.6、pandas 2.3.3、scikit-image 0.25.2、SciPy 1.15.3、Matplotlib 3.10.9、tqdm 4.70.1、PyYAML 6.0.3。该环境仅用于检查和制图。Python 3.10 可以运行项目代码；跨设备和依赖版本不保证逐位一致。

安装适配平台的 PyTorch/torchvision 后，再按 requirements-runtime.txt 安装其余依赖。Kaggle 已有 GPU 版本时直接使用，不应覆盖安装 CPU wheel。公开项目不会自动创建 conda 环境。

## 从零准备与训练

在项目根目录执行；完整队列需要 GPU，CPU 检查不代替正式训练。

```bash
python src/prepare_data.py
# 如果系统代理导致下载不完整，可改用：python src/prepare_data.py --no-proxy
python src/check_pipeline.py
python src/run_experiments.py --stage all --seeds 42 2026 3407 --device cuda --list
python src/run_experiments.py --stage all --seeds 42 2026 3407 --device cuda
```

官方 60,000 张训练图以种子 2026 固定划分为 55,000/5,000；官方 10,000 张测试图保持独立。训练用动态截断高斯噪声；验证与测试噪声按图像 ID 固定，噪声种子分别为 9300、19300。

每组训练 20 轮、batch 128、Adam lr=0.001，默认 betas=(0.9,0.999)、eps=1e-8、weight_decay=0；无调度器。DataLoader num_workers=0，训练打乱、验证不打乱；开启确定性算法。主结果固定使用 last.pt，而非按测试分数选轮次。梯度权重候选 0.01、0.05、0.1，仅根据验证 SSIM 选择，最终 λ=0.1。

队列会跳过已完整完成的任务，在轮次边界恢复未完成任务；应保留完整 run、数据划分和 gradient_selection.json。中途关闭或丢失 Kaggle 工作目录后，必须先从下载备份恢复这些文件。

## 冻结后的测试与汇总

```bash
python src/evaluate.py --split test --device cuda
python src/summarize.py --split test --device cpu
python src/collect_results.py
```

测试入口会检查正式训练是否齐全。测试 σ 为 0、0.1、0.2、0.3、0.4、0.5、0.6；同图同噪声在模型间共享。高斯滤波宽度仅由验证集选择。SSIM 使用 data_range=1、win_size=7、均匀窗口和样本协方差；PSNR 逐图计算后平均。跨种子波动为样本标准差，不是置信区间。

## 论文图表与结果范围

results/ 提供原始精度的总体汇总、逐种子记录、配对消融和逐图差值摘要；figures/ 提供可编辑 SVG。负向残差结果被保留，没有在测试后删除失败配置。

src/build_paper_assets.py 和 src/refine_paper_figures.py 使用完整本地实验归档生成论文图表。前者还核对归档 SHA256，后者仅读取冻结逐图指标，不训练或推理。运行时需要先恢复原始结果归档和 outputs/paper_assets；仅下载轻量仓库无法重建逐图直方图。完整权重及逐图归档不随 Git 上传，也没有公开下载地址。教师 Word 模板和个人封面不属于公开代码仓库。

本研究只涉及 28×28 灰度数字与截断高斯合成噪声，不代表自然彩色图像或真实传感器噪声的结果。条件模型获得真实 σ，不属于未知噪声估计。
