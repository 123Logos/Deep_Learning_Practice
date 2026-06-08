# 易拉罐盖缺陷检测 (YOLOv8)

该项目使用 Ultralytics YOLOv8 实现易拉罐盖缺陷检测、训练、测试与简单交互式推理。

## 目录结构

- `data/data.yaml`：YOLO 数据配置文件，包含训练集和验证集路径、类别名称。
- `data/images/train/`：训练图片及对应的 YOLO 标签文件。
- `data/images/val/`：验证/测试图片及标签文件。
- `main.py`：训练、评估、推理与交互式界面脚本。

## 安装依赖

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 训练模型

使用预训练权重 `yolov8n.pt` 开始训练：

```powershell
python main.py train --data data/data.yaml --weights yolov8n.pt --epochs 50 --batch 16 --imgsz 640 --augment
```

训练完成后，最佳模型权重会保存在：

- `runs/train/yolov8_defect/weights/best.pt`

## 评估测试集

对测试集（`data.yaml` 中 `val` 路径）做分类评估，输出混淆矩阵与分类报告：

```powershell
python main.py eval --weights runs/train/yolov8_defect/weights/best.pt --data data/data.yaml --output runs/eval
```

评估结果包含：

- `runs/eval/classification_results.csv`
- `runs/eval/confusion_matrix.png`
- `runs/eval/classification_report.txt`

## 单图 / 批量图像推理

对单张图片或整个文件夹进行缺陷检测并保存可视化结果：

```powershell
python main.py infer --weights runs/train/yolov8_defect/weights/best.pt --source data/images/val --output runs/infer
```

## 交互式界面

1. 运行 GUI 界面：

```powershell
python main.py gui
```

2. 在界面中：
- 指定 `weights` 模型权重文件
- 指定 `data.yaml` 数据配置文件
- 指定图像文件或图像目录
- 点击“运行检测”，即可在窗口中查看带缺陷框和类别标签的结果图像
- 如果当前图像没有检测到缺陷，则显示原图

## 打包成可执行文件

项目已经包含 `pyinstaller` 依赖，并且默认会把 `data.yaml` 与 `yolov8n.pt` 打包进 EXE。

可以使用以下命令生成单文件 EXE：

```powershell
uv run pyinstaller --noconfirm --onefile --windowed --add-data "data.yaml;." --add-data "yolov8n.pt;." main.py
```

生成完成后，运行目录中 `dist\main.exe` 即可直接启动 GUI 界面，并默认加载内置的权重与 yaml 配置。

也可以使用项目根目录中的一键打包脚本：

```powershell
build_exe.bat
```

## 说明

- `main.py` 中的 `eval` 模式以图像级分类方式计算混淆矩阵和分类指标。
- `infer` 模式会把检测结果绘制在原图上，并保存到指定输出目录。
- 训练时可使用 `--augment` 开启 YOLOv8 内置数据增强。

## 目标类别

当前 `data/data.yaml` 配置了 3 类缺陷：

- `凹边`
- `漏胶`
- `凸边`
- 另有 `正常` 类作为无缺陷样本的推理结果分类标签。
