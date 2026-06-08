import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageTk
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

DEFAULT_DATA_YAML = "data.yaml"
DEFAULT_WEIGHTS = "best.pt"

NORMAL_CLASS_NAME = "正常"
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
import platform

# 配置 matplotlib 中文字体
if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # Windows 常用中文字体
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']  # Linux/macOS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class YoloDefectDetector:
    def __init__(self, weights: str, data_yaml: str, device: Optional[str] = None):
        self.weights = weights
        self.data_yaml = Path(data_yaml)
        if self.weights == DEFAULT_WEIGHTS and not self.data_yaml.exists():
            self.data_yaml = Path(get_resource_path(DEFAULT_DATA_YAML))
        if self.weights == DEFAULT_WEIGHTS and not Path(self.weights).exists():
            self.weights = get_resource_path(DEFAULT_WEIGHTS)
        if self.data_yaml == Path(DEFAULT_DATA_YAML) and not self.data_yaml.exists():
            self.data_yaml = Path(get_resource_path(DEFAULT_DATA_YAML))
        self.data_config = self._load_data_yaml(self.data_yaml)
        self.class_names = list(self.data_config.get("names", []))
        self.class_names.append(NORMAL_CLASS_NAME)
        self.model = YOLO(self.weights)
        self.device = device

    @staticmethod
    def _load_data_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data

    def _draw_text(self, image: np.ndarray, text: str, position: Tuple[int, int], font_size: int = 20) -> np.ndarray:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        font = None
        try:
            if platform.system() == "Windows":
                font = ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", font_size)
            else:
                font = ImageFont.truetype("/usr/share/fonts/truetype/wenquanyi/WeiRuanYaHei.ttf", font_size)
        except Exception:
            try:
                if platform.system() == "Windows":
                    font = ImageFont.truetype("C:\\Windows\\Fonts\\simhei.ttf", font_size)
                else:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

        try:
            text_width, text_height = font.getsize(text)
        except Exception:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        x, y = position
        background = (0, 0, 0)
        draw.rectangle([x, y, x + text_width + 4, y + text_height + 4], fill=background)
        draw.text((x + 2, y + 2), text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _load_font(self, font_size: int = 20):
        try:
            if platform.system() == "Windows":
                return ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", font_size)
            return ImageFont.truetype("/usr/share/fonts/truetype/wenquanyi/WeiRuanYaHei.ttf", font_size)
        except Exception:
            try:
                if platform.system() == "Windows":
                    return ImageFont.truetype("C:\\Windows\\Fonts\\simhei.ttf", font_size)
                return ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", font_size)
            except Exception:
                try:
                    return ImageFont.truetype("arial.ttf", font_size)
                except Exception:
                    return ImageFont.load_default()

    def predict_image_pil(self, source: str, conf: float = 0.25, iou: float = 0.45):
        results = self.model.predict(source=source, conf=conf, iou=iou, imgsz=640, max_det=100)
        result = results[0]
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {source}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        font = self._load_font(font_size=20)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            label = self.class_names[cls_id]
            text = f"{label} {score:.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x1
            text_y = max(y1 - text_height - 6, 0)
            draw.rectangle([x1, y1, x2, y2], outline=(8, 143, 255), width=3)
            draw.rectangle([text_x, text_y, text_x + text_width + 6, text_y + text_height + 6], fill=(0, 0, 0))
            draw.text((text_x + 3, text_y + 3), text, fill=(255, 255, 255), font=font)

        return pil_image

    def train(self, epochs: int = 50, imgsz: int = 640, batch: int = 16, augment: bool = False, lr0: Optional[float] = None):
        print(f"[TRAIN] start training with weights={self.weights}, data={self.data_yaml}")
        train_args = {
            "data": str(self.data_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": "runs/train",
            "name": "yolov8_defect",
            "exist_ok": True,
            "augment": augment,
        }
        if self.device:
            train_args["device"] = self.device
        if lr0 is not None:
            train_args["lr0"] = lr0
        self.model.train(**train_args)
        best_path = Path("runs/train/yolov8_defect/weights/best.pt")
        if best_path.exists():
            print(f"[TRAIN] training finished. Best weights saved to: {best_path}")
        else:
            print("[TRAIN] training finished. Best weights not found; please inspect runs/train/yolov8_defect/weights")

    def predict_image(self, source: str, conf: float = 0.25, iou: float = 0.45, save_path: Optional[str] = None):
        results = self.model.predict(source=source, conf=conf, iou=iou, imgsz=640, max_det=100)
        result = results[0]
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {source}")

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            label = self.class_names[cls_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), (8, 143, 255), 2)
            text = f"{label} {score:.2f}"
            text_y = max(y1 - 30, 0)
            image = self._draw_text(image, text, (x1, text_y), font_size=20)

        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), image)
            print(f"[INFER] saved result to {out_path}")
            return str(out_path)
        return image

    def evaluate_classification(self, test_dir: str, conf: float = 0.25, iou: float = 0.45, output_dir: Optional[str] = None):
        test_path = Path(test_dir)
        if not test_path.exists():
            test_path = self.data_yaml.parent / test_path
        if not test_path.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        image_paths = sorted(
            [p for p in test_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        )
        if not image_paths:
            raise ValueError(f"No supported images found in {test_dir}")

        actual_labels = []
        predicted_labels = []
        rows = []
        print(f"[EVAL] evaluating {len(image_paths)} images in {test_dir}")

        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"[WARN] cannot open image {image_path}, skip")
                continue
            height, width = img.shape[:2]
            actual = self._load_image_label(image_path, width, height)
            predicted = self._classify_image(image_path, conf, iou)
            actual_labels.append(actual)
            predicted_labels.append(predicted)
            rows.append({
                "image": str(image_path),
                "actual": self.class_names[actual],
                "predicted": self.class_names[predicted],
            })

        cm = confusion_matrix(actual_labels, predicted_labels, labels=list(range(len(self.class_names))))
        report = classification_report(actual_labels, predicted_labels, labels=list(range(len(self.class_names))), target_names=self.class_names, zero_division=0)

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "classification_results.csv", index=False, encoding="utf-8-sig")
            self._save_confusion_matrix(cm, out_dir / "confusion_matrix.png")
            with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            print(f"[EVAL] saved evaluation outputs to {out_dir}")

        print("\n=== Confusion Matrix ===")
        print(cm)
        print("\n=== Classification Report ===")
        print(report)
        return cm, report

    def _get_label_path(self, image_path: Path) -> Path:
        # data/images/train and data/images/val 使用独立 labels 目录保存标注
        parts = list(image_path.parts)
        if "images" in parts:
            idx = parts.index("images")
            label_parts = list(parts[:idx]) + ["labels"] + list(parts[idx + 1:])
            label_path = Path(*label_parts).with_suffix(".txt")
            if label_path.exists():
                return label_path

        adjacent_label = image_path.with_suffix(".txt")
        if adjacent_label.exists():
            return adjacent_label

        return label_path if 'label_path' in locals() else adjacent_label

    def _load_image_label(self, image_path: Path, width: int, height: int) -> int:
        label_path = self._get_label_path(image_path)
        if not label_path.exists():
            return len(self.class_names) - 1

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return len(self.class_names) - 1

        classes = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                classes.append(int(parts[0]))
            except ValueError:
                continue
        if not classes:
            return len(self.class_names) - 1
        return int(max(set(classes), key=classes.count))

    def _classify_image(self, image_path: Path, conf: float, iou: float) -> int:
        results = self.model.predict(source=str(image_path), conf=conf, iou=iou, imgsz=640, max_det=100)
        result = results[0]
        if not result.boxes:
            return len(self.class_names) - 1
        best_box = None
        best_score = -1.0
        for box in result.boxes:
            score = float(box.conf[0].item())
            if score > best_score:
                best_score = score
                best_box = box
        if best_box is None:
            return len(self.class_names) - 1
        return int(best_box.cls[0].item())

    def _save_confusion_matrix(self, cm: np.ndarray, path: Path):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("分类混淆矩阵")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def get_resource_path(relative_path: str) -> str:
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent
    return str(base_path / relative_path)


def parse_args() -> argparse.Namespace:
    if sys.stderr is None:
        sys.stderr = sys.__stderr__
    if sys.stdout is None:
        sys.stdout = sys.__stdout__

    parser = argparse.ArgumentParser(description="YOLOv8 罐盖缺陷检测训练与测试脚本")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练 YOLOv8 模型")
    train_parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="数据集 yaml 配置")
    train_parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="初始权重或预训练模型")
    train_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    train_parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    train_parser.add_argument("--batch", type=int, default=16, help="批大小")
    train_parser.add_argument("--augment", action="store_true", help="启用训练时的数据增强")
    train_parser.add_argument("--device", type=str, default=None, help="设备，例如 cpu 或 0")
    train_parser.add_argument("--lr0", type=float, default=None, help="初始学习率")

    eval_parser = subparsers.add_parser("eval", help="对测试集做分类评估")
    eval_parser.add_argument("--weights", type=str, required=True, help="训练好的模型权重")
    eval_parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="数据集 yaml 配置")
    eval_parser.add_argument("--test-dir", type=str, default=None, help="测试集图片目录；缺省使用 yaml 中 val 路径")
    eval_parser.add_argument("--conf", type=float, default=0.25, help="推理置信度阈值")
    eval_parser.add_argument("--iou", type=float, default=0.45, help="推理 IoU 阈值")
    eval_parser.add_argument("--output", type=str, default="runs/eval", help="结果输出目录")
    eval_parser.add_argument("--device", type=str, default=None, help="设备，例如 cpu 或 0")

    infer_parser = subparsers.add_parser("infer", help="对单张图像或文件夹做缺陷检测并保存可视化结果")
    infer_parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="数据集 yaml 配置")
    infer_parser.add_argument("--weights", type=str, required=True, help="训练好的模型权重")
    infer_parser.add_argument("--source", type=str, required=True, help="输入图像或目录")
    infer_parser.add_argument("--output", type=str, default="runs/infer", help="可视化结果保存目录")
    infer_parser.add_argument("--conf", type=float, default=0.25, help="推理置信度阈值")
    infer_parser.add_argument("--iou", type=float, default=0.45, help="推理 IoU 阈值")
    infer_parser.add_argument("--device", type=str, default=None, help="设备，例如 cpu 或 0")

    gui_parser = subparsers.add_parser("gui", help="启动交互式界面")
    gui_parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="数据集 yaml 配置")
    gui_parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="初始权重或预训练模型")
    gui_parser.add_argument("--device", type=str, default=None, help="设备，例如 cpu 或 0")

    if len(sys.argv) == 1:
        return argparse.Namespace(command="gui", data=DEFAULT_DATA_YAML, weights=DEFAULT_WEIGHTS, device=None)
    return parser.parse_args()


def choose_path(prompt: str, folder: bool = False) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        if folder:
            path = filedialog.askdirectory(title=prompt)
        else:
            path = filedialog.askopenfilename(title=prompt, filetypes=[("All files", "*.*")])
        root.update()
        root.destroy()
        return path or None
    except Exception:
        return None


def run_gui_app(data_yaml: str, weights: str, device: Optional[str]):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("YOLOv8 缺陷检测 GUI")
    root.geometry("1024x800")
    root.resizable(True, True)

    weights_var = tk.StringVar(value=weights)
    data_var = tk.StringVar(value=data_yaml)
    source_var = tk.StringVar(value="")
    output_var = tk.StringVar(value="runs/infer")
    test_dir_var = tk.StringVar(value="")
    eval_output_var = tk.StringVar(value="runs/eval")
    conf_var = tk.DoubleVar(value=0.25)
    iou_var = tk.DoubleVar(value=0.45)
    status_var = tk.StringVar(value="请选择模型、数据和文件/目录，然后点击运行。")

    image_paths = []
    current_index = 0
    detector = None
    photo_image = None
    report_text_widget = None

    def browse_file():
        path = filedialog.askopenfilename(title="选择图像文件", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if path:
            source_var.set(path)
            output_var.set("runs/infer")

    def browse_folder():
        path = filedialog.askdirectory(title="选择图像目录")
        if path:
            source_var.set(path)
            output_var.set("runs/infer")

    def browse_weights():
        path = filedialog.askopenfilename(title="选择模型权重文件", filetypes=[("PyTorch Weights", "*.pt *.pth"), ("All Files", "*.*")])
        if path:
            weights_var.set(path)

    def browse_data():
        path = filedialog.askopenfilename(title="选择数据 yaml 文件", filetypes=[("YAML Files", "*.yaml *.yml"), ("All Files", "*.*")])
        if path:
            data_var.set(path)

    def browse_output():
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            output_var.set(path)

    def browse_eval_output():
        path = filedialog.askdirectory(title="选择评估输出目录")
        if path:
            eval_output_var.set(path)

    def browse_test_dir():
        path = filedialog.askdirectory(title="选择测试集目录")
        if path:
            test_dir_var.set(path)

    def load_source_list():
        nonlocal image_paths, current_index
        source = source_var.get().strip()
        if not source:
            image_paths = []
            return
        source_path = Path(source)
        if source_path.is_dir():
            image_paths = sorted([str(p) for p in source_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
        elif source_path.is_file():
            image_paths = [str(source_path)]
        else:
            image_paths = []
        current_index = 0

    def update_controls():
        prev_button.config(state=("normal" if current_index > 0 else "disabled"))
        next_button.config(state=("normal" if current_index + 1 < len(image_paths) else "disabled"))
        if image_paths:
            status_var.set(f"当前: {current_index + 1}/{len(image_paths)}  {Path(image_paths[current_index]).name}")
        else:
            status_var.set("没有可检测的图像，请选择文件或文件夹。")

    def show_image(pil_image: Image.Image):
        nonlocal photo_image
        max_w, max_h = 960, 560
        width, height = pil_image.size
        scale = min(max_w / width, max_h / height, 1.0)
        display_image = pil_image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        photo_image = ImageTk.PhotoImage(display_image)
        image_label.config(image=photo_image)
        image_label.image = photo_image

    def detect_and_display(path: str):
        nonlocal detector
        try:
            if detector is None or detector.weights != weights_var.get() or str(detector.data_yaml) != data_var.get():
                detector = YoloDefectDetector(weights_var.get(), data_var.get(), device=device)
            pil_result = detector.predict_image_pil(path, conf=conf_var.get(), iou=iou_var.get())
            output_dir = output_var.get().strip()
            if output_dir:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                save_path = out_dir / Path(path).name
                pil_result.convert("RGB").save(save_path)
            show_image(pil_result)
            status_var.set(f"检测完成: {Path(path).name}  ({current_index + 1}/{len(image_paths)})")
            return pil_result
        except Exception as exc:
            messagebox.showerror("推理错误", str(exc))
            status_var.set(f"推理失败: {exc}")
            return None

    def on_eval():
        nonlocal detector, report_text_widget
        if not weights_var.get().strip() or not data_var.get().strip():
            messagebox.showwarning("参数缺失", "请先指定模型权重和 data.yaml 文件。")
            return
        test_dir = test_dir_var.get().strip()
        if not test_dir:
            data_config = YoloDefectDetector._load_data_yaml(Path(data_var.get()))
            test_dir = data_config.get("val", "")
        if not test_dir:
            messagebox.showwarning("参数缺失", "请指定测试集目录或在 data.yaml 中设置 val 路径。")
            return
        output_dir = eval_output_var.get().strip() or "runs/eval"
        try:
            detector = YoloDefectDetector(weights_var.get(), data_var.get(), device=device)
            cm, report = detector.evaluate_classification(test_dir, conf=conf_var.get(), iou=iou_var.get(), output_dir=output_dir)
            status_var.set(f"评估完成，结果已保存到 {output_dir}")
            if report_text_widget:
                report_text_widget.config(state=tk.NORMAL)
                report_text_widget.delete("1.0", tk.END)
                report_text_widget.insert(tk.END, report)
                report_text_widget.config(state=tk.DISABLED)
        except Exception as exc:
            messagebox.showerror("评估错误", str(exc))
            status_var.set(f"评估失败: {exc}")

    def on_run():
        nonlocal current_index
        if not weights_var.get().strip() or not data_var.get().strip():
            messagebox.showwarning("参数缺失", "请先指定模型权重和 data.yaml 文件。")
            return
        load_source_list()
        if not image_paths:
            messagebox.showwarning("无效输入", "请选择有效的图像文件或目录。")
            return
        current_index = 0
        update_controls()
        detect_and_display(image_paths[current_index])

    def on_prev():
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            update_controls()
            detect_and_display(image_paths[current_index])

    def on_next():
        nonlocal current_index
        if current_index + 1 < len(image_paths):
            current_index += 1
            update_controls()
            detect_and_display(image_paths[current_index])

    frame = tk.Frame(root)
    frame.pack(fill=tk.X, padx=12, pady=8)

    tk.Label(frame, text="模型权重:").grid(row=0, column=0, sticky=tk.W, pady=4)
    tk.Entry(frame, textvariable=weights_var, width=70).grid(row=0, column=1, sticky=tk.W, padx=4)
    tk.Button(frame, text="浏览...", command=browse_weights).grid(row=0, column=2, padx=4)

    tk.Label(frame, text="data.yaml:").grid(row=1, column=0, sticky=tk.W, pady=4)
    tk.Entry(frame, textvariable=data_var, width=70).grid(row=1, column=1, sticky=tk.W, padx=4)
    tk.Button(frame, text="浏览...", command=browse_data).grid(row=1, column=2, padx=4)

    tk.Label(frame, text="输入文件/目录:").grid(row=2, column=0, sticky=tk.W, pady=4)
    tk.Entry(frame, textvariable=source_var, width=70).grid(row=2, column=1, sticky=tk.W, padx=4)
    button_frame = tk.Frame(frame)
    button_frame.grid(row=2, column=2, padx=4)
    tk.Button(button_frame, text="选择文件", command=browse_file).pack(side=tk.TOP, fill=tk.X)
    tk.Button(button_frame, text="选择文件夹", command=browse_folder).pack(side=tk.TOP, fill=tk.X, pady=4)

    tk.Label(frame, text="输出目录:").grid(row=3, column=0, sticky=tk.W, pady=4)
    tk.Entry(frame, textvariable=output_var, width=70).grid(row=3, column=1, sticky=tk.W, padx=4)
    tk.Button(frame, text="浏览...", command=browse_output).grid(row=3, column=2, padx=4)

    tk.Label(frame, text="测试集目录:").grid(row=4, column=0, sticky=tk.W, pady=4)
    tk.Entry(frame, textvariable=test_dir_var, width=70).grid(row=4, column=1, sticky=tk.W, padx=4)
    tk.Button(frame, text="浏览...", command=browse_test_dir).grid(row=4, column=2, padx=4)

    tk.Label(frame, text="评估输出目录:").grid(row=5, column=0, sticky=tk.W, pady=4)
    tk.Entry(frame, textvariable=eval_output_var, width=70).grid(row=5, column=1, sticky=tk.W, padx=4)
    tk.Button(frame, text="浏览...", command=browse_eval_output).grid(row=5, column=2, padx=4)

    tk.Label(frame, text="置信度:").grid(row=6, column=0, sticky=tk.W, pady=4)
    tk.Entry(frame, textvariable=conf_var, width=10).grid(row=6, column=1, sticky=tk.W, padx=4)
    tk.Label(frame, text="IoU:").grid(row=6, column=1, sticky=tk.W, padx=140)
    tk.Entry(frame, textvariable=iou_var, width=10).grid(row=6, column=1, sticky=tk.W, padx=190)

    tk.Button(frame, text="运行检测", command=on_run, bg="#4CAF50", fg="white", width=16).grid(row=7, column=1, sticky=tk.W, pady=12)
    tk.Button(frame, text="评估测试集", command=on_eval, bg="#2196F3", fg="white", width=16).grid(row=7, column=2, sticky=tk.W, pady=12)

    nav_frame = tk.Frame(root)
    nav_frame.pack(fill=tk.X, padx=12)
    prev_button = tk.Button(nav_frame, text="上一张", command=on_prev, state=tk.DISABLED, width=12)
    next_button = tk.Button(nav_frame, text="下一张", command=on_next, state=tk.DISABLED, width=12)
    prev_button.pack(side=tk.LEFT, padx=4, pady=4)
    next_button.pack(side=tk.LEFT, padx=4, pady=4)

    status_label = tk.Label(root, textvariable=status_var, anchor="w")
    status_label.pack(fill=tk.X, padx=12, pady=4)

    image_panel = tk.Frame(root, bg="#222222")
    image_panel.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
    image_label = tk.Label(image_panel, bg="#222222")
    image_label.pack(expand=True)

    report_frame = tk.Frame(root)
    report_frame.pack(fill=tk.BOTH, expand=False, padx=12, pady=4)
    tk.Label(report_frame, text="评估报告输出:").pack(anchor=tk.W)
    report_text_widget = tk.Text(report_frame, height=10, wrap=tk.CHAR, state=tk.DISABLED)
    report_text_widget.pack(fill=tk.BOTH, expand=True)

    update_controls()
    root.mainloop()


def interactive_menu(data_yaml: str, weights: str, device: Optional[str]):
    print("=== YOLOv8 易拉罐盖缺陷检测交互界面 ===")
    while True:
        print("\n请选择操作:")
        print("1. 训练模型")
        print("2. 对测试集做评估")
        print("3. 单图/批量图像测试")
        print("q. 退出")
        choice = input("输入编号: ").strip().lower()
        if choice == "q":
            break
        if choice == "1":
            epochs = int(input("训练轮数 (默认50): ") or "50")
            imgsz = int(input("图像尺寸 (默认640): ") or "640")
            batch = int(input("批大小 (默认16): ") or "16")
            augment = input("是否启用增强? [y/N]: ").strip().lower() == "y"
            detector = YoloDefectDetector(weights, data_yaml, device=device)
            detector.train(epochs=epochs, imgsz=imgsz, batch=batch, augment=augment)
        elif choice == "2":
            detector = YoloDefectDetector(weights, data_yaml, device=device)
            data = YoloDefectDetector._load_data_yaml(Path(data_yaml))
            default_test_dir = data.get("val")
            print(f"测试集目录 (默认 {default_test_dir})")
            test_dir = input("输入测试集目录或直接回车: ").strip() or default_test_dir
            output_dir = input("结果输出目录 (默认 runs/eval): ").strip() or "runs/eval"
            detector.evaluate_classification(test_dir, output_dir=output_dir)
        elif choice == "3":
            source = choose_path("选择要检测的图像或目录", folder=False)
            if not source:
                source = input("请输入图像或目录路径: ").strip()
            output = input("输出目录 (默认 runs/infer): ").strip() or "runs/infer"
            detector = YoloDefectDetector(weights, data_yaml, device=device)
            source_path = Path(source)
            if source_path.is_dir():
                for img_path in sorted([p for p in source_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]):
                    out_path = Path(output) / img_path.name
                    detector.predict_image(str(img_path), save_path=str(out_path), conf=0.25, iou=0.45)
            else:
                out_path = Path(output) / source_path.name
                detector.predict_image(str(source_path), save_path=str(out_path), conf=0.25, iou=0.45)
        else:
            print("无效选项，请重试。")


def main():
    args = parse_args()
    if args.command == "train":
        detector = YoloDefectDetector(args.weights, args.data, device=args.device)
        detector.train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, augment=args.augment, lr0=args.lr0)
    elif args.command == "eval":
        detector = YoloDefectDetector(args.weights, args.data, device=args.device)
        data_cfg = detector.data_config
        test_dir = args.test_dir or data_cfg.get("val")
        if test_dir is None:
            raise ValueError("未在 data.yaml 中找到 val 路径，请通过 --test-dir 指定测试集目录")
        detector.evaluate_classification(test_dir, conf=args.conf, iou=args.iou, output_dir=args.output)
    elif args.command == "infer":
        detector = YoloDefectDetector(args.weights, args.data, device=args.device)
        source_path = Path(args.source)
        if source_path.is_dir():
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            for img_path in sorted([p for p in source_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]):
                out_path = output_dir / img_path.name
                detector.predict_image(str(img_path), conf=args.conf, iou=args.iou, save_path=str(out_path))
        else:
            out_path = Path(args.output) / source_path.name
            Path(args.output).mkdir(parents=True, exist_ok=True)
            detector.predict_image(str(source_path), conf=args.conf, iou=args.iou, save_path=str(out_path))
    elif args.command == "gui":
        run_gui_app(args.data, args.weights, args.device)
    else:
        print("未知命令")
        sys.exit(1)


if __name__ == "__main__":
    main()
