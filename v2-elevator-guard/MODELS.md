# 模型与权重说明

本仓库不包含任何模型权重。三个环节的默认行为和替代方案如下。

## 1. 姿态估计（必需，二选一）

### YOLO11n-Pose —— 默认，零配置

```env
POSE_ENGINE=yolo11n
```

权重文件不存在时，ultralytics 会在首次推理时**自动下载** `yolo11n-pose.pt` 到当前工作目录。
只要机器能联网，`git clone` 之后不需要做任何配置就能跑通。

想指定路径（比如放到 `models/` 下）：

```env
YOLO11N_WEIGHTS=models/yolo11n-pose.pt
```

### MEC-Pose —— 自训练权重，需要自行提供

这是用 YOLOv5-Pose 格式训练的自定义权重（COCO 17 关键点）。启用它需要**同时**提供两个路径：

```env
POSE_ENGINE=mec
MEC_WEIGHTS=<你的仓库>/runs/train/exp102/weights/best.pt
MEC_REPO=<你的仓库根目录>
```

**为什么必须给 `MEC_REPO`**：这个 checkpoint 是 pickle 序列化的，反序列化时需要一个名叫
`models` 的模块。程序会把 `MEC_REPO` 插入 `sys.path`，再从里面导入
`models.experimental.attempt_load`。只给权重路径会报
`No module named 'models'`。

两个路径任一不存在时，`mec` 引擎在页面上会显示为「不可用」并给出具体原因，不会让服务起不来。

> 顺带一提：YOLOv5 自带的 `utils.torch_utils.select_device("cuda:0")` 在 PyTorch 2.6+
> 上会失败。它的做法是**在运行时**把 `CUDA_VISIBLE_DEVICES` 改成 `"0"` 再断言
> `torch.cuda.is_available()`，而 CUDA 在 torch 导入后就已完成设备枚举，改环境变量不再生效，
> 断言直接炸掉（`AssertionError: CUDA unavailable, invalid device cuda:0 requested`）。
> 本项目因此**不使用** `select_device`，而是直接构造 `torch.device("cuda:0")`。

## 2. 行为识别（可选，默认关闭）

```env
BEHAVIOR_ENGINE=none    # 只出关键点，不判行为（默认）
BEHAVIOR_ENGINE=rule    # 规则引擎：摔倒/打架/踢门/推门/扒门/靠门
BEHAVIOR_ENGINE=llm     # 交给大模型判定
BEHAVIOR_ENGINE=stgcn   # PCG-GCN，需自行提供权重
```

`rule` 与 `llm` 都不需要权重文件。

### PCG-GCN（stgcn）

保留但默认不启用。原因是有实测依据的：

| 视频 | PCG-GCN 原始输出 | 置信度 | 能否映射到电梯行为 |
|---|---|---|---|
| 4月22日.mp4 | `sit down` | 0.53 | 否 |
| 4月15日(2).mp4 | `play with phone` | 0.49 | 否 |
| 4月16日.mp4 | `cheer up` | **1.00** | 否 |
| 4月15日.mp4 | `hopping` | **1.00** | 否 |

现有权重是 **NTU RGB+D 60 类通用行为模型**，没有用电梯数据训练过。
置信度 1.00 却是错判——模型没见过电梯场景，只能在 60 个类别里挑最像的。

用电梯数据微调后，把 `BEHAVIOR_ENGINE` 改成 `stgcn` 即可启用：

```env
STGCN_WEIGHTS=<权重路径>/epoch80_model.pt
STGCN_REPO=<st-gcn 仓库根目录>
```

代码在 `backend/app/engines/behavior/stgcn_engine.py`，里面的
`NTU_TO_BEHAVIOR` 映射表就是"哪些 NTU 类别可以对应到电梯行为"的对照，
微调后按新的类别体系改这张表即可。

## 3. 建议生成（可选）

```env
ADVICE_ENGINE=template   # 手写模板，零成本（默认）
ADVICE_ENGINE=llm        # 大模型生成，需要 LLM_API_KEY
ADVICE_ENGINE=rag        # 大模型 + 知识库检索
```

不涉及模型权重。上传时勾选「对比全部建议引擎」可以一次生成三份并排比较。
