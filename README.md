# Pose Insight：YOLO-Pose + LLM 动作分析

一个从图片上传到网页分析报告的可运行项目，流程如下：

```text
上传图片 → YOLO-Pose 检测 → 骨架/关键点可视化 → LLM 动作分析 → 网页报告
```

## 已包含的能力

- CV：Ultralytics YOLO-Pose 检测 COCO 17 个关键点，并输出带骨架的可视化图片。
- 后端：FastAPI 提供健康检查、图片校验、上传、推理、静态文件托管和分析 API。
- Agent：将结构化关键点传给 LLM，要求其只依据二维证据生成谨慎的中文结论；未配置 Key 或模型调用失败时自动使用本地可解释规则。
- 前端：零构建依赖的单页网页，支持拖放上传、结果对比和报告展示。
- 工程化：Dockerfile、docker-compose 持久化 `data/` 和 `models/`。

## 启动

### Docker（推荐）

确保已安装 Docker Desktop，在项目根目录执行：

```powershell
docker compose up --build
```

访问 [http://localhost:8000](http://localhost:8000)。首次分析时 Ultralytics 会下载 `yolo11n-pose.pt` 权重；下载完成后会使用 Docker 挂载的 `models/` 目录缓存。

### 本地 Python

推荐 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

## 配置 LLM

编辑 `.env`：

```env
OPENAI_API_KEY=你的密钥
OPENAI_MODEL=gpt-4.1-mini
# 如使用兼容 API，再填写：OPENAI_BASE_URL=https://...
```

无需重写前端。每次请求会把 YOLO 输出的关键点 JSON 交给 Agent；响应会标识为“LLM 分析”。没有 Key 时显示“本地规则分析”，仍可完整验证上传、检测、可视化和报告流程。

## API

| 方法 | 地址 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 服务与模型配置检查 |
| `POST` | `/api/analyze` | 上传字段名为 `file` 的 JPG/PNG/WEBP，返回图片 URL、关键点和报告 |

接口文档：`http://localhost:8000/docs`。

## 目录

```text
pose-web/
├─ backend/app/
│  ├─ main.py                 # FastAPI 路由和上传流程
│  └─ services/
│     ├─ pose_service.py       # YOLO-Pose 推理与可视化
│     └─ llm_service.py        # 动作分析 Agent
├─ frontend/                  # 单页网页
├─ data/uploads/              # 原始上传图片（运行时生成）
├─ data/reports/              # 姿态可视化结果（运行时生成）
├─ models/                    # 模型权重缓存
└─ docker-compose.yml
```

## 重要限制

单张图片只提供二维姿态，不能可靠识别连续运动、三维角度或健康状况。若要分析深蹲/挥拍等完整动作，应扩展视频上传、逐帧跟踪和时序模型。

