# Elevator Guard · 电梯危险行为智能预警系统

上传一段电梯轿厢监控视频，自动完成：**关键点提取 → 骨架可视化视频 → 危险行为判定 → 风险评估 → 处置建议 → PDF 报告 → 入库与网页展示**。

面向电梯轿厢场景的**六种危险行为**：

| 行为 | 判定依据 | 风险 |
|---|---|---|
| **摔倒** | 躯干偏离竖直 + 重心骤降 | 高 |
| **打架** | 多人近距离 + 上肢动作剧烈 | 高 |
| **踢门** | 抬腿 + 脚部高速接近门区域 | 高 |
| **扒门** | 双手贴近门区域并持续开合 | 高 |
| **推门** | 重心持续朝门区域推进 | 中 |
| **靠门** | 长时间静止贴近门区域 | 中 |

其中后四种都与轿门有关，因此引入了**门区域标定**——这是本项目区别于通用人体行为识别的关键设计，详见下文。

## 目录

- [核心设计：三个可插拔环节](#核心设计三个可插拔环节)
- [快速开始](#快速开始)
- [门区域标定（重要）](#门区域标定重要)
- [六种行为怎么判的](#六种行为怎么判的)
- [配置大模型与知识库](#配置大模型与知识库)
- [记录管理](#记录管理)
- [API](#api)
- [工程与部署](#工程与部署)
- [实测性能](#实测性能)
- [已知边界](#已知边界)

## 核心设计：三个可插拔环节

整条链路拆成三个互不依赖的环节，每个环节都是一组可替换的引擎，
通过页面下拉框或 `.env` 切换，**换模型不需要改任何代码**。

```
视频 / 图片
    │
    ▼
┌─ 姿态估计 ────────────────────────────────────┐
│  yolo11n  YOLO11n-Pose（通用，零配置可用）      │
│  mec      MEC-Pose（自训练 YOLOv5-Pose，可注入）│
└───────────────────────────────────────────────┘
    │  COCO-17 关键点 + 轨迹跟踪
    ▼
┌─ 行为识别 ────────────────────────────────────┐
│  none     只出关键点，不判行为                  │
│  rule     规则引擎（六种电梯行为，阈值可配）      │
│  llm      交给大模型判定                        │
│  stgcn    PCG-GCN（保留待微调）                 │
└───────────────────────────────────────────────┘
    │  行为结论 + 实测数值
    ▼
┌─ 建议生成 ────────────────────────────────────┐
│  template 手写模板（基线对照）                  │
│  llm      大模型生成                            │
│  rag      大模型 + 电梯安全知识库检索            │
└───────────────────────────────────────────────┘
    │
    ▼
风险评分 → PDF 报告 → SQLite → 网页
```

注册表在 `backend/app/engines/`，新增一个模型只需实现接口 + 注册一行：

```python
# backend/app/engines/pose/my_model.py
class MyPoseEngine(PoseEngine):
    key = "my_model"
    label = "我的模型"

    @property
    def available(self) -> bool: ...

    def detect(self, frame) -> list[PersonPose]: ...

# backend/app/engines/pose/__init__.py
POSE_ENGINES = {"yolo11n": ..., "mec": ..., "my_model": MyPoseEngine}
```

`pipeline.py` 只认接口不认实现，所以新增引擎不会牵连流程代码。

## 快速开始

```powershell
cd backend

# 1) 配置（可选：不配也能跑，全程走本地规则）
Copy-Item ..\.env.example .env

# 2) 启动
cd ..
.\run.ps1
```

浏览器打开 <http://127.0.0.1:8001>，接口文档在 `/docs`。

启动脚本会依次尝试：`-Python` 参数 → 环境变量 `POSE_WEB_PYTHON` → 常见 conda 路径 → PATH 里的 `python`，
所以换机器不用改脚本。端口被占用时会直接告诉你被谁占了，并给出可用端口的建议。

```powershell
.\run.ps1 -Port 8020
.\run.ps1 -Python "C:\path\to\python.exe"
```

### 依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

`requirements.txt` **刻意不含 torch / torchvision**：这两个包体积大、且必须和你的
CUDA 版本严格匹配，让 pip 自动解析容易装错。请先自行安装匹配的版本，例如：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

只要环境里已有满足要求的 torch，`pip install -r requirements.txt` 不会替换它。

## 门区域标定（重要）

**这是本项目最容易踩的坑，也是它区别于通用人体行为识别的关键。**

「踢门 / 推门 / 扒门 / 靠门」四类行为的共性是"人对门做了什么"，
但关键点里**没有"门"这个对象**——骨架只有人。不告诉系统门在哪，
这四条规则就只能瞎猜，结果是"站在轿厢中间"也会被判成"贴近门"。

### 标定方式

1. 先跑一次任务（此时行为引擎选 `rule`）
2. 在结果页的「门区域标定」里，于关键帧上**拖拽框出电梯门**
3. 点「用当前门区域重新分析」

想让它成为以后所有新任务的默认值，再点「把当前门区域设为默认值」，会写进 `backend/.env`。

### 没标定会怎样

不会瞎报。门相关规则**直接不参与判定**，结果里会写明：

> 门区域尚未标定，「踢门 / 推门 / 扒门 / 靠门」四类行为本次没有参与判定，
> 只能给出摔倒与打架的结论。

### 实测教训

开发过程中用一个覆盖画面 54% 的"默认区域"试过：一个只是站在电梯口按按钮的人，
被判成「踢门 0.92」和「扒门 0.99」——因为他的手脚本来就落在那个巨大的"门区域内"。
把区域收紧到真实的门之后，同样的视频只剩下正确的那条结论。

所以本项目的 `DOOR_ROI` **默认留空**，而不是给一个猜的默认值。

## 六种行为怎么判的

规则全部是确定性的阈值判断，定义在 `backend/app/engines/behavior/rules.yaml`，
改完重启即可生效。所有距离与速度都按**躯干长度**归一化，因此不随人物远近变化。

```yaml
kick_door:
  label: 踢门
  risk_level: high
  params:
    ankle_speed_min: 1.6      # 脚踝速度阈值（躯干长度/秒）
    ankle_door_max_dist: 1.0  # 脚踝距门区域上限
    min_ankle_split: 0.25     # 两脚踝垂直落差，用来区分"踢"和"原地站立"
```

每条结论都附带实测数值，报告里能看出"为什么这么判"：

```json
{
  "说明": "有抬腿动作且脚部高速接近门区域",
  "脚踝速度(躯干倍数/秒)": 4.91,
  "脚踝距门(躯干倍数)": 0.42,
  "两脚踝落差(躯干倍数)": 0.62
}
```

### 两条值得说明的工程细节

**1. 脚踝落差（ankle_split）**：最初只用"脚踝速度 + 距门距离"判踢门，
结果静止站立的人也会被误报——因为他的脚本来就在门区域附近。
加上"两条腿必须有明显垂直落差"这个条件后，误报消失。

**2. 跟踪器要能扛住快速运动**：摔倒本身就是一帧内大幅位移的事件，
纯 IoU 跟踪会把它拆成两条轨迹，"下降过程"被切掉，摔倒反而检测不到。
因此跟踪用了两轮匹配：先按 IoU，匹配不上的再用**中心距兜底**（位移不超过大半个身高）。
`tests/test_rule_engine.py::test_tracker_survives_fast_motion` 专门守着这个行为。

### 行为为什么不全交给大模型

规则引擎产出**候选和硬证据**（确定性、可复现、零成本、可审计），
LLM 负责**复核和排除误报**。把 `rules.yaml` 的判据原文写进 prompt，
模型用的是和你完全一致的判定标准，而不是自己另立一套。

反过来让 LLM 从零看裸坐标做判断，它就会编——这是要避免的。

## 配置大模型与知识库

```env
LLM_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
```

**关键：`LLM_MODEL` 必须填 `LLM_BASE_URL` 那个端点实际提供的模型名。**
两者不匹配时不会报错、界面也不异常，只是悄悄退回本地规则。
用 5 行代码先验证一下更稳妥：

```python
from openai import OpenAI
client = OpenAI(api_key="你的密钥", base_url="你的 BASE_URL")
print(client.chat.completions.create(
    model="你的模型名",
    messages=[{"role": "user", "content": "只回复两个字：测试"}],
).choices[0].message.content)
```

能打印出「测试」就对了；报 `404 Model not exist` 说明模型名写错了。

**不配置也能跑完整流程**：行为判定退回规则引擎，风险评估退回知识库声明的等级，
建议退回模板，任务不会失败，只是结果里会标注用的是哪个引擎。

### 知识库

语料在 `backend/knowledge/*.md`，当前是一份通用电梯乘用安全语料，
**不含编造的国家标准条款号**，正式引用前请核对现行规程。

检索默认用内置的 **BM25 关键词检索**，零依赖开箱可用；
装了 `chromadb` 且配了 `EMBEDDING_API_KEY` 时会自动切换为向量检索。
替换语料后调 `POST /api/knowledge/rebuild` 重建索引。

### 建议引擎对比

上传时勾选「对比全部建议引擎」，同一次分析会同时生成三份建议并排展示，PDF 里也有对比表。
用途是评估大模型到底带来了多少增量、以及知识库检索是否改变了结论。

## 记录管理

分析记录多了之后，一个长列表是没法用的。现在提供：

- **筛选**：状态、风险等级、文件名搜索、当前/已归档切换
- **分页**：默认每页 12 条
- **删除**：单条 / 批量勾选，同时级联删除该任务的全部产物文件
- **归档**：从主列表隐藏但保留数据，可切回查看
- **清理失败任务**：一键删除
- **清理过期原始视频**：按天数清理，**报告与关键帧保留**

存储构成参考：清理前 7 个任务的**原始视频共 121 MB，而 PDF 报告不到 1 MB**。
空间几乎全被原始视频占据，而它恰恰是分析完最没用的东西。
设置 `RAW_VIDEO_RETAIN_DAYS=7` 即可在服务启动时自动清理。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 设备、三组引擎可用性、知识库、LLM 状态、当前门区域 |
| GET | `/api/engines` | 引擎清单与默认值 |
| POST | `/api/analyses` | 上传并分析 |
| GET | `/api/analyses` | 列表（`status`/`risk`/`q`/`archived`/`page`/`page_size`） |
| GET | `/api/analyses/{id}` | 详情 |
| POST | `/api/analyses/{id}/rerun` | 换引擎或换门区域后重跑 |
| POST | `/api/analyses/{id}/archive` | 归档 / 取消归档 |
| DELETE | `/api/analyses/{id}` | 删除任务与产物 |
| POST | `/api/analyses/bulk-delete` | 批量删除 |
| GET | `/api/analyses/{id}/report` | 下载 PDF |
| GET | `/api/analyses/{id}/video` | 下载可视化视频（WebM） |
| GET | `/api/analyses/{id}/keyframe/{n}` | 关键帧图片 |
| POST | `/api/settings/door-roi` | 把门区域写回 `.env` |
| GET | `/api/rules` | 查看六条规则与阈值 |
| GET/POST | `/api/knowledge`、`/api/knowledge/rebuild`、`/api/knowledge/search` | 知识库 |
| POST | `/api/maintenance/cleanup-failed` | 清理失败任务 |
| POST | `/api/maintenance/cleanup-raw?days=7` | 清理过期原始视频 |

上传参数：`video`（文件）、`pose_engine`、`behavior_engine`、`advice_engine`、
`compare_advice`、`door_roi`（`x1,y1,x2,y2`，归一化）。

## 工程与部署

### 目录结构

```text
backend/
  app/
    engines/              可插拔引擎
      pose/               yolo11n / mec
      behavior/           none / rule / llm / stgcn  + rules.yaml
      advice/             template / llm / rag
    services/
      pipeline.py         全流程编排
      visualizer.py       骨架绘制 + 视频编码（含编码器回退与回读校验）
      risk_agent.py       确定性风险评分
      report_service.py   PDF 报告
      knowledge.py        知识库检索（BM25 / Chroma）
      llm_client.py       OpenAI 兼容客户端
    static/               零构建单页前端（原生 HTML/CSS/JS，无需 npm）
    data/                 运行产物（已 gitignore）
  knowledge/              知识库语料
  tests/                  单元测试
  requirements.txt
Dockerfile
docker-compose.yml
MODELS.md                 权重配置说明
run.ps1                   启动脚本
```

### Docker

```powershell
docker compose up --build
```

默认构建 **CPU 版镜像**，不需要 nvidia runtime。环境变量（含 LLM Key）
在 `docker-compose.yml` 里透传，`.env` 不必挂进容器。

需要 GPU 时把 `Dockerfile` 的基础镜像换成 `nvidia/cuda:12.8-runtime-ubuntu22.04`
并安装对应 CUDA 版 torch，同时在 compose 里加 `deploy.resources.reservations.devices`。
注意那样镜像会到 5GB 以上。

### 测试

```powershell
cd backend
python -m pytest -q
```

当前 **29 个用例**，全部用合成数据，不依赖模型权重，0.3 秒跑完：
六种行为的判定、门区域开关的影响、跟踪器对快速运动的鲁棒性、
规则配置完整性、引擎注册表、RiskAgent 的评分逻辑。

### 可视化视频的编码选择

```text
VP8 / WebM  →  VP9 / WebM  →  mp4v / MP4
```

**为什么不用更常见的 H.264**：在开发机上测出 OpenCV 缺 `openh264` 库，
用 `avc1` 写视频时 `isOpened()` 返回 `True`，但写出的是**一个 0 字节的空文件**——
典型的静默失败，最难排查。所以改用浏览器原生支持的 WebM。

**为什么 VP8 排在 VP9 前面**：实测 1280×720 每帧编码耗时，**VP9 约 257ms，VP8 只要 24ms**，
相差约 10 倍，而两者在 Chrome/Edge/Firefox 里播放体验没有差别。
一个 14 秒视频的可视化导出因此从 108 秒降到 10 秒。

写出后还会**回读校验**（能否打开、帧数是否大于 0），不通过就丢弃，
避免"文件生成了但播不了"这种情况混进结果。

## 实测性能

1906×1080、14 秒、140 个采样帧（`FRAME_STRIDE=3`）的电梯监控视频：

| 阶段 | 耗时 |
|---|---|
| 姿态推理（MEC-Pose / YOLOv5-Pose） | ~6.3 s |
| 绘制与视频编码 | ~20 s ← 瓶颈 |
| 行为判定（规则引擎） | 0.03 s |
| 风险评分 | 0.00 s |
| 建议生成（模板） | 0.00 s |
| PDF 报告 | ~0.4 s |
| **端到端** | **约 30 s** |

**可视化视频的编码是唯一的大头**，其余环节加起来不到 7 秒。三个旋钮的实测效果：

| 配置 | 编码耗时 | 端到端 |
|---|---|---|
| `OVERLAY_MAX_WIDTH=1280`、`OVERLAY_FPS=0`（默认） | 20.0 s | 30 s |
| `OVERLAY_FPS=10` | 15.5 s | 26 s |
| `OVERLAY_MAX_WIDTH=854` + `OVERLAY_FPS=10` | **7.7 s** | **16 s** |
| `OVERLAY_ENABLED=0`（不要视频） | 0 s | ~7 s |

两个结论：

- **分辨率是主杠杆**，耗时大致与像素数成正比（1280→854 是 0.44 倍像素，基本对应 0.5 倍耗时）；
- **帧率的效果有限**，因为重复帧编码很便宜，真正花钱的是每个"唯一帧"。
  30fps 输出 420 帧要 20 秒，10fps 输出 140 帧仍要 15.5 秒。

用 YOLO11n-Pose 姿态推理只占 2.6 s 左右，比 MEC-Pose 快一倍多。
每次分析都会把这份耗时明细存进 `result.timings`，页面上可以查看，
方便判断该调 `FRAME_STRIDE`、`OVERLAY_MAX_WIDTH` 还是换模型。

## 已知边界

- **规则阈值是按少量样本给的初值**，正式使用前需用本单位素材重新标定，
  并统计误报率与漏报率。"推门 / 靠门 / 扒门"三者最容易混淆，
  因为差别集中在手部动作和接触状态上，建议优先标定门区域再调这三条的阈值。
- **门区域的目标是"门板本身"**，不包含门前的地面。框太大等于没有约束。
- **知识库是通用语料**，不含正式标准条款，引用前请核对现行规程。
- **PCG-GCN 保留但默认不启用**：现有公开权重是 NTU RGB+D 60 类通用模型，
  实测会把画面判成 `cheer up`、`hopping` 之类（置信度甚至是 1.00），无法对应电梯行为。
  细节与启用方式见 [MODELS.md](MODELS.md)。
- 前端资源已设置 `Cache-Control: no-store`。如果你改完前端却看不到变化，
  说明浏览器还持有旧缓存，按一次 `Ctrl + F5`。
- 系统输出仅作为安全管理辅助依据，**高风险结论须经人工确认后执行处置**。
