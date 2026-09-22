# Pose Insight：YOLO-Pose + LLM 动作分析

> **这是本仓库的第一代（v1）：图片单帧姿态检测。**
> 只处理图片，出结果只要约 3 秒，零配置、不依赖数据库。
>
> 如果你需要**视频分析、电梯危险行为判定（摔倒/打架/踢门/推门/扒门/靠门）、
> 可视化视频、PDF 报告、记录管理**，请看第二代：
> **[`../v2-elevator-guard/`](../v2-elevator-guard/)** —— 它是本项目的能力增强版。
>
> 两代相互独立，各自 `.env.example` 与启动方式都在本目录内，可以同时运行。
>
> 下面所有命令都请在本目录（`v1-image/`）下执行。

一个从图片上传到网页分析报告的可运行项目，流程如下：

```text
上传图片 → YOLO-Pose 检测 → 骨架/关键点可视化 → LLM 动作分析 → 网页报告
```

## 更新记录

### v0.1.1 — 修复两个导致功能不可用的问题

这一版修掉了两个必须修的缺陷：一个是上传直接失败，一个是配好 LLM 也永远走不到大模型。

---

#### 修复 1：任何图片都传不上去（HTTP 413）

**现象**

网页选好图片点上传，返回：

```
{"detail":"图片不能超过 0 MB"}
```

任何非空文件都被拒。

**原因**

`.env` 里约定 `MAX_UPLOAD_MB=0` 表示「不限制大小」，但代码直接把 0 拿去比较，上限变成了 0 字节：

```python
# 修改前
if len(content) > settings.max_upload_mb * 1024 * 1024:   # 0 * 1024 * 1024 = 0
    raise HTTPException(413, f"图片不能超过 {settings.max_upload_mb} MB")
```

**修复**（`backend/app/main.py`）

先判断是否为正数，负数或 0 视为不限制：

```python
# 修改后
if settings.max_upload_mb > 0 and len(content) > settings.max_upload_mb * 1024 * 1024:
    raise HTTPException(413, f"图片不能超过 {settings.max_upload_mb} MB")
```

---

#### 修复 2：配置了 LLM，却始终显示「本地规则分析」

**现象**

`.env` 里已经填好 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`，接口却能正常返回报告、界面也不报错，
但 `llm_provider` 一直是 `local-rules`。换成正确的模型名之后，**分析耗时从 3 秒涨到 40 秒以上，
报告却仍然是本地规则**——等于白等了 40 秒。

**原因**（三个环节叠加）

1. **prompt 的类型说明有歧义。** 发给模型的 prompt 里，
   `observations` 和 `recommendations` 都标注了「字符串数组」，唯独 `limitations` 没写类型。
   `limitations` 语义上就是「若干条使用限制」，模型因此经常返回数组：

   ```json
   {"limitations": ["单张图片无法确认连续动作；", "部分关键点置信度不足；", ...]}
   ```

2. **数据模型是严格类型。** `schemas.py` 里定义的是 `limitations: str`，收到数组直接判为不合法：

   ```
   pydantic ValidationError: 1 validation error for AnalysisReport
   limitations
     Input should be a valid string [input_type=list]
   ```

3. **异常被静默吞掉。** `llm_service.py` 的兜底逻辑连失败原因都不留：

   ```python
   # 修改前
   try:
       return self._analyze_with_llm(detections), "openai"
   except Exception:
       # Provider errors should not break the visual-inspection workflow.
       pass                      # ← 整份 LLM 结果被丢弃，且无任何日志
   return self._analyze_locally(detections), "local-rules"
   ```

   于是流程变成：**调用大模型 → 等 40 秒 → 拿到结果 → 因格式不合整份丢掉 → 退回本地规则。**

   实测统计：连续 7 次调用中 **6 次返回数组、仅 1 次返回字符串**，也就是约 86% 的概率会触发。
   因为 temperature 不是 0，同一段 prompt 的输出格式本来就不稳定——这也是这个问题很难被发现的原因。

**修复**（三处）

**① `backend/app/schemas.py` — 让字段容忍两种形状（关键修复）**

```python
from pydantic import field_validator

def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "；".join(str(item) for item in value if str(item).strip())
    return str(value)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else [str(value)]


class AnalysisReport(BaseModel):
    title: str
    summary: str
    posture: str
    observations: list[str]
    recommendations: list[str]
    limitations: str

    @field_validator("title", "summary", "posture", "limitations", mode="before")
    @classmethod
    def _coerce_text(cls, value):
        return _as_text(value)

    @field_validator("observations", "recommendations", mode="before")
    @classmethod
    def _coerce_list(cls, value):
        return _as_list(value)
```

这样无论模型返回字符串还是数组都能通过校验，那 40 秒不会白花。

**② `backend/app/services/llm_service.py` — prompt 里把类型写死**

```diff
-"recommendations（字符串数组）、limitations。"
+"recommendations（字符串数组）、limitations（字符串，不是数组）。"
```

从源头减少歧义。**但不能只靠它**——模型不保证听话，第 ① 条的 schema 容错才是保底。

**③ `backend/app/services/llm_service.py` — 不再静默吞异常**

```diff
-except Exception:
-    pass
+except Exception as exc:
+    self.last_error = f"{type(exc).__name__}: {exc}"
+    logger.warning("LLM 分析失败，回退本地规则：%s", self.last_error)
```

否则以后再出现别的格式问题，仍然只能看到「为什么配了 key 却没用 LLM」。

---

### 修复效果

| 场景 | 修改前 | 修改后 |
| --- | --- | --- |
| 上传任意图片 | `413 图片不能超过 0 MB` | 正常分析 |
| LLM 返回数组格式 | 静默丢弃，退回本地规则 | 通过校验，正常使用 |
| LLM 调用失败 | 无任何日志 | 记录 `last_error` 并打警告 |
| 正确配置时的 `llm_provider` | 大概率 `local-rules` | `openai` |

## ⚠️ 配置提醒：模型名必须和端点匹配

这是 clone 之后最容易踩的坑。**`OPENAI_MODEL` 必须填你的端点实际提供的模型名**，
填错会得到一个很像"没生效"的结果：接口正常返回、界面不报错、日志里也没有异常，
但实际上大模型只用了 0.8 秒就失败了（`404 Model not exist`），然后悄悄退回本地规则。

举例说明两种常见组合：

```env
# 官方 OpenAI 接口
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini

# 阿里云百炼 / MaaS 兼容接口
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen3.7-plus
```

**不要把 OpenAI 的模型名配到 Qwen 的端点上**，反之亦然。
如果模型名不确定，可以用下面这段代码直接验证，比翻页面快：

```python
from openai import OpenAI
client = OpenAI(api_key="你的密钥", base_url="你的BASE_URL")
print(client.chat.completions.create(
    model="你的模型名",
    messages=[{"role": "user", "content": "只回复两个字：测试"}],
).choices[0].message.content)
```

能打印出「测试」说明配置正确；报 `404 Model not exist` 就是模型名错了。

### 速度取舍

- **不配 Key**：走本地规则，约 **3 秒** 出结果，稳定、可复现。
- **配好 Key**：走大模型，耗时取决于端点，实测约 **40~60 秒**（共享 MaaS 实例较慢，
  官方接口通常在数秒级）。

如果只是要快速看姿态检测和可视化，把 `OPENAI_API_KEY` 留空即可，全流程仍然完整可用。

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

**注意**：`.env` 是按「当前工作目录」查找的，所以必须在项目根目录启动上面这条命令。
在别的目录启动会找不到 `.env`，表现为「明明配了 Key 却不生效」。

## 配置 LLM

编辑项目根目录的 `.env`：

```env
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://你的端点/compatible-mode/v1
OPENAI_MODEL=你的端点实际提供的模型名
```

无需重写前端。每次请求会把 YOLO 输出的关键点 JSON 交给 Agent；响应中的 `llm_provider`
字段会标识这次用的是 `openai` 还是 `local-rules`，前端右上角的角标显示同一个值。
没有 Key 时显示「本地规则分析」，仍可完整验证上传、检测、可视化和报告流程。

## API

| 方法 | 地址 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 服务与模型配置检查 |
| `POST` | `/api/analyze` | 上传字段名为 `file` 的 JPG/PNG/WEBP，返回图片 URL、关键点和报告 |

接口文档：`http://localhost:8000/docs`。

响应里的 `llm_provider` 取值：

- `openai` —— 本次报告由大模型生成；
- `local-rules` —— 未配置 Key，或大模型调用/校验失败，已回退本地规则。此时可查看服务日志里的
  `LLM 分析失败，回退本地规则：...` 一行，里面会写明具体原因。

## 目录

```text
pose-web/
├─ backend/app/
│  ├─ main.py                 # FastAPI 路由和上传流程
│  ├─ schemas.py              # 出入参模型（含字段类型容错）
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
