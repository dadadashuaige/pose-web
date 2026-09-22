const $ = (selector) => document.querySelector(selector);
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

const RISK_CLASS = { "高风险": "high", "中风险": "medium", "低风险": "low" };
const STATUS_TEXT = { queued: "排队中", processing: "分析中", completed: "已完成", failed: "失败" };

const state = {
  health: null,
  engines: { pose: [], behavior: [], advice: [] },
  records: [],
  total: 0,
  page: 1,
  pageSize: 12,
  stats: {},
  selected: null,
  selectedIds: new Set(),
  file: null,
  previewUrl: null,
  poll: null,
  // 门区域（归一化 x1,y1,x2,y2）。全零表示未标定。
  roi: [0, 0, 0, 0],
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) { /* 忽略非 JSON 响应 */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function riskBadge(level) {
  if (!level) return "";
  return `<span class="badge ${RISK_CLASS[level] || ""}">${esc(level)}</span>`;
}

/* ------------------------------------------------------------ 系统状态 */
async function loadHealth() {
  try {
    state.health = await api("/api/health");
  } catch (error) {
    $("#health").innerHTML = `<span class="chip bad">服务异常：${esc(error.message)}</span>`;
    return;
  }
  const h = state.health;
  const gpu = h.device.gpu || h.device.device;
  const kb = `${h.knowledge_entries} 条 / ${h.knowledge_backend}`;
  $("#health").innerHTML = [
    `<span class="chip">推理设备 <b>${esc(gpu)}</b></span>`,
    `<span class="chip ${h.llm_configured ? "ok" : "bad"}">LLM <b>${h.llm_configured ? esc(h.llm_model) : "未配置"}</b></span>`,
    `<span class="chip">知识库 <b>${esc(kb)}</b></span>`,
    `<span class="chip">视频编码 <b>${esc((h.overlay_codecs || []).join("/") || "—")}</b></span>`,
  ].join("");

  state.engines = h.engines;
  fillSelect("#poseEngine", h.engines.pose, h.defaults.pose);
  fillSelect("#behaviorEngine", h.engines.behavior, h.defaults.behavior);
  fillSelect("#adviceEngine", h.engines.advice, h.defaults.advice);
  state.roi = Array.isArray(h.door_roi) && h.door_roi.length === 4
    ? h.door_roi.slice()
    : [0, 0, 0, 0];
}

const roiText = (roi) =>
  (roi[2] > roi[0] && roi[3] > roi[1])
    ? roi.map((v) => Number(v).toFixed(3)).join(", ")
    : "未标定";

function fillSelect(selector, list, current) {
  const element = $(selector);
  if (!element) return;
  element.innerHTML = (list || [])
    .map((engine) => {
      const disabled = engine.available ? "" : "disabled";
      const selected = engine.key === current ? "selected" : "";
      const suffix = engine.available ? "" : "（不可用）";
      return `<option value="${engine.key}" ${disabled} ${selected} title="${esc(engine.note)}">${esc(engine.label)}${suffix}</option>`;
    })
    .join("");
}

/* ------------------------------------------------------------ 记录列表 */
function currentFilters() {
  return {
    status: $("#statusFilter").value,
    risk: $("#riskFilter").value,
    archived: $("#archiveFilter").value,
    q: $("#searchInput").value.trim(),
    page: state.page,
    page_size: state.pageSize,
  };
}

async function loadRecords() {
  const params = new URLSearchParams();
  const filters = currentFilters();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== "" && value !== undefined) params.set(key, value);
  });
  const data = await api(`/api/analyses?${params.toString()}`);
  state.records = data.items;
  state.total = data.total;
  state.stats = data.stats || {};
  renderRecords();
  renderStats();
  schedulePolling();
}

function renderStats() {
  const stats = state.stats || {};
  const parts = [`共有 ${stats.total ?? state.total} 个任务`];
  Object.entries(stats.by_status || {}).forEach(([key, value]) => {
    parts.push(`${STATUS_TEXT[key] || key} ${value}`);
  });
  Object.entries(stats.by_risk || {}).forEach(([key, value]) => {
    if (key !== "未判定") parts.push(`${key} ${value}`);
  });
  if (stats.archived) parts.push(`已归档 ${stats.archived}`);
  $("#stats").innerHTML = parts.map((text) => `<span>${esc(text)}</span>`).join("");
}

function renderRecords() {
  const list = $("#list");
  if (!state.records.length) {
    list.innerHTML = `<p class="muted">没有符合条件的记录。</p>`;
  } else {
    list.innerHTML = "";
    state.records.forEach((item) => list.append(recordNode(item)));
  }
  const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
  $("#pageInfo").textContent = `第 ${state.page} / ${pages} 页`;
  $("#prevPage").disabled = state.page <= 1;
  $("#nextPage").disabled = state.page >= pages;
  $("#bulkDelete").disabled = state.selectedIds.size === 0;
  $("#bulkDelete").textContent = state.selectedIds.size
    ? `删除选中（${state.selectedIds.size}）`
    : "删除选中";
}

function recordNode(item) {
  const node = $("#recordTpl").content.cloneNode(true);
  const root = node.querySelector(".record");
  root.dataset.id = item.id;
  if (state.selected === item.id) root.classList.add("active");

  const checkbox = root.querySelector("input");
  checkbox.checked = state.selectedIds.has(item.id);
  checkbox.addEventListener("click", (event) => {
    event.stopPropagation();
    if (checkbox.checked) state.selectedIds.add(item.id);
    else state.selectedIds.delete(item.id);
    renderRecords();
  });
  root.querySelector(".record-check").addEventListener("click", (event) => event.stopPropagation());

  root.querySelector(".r-name").textContent = item.filename;
  const status = root.querySelector(".status");
  status.textContent = STATUS_TEXT[item.status] || item.status;
  status.classList.add(item.status);

  const bits = [];
  bits.push(new Date(item.created_at).toLocaleString());
  if (item.status === "processing" || item.status === "queued") {
    bits.push(`${item.stage || "处理中"} ${item.progress || 0}%`);
  } else {
    if (item.frames_sampled) bits.push(`${item.frames_sampled} 帧`);
    if (item.people_max) bits.push(`最多 ${item.people_max} 人`);
    if (item.duration_s) bits.push(`${item.duration_s.toFixed(1)}s`);
  }
  root.querySelector(".record-meta").textContent = bits.join(" · ");

  const badges = [];
  if (item.status === "completed") {
    badges.push(riskBadge(item.risk_level));
    if (item.behavior_label) badges.push(`<span class="badge">${esc(item.behavior_label)}</span>`);
    badges.push(`<span class="badge">${esc(item.behavior_engine || "—")}</span>`);
    if (item.has_video) badges.push(`<span class="badge">可视化</span>`);
  }
  if (item.archived) badges.push(`<span class="badge">已归档</span>`);
  root.querySelector(".record-badges").innerHTML = badges.join("");

  root.addEventListener("click", () => selectRecord(item.id));
  return node;
}

async function selectRecord(id) {
  state.selected = id;
  renderRecords();
  try {
    const item = await api(`/api/analyses/${id}`);
    renderPipeline(item);
    renderDetail(item);
  } catch (error) {
    $("#detailBody").innerHTML = `<p class="muted">加载失败：${esc(error.message)}</p>`;
  }
}

/* ------------------------------------------------------------ 流水线 */
const PIPELINE_STEPS = [
  { key: "upload", label: "视频上传", target: "#sec-upload" },
  { key: "pose", label: "姿态估计", target: "#sec-pose" },
  { key: "keypoints", label: "关键点序列", target: "#sec-pose" },
  { key: "behavior", label: "行为识别", target: "#sec-behavior" },
  { key: "risk", label: "风险评分", target: "#sec-risk" },
  { key: "advice", label: "处置建议", target: "#sec-advice" },
  { key: "output", label: "报告与归档", target: "#sec-artifacts" },
];

function pipelineMeta(item, key) {
  const result = item.result || {};
  const behavior = result.behavior || {};
  const risk = result.risk || {};
  switch (key) {
    case "upload":
      return { text: `${item.filename}`, skipped: false };
    case "pose":
      return {
        text: `${item.pose_engine || "—"} · ${item.frames_sampled || 0} 帧 · 最多 ${item.people_max || 0} 人`,
        skipped: false,
      };
    case "keypoints":
      return {
        text: `${result.resolution || "—"} @ ${result.fps || "—"}fps · 每 ${result.frame_stride || "—"} 帧取样`,
        skipped: false,
      };
    case "behavior":
      return {
        text: `${item.behavior_engine || "—"} → ${behavior.label || "未判定"}`,
        skipped: item.behavior_engine === "none",
      };
    case "risk":
      return {
        text: `${risk.level || item.risk_level || "—"} · 分数 ${(risk.score ?? item.risk_score ?? 0).toFixed(2)}`,
        skipped: false,
      };
    case "advice":
      return {
        text: `${(item.advice || {}).engine || "—"} → ${(item.advice || {}).priority || ""}`,
        skipped: false,
      };
    case "output":
      return {
        text: `${item.has_video ? "可视化视频 ✓" : "无视频"} · ${item.has_report ? "PDF ✓" : "无报告"}`,
        skipped: false,
      };
    default:
      return { text: "", skipped: false };
  }
}

function renderPipeline(item) {
  const flow = $("#flow");
  $("#pipelineHint").textContent = item
    ? `任务 ${item.id.slice(0, 8)} · ${STATUS_TEXT[item.status] || item.status}`
    : "选择一个任务查看每一步的真实状态";

  if (!item) {
    flow.innerHTML = PIPELINE_STEPS.map(
      (step, index) => `
      <div class="flow-node">
        <span class="idx">0${index + 1}</span>
        <b>${step.label}</b>
        <small>等待任务</small>
      </div>`
    ).join('<span class="flow-arrow">→</span>');
    return;
  }

  flow.innerHTML = PIPELINE_STEPS.map((step, index) => {
    const meta = pipelineMeta(item, step.key);
    const classes = ["flow-node"];
    if (meta.skipped) classes.push("skipped");
    if (item.status === "failed" && step.key === "output") classes.push("failed");
    const arrow = index < PIPELINE_STEPS.length - 1 ? '<span class="flow-arrow">→</span>' : "";
    return `<div class="${classes.join(" ")}" data-target="${step.target}">
        <span class="idx">0${index + 1}</span>
        <b>${step.label}</b>
        <small>${esc(meta.text)}</small>
      </div>${arrow}`;
  }).join("");

  flow.querySelectorAll(".flow-node").forEach((node) => {
    node.addEventListener("click", () => {
      const target = document.querySelector(node.dataset.target);
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      target.classList.add("flash");
      setTimeout(() => target.classList.remove("flash"), 1200);
    });
  });
}

/* ------------------------------------------------------------ 详情 */
function renderDetail(item) {
  const body = $("#detailBody");
  const actions = $("#detailActions");
  body.classList.remove("empty");

  if (item.status === "queued" || item.status === "processing") {
    actions.innerHTML = "";
    body.innerHTML = `
      <div class="verdict"><span class="level">正在分析</span>
        <div class="sub">${esc(item.stage || "处理中")} · ${item.progress || 0}%</div></div>
      <div class="progress-inline"><i style="width:${item.progress || 0}%"></i></div>
      <p class="muted">姿态推理按帧串行执行，页面会自动刷新。</p>`;
    return;
  }

  if (item.status === "failed") {
    actions.innerHTML = rerunButton(item) + deleteButton();
    body.innerHTML = `
      <div class="verdict"><span class="level" style="color:#ef4444">分析失败</span></div>
      <pre class="muted" style="white-space:pre-wrap">${esc(item.error)}</pre>`;
    bindActions(item);
    return;
  }

  const result = item.result || {};
  const behavior = result.behavior || {};
  const risk = result.risk || {};
  const advice = item.advice || {};
  const variants = result.advice_variants || {};
  const warnings = result.warnings || [];
  const levelClass = RISK_CLASS[item.risk_level] || "low";

  const keyframes = (result.keyframes || [])
    .map((name, index) => `
      <figure>
        <img src="/api/analyses/${item.id}/keyframe/${index}" alt="${esc(name)}" data-zoom>
        <figcaption>${esc(name)}</figcaption>
      </figure>`)
    .join("");

  const evidenceRows = Object.entries(behavior.evidence || {})
    .map(([key, value]) => `<tr><td>${esc(key)}</td><td>${esc(Array.isArray(value) ? value.join("；") : value)}</td></tr>`)
    .join("");

  const factorRows = (risk.factors || [])
    .map((factor) => `<tr><td>${esc(factor.name)}</td><td>${esc(factor.value)}</td></tr>`)
    .join("");

  const variantCards = Object.entries(variants)
    .map(([key, payload]) => `
      <article class="variant">
        <h4>${esc(engineLabel("advice", key))}</h4>
        <div class="tag">${esc(payload.priority || "")} · ${esc(payload.title || "")}
          ${payload.latency_ms ? ` · ${payload.latency_ms}ms` : ""}</div>
        <ol>${(payload.actions || []).map((text) => `<li>${esc(text)}</li>`).join("")}</ol>
        ${payload.note ? `<div class="refs">${esc(payload.note)}</div>` : ""}
        ${(payload.references || []).length
          ? `<div class="refs">引用：${(payload.references || []).map((ref) => esc(ref.entry_id)).join("、")}</div>`
          : ""}
      </article>`)
    .join("");

  actions.innerHTML =
    (item.has_report ? `<a href="/api/analyses/${item.id}/report" target="_blank"><button>下载 PDF</button></a>` : "") +
    (item.has_video ? `<a href="/api/analyses/${item.id}/video" download><button class="ghost">下载可视化视频</button></a>` : "") +
    rerunButton(item) +
    `<button class="ghost" id="archiveBtn">${item.archived ? "取消归档" : "归档"}</button>` +
    deleteButton();

  body.innerHTML = `
    <div id="sec-upload" class="anchor"></div>
    <div class="verdict ${levelClass}">
      <span class="level">${esc(item.risk_level || "低风险")}</span>
      <span class="action">${esc(behavior.label || item.behavior_label || "未判定")}</span>
      <div class="sub">
        置信度 ${(behavior.confidence ?? item.behavior_score ?? 0).toFixed(2)} ·
        风险分数 ${(risk.score ?? item.risk_score ?? 0).toFixed(2)} ·
        姿态 ${esc(item.pose_engine || "—")} ·
        行为引擎 ${esc(item.behavior_engine || "—")}
      </div>
    </div>

    ${warnings.length ? `<div class="warnbox"><b>结论使用提示</b><ul>${warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>` : ""}

    <h3 class="section" id="sec-artifacts">可视化结果</h3>
    ${item.has_video ? `
      <div class="video-wrap">
        <video controls preload="metadata" src="/api/analyses/${item.id}/video"></video>
        <div class="video-note">编码 ${esc(item.overlay_codec || "—")} ·
          ${esc((result.overlay || {}).frames || 0)} 帧 ·
          ${esc((result.overlay || {}).readable ? "已通过回读校验" : "未通过校验")}</div>
      </div>` : `<p class="muted">本次未生成可视化视频${result.overlay_warning ? "：" + esc(result.overlay_warning) : "。"}</p>`}
    ${keyframes ? `<div class="gallery">${keyframes}</div>` : ""}

    <h3 class="section" id="sec-door">门区域标定</h3>
    <p class="muted" style="font-size:12px">
      「踢门 / 推门 / 扒门 / 靠门」四类行为都依赖这块区域。
      在图上拖拽框出电梯门，再点上面的「用当前门区域重新分析」。
      当前：<b id="roiText">${roiText(item.door_roi || [0, 0, 0, 0])}</b>
    </p>
    ${(result.keyframes || []).length ? `
      <div class="roi-wrap" id="roiWrap">
        <img id="roiImage" src="/api/analyses/${item.id}/keyframe/0" alt="关键帧" />
        <div class="roi-rect" id="roiRect"></div>
      </div>
      <div class="actions-row">
        <button class="ghost" id="roiSaveDefault">把当前门区域设为默认值</button>
        <button class="ghost" id="roiClear">清除门区域</button>
      </div>` : `<p class="muted">本次没有生成关键帧，无法在图上标定。</p>`}

    <h3 class="section" id="sec-pose">姿态与采样</h3>
    <table class="grid">
      <tr><th style="width:140px">原始文件</th><td>${esc(item.filename)}</td></tr>
      <tr><th>画面与时长</th><td>${esc(result.resolution || "—")} @ ${esc(result.fps || "—")}fps · ${(item.duration_s || 0).toFixed(1)} 秒</td></tr>
      <tr><th>采样</th><td>${item.frames_sampled || 0} 帧（总 ${item.frames_total || 0} 帧，每 ${result.frame_stride || "—"} 帧取 1）</td></tr>
      <tr><th>检测统计</th><td>最大同时人数 ${item.people_max || 0} · 关键点平均置信度 ${(item.posture_quality || 0).toFixed(3)}</td></tr>
      <tr><th>姿态模型</th><td>${esc(item.pose_engine || "—")}</td></tr>
    </table>

    <h3 class="section" id="sec-behavior">行为判定</h3>
    <p>${esc(behavior.label || "未判定")}（${esc(behavior.behavior || "—")}），置信度 ${(behavior.confidence ?? 0).toFixed(2)}。</p>
    ${behavior.note ? `<p class="muted">${esc(behavior.note)}</p>` : ""}
    ${behavior.reasoning ? `<p class="muted">判定理由：${esc(behavior.reasoning)}</p>` : ""}
    ${evidenceRows ? `<table class="grid"><tr><th style="width:200px">实测依据</th><th>取值</th></tr>${evidenceRows}</table>` : ""}

    <h3 class="section" id="sec-risk">风险评分</h3>
    <p>等级 <b>${esc(risk.level || "—")}</b>，分数 ${(risk.score ?? 0).toFixed(2)}。</p>
    ${risk.reason ? `<p class="muted">${esc(risk.reason)}</p>` : ""}
    ${factorRows ? `<table class="grid"><tr><th style="width:200px">风险因子</th><th>取值</th></tr>${factorRows}</table>` : ""}

    <h3 class="section" id="sec-advice">处置建议</h3>
    <p><b>${esc(advice.priority || "")} · ${esc(advice.title || "")}</b></p>
    <ol>${(advice.actions || []).map((text) => `<li>${esc(text)}</li>`).join("")}</ol>
    ${advice.reasoning ? `<p class="muted">建议依据：${esc(advice.reasoning)}</p>` : ""}
    ${advice.note ? `<p class="muted">${esc(advice.note)}</p>` : ""}
    ${advice.disclaimer ? `<p class="muted">${esc(advice.disclaimer)}</p>` : ""}

    ${Object.keys(variants).length > 1 ? `
      <h3 class="section">建议引擎对比</h3>
      <div class="variants">${variantCards}</div>` : ""}
  `;

  bindActions(item);
  body.querySelectorAll("[data-zoom]").forEach((image) => {
    image.addEventListener("click", () => showLightbox(image.src));
  });
}

function rerunButton(item) {
  return `<button class="ghost" id="rerunBtn" ${item.raw_video_deleted ? "disabled title='原始视频已清理'" : ""}>用当前引擎重新分析</button>`;
}
function deleteButton() {
  return `<button class="danger" id="deleteBtn">删除</button>`;
}

function bindActions(item) {
  $("#rerunBtn")?.addEventListener("click", async () => {
    const form = new FormData();
    form.append("pose_engine", $("#poseEngine").value);
    form.append("behavior_engine", $("#behaviorEngine").value);
    form.append("advice_engine", $("#adviceEngine").value);
    form.append("compare_advice", $("#compareAdvice").checked ? "1" : "0");
    form.append("door_roi", state.roi.join(","));
    try {
      await api(`/api/analyses/${item.id}/rerun`, { method: "POST", body: form });
      await loadRecords();
      await selectRecord(item.id);
    } catch (error) {
      alert("重新分析失败：" + error.message);
    }
  });

  $("#archiveBtn")?.addEventListener("click", async () => {
    const form = new FormData();
    form.append("archived", item.archived ? "0" : "1");
    await api(`/api/analyses/${item.id}/archive`, { method: "POST", body: form });
    await loadRecords();
    await selectRecord(item.id);
  });

  $("#deleteBtn")?.addEventListener("click", async () => {
    if (!confirm(`删除任务「${item.filename}」及其全部产物？`)) return;
    await api(`/api/analyses/${item.id}`, { method: "DELETE" });
    state.selected = null;
    state.selectedIds.delete(item.id);
    renderPipeline(null);
    $("#detailBody").innerHTML = `<p class="muted">请选择一个任务</p>`;
    $("#detailActions").innerHTML = "";
    await loadRecords();
  });

  // ---- 门区域拖拽标定 ------------------------------------------------
  const wrap = $("#roiWrap");
  const rect = $("#roiRect");
  if (wrap && rect) {
    const paint = () => {
      const [x1, y1, x2, y2] = state.roi;
      rect.style.left = `${x1 * 100}%`;
      rect.style.top = `${y1 * 100}%`;
      rect.style.width = `${Math.max(0, x2 - x1) * 100}%`;
      rect.style.height = `${Math.max(0, y2 - y1) * 100}%`;
    };
    paint();

    let dragging = false;
    let start = null;
    const point = (event) => {
      const bounds = wrap.getBoundingClientRect();
      return [
        Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
        Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
      ];
    };
    wrap.addEventListener("mousedown", (event) => {
      dragging = true;
      start = point(event);
      event.preventDefault();
    });
    window.addEventListener("mousemove", (event) => {
      // 详情面板重新渲染后这个 wrap 会失效，用 contains 兜一下
      if (!dragging || !document.body.contains(wrap)) return;
      const now = point(event);
      state.roi = [
        Math.min(start[0], now[0]), Math.min(start[1], now[1]),
        Math.max(start[0], now[0]), Math.max(start[1], now[1]),
      ];
      paint();
      const label = $("#roiText");
      if (label) label.textContent = roiText(state.roi);
    });
    window.addEventListener("mouseup", () => { dragging = false; });
  }

  $("#roiClear")?.addEventListener("click", () => {
    state.roi = [0, 0, 0, 0];
    const box = $("#roiRect");
    if (box) { box.style.width = "0"; box.style.height = "0"; }
    const label = $("#roiText");
    if (label) label.textContent = "未标定";
  });

  $("#roiSaveDefault")?.addEventListener("click", async () => {
    if (!(state.roi[2] > state.roi[0] && state.roi[3] > state.roi[1])) {
      alert("请先在图上拖拽框出电梯门。");
      return;
    }
    try {
      const res = await api("/api/settings/door-roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ roi: state.roi }),
      });
      alert(res.note || "已保存");
    } catch (error) {
      alert("保存失败：" + error.message);
    }
  });
}

function engineLabel(kind, key) {
  const found = (state.engines[kind] || []).find((engine) => engine.key === key);
  return found ? found.label : key;
}

function showLightbox(src) {
  const box = document.createElement("div");
  box.className = "lightbox";
  box.innerHTML = `<img src="${src}" alt="preview">`;
  box.addEventListener("click", () => box.remove());
  document.body.append(box);
}

/* ------------------------------------------------------------ 上传 */
$("#video").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) setFile(file);
});

function setFile(file) {
  state.file = file;
  $("#fileInfo").classList.remove("hidden");
  $("#fileName").textContent = file.name;
  $("#fileSize").textContent = `${formatBytes(file.size)} · ${file.type || "未知类型"}`;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  const preview = $("#preview");
  preview.src = state.previewUrl;
  preview.style.display = file.type.startsWith("video") ? "block" : "none";
}

$("#clearFile").addEventListener("click", () => {
  state.file = null;
  $("#video").value = "";
  $("#fileInfo").classList.add("hidden");
  $("#uploadStatus").textContent = "";
});

["dragenter", "dragover"].forEach((type) =>
  $("#dropZone").addEventListener(type, (event) => {
    event.preventDefault();
    $("#dropZone").classList.add("hot");
  })
);
["dragleave", "drop"].forEach((type) =>
  $("#dropZone").addEventListener(type, (event) => {
    event.preventDefault();
    $("#dropZone").classList.remove("hot");
  })
);
$("#dropZone").addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) setFile(file);
});

$("#uploadForm").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!state.file) {
    alert("请先选择视频或图片。");
    return;
  }
  const form = new FormData();
  form.append("video", state.file);
  form.append("pose_engine", $("#poseEngine").value);
  form.append("behavior_engine", $("#behaviorEngine").value);
  form.append("advice_engine", $("#adviceEngine").value);
  form.append("compare_advice", $("#compareAdvice").checked ? "1" : "0");
  form.append("door_roi", state.roi.join(","));

  const button = $("#submitBtn");
  button.disabled = true;
  $("#progressWrap").classList.remove("hidden");
  $("#progressBar").style.width = "0%";
  $("#uploadStatus").textContent = "正在上传…";

  const request = new XMLHttpRequest();
  request.open("POST", "/api/analyses");
  request.upload.addEventListener("progress", (progressEvent) => {
    if (!progressEvent.lengthComputable) return;
    const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
    $("#progressBar").style.width = `${percent}%`;
    $("#uploadStatus").textContent = `正在上传… ${percent}%（${formatBytes(progressEvent.loaded)} / ${formatBytes(progressEvent.total)}）`;
  });
  request.addEventListener("load", async () => {
    button.disabled = false;
    if (request.status >= 200 && request.status < 300) {
      const item = JSON.parse(request.responseText);
      $("#progressBar").style.width = "100%";
      $("#uploadStatus").innerHTML = `已进入分析队列：<b>${esc(item.filename)}</b>（任务 ${item.id.slice(0, 8)}）`;
      state.page = 1;
      await loadRecords();
      await selectRecord(item.id);
    } else {
      let detail = request.statusText;
      try { detail = JSON.parse(request.responseText).detail || detail; } catch (_) { /* ignore */ }
      $("#uploadStatus").textContent = "上传失败：" + detail;
    }
  });
  request.addEventListener("error", () => {
    button.disabled = false;
    $("#uploadStatus").textContent = "上传失败：网络错误";
  });
  request.send(form);
});

/* ------------------------------------------------------------ 筛选与维护 */
["#statusFilter", "#riskFilter", "#archiveFilter"].forEach((selector) =>
  $(selector).addEventListener("change", () => {
    state.page = 1;
    loadRecords();
  })
);

let searchTimer = null;
$("#searchInput").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.page = 1;
    loadRecords();
  }, 300);
});

$("#prevPage").addEventListener("click", () => {
  if (state.page > 1) {
    state.page -= 1;
    loadRecords();
  }
});
$("#nextPage").addEventListener("click", () => {
  state.page += 1;
  loadRecords();
});
$("#refresh").addEventListener("click", async () => {
  await loadHealth();
  await loadRecords();
  if (state.selected) await selectRecord(state.selected);
});

$("#bulkDelete").addEventListener("click", async () => {
  const ids = Array.from(state.selectedIds);
  if (!ids.length) return;
  if (!confirm(`删除选中的 ${ids.length} 个任务及其全部产物？`)) return;
  const result = await api("/api/analyses/bulk-delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  state.selectedIds.clear();
  if (ids.includes(state.selected)) {
    state.selected = null;
    renderPipeline(null);
    $("#detailBody").innerHTML = `<p class="muted">请选择一个任务</p>`;
    $("#detailActions").innerHTML = "";
  }
  alert(`已删除 ${result.deleted} 个任务。${result.detail || ""}`);
  await loadRecords();
});

$("#cleanupFailed").addEventListener("click", async () => {
  if (!confirm("删除所有失败的分析任务？")) return;
  const result = await api("/api/maintenance/cleanup-failed", { method: "POST" });
  alert(`已删除 ${result.deleted} 个失败任务。${result.detail || ""}`);
  await loadRecords();
});

$("#cleanupRaw").addEventListener("click", async () => {
  const days = prompt("清理多少天前的原始视频？（报告与关键帧会保留）", "7");
  if (!days) return;
  const result = await api(`/api/maintenance/cleanup-raw?days=${encodeURIComponent(days)}`, {
    method: "POST",
  });
  alert(result.detail || `已清理 ${result.deleted} 个文件`);
  await loadRecords();
});

/* ------------------------------------------------------------ 轮询 */
function schedulePolling() {
  if (state.poll) clearTimeout(state.poll);
  const busy = state.records.some((item) => item.status === "queued" || item.status === "processing");
  if (!busy) return;
  state.poll = setTimeout(async () => {
    await loadRecords();
    if (state.selected) {
      try {
        const item = await api(`/api/analyses/${state.selected}`);
        renderPipeline(item);
        renderDetail(item);
      } catch (_) { /* 忽略 */ }
    }
  }, 2500);
}

(async function init() {
  await loadHealth();
  await loadRecords();
  renderPipeline(null);
})();
