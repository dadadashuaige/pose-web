const input = document.querySelector('#file-input');
const dropZone = document.querySelector('#drop-zone');
const selectButton = document.querySelector('#select-button');
const progress = document.querySelector('#progress');
const progressText = document.querySelector('#progress-text');
const error = document.querySelector('#error');
const results = document.querySelector('#results');

selectButton.addEventListener('click', () => input.click());
input.addEventListener('change', () => upload(input.files[0]));
['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault(); dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault(); dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (event) => upload(event.dataTransfer.files[0]));

async function upload(file) {
  if (!file) return;
  document.querySelector('#file-name').textContent = file.name;
  error.textContent = '';
  results.hidden = true;
  progress.hidden = false;
  progressText.textContent = '正在上传并进行 YOLO-Pose 检测…';
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || '分析请求失败');
    render(body);
  } catch (exception) {
    error.textContent = exception.message;
  } finally {
    progress.hidden = true;
  }
}

function fillList(selector, items) {
  const list = document.querySelector(selector);
  list.replaceChildren(...items.map((item) => {
    const element = document.createElement('li'); element.textContent = item; return element;
  }));
}

function render(data) {
  document.querySelector('#original-image').src = data.original_url;
  document.querySelector('#pose-image').src = data.visualization_url;
  document.querySelector('#report-title').textContent = data.report.title;
  document.querySelector('#report-summary').textContent = data.report.summary;
  document.querySelector('#report-posture').textContent = data.report.posture;
  document.querySelector('#detection-meta').textContent = `检测到 ${data.detections.length} 人 · 模型：${data.model}`;
  document.querySelector('#provider').textContent = data.llm_provider === 'openai' ? 'LLM 分析' : '本地规则分析';
  document.querySelector('#limitations').textContent = data.report.limitations;
  fillList('#observations', data.report.observations);
  fillList('#recommendations', data.report.recommendations);
  results.hidden = false;
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

