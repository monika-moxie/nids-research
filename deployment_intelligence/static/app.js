const jsonInput = document.querySelector('#flow-json');
const results = document.querySelector('#results');
const emptyState = document.querySelector('#empty-state');
const toast = document.querySelector('#toast');
let latestReport = null;
let batchAlerts = [];

function showToast(message) { toast.textContent = message; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 2600); }
function pretty(value) { return JSON.stringify(value, null, 2); }
function percent(value) { return `${(value * 100).toFixed(1)}%`; }

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || 'The request could not be completed.');
  return body;
}

async function loadDemo() {
  try {
    const data = await request('/demo-flow');
    jsonInput.value = pretty(data.features);
    showToast('Demo flow loaded');
  } catch (error) { showToast(error.message); }
}

function renderFeatures(features) {
  const max = Math.max(...features.map(item => Math.abs(item.contribution)), 0.001);
  document.querySelector('#explanation-list').innerHTML = features.map(item => {
    const width = Math.max(8, Math.abs(item.contribution) / max * 100);
    const direction = item.direction === 'toward_attack' ? 'attack' : 'normal';
    return `<div class="feature-row"><div class="feature-copy"><strong title="${item.feature}">${item.feature}</strong><span>${item.direction.replace('_', ' ')}</span><div class="bar-track"><div class="bar ${direction}" style="width:${width}%"></div></div></div><span class="feature-value">${item.contribution.toFixed(3)}</span></div>`;
  }).join('');
}

function render(data) {
  emptyState.hidden = true; results.hidden = false;
  document.querySelector('#action-title').textContent = data.decision.action === 'escalate' ? 'Escalate for analyst review' : 'Suppress and retain for audit';
  document.querySelector('#decision-reason').textContent = data.decision.reason;
  const badge = document.querySelector('#severity-badge'); badge.textContent = data.decision.severity; badge.className = `severity ${data.decision.severity}`;
  document.querySelector('#attack-probability').textContent = percent(data.attack_probability);
  document.querySelector('#model-label').textContent = data.predicted_label;
  document.querySelector('#certified-radius').textContent = data.certification.abstained ? 'Abstained' : data.certification.certified_radius.toFixed(3);
  document.querySelector('#certificate-status').textContent = data.certification.abstained ? 'No radius certified' : 'L2 preprocessed radius';
  document.querySelector('#plain-summary').textContent = data.plain_english_explanation;
  document.querySelector('#report-id').textContent = data.incident_report.report_id;
  document.querySelector('#report-status').textContent = data.incident_report.status;
  document.querySelector('#next-step').textContent = data.incident_report.recommended_next_step;
  document.querySelector('#report-limitation').textContent = data.incident_report.limitation;
  renderFeatures(data.explanation.top_features);
  latestReport = data.incident_report;
}

async function analyze() {
  let features;
  try { features = JSON.parse(jsonInput.value); } catch { showToast('Enter valid JSON before analysis'); return; }
  const button = document.querySelector('#analyze'); button.disabled = true; button.innerHTML = '<i data-lucide="loader-circle"></i>Analyzing…'; lucide.createIcons();
  try { render(await request('/intelligence', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({features}) })); }
  catch (error) { showToast(error.message); }
  finally { button.disabled = false; button.innerHTML = '<i data-lucide="scan-search"></i>Analyze flow'; lucide.createIcons(); }
}

function renderQueue(data) {
  batchAlerts = data.alerts;
  const queue = document.querySelector('#alert-queue');
  const list = document.querySelector('#queue-list');
  queue.hidden = false;
  document.querySelector('#queue-summary').textContent = `${data.flagged_count} flagged / ${data.total_rows} screened`;
  if (!data.alerts.length) {
    list.innerHTML = '<p class="queue-empty">No flows crossed the 50% attack threshold in this batch.</p>';
    return;
  }
  list.innerHTML = data.alerts.map((alert, index) => `<button class="queue-item" data-alert-index="${index}" title="Analyze flagged CSV row ${alert.row_number}"><span>Row ${alert.row_number}</span><strong>${percent(alert.attack_probability)}</strong><i data-lucide="arrow-up-right"></i></button>`).join('');
  list.querySelectorAll('[data-alert-index]').forEach(button => button.addEventListener('click', () => selectAlert(Number(button.dataset.alertIndex))));
  lucide.createIcons();
}

async function scoreBatch() {
  const file = document.querySelector('#batch-file').files[0];
  if (!file) { showToast('Choose a CSV batch first'); return; }
  const button = document.querySelector('#score-batch'); button.disabled = true; button.innerHTML = '<i data-lucide="loader-circle"></i>Screeningâ€¦'; lucide.createIcons();
  try {
    const form = new FormData(); form.append('file', file);
    const data = await request('/batch-score', { method: 'POST', body: form });
    renderQueue(data); showToast(`${data.flagged_count} flagged flow${data.flagged_count === 1 ? '' : 's'} found`);
  } catch (error) { showToast(error.message); }
  finally { button.disabled = false; button.innerHTML = '<i data-lucide="list-filter"></i>Screen batch'; lucide.createIcons(); }
}

function selectAlert(index) {
  const alert = batchAlerts[index];
  jsonInput.value = pretty(alert.features);
  showToast(`Row ${alert.row_number} loaded for detailed analysis`);
  analyze();
}

document.querySelector('#load-demo').addEventListener('click', loadDemo);
document.querySelector('#analyze').addEventListener('click', analyze);
document.querySelector('#score-batch').addEventListener('click', scoreBatch);
document.querySelector('#copy-report').addEventListener('click', async () => { if (latestReport) { await navigator.clipboard.writeText(pretty(latestReport)); showToast('Incident report copied'); } });

request('/metadata').then(data => document.querySelector('#schema-note').textContent = `Required: ${data.model_input_fields.length} raw fields. Excludes: ${data.excluded_fields.join(', ')}.`).catch(() => { document.querySelector('#api-status').textContent = 'API unavailable'; });
lucide.createIcons();
