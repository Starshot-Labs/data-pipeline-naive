// Judging the judge: every verdict the review folder holds, with the exterior and cutaway
// frames it was made from, so the filter can be spot-checked before anything is deleted.
//
// Your own agree/disagree marks live in localStorage — this page never writes to the dataset,
// and the point of it is to measure the filter, not to apply it. "Copy disagreements" puts the
// ids you flagged on the clipboard, which is the input to deciding whether the model's
// threshold needs moving.

export {};

interface Decision {
  id: string;
  verdict: boolean;
  reason: string;
  placement: string;
  placed?: string;
  anchor?: string;
  model?: string;
  views: string[];
}

type Mark = 'ok' | 'bad';
type Filter = 'all' | 'keep' | 'drop' | 'unmarked' | 'disagreed';

const MARKS_KEY = 'review-marks-v1';
const marks: Record<string, Mark> = JSON.parse(localStorage.getItem(MARKS_KEY) ?? '{}');
const saveMarks = () => localStorage.setItem(MARKS_KEY, JSON.stringify(marks));

const list = document.getElementById('list') as HTMLElement;
const stats = document.getElementById('stats') as HTMLElement;
const filters = document.getElementById('filters') as HTMLElement;

let samples: Decision[] = [];
let filter: Filter = 'all';

const matches = (sample: Decision): boolean => {
  const mark = marks[sample.id];
  if (filter === 'keep') return sample.verdict;
  if (filter === 'drop') return !sample.verdict;
  if (filter === 'unmarked') return !mark;
  if (filter === 'disagreed') return mark === 'bad';
  return true;
};

function renderStats(): void {
  const kept = samples.filter((s) => s.verdict).length;
  const marked = samples.filter((s) => marks[s.id]).length;
  const wrong = samples.filter((s) => marks[s.id] === 'bad').length;
  const agreement = marked ? `${(((marked - wrong) / marked) * 100).toFixed(0)}%` : '—';
  stats.innerHTML =
    `<b>${samples.length}</b> judged · <b>${kept}</b> keep / <b>${samples.length - kept}</b> drop` +
    ` · you reviewed <b>${marked}</b>, disagreed with <b>${wrong}</b> · model agreement <b>${agreement}</b>`;
}

function renderFilters(): void {
  const options: Filter[] = ['all', 'keep', 'drop', 'unmarked', 'disagreed'];
  filters.innerHTML = '';
  for (const option of options) {
    const button = document.createElement('button');
    button.textContent = option;
    button.className = option === filter ? 'on' : '';
    button.onclick = () => {
      filter = option;
      render();
    };
    filters.append(button);
  }
}

function card(sample: Decision): HTMLElement {
  const el = document.createElement('section');
  const mark = marks[sample.id];
  el.className = `sample${mark === 'ok' ? ' marked-ok' : mark === 'bad' ? ' marked-bad' : ''}`;

  const head = document.createElement('div');
  head.className = 'head';
  head.innerHTML =
    `<span class="verdict ${sample.verdict ? 'keep' : 'drop'}">${sample.verdict ? 'KEEP' : 'DROP'}</span>` +
    `<span class="phrase">“${sample.placement}”</span>` +
    `<span class="id">${sample.id}</span>`;
  el.append(head);

  if (sample.reason) {
    const reason = document.createElement('div');
    reason.className = 'reason';
    reason.textContent = sample.reason;
    el.append(reason);
  }

  if (sample.views.length) {
    const frames = document.createElement('div');
    frames.className = 'frames';
    for (const view of sample.views) {
      const cell = document.createElement('div');
      cell.className = 'frame';
      const img = document.createElement('img');
      img.loading = 'lazy';
      img.src = `/review/${encodeURIComponent(sample.id)}/${encodeURIComponent(view)}`;
      const label = document.createElement('span');
      // "3-back.png" reads better as "3 back".
      label.textContent = view.replace(/\.png$/, '').replace('-', ' ');
      cell.append(img, label);
      frames.append(cell);
    }
    el.append(frames);
  } else {
    const none = document.createElement('div');
    none.className = 'unseen';
    none.textContent = 'No frames — this sample was dropped without being looked at.';
    el.append(none);
  }

  const agree = document.createElement('div');
  agree.className = 'agree';
  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = 'The verdict above is';
  agree.append(label);
  for (const [value, text] of [['ok', 'right'], ['bad', 'wrong']] as [Mark, string][]) {
    const button = document.createElement('button');
    button.textContent = text;
    button.className = mark === value ? 'on' : '';
    button.onclick = () => {
      if (marks[sample.id] === value) delete marks[sample.id];
      else marks[sample.id] = value;
      saveMarks();
      render();
    };
    agree.append(button);
  }
  el.append(agree);
  return el;
}

function render(): void {
  renderFilters();
  renderStats();
  list.innerHTML = '';
  const shown = samples.filter(matches);
  if (!shown.length) {
    const empty = document.createElement('p');
    empty.style.color = 'var(--dim)';
    empty.textContent = samples.length
      ? 'Nothing matches that filter.'
      : 'No verdicts yet. Pull them down with: modal volume get trellis-scene-vol-v2 datasets/raw/review review';
    list.append(empty);
    return;
  }
  for (const sample of shown) list.append(card(sample));
}

(document.getElementById('export') as HTMLButtonElement).onclick = async () => {
  const wrong = samples.filter((s) => marks[s.id] === 'bad');
  const text = JSON.stringify(
    wrong.map((s) => ({ id: s.id, verdict: s.verdict, placement: s.placement, reason: s.reason })),
    null,
    2,
  );
  await navigator.clipboard.writeText(text);
  const button = document.getElementById('export') as HTMLButtonElement;
  button.textContent = `Copied ${wrong.length}`;
  setTimeout(() => (button.textContent = 'Copy disagreements'), 1500);
};

const response = await fetch('/api/review');
const payload = (await response.json()) as { dir: string; samples: Decision[] };
// Drops first: a filter is judged on what it throws away, so that is what wants looking at.
samples = payload.samples.sort((a, b) => Number(a.verdict) - Number(b.verdict) || a.id.localeCompare(b.id));
render();
