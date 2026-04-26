// SSE consumer for the per-user "Ask Qwen" RAG.
// Output renders only here, in the streamer's browser. Nothing leaves the box.

function chatterbotAsk(userId) {
  const submit   = document.getElementById('ask-submit');
  const qInput   = document.getElementById('ask-q');
  const status   = document.getElementById('ask-status');
  const answer   = document.getElementById('ask-answer');
  const cWrap    = document.getElementById('ask-citations-wrap');
  const cArea    = document.getElementById('ask-citations');
  if (!submit) return;

  let es = null;

  function start() {
    const q = (qInput.value || '').trim();
    if (!q) return;
    if (es) es.close();

    answer.textContent = '';
    cArea.innerHTML = '';
    cWrap.classList.add('hidden');
    status.textContent = 'thinking…';
    status.classList.remove('hidden');
    submit.disabled = true;

    const url = `/users/${encodeURIComponent(userId)}/ask?q=${encodeURIComponent(q)}`;
    es = new EventSource(url);

    es.addEventListener('citations', (e) => {
      try {
        const data = JSON.parse(e.data);
        renderCitations(data);
      } catch (_) {
        // ignore malformed payload
      }
    });

    es.addEventListener('chunk', (e) => {
      // Server escaped \n as \\n so each SSE message is one logical line.
      answer.textContent += e.data.replace(/\\n/g, '\n');
    });

    es.addEventListener('error', (e) => {
      if (e.data) {
        answer.textContent += `\n[error: ${e.data}]`;
      }
    });

    es.addEventListener('done', () => {
      status.textContent = '';
      status.classList.add('hidden');
      submit.disabled = false;
      es.close();
    });

    es.onerror = () => {
      status.textContent = 'connection lost';
      submit.disabled = false;
      if (es) es.close();
    };
  }

  function renderCitations(data) {
    const notes = (data.notes || []);
    const messages = (data.messages || []);
    if (!notes.length && !messages.length) return;
    cWrap.classList.remove('hidden');

    if (notes.length) {
      const h = document.createElement('div');
      h.className = 'text-xs uppercase tracking-wider text-slate-500 mt-1';
      h.textContent = 'notes';
      cArea.appendChild(h);
      const ul = document.createElement('ul');
      ul.className = 'list-disc pl-5 space-y-1';
      for (const n of notes) {
        const li = document.createElement('li');
        li.textContent = n.text;
        ul.appendChild(li);
      }
      cArea.appendChild(ul);
    }
    if (messages.length) {
      const h = document.createElement('div');
      h.className = 'text-xs uppercase tracking-wider text-slate-500 mt-3';
      h.textContent = 'messages';
      cArea.appendChild(h);
      const ul = document.createElement('ul');
      ul.className = 'list-disc pl-5 space-y-1 text-slate-300';
      for (const m of messages) {
        const li = document.createElement('li');
        const ts = (m.ts || '').replace('T', ' ').slice(0, 16);
        li.textContent = `[${ts}] ${m.content}`;
        ul.appendChild(li);
      }
      cArea.appendChild(ul);
    }
  }

  submit.addEventListener('click', start);
  qInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); start(); }
  });
}
