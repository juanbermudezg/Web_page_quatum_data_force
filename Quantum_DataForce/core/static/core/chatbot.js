document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('chatForm');
  const input = document.getElementById('msgInput');
  const messages = document.getElementById('messages');

  function appendMessage(who, text) {
    const el = document.createElement('div');
    el.className = who === 'user' ? 'text-end mb-2' : 'text-start mb-2';
    el.innerHTML = `<small class="text-muted">${who}</small><div class="p-2 border rounded">${text}</div>`;
    messages.appendChild(el);
    messages.parentElement.scrollTop = messages.parentElement.scrollHeight;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    appendMessage('user', text);
    input.value = '';

    try {
      const res = await fetch('/api/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const data = await res.json();
      appendMessage('bot', data.reply || 'No reply');
    } catch (err) {
      appendMessage('bot', 'Error de conexión');
      console.error(err);
    }
  });
});
