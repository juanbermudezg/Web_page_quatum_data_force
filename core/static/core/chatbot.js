document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('chatForm');
  const input = document.getElementById('msgInput');
  const messages = document.getElementById('messages');

  function appendMessage(who, text) {
    const el = document.createElement('div');
    el.className = who === 'user' ? 'text-end mb-2' : 'text-start mb-2';
    el.innerHTML = `
      <small class="text-muted">${who}</small>
      <div class="p-2 border rounded">${text}</div>
    `;
    messages.appendChild(el);
    messages.parentElement.scrollTop = messages.parentElement.scrollHeight;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const text = input.value.trim();
    if (!text) return;

    appendMessage('user', text);
    input.value = '';

    const csrfToken = document
      .querySelector('meta[name="csrf-token"]')
      .getAttribute('content');

    try {
      const res = await fetch('/api/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ message: text })
      });

      if (!res.ok) {
        throw new Error(`HTTP error ${res.status}`);
      }

      const data = await res.json();
      appendMessage('bot', data.reply ?? 'Sin respuesta');

    } catch (err) {
      console.error(err);
      appendMessage('bot', 'Error de conexión con el servidor');
    }
  });
});
