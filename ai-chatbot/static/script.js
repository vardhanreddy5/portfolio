async function sendMessage() {
  const input = document.getElementById("message");
  const chatBox = document.getElementById("chat-box");

  if (!input.value) return;

  chatBox.innerHTML += `<div class="message user">${input.value}</div>`;

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: input.value })
  });

  const data = await res.json();

  chatBox.innerHTML += `<div class="message bot">${data.reply}</div>`;
  input.value = "";
}