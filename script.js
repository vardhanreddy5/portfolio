// Simple contact form handler
const form = document.getElementById('contact-form');
const response = document.getElementById('form-response');

form.addEventListener('submit', (e) => {
  e.preventDefault();
  response.textContent = "Thanks for your message! I will get back to you soon.";
  form.reset();
});
