const form = document.querySelector(".prediction-form");

if (form) {
  form.addEventListener("submit", (event) => {
    const invalid = [...form.querySelectorAll("input, select")].find((field) => !field.value);
    if (invalid) {
      event.preventDefault();
      invalid.focus();
    }
  });
}

