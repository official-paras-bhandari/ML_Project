// === Academic Assignment Generator - Frontend Logic ===

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("genForm");
    const submitBtn = document.getElementById("submitBtn");
    const fileInput = document.getElementById("reference_file");
    const fileNameSpan = document.getElementById("fileName");
    const dropZone = document.getElementById("dropZone");

    // --- Form submission: show loading state ---
    if (form && submitBtn) {
        form.addEventListener("submit", (e) => {
            const topic = document.getElementById("topic").value.trim();
            if (!topic) {
                e.preventDefault();
                return;
            }
            const btnText = submitBtn.querySelector(".btn-text");
            const btnLoading = submitBtn.querySelector(".btn-loading");
            if (btnText) btnText.style.display = "none";
            if (btnLoading) btnLoading.style.display = "inline-flex";
            submitBtn.disabled = true;
            submitBtn.style.opacity = "0.7";
        });
    }

    // --- File input: show selected filename ---
    if (fileInput && fileNameSpan) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                fileNameSpan.textContent = fileInput.files[0].name;
            } else {
                fileNameSpan.textContent = "";
            }
        });
    }

    // --- Drag and drop visual feedback ---
    if (dropZone) {
        ["dragenter", "dragover"].forEach((evt) => {
            dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropZone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((evt) => {
            dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropZone.classList.remove("dragover");
            });
        });
    }
});
