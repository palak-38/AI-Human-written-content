document.getElementById('analyze-button').addEventListener('click', () => {
    const text = document.getElementById('text-input').value;
    fetch('/analyze-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('analysis-result').style.display = 'block';
        document.getElementById('prediction-value').innerText = data.prediction;
    });
});

document.getElementById('upload-button').addEventListener('click', () => {
    const files = document.getElementById('file-input').files;
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    fetch('/upload-files', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const batchResults = document.getElementById('batch-results');
        batchResults.innerHTML = '';
        for (const [filename, prediction] of Object.entries(data)) {
            batchResults.innerHTML += `<p>${filename}: ${prediction}</p>`;
        }
        batchResults.style.display = 'block';
    });
});
