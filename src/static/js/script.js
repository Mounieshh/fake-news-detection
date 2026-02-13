// Global state
window.currentTab = 'text';

function switchTab(tabName) {
    window.currentTab = tabName;

    // Update Tab Buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.innerText.toLowerCase() === tabName) {
            btn.classList.add('active');
        }
    });

    // Update Content Areas
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
    });

    const activeContent = document.getElementById(tabName + '-tab');
    if (activeContent) {
        activeContent.classList.add('active');
        activeContent.style.display = 'block';
    }

    // Hide previous results
    document.getElementById('result-area').style.display = 'none';
}

async function analyzeContent() {
    const btn = document.getElementById('analyze-btn');
    const resultArea = document.getElementById('result-area');

    // Get Input Data
    let payload = {};
    if (window.currentTab === 'text') {
        const textVal = document.getElementById('text-input').value.trim();
        if (!textVal) {
            alert("Please enter text to analyze.");
            return;
        }
        payload = { text: textVal };
    } else {
        const urlVal = document.getElementById('url-input').value.trim();
        if (!urlVal) {
            alert("Please enter a valid URL.");
            return;
        }
        payload = { url: urlVal };
    }

    // Set Loading State
    const originalBtnText = btn.innerText;
    btn.innerText = "Processing...";
    btn.disabled = true;
    resultArea.style.display = 'none';

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok || data.prediction === 'ERROR') {
            throw new Error(data.explanation || data.error || "Analysis failed.");
        }

        // Display Results
        displayResult(data);

    } catch (error) {
        console.error("Error:", error);
        alert("Error: " + error.message);
    } finally {
        btn.innerText = originalBtnText;
        btn.disabled = false;
    }
}

function displayResult(data) {
    const resultArea = document.getElementById('result-area');
    const predText = document.getElementById('prediction-text');
    const confBar = document.getElementById('confidence-bar');
    const explanation = document.getElementById('explanation');

    // Set Prediction Text & Color
    predText.innerText = data.prediction;
    predText.className = 'prediction-value';
    if (data.prediction === 'REAL') {
        predText.classList.add('prediction-real');
    } else {
        predText.classList.add('prediction-fake');
    }

    // Set Confidence Bar
    // Animated fill
    confBar.style.width = '0%';
    setTimeout(() => {
        confBar.style.width = (data.confidence || 0) + '%';
    }, 100);

    // Set Explanation
    explanation.innerText = data.explanation || "No details provided.";

    // Show Result Box
    resultArea.style.display = 'block';
}

// Theme Logic
const themeBtn = document.getElementById('theme-toggle');
const root = document.documentElement;

function setTheme(themeName) {
    root.setAttribute('data-theme', themeName);
    localStorage.setItem('theme', themeName);

    // Update button content based on *current* theme
    // If current is light, show "Dark Mode" option
    // If current is dark, show "Light Mode" option
    const isLight = themeName === 'light';
    themeBtn.innerHTML = isLight
        ? '<span id="theme-icon">🌙</span> Dark Mode'
        : '<span id="theme-icon">☀️</span> Light Mode';
}

// Toggle Event
if (themeBtn) {
    themeBtn.addEventListener('click', () => {
        const currentTheme = root.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    switchTab('text');

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
});
