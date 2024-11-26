document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabs = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and panes
            tabs.forEach(t => t.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            // Add active class to clicked tab and corresponding pane
            tab.classList.add('active');
            document.getElementById(tab.dataset.tab).classList.add('active');
        });
    });

    // Chat functionality
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');

    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessage(message, 'user');
        userInput.value = '';

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            });

            const data = await response.json();
            addMessage(data.response, 'assistant');
        } catch (error) {
            addMessage('Sorry, there was an error processing your request.', 'system');
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Market Analysis functionality
    const symbolInput = document.getElementById('symbolInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const marketResult = document.getElementById('marketResult');

    analyzeBtn.addEventListener('click', async () => {
        const symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) return;

        marketResult.innerHTML = 'Analyzing...';

        try {
            const response = await fetch('/analyze-market', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symbol: symbol }),
            });

            const data = await response.json();
            marketResult.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        } catch (error) {
            marketResult.innerHTML = 'Error analyzing market data.';
        }
    });

    // CSV Analysis functionality
    const csvFile = document.getElementById('csvFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const csvResult = document.getElementById('csvResult');

    uploadBtn.addEventListener('click', async () => {
        const file = csvFile.files[0];
        if (!file) {
            csvResult.innerHTML = 'Please select a CSV file first.';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        csvResult.innerHTML = 'Analyzing CSV data...';

        try {
            const response = await fetch('/analyze-csv', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            csvResult.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        } catch (error) {
            csvResult.innerHTML = 'Error analyzing CSV file.';
        }
    });
});
