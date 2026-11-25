// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;
let currentRequest = null; // AbortController of the in-flight query

// DOM elements
let chatMessages, chatInput, sendButton, newChatButton, totalCourses, courseTitles;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    newChatButton = document.getElementById('newChatButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    
    setupThemeToggle();
    setupEventListeners();
    createNewSession();
    loadCourseStats();
});

// Theme toggle: dark is the default, light sets data-theme="light" on <html>
function setupThemeToggle() {
    const toggle = document.getElementById('themeToggle');
    const root = document.documentElement;
    let transitionTimer;

    const syncLabel = () => {
        const isLight = root.dataset.theme === 'light';
        toggle.setAttribute('aria-label', isLight ? 'Switch to dark theme' : 'Switch to light theme');
        toggle.title = toggle.getAttribute('aria-label');
    };

    toggle.addEventListener('click', () => {
        const next = root.dataset.theme === 'light' ? 'dark' : 'light';
        root.classList.add('theme-transition');
        clearTimeout(transitionTimer);
        transitionTimer = setTimeout(() => root.classList.remove('theme-transition'), 300);
        root.dataset.theme = next;
        try { localStorage.setItem('theme', next); } catch (e) {}
        syncLabel();
    });

    syncLabel();
}

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    newChatButton.addEventListener('click', startNewChat);

    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const controller = new AbortController();
    currentRequest = controller;

    try {
        const response = await fetch(`${API_URL}/query`, {
            signal: controller.signal,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        // Aborted by "New Chat": the chat was already reset
        if (error.name === 'AbortError') return;
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        if (currentRequest === controller) currentRequest = null;
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function startNewChat() {
    if (currentRequest) currentRequest.abort();
    if (currentSessionId) {
        fetch(`${API_URL}/session/${encodeURIComponent(currentSessionId)}`, { method: 'DELETE' })
            .catch(error => console.error('Failed to delete session:', error));
    }
    createNewSession();
    chatInput.value = '';
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        // Several chunks often come from the same lesson; show each source once
        const uniqueSources = [...new Map(sources.map(s => [s.text, s])).values()];
        const sourceItems = uniqueSources.map(s => s.url
            ? `<a class="source-item" href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.text)}<span class="source-icon" aria-hidden="true">↗</span></a>`
            : `<span class="source-item">${escapeHtml(s.text)}</span>`
        );
        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources (${uniqueSources.length})</summary>
                <div class="sources-content">${sourceItems.join('')}</div>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}