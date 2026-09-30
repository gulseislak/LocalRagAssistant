/**
 * Local RAG Assistant - Presentation Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const uploadBtn = document.getElementById('uploadBtn');
  const selectedFileName = document.getElementById('selectedFileName');
  const uploadProgressContainer = document.getElementById('uploadProgressContainer');
  const uploadProgressBar = document.getElementById('uploadProgressBar');
  const uploadProgressText = document.getElementById('uploadProgressText');

  const statTotalChunks = document.getElementById('statTotalChunks');
  const chunkCounterBadge = document.getElementById('chunkCounterBadge');
  const docList = document.getElementById('docList');
  const clearBtn = document.getElementById('clearBtn');

  const chatMessages = document.getElementById('chatMessages');
  const questionInput = document.getElementById('questionInput');
  const sendBtn = document.getElementById('sendBtn');
  const topKSelect = document.getElementById('topKSelect');
  const toastContainer = document.getElementById('toastContainer');

  const chipQuestion1 = document.getElementById('chipQuestion1');
  const chipQuestion2 = document.getElementById('chipQuestion2');
  const chipQuestion3 = document.getElementById('chipQuestion3');

  let selectedFile = null;
  let isGenerating = false;

  // --------------------------------------------------------------------------
  // Initial Status Fetch
  // --------------------------------------------------------------------------
  fetchStatus();

  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error(`Status HTTP error: ${res.status}`);
      const data = await res.json();
      updateUIStatus(data);
    } catch (err) {
      console.error('Failed to fetch system status:', err);
    }
  }

  function updateUIStatus(data) {
    const count = data.total_chunks || 0;
    statTotalChunks.textContent = count;
    chunkCounterBadge.textContent = `${count} Chunks`;

    // Render documents list
    docList.innerHTML = '';
    if (data.documents && data.documents.length > 0) {
      data.documents.forEach((doc) => {
        const item = document.createElement('div');
        item.className = 'doc-item';
        item.innerHTML = `
          <span class="doc-name" title="${escapeHtml(doc.source)}">${escapeHtml(doc.source)}</span>
          <span class="doc-meta">${doc.chunks} chunks &bull; ${doc.pages}p</span>
        `;
        docList.appendChild(item);
      });
    } else {
      docList.innerHTML = '<div class="empty-docs-state">No documents indexed yet.</div>';
    }
  }

  // --------------------------------------------------------------------------
  // File Selection and Drag & Drop
  // --------------------------------------------------------------------------
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showToast('Please select a valid PDF (.pdf) file.', 'error');
      return;
    }
    selectedFile = file;
    selectedFileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    uploadBtn.disabled = false;
  }

  // --------------------------------------------------------------------------
  // Document Upload and Indexing
  // --------------------------------------------------------------------------
  uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    uploadBtn.disabled = true;
    uploadProgressContainer.style.display = 'block';
    uploadProgressBar.style.width = '35%';
    uploadProgressText.textContent = 'Extracting pages & generating chunks...';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      uploadProgressBar.style.width = '65%';
      uploadProgressText.textContent = 'Computing 1024d embeddings & storing in SQLite...';

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || 'Failed to upload document');
      }

      uploadProgressBar.style.width = '100%';
      uploadProgressText.textContent = 'Ingestion complete!';

      showToast(`Indexed ${data.indexed_chunks} chunks from '${data.filename}'!`, 'success');
      selectedFile = null;
      selectedFileName.textContent = 'Max 50MB per file';
      fileInput.value = '';

      setTimeout(() => {
        uploadProgressContainer.style.display = 'none';
        uploadProgressBar.style.width = '0%';
      }, 1500);

      fetchStatus();
    } catch (err) {
      console.error('Upload failed:', err);
      showToast(err.message, 'error');
      uploadProgressContainer.style.display = 'none';
      uploadBtn.disabled = false;
    }
  });

  // --------------------------------------------------------------------------
  // Clear Database Index
  // --------------------------------------------------------------------------
  clearBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear all document chunks from SQLite?')) {
      return;
    }

    try {
      const res = await fetch('/api/clear', { method: 'POST' });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || 'Failed to clear index');

      showToast('Document index cleared successfully.', 'info');
      fetchStatus();
    } catch (err) {
      console.error('Clear failed:', err);
      showToast(err.message, 'error');
    }
  });

  // --------------------------------------------------------------------------
  // Chat Interaction & RAG Query
  // --------------------------------------------------------------------------
  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  sendBtn.addEventListener('click', () => {
    handleSend();
  });

  function handleSend() {
    const question = questionInput.value.trim();
    if (!question || isGenerating) return;

    questionInput.value = '';
    questionInput.style.height = 'auto';
    executeRAGQuery(question);
  }

  async function executeRAGQuery(question) {
    isGenerating = true;
    sendBtn.disabled = true;

    // Append User Message
    appendUserMessage(question);

    // Append Typing Indicator
    const typingId = appendTypingIndicator();

    const topK = parseInt(topKSelect.value, 10) || 3;

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: topK }),
      });

      const data = await res.json();
      removeTypingIndicator(typingId);

      if (!res.ok || data.error) {
        appendAssistantMessage({
          answer: `Error executing query: ${data.error || 'Unknown error'}`,
          is_unsupported: true,
          sources: [],
        });
      } else {
        appendAssistantMessage(data);
      }
    } catch (err) {
      console.error('Query execution error:', err);
      removeTypingIndicator(typingId);
      appendAssistantMessage({
        answer: `Connection error: Could not reach local RAG server (${err.message}).`,
        is_unsupported: true,
        sources: [],
      });
    } finally {
      isGenerating = false;
      sendBtn.disabled = false;
      questionInput.focus();
    }
  }

  // --------------------------------------------------------------------------
  // Message Rendering
  // --------------------------------------------------------------------------
  function appendUserMessage(text) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message-wrapper user-wrapper';
    wrapper.innerHTML = `
      <div class="avatar user-avatar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
          <circle cx="12" cy="7" r="4"></circle>
        </svg>
      </div>
      <div class="message-content">
        <div class="message-header" style="justify-content: flex-end;">
          <span class="message-time">You</span>
        </div>
        <div class="message-body">${escapeHtml(text)}</div>
      </div>
    `;
    chatMessages.appendChild(wrapper);
    scrollToBottom();
  }

  function appendTypingIndicator() {
    const id = 'typing_' + Date.now();
    const wrapper = document.createElement('div');
    wrapper.id = id;
    wrapper.className = 'message-wrapper assistant-wrapper';
    wrapper.innerHTML = `
      <div class="avatar assistant-avatar">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
          <line x1="8" y1="21" x2="16" y2="21"></line>
          <line x1="12" y1="17" x2="12" y2="21"></line>
        </svg>
      </div>
      <div class="message-content">
        <div class="message-header">
          <span class="sender-name">Local RAG Assistant</span>
          <span class="message-time">Searching SQLite & Generating...</span>
        </div>
        <div class="message-body typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    `;
    chatMessages.appendChild(wrapper);
    scrollToBottom();
    return id;
  }

  function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function appendAssistantMessage(data) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message-wrapper assistant-wrapper';

    const sources = data.sources || [];
    const isUnsupported = data.is_unsupported;

    let statusPillHtml = '';
    if (isUnsupported) {
      statusPillHtml = `
        <div class="grounding-status-bar">
          <span class="refusal-pill">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
              <line x1="12" y1="9" x2="12" y2="13"></line>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            Refusal / Information Not in Context
          </span>
        </div>
      `;
    } else if (sources.length > 0) {
      statusPillHtml = `
        <div class="grounding-status-bar">
          <span class="grounded-pill">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Grounded in ${sources.length} Retrieved Chunks
          </span>
        </div>
      `;
    }

    let sourcesAccordionHtml = '';
    if (sources.length > 0) {
      const sourceItems = sources
        .map((s, idx) => {
          const scorePercent = (s.score * 100).toFixed(1);
          return `
            <div class="source-item">
              <div class="source-item-top">
                <span class="source-citation">#${idx + 1} &bull; ${escapeHtml(s.source)} (Page ${s.page_number})</span>
                <span class="source-score">${scorePercent}% Match</span>
              </div>
              <p class="source-snippet">"${escapeHtml(s.text)}"</p>
            </div>
          `;
        })
        .join('');

      sourcesAccordionHtml = `
        <div class="sources-card">
          <div class="sources-header" onclick="this.parentElement.classList.toggle('open')">
            <span class="sources-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
              </svg>
              View ${sources.length} Retrieved Sources & Citations
            </span>
            <span style="font-size: 11px; color: var(--text-muted);">&#9660;</span>
          </div>
          <div class="sources-content">
            ${sourceItems}
          </div>
        </div>
      `;
    }

    wrapper.innerHTML = `
      <div class="avatar assistant-avatar">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
          <line x1="8" y1="21" x2="16" y2="21"></line>
          <line x1="12" y1="17" x2="12" y2="21"></line>
        </svg>
      </div>
      <div class="message-content">
        <div class="message-header">
          <span class="sender-name">Local RAG Assistant</span>
          <span class="message-time">Just now</span>
        </div>
        <div class="message-body">
          <p>${formatMarkdown(data.answer)}</p>
          ${statusPillHtml}
          ${sourcesAccordionHtml}
        </div>
      </div>
    `;

    chatMessages.appendChild(wrapper);
    scrollToBottom();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // --------------------------------------------------------------------------
  // Quick Prompt Buttons
  // --------------------------------------------------------------------------
  [chipQuestion1, chipQuestion2, chipQuestion3].forEach((chip) => {
    if (chip) {
      chip.addEventListener('click', () => {
        const query = chip.getAttribute('data-query');
        if (query) {
          questionInput.value = query;
          handleSend();
        }
      });
    }
  });

  // --------------------------------------------------------------------------
  // Toast Helper
  // --------------------------------------------------------------------------
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // --------------------------------------------------------------------------
  // Utilities
  // --------------------------------------------------------------------------
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatMarkdown(text) {
    if (!text) return '';
    let escaped = escapeHtml(text);
    // Bold
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Code
    escaped = escaped.replace(/`(.*?)`/g, '<code>$1</code>');
    // Line breaks
    escaped = escaped.replace(/\n/g, '<br>');
    return escaped;
  }
});
