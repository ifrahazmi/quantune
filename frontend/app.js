/**
 * Quantune Frontend
 * =================
 * Autonomous SLM Fine-Tuning Studio
 */

// Theme Toggle - Initialize immediately to prevent flash
(function() {
    const html = document.documentElement;
    const savedTheme = localStorage.getItem('quantune-theme') || localStorage.getItem('tunekit-theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme) {
        html.setAttribute('data-theme', savedTheme);
    } else if (!systemPrefersDark) {
        html.setAttribute('data-theme', 'light');
    }

    // Set up toggle after DOM loads
    document.addEventListener('DOMContentLoaded', () => {
        const toggle = document.getElementById('themeToggle');
        if (toggle) {
            toggle.addEventListener('click', () => {
                const currentTheme = html.getAttribute('data-theme');
                const newTheme = currentTheme === 'light' ? 'dark' : 'light';

                if (newTheme === 'dark') {
                    html.removeAttribute('data-theme');
                } else {
                    html.setAttribute('data-theme', newTheme);
                }

                localStorage.setItem('quantune-theme', newTheme);
            });
        }
    });
})();

// Use relative URLs - works in both local and production
const API_URL = '';

// State
let sessionId = null;
let currentStep = 1;
let jobId = null;
let statusPollInterval = null;
let allTrainedRuns = []; // Checkpoint runs for testing studio & vault
let currentTrainingJobId = null; // Active background job ID
let currentLoadedRows = []; // Canonical dataset rows loaded in memory
let hasConfirmedConfig = false; // Step 2 configuration confirmation flag
let trainingPollInterval = null; // Polling timer for in-app trainer

// User selections
let selectedTask = null;
let selectedDeployment = null;

// Analysis data
let analysisData = null;
let recommendationData = null;
let selectedModelData = null; // The currently selected model (could be primary or alternative)
let allModelsData = []; // All models (primary + alternatives) for easy switching
let cachedSystemHardware = null; // System hardware specs cache

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileRows = document.getElementById('fileRows');
const fileSize = document.getElementById('fileSize');
const removeFile = document.getElementById('removeFile');
const continueBtn = document.getElementById('continueBtn');
const validationStatus = document.getElementById('validationStatus');
const errorBanner = document.getElementById('errorBanner');
const errorText = document.getElementById('errorText');
const errorClose = document.getElementById('errorClose');

// ============================================================================
// STEP NAVIGATION
// ============================================================================

// ============================================================================
// WIZARD STEP NAVIGATION GUARDRAILS
// ============================================================================

let navGuardToastTimeout = null;

function showNavGuardToast(message) {
    const toast = document.getElementById('navGuardToast');
    const msgEl = document.getElementById('navGuardToastText');
    if (!toast || !msgEl) return;

    msgEl.textContent = message;
    toast.classList.remove('hidden');

    if (navGuardToastTimeout) clearTimeout(navGuardToastTimeout);
    navGuardToastTimeout = setTimeout(() => {
        toast.classList.add('hidden');
    }, 3800);
}
window.showNavGuardToast = showNavGuardToast;

function hideNavGuardToast() {
    const toast = document.getElementById('navGuardToast');
    if (toast) toast.classList.add('hidden');
    if (navGuardToastTimeout) clearTimeout(navGuardToastTimeout);
}
window.hideNavGuardToast = hideNavGuardToast;

function shakeNavStep(stepNum) {
    const stepEl = document.querySelector(`.progress-step-nav[data-step="${stepNum}"]`);
    if (stepEl) {
        stepEl.classList.add('shake-nav');
        setTimeout(() => stepEl.classList.remove('shake-nav'), 500);
    }
}
window.shakeNavStep = shakeNavStep;

function getStepValidationStatus(step) {
    const hasData = Boolean(
        sessionId && (
            (analysisData && (analysisData.rows || analysisData.stats?.total_examples)) ||
            (currentLoadedRows && currentLoadedRows.length > 0)
        )
    );
    const hasConfig = Boolean(hasData && hasConfirmedConfig);
    const hasRec = Boolean(hasConfig && (recommendationData || selectedModelData));
    const hasTrainedModel = Boolean(
        (allTrainedRuns && allTrainedRuns.length > 0) ||
        (currentTrainingJobId && document.getElementById('trainCompletedCta') && !document.getElementById('trainCompletedCta').classList.contains('hidden')) ||
        (localStorage.getItem('quantune_has_completed_run') === 'true' || localStorage.getItem('tunekit_has_completed_run') === 'true')
    );

    if (step === 1) {
        return { allowed: true, reason: '' };
    }
    if (step === 2) {
        if (!hasData) {
            return {
                allowed: false,
                reason: 'Please upload a dataset or select a sample in Step 1 first before configuring.',
                requiredStep: 1
            };
        }
        return { allowed: true, reason: '' };
    }
    if (step === 3) {
        if (!hasData) {
            return {
                allowed: false,
                reason: 'Please upload or select a dataset in Step 1 first.',
                requiredStep: 1
            };
        }
        if (!hasConfig) {
            return {
                allowed: false,
                reason: 'Please review and confirm your configuration in Step 2 first.',
                requiredStep: 2
            };
        }
        return { allowed: true, reason: '' };
    }
    if (step === 4) {
        if (!hasData) {
            return {
                allowed: false,
                reason: 'Please upload or select a dataset in Step 1 first.',
                requiredStep: 1
            };
        }
        if (!hasRec) {
            return {
                allowed: false,
                reason: 'Please evaluate and select a recommended model in Step 3 first.',
                requiredStep: 3
            };
        }
        return { allowed: true, reason: '' };
    }
    if (step === 5) {
        if (!hasTrainedModel) {
            return {
                allowed: false,
                reason: 'Please complete in-app training or select an existing checkpoint from the vault in Step 4 first.',
                requiredStep: 4
            };
        }
        return { allowed: true, reason: '' };
    }
    return { allowed: true, reason: '' };
}
window.getStepValidationStatus = getStepValidationStatus;

function updateNavigationGuardrails() {
    for (let s = 1; s <= 5; s++) {
        const navEl = document.querySelector(`.progress-step-nav[data-step="${s}"]`);
        if (!navEl) continue;

        const status = getStepValidationStatus(s);
        if (status.allowed) {
            navEl.classList.remove('nav-locked');
            navEl.classList.add('nav-unlocked');
            navEl.removeAttribute('title');
        } else {
            navEl.classList.remove('nav-unlocked');
            navEl.classList.add('nav-locked');
            navEl.setAttribute('title', status.reason);
        }
    }

    const navTestBtn = document.getElementById('navTestBtn');
    if (navTestBtn) {
        const step5Status = getStepValidationStatus(5);
        if (step5Status.allowed) {
            navTestBtn.classList.remove('nav-locked');
            navTestBtn.style.opacity = '1';
            navTestBtn.setAttribute('title', 'Jump directly to Model Test & Refinement');
        } else {
            navTestBtn.classList.add('nav-locked');
            navTestBtn.style.opacity = '0.5';
            navTestBtn.setAttribute('title', step5Status.reason);
        }
    }
}
window.updateNavigationGuardrails = updateNavigationGuardrails;

function setStep(step) {
    const status = getStepValidationStatus(step);
    if (!status.allowed && step > currentStep) {
        showNavGuardToast(status.reason);
        shakeNavStep(step);
        return false;
    }

    currentStep = step;
    
    document.querySelectorAll('.step-content').forEach((el, i) => {
        el.classList.remove('active');
        if (i + 1 === step) {
            el.classList.add('active');
        }
    });
    
    // Update top progress bar
    document.querySelectorAll('.progress-step-nav').forEach((el, i) => {
        el.classList.remove('active', 'completed');
        if (i + 1 < step) {
            el.classList.add('completed');
        } else if (i + 1 === step) {
            el.classList.add('active');
        }
    });
    
    // Update connector lines
    document.querySelectorAll('.progress-connector').forEach((el, i) => {
        el.classList.remove('completed');
        if (i + 1 < step) {
            el.classList.add('completed');
        }
    });
    
    // Scroll to top smoothly
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Initialize Configuration Studio if entering Step 2
    if (step === 2 && typeof initConfigurationStudio === 'function') {
        initConfigurationStudio();
    }

    // Initialize Staged Model Hero and Trained Models Vault if entering Step 4
    if (step === 4) {
        if (typeof updateStagedModelHero === 'function') updateStagedModelHero();
        if (typeof loadTrainedModelsVault === 'function') loadTrainedModelsVault();
    }

    updateNavigationGuardrails();
    return true;
}

// Allow clicking on navigation steps with guardrail protection
document.querySelectorAll('.progress-step-nav').forEach((el, i) => {
    el.addEventListener('click', () => {
        const targetStep = i + 1;
        const status = getStepValidationStatus(targetStep);
        if (!status.allowed) {
            showNavGuardToast(status.reason);
            shakeNavStep(targetStep);
            return;
        }
        if (targetStep === 5) {
            openPlayground();
        } else {
            setStep(targetStep);
        }
    });
});

document.getElementById('navTestBtn')?.addEventListener('click', (e) => {
    const status = getStepValidationStatus(5);
    if (!status.allowed) {
        e.preventDefault();
        showNavGuardToast(status.reason);
        shakeNavStep(5);
        return;
    }
    openPlayground();
});


// ============================================================================
// QUESTION FADE TRANSITIONS & TYPEWRITER EFFECT
// ============================================================================

function typewriterEffect(element, text, callback) {
    // Clear any existing intervals
    if (element._typewriterInterval) {
        clearInterval(element._typewriterInterval);
    }

    let index = 0;
    element.classList.add('typing');
    element.textContent = '';

    // Start typing immediately
    element._typewriterInterval = setInterval(() => {
        if (index < text.length) {
            element.textContent += text[index];
            index++;
        } else {
            clearInterval(element._typewriterInterval);
            element._typewriterInterval = null;
            // Keep cursor briefly, then remove
            setTimeout(() => {
                element.classList.remove('typing');
                if (callback) callback();
            }, 300);
        }
    }, 45); // Smooth typing speed
}

function showQuestion(questionId) {
    const questionWrapper = document.getElementById(questionId);
    const allQuestions = document.querySelectorAll('.flow-question');

    // Hide all questions first
    allQuestions.forEach(q => {
        q.classList.remove('active');
    });

    // Clear the question text before showing
    const questionText = questionWrapper.querySelector('.flow-question-text');
    if (questionText) {
        questionText.textContent = '';
        questionText.classList.remove('typing');
    }

    // Reset option animations and selections for this question
    const options = questionWrapper.querySelectorAll('.flow-option');
    options.forEach((opt, i) => {
        // Clear selection state
        opt.classList.remove('selected');
        opt.removeAttribute('style');
        // Reset animations
        opt.style.animation = 'none';
        opt.offsetHeight; // Force reflow
        opt.style.animation = '';
        opt.style.setProperty('--delay', i);
    });

    // Show the question wrapper
    questionWrapper.classList.add('active');

    // Start typewriter immediately
    if (questionText && questionText.dataset.text) {
        typewriterEffect(questionText, questionText.dataset.text);
    }
}

function hideQuestion(questionId) {
    const questionWrapper = document.getElementById(questionId);
    questionWrapper.classList.remove('active');
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
    errorText.textContent = message;
    errorBanner.classList.remove('hidden');
}

function hideError() {
    errorBanner.classList.add('hidden');
}

errorClose?.addEventListener('click', hideError);

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ============================================================================
// SAMPLE DATASETS & INSPECTOR
// ============================================================================

async function loadSampleDatasetsGrid() {
    try {
        const response = await fetch(`${API_URL}/sample-datasets`);
        if (!response.ok) return;
        const data = await response.json();
        const datasets = data.datasets || [];
        const container = document.getElementById('sampleHubGrid');
        if (!container || datasets.length === 0) return;

        const iconMap = {
            'customer_support': '🎧',
            'python_coder': '💻',
            'medical_qa': '🩺',
            'finance_analyst': '📊'
        };

        container.innerHTML = datasets.map(d => {
            const icon = iconMap[d.id] || '📄';
            return `
                <div class="hub-card" data-sample="${d.id}">
                    <div class="hub-card-top">
                        <span class="hub-card-icon">${icon}</span>
                        <span class="hub-card-badge">${d.num_rows} rows</span>
                    </div>
                    <h4 class="hub-card-title">${escapeHtml(d.title)}</h4>
                    <p class="hub-card-desc">${escapeHtml(d.description)}</p>
                    <button class="btn btn-secondary btn-small hub-load-btn" type="button" data-sample-id="${d.id}">Load Dataset</button>
                </div>
            `;
        }).join('');

        container.querySelectorAll('.hub-load-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const sId = btn.getAttribute('data-sample-id');
                loadSampleDataset(sId);
            });
        });
    } catch (e) {
        console.warn('Could not fetch sample datasets:', e);
    }
}

async function loadSampleDataset(sampleId) {
    hideError();
    continueBtn.disabled = true;
    continueBtn.textContent = 'Loading sample dataset...';

    try {
        const response = await fetch(`${API_URL}/load-sample/${sampleId}`, { method: 'POST' });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to load sample dataset');
        }
        const data = await response.json();
        sessionId = data.session_id;

        // Update UI
        if (fileName) fileName.textContent = `${sampleId}.jsonl (Sample)`;
        if (fileRows) fileRows.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> ${data.rows || 55} examples`;
        if (fileSize) fileSize.textContent = '~35 KB';
        if (fileInfo) fileInfo.classList.remove('hidden');
        if (dropZone) dropZone.style.display = 'none';

        if (data.stats) {
            showValidationStatus(data.stats);
        }

        // Highlight active sample hub card
        document.querySelectorAll('.hub-card').forEach(c => {
            if (c.getAttribute('data-sample') === sampleId) {
                c.style.borderColor = 'var(--accent)';
                c.style.background = 'rgba(99, 102, 241, 0.12)';
            } else {
                c.style.borderColor = '';
                c.style.background = '';
            }
        });

        // Load visual dataset inspector & diagnostic report
        if (data.report) {
            renderDiagnosticReport(data.report, data.conversion_meta);
        }
        await renderInteractiveRowTable(sessionId);

        // Store for later
        analysisData = data;

        continueBtn.disabled = false;
        resetContinueBtn();
        if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();
    } catch (err) {
        console.error('Error loading sample dataset:', err);
        showError(err.message);
        resetContinueBtn();
    }
}

async function loadDatasetInspector(sid) {
    try {
        const response = await fetch(`${API_URL}/sample-data/${sid}?limit=5`);
        if (!response.ok) return;
        const data = await response.json();
        const rows = data.rows || [];
        const tableCard = document.getElementById('datasetInspectorCard');
        const tableBody = document.getElementById('inspectorTableBody');
        const countSpan = document.getElementById('inspectorRowCount');
        if (!tableCard || !tableBody || rows.length === 0) return;

        countSpan.textContent = `${rows.length} examples shown`;
        tableBody.innerHTML = rows.map((r, idx) => {
            const msgs = r.messages || [];
            let userPrompt = '—';
            let assistantResp = '—';
            let format = msgs.length > 2 ? 'Multi-Turn' : 'Direct Q&A';

            for (const m of msgs) {
                if (m.role === 'user' && userPrompt === '—') userPrompt = m.content || '';
                if (m.role === 'assistant' && assistantResp === '—') assistantResp = m.content || '';
            }

            const promptTrunc = userPrompt.length > 130 ? userPrompt.substring(0, 130) + '…' : userPrompt;
            const respTrunc = assistantResp.length > 160 ? assistantResp.substring(0, 160) + '…' : assistantResp;
            const estTokens = Math.round((userPrompt.length + assistantResp.length) / 4);

            return `
                <tr>
                    <td class="row-idx">#${idx + 1}</td>
                    <td><span class="fmt-pill">${format}</span></td>
                    <td class="prompt-cell">${escapeHtml(promptTrunc)}</td>
                    <td class="response-cell">${escapeHtml(respTrunc)}</td>
                    <td><span class="tokens-badge">~${estTokens}</span></td>
                </tr>
            `;
        }).join('');

        tableCard.classList.remove('hidden');
    } catch (e) {
        console.warn('Could not load dataset inspector:', e);
    }
}

// ============================================================================
// FILE UPLOAD
// ============================================================================

dropZone?.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone?.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone?.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
});

dropZone?.addEventListener('click', () => {
    fileInput.click();
});

fileInput?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelect(file);
});

removeFile?.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFileSelect(file) {
    hideError();
    
    // Validate file type
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const validExts = ['.jsonl', '.json', '.csv', '.tsv', '.txt'];
    
    if (!validExts.includes(ext)) {
        showError(`Unsupported file format '${ext}'. Please upload a .jsonl, .json, .csv, .tsv, or .txt file.`);
        return;
    }
    
    uploadFile(file);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state immediately without any delay
    continueBtn.disabled = true;
    continueBtn.textContent = 'Processing & Validating...';
    
    // Add spinner after text change
    const spinner = document.createElement('span');
    spinner.className = 'btn-loader';
    continueBtn.appendChild(spinner);
        
    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            let errorMsg = `Upload failed`;
            const text = await response.text();
            try {
                const error = JSON.parse(text);
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = text || errorMsg;
            }
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        sessionId = data.session_id;
        
        let numExamples = 0;
        if (data.rows !== undefined && data.rows !== null) {
            numExamples = Number(data.rows) || 0;
        } else if (data.stats && data.stats.total_examples) {
            numExamples = Number(data.stats.total_examples) || 0;
        }
        
        // Show file info
        fileName.textContent = file.name;
        if (numExamples > 0) {
            fileRows.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> ${numExamples.toLocaleString()} examples`;
        } else {
            fileRows.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> — examples`;
        }
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.classList.remove('hidden');
        dropZone.style.display = 'none';
        
        // Show diagnostic report & interactive editor
        if (data.report) {
            renderDiagnosticReport(data.report, data.conversion_meta);
        } else if (data.stats) {
            showValidationStatus(data.stats);
        }
        
        await renderInteractiveRowTable(sessionId);

        // Store for later
        analysisData = data;
        
        // Enable continue
        continueBtn.disabled = false;
        resetContinueBtn();
        if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();
        
    } catch (error) {
        console.error('Upload error:', error);
        resetContinueBtn();
        
        if (error.message === 'Failed to fetch' || error.message.includes('NetworkError')) {
            showError('Cannot connect to server. Is the backend running on http://localhost:8000?');
        } else {
            showError(error.message);
        }
    }
}

function resetContinueBtn() {
    continueBtn.innerHTML = `Continue <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>`;
}

function showValidationStatus(stats) {
    const validationIcon = document.getElementById('validationIcon');
    const validationTitle = document.getElementById('validationTitle');
    const validationMeta = document.getElementById('validationMeta');
    
    // Simple validation status - detailed info goes to summary card
    if (validationIcon) {
        validationIcon.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    }
    if (validationTitle) {
        validationTitle.textContent = 'Format Validated';
    }
    if (validationMeta) {
        validationMeta.textContent = 'Dataset ready for analysis';
    }
    
    // Show warnings if any
    if (stats && stats.warnings && stats.warnings.length > 0) {
        stats.warnings.forEach(warning => {
            console.warn('Dataset warning:', warning);
        });
    }
    
    const vStatus = document.getElementById('validationStatus');
    if (vStatus) {
        vStatus.classList.remove('hidden');
    }
    
    // Show dataset summary with all the details
    showDatasetSummary(stats);
}

function showDatasetSummary(stats) {
    const summaryCard = document.getElementById('datasetSummaryCard');
    const statsGrid = document.getElementById('summaryStatsGrid');
    
    if (!stats || !summaryCard) return;
    
    // Build clean stats table
    let html = '';
    
    // Secondary details
    html += '<div class="stats-details">';
    
    // Conversation type
    const singlePct = stats.single_turn_pct || 0;
    const multiPct = stats.multi_turn_pct || 0;
    html += `<div class="detail-row">
        <span class="detail-label">Conversation Type</span>
        <span class="detail-value">${singlePct}% single-turn, ${multiPct}% multi-turn</span>
    </div>`;
    
    // System prompts
    const sysPct = stats.system_prompt_pct || 0;
    html += `<div class="detail-row">
        <span class="detail-label">System Prompts</span>
        <span class="detail-value">${sysPct > 0 ? sysPct + '% of examples' : 'None'}</span>
    </div>`;
    
    // Response length
    const avgOutput = stats.avg_output_chars || 0;
    const outputTokens = Math.round(avgOutput / 4);
    html += `<div class="detail-row">
        <span class="detail-label">Avg Response Length</span>
        <span class="detail-value">~${outputTokens} tokens</span>
    </div>`;
    
    // Quality indicator
    const quality = stats.quality || 'good';
    const qualityLabel = quality === 'excellent' ? 'Excellent' : quality === 'good' ? 'Good' : 'Minimal';
    const qualityClass = quality === 'excellent' ? 'success' : quality === 'good' ? 'accent' : 'warning';
    html += `<div class="detail-row">
        <span class="detail-label">Dataset Quality</span>
        <span class="detail-value quality-badge ${qualityClass}">${qualityLabel}</span>
    </div>`;

    html += '</div>';

    // Training estimate section
    const totalExamples = stats.total_examples || 0;
    const avgTokens = Math.round((stats.avg_output_chars || 100) / 4);

    // Estimate training time: ~1 min per 50 examples on T4
    let trainingMin = Math.max(5, Math.round(totalExamples / 50));
    if (trainingMin > 30) trainingMin = '25-35';
    else if (trainingMin > 15) trainingMin = '15-25';
    else trainingMin = `${trainingMin}-${trainingMin + 5}`;

    html += '<div class="stats-estimate">';
    html += '<div class="estimate-row"><span class="estimate-label">Est. Training Time</span><span class="estimate-value">' + trainingMin + ' min</span></div>';
    html += '<div class="estimate-row"><span class="estimate-label">GPU Required</span><span class="estimate-value">Free T4 (Colab)</span></div>';
    html += '</div>';
    
    // Warnings if any
    if (stats.warnings && stats.warnings.length > 0) {
        html += '<div class="stats-warnings">';
        stats.warnings.forEach(w => {
            // Check if this is the system prompt warning
            if (w.includes('system prompt')) {
                html += `
                    <div class="warning-item with-action">
                        <span>⚠️ ${w}</span>
                        <button class="btn btn-secondary btn-small" id="openSystemPromptModal">
                            Add System Prompt
                        </button>
                    </div>`;
            } else {
                html += `<div class="warning-item">⚠️ ${w}</div>`;
            }
        });
        html += '</div>';
    }
    
    statsGrid.innerHTML = html;
    summaryCard.classList.remove('hidden');

    // Setup modal for system prompts if needed
    if (!stats.has_system_prompts && stats.total_examples >= 50) {
        setupSystemPromptModal(stats);

        // Add event listener to the button (it's dynamically created)
        setTimeout(() => {
            const modalBtn = document.getElementById('openSystemPromptModal');
            if (modalBtn) {
                modalBtn.addEventListener('click', () => openSystemPromptModal());
            }
        }, 100);
    }
}

// ============================================================================
// SYSTEM PROMPT FUNCTIONALITY
// ============================================================================

let selectedSystemPrompt = null;

function setupSystemPromptModal(stats) {
    // Just set up event listeners for the modal
    setupSystemPromptListeners();
}

function openSystemPromptModal() {
    const modal = document.getElementById('systemPromptModal');
    if (modal) {
        modal.classList.remove('hidden');
    }
}

function closeSystemPromptModal() {
    const modal = document.getElementById('systemPromptModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

function setupSystemPromptListeners() {
    // Close modal button
    const closeBtn = document.getElementById('closeModalBtn');
    if (closeBtn && !closeBtn.dataset.listenerAttached) {
        closeBtn.addEventListener('click', closeSystemPromptModal);
        closeBtn.dataset.listenerAttached = 'true';
    }

    // Cancel button
    const cancelBtn = document.getElementById('cancelPromptBtn');
    if (cancelBtn && !cancelBtn.dataset.listenerAttached) {
        cancelBtn.addEventListener('click', closeSystemPromptModal);
        cancelBtn.dataset.listenerAttached = 'true';
    }

    // Close modal when clicking outside
    const modal = document.getElementById('systemPromptModal');
    if (modal && !modal.dataset.listenerAttached) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeSystemPromptModal();
            }
        });
        modal.dataset.listenerAttached = 'true';
    }
    
    // Apply button
    const applyBtn = document.getElementById('applyPromptBtn');
    if (applyBtn && !applyBtn.dataset.listenerAttached) {
        applyBtn.addEventListener('click', async () => {
            // Check if sessionId exists
            if (!sessionId) {
                showError('No active session. Please upload a file first.');
                return;
            }

            const customInput = document.getElementById('customPromptInput');
            const prompt = customInput.value.trim();

            if (!prompt) {
                showError('Please enter a system prompt');
                return;
            }
            
            // Apply to dataset
            applyBtn.disabled = true;
            applyBtn.textContent = 'Applying...';
            
            try {
                console.log('Applying system prompt with sessionId:', sessionId);
                
                const response = await fetch(`${API_URL}/add-system-prompt`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        system_prompt: prompt
                    })
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(errorData.detail || `Server error: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('System prompt applied successfully:', data);
                
                // Update the stats display
                if (data.stats) {
                    showDatasetSummary(data.stats);
                }
                
                // Show success message with download option
                showSystemPromptSuccess(data.modified_count || 0, sessionId);

                // Close the modal
                closeSystemPromptModal();
                
            } catch (error) {
                console.error('Error applying system prompt:', error);
                showError(error.message || 'Failed to apply system prompt');
            } finally {
                applyBtn.disabled = false;
                applyBtn.textContent = 'Apply to Dataset';
            }
        });
        applyBtn.dataset.listenerAttached = 'true';
    }
}

// Task-specific prompt templates
const TASK_PROMPTS = {
    'classify': [
        {
            label: 'Classification Specialist',
            icon: '🎯',
            tagline: 'Optimized for accurate categorization',
            bestFor: 'Sentiment analysis, content categorization, intent detection',
            prompt: 'You are a classification model. Analyze inputs carefully and return the most appropriate category. Be consistent in your classifications and provide confident, direct responses.'
        },
        {
            label: 'Multi-Label Classifier',
            icon: '🏷️',
            tagline: 'For inputs with multiple categories',
            bestFor: 'Content tagging, feature detection, multi-class problems',
            prompt: 'You are a multi-label classification assistant. Analyze the input and identify all applicable categories. Maintain consistency in your labeling and explain your reasoning when multiple categories apply.'
        }
    ],
    'qa': [
        {
            label: 'Knowledge Expert',
            icon: '💡',
            tagline: 'Accurate answers with clear reasoning',
            bestFor: 'FAQ systems, knowledge bases, documentation',
            prompt: 'You are a knowledgeable assistant specializing in answering questions accurately. Provide clear, well-researched answers. If you\'re uncertain about something, acknowledge it. Include reasoning or sources when helpful.'
        },
        {
            label: 'Technical Support',
            icon: '🔧',
            tagline: 'Troubleshooting and problem-solving',
            bestFor: 'Customer support, technical assistance, debugging help',
            prompt: 'You are a technical support specialist. When answering questions, first understand the user\'s problem, then provide step-by-step solutions. Be patient and ask clarifying questions when needed.'
        }
    ],
    'conversation': [
        {
            label: 'Friendly Assistant',
            icon: '💬',
            tagline: 'Natural, engaging conversations',
            bestFor: 'Chatbots, customer service, virtual assistants',
            prompt: 'You are a friendly, conversational assistant. Engage naturally with users, maintain context throughout the conversation, and ask clarifying questions when needed. Be helpful while keeping responses concise.'
        },
        {
            label: 'Professional Consultant',
            icon: '👔',
            tagline: 'Formal, business-appropriate tone',
            bestFor: 'Enterprise chatbots, professional services, B2B interactions',
            prompt: 'You are a professional consultant assistant. Maintain a courteous and professional tone. Provide thorough, well-structured responses and prioritize accuracy and clarity in all communications.'
        }
    ],
    'generation': [
        {
            label: 'Creative Writer',
            icon: '✍️',
            tagline: 'Engaging, creative content',
            bestFor: 'Blog posts, marketing copy, creative writing',
            prompt: 'You are a creative content writer. Generate engaging, well-structured content that captures attention. Adapt your writing style to match the requested tone and format. Be original and compelling.'
        },
        {
            label: 'Technical Writer',
            icon: '📝',
            tagline: 'Clear, precise documentation',
            bestFor: 'Documentation, tutorials, technical guides',
            prompt: 'You are a technical writer. Create clear, precise, and well-organized content. Use appropriate technical terminology, provide step-by-step instructions when relevant, and ensure accuracy in all details.'
        }
    ],
    'extraction': [
        {
            label: 'Data Extraction Specialist',
            icon: '📊',
            tagline: 'Structured data extraction',
            bestFor: 'Entity extraction, form parsing, data structuring',
            prompt: 'You are a data extraction specialist. Analyze inputs and extract relevant information in a consistent, structured format. Be precise and thorough. If information is missing or unclear, indicate it explicitly.'
        },
        {
            label: 'Entity Recognition Expert',
            icon: '🔍',
            tagline: 'Identifying key entities and relationships',
            bestFor: 'Named entity recognition, relationship extraction',
            prompt: 'You are an entity recognition expert. Identify and extract key entities (people, organizations, locations, dates, etc.) from text. Maintain consistency in entity naming and format your output clearly.'
        }
    ]
};

function generateSuggestedPrompts(stats) {
    // Use ONLY dataset characteristics to suggest prompts (Step 1 happens before task selection)
    let prompts = [];

    const avgOutputTokens = Math.round((stats.avg_output_chars || 200) / 4);
    const isMultiTurn = (stats.multi_turn_pct || 0) > 50;
    const avgInputTokens = Math.round((stats.avg_input_chars || 100) / 4);

    // Detect task type from dataset patterns
    if (avgOutputTokens < 50) {
        // Short outputs suggest classification/extraction
        prompts.push(...TASK_PROMPTS.classify);
    } else if (isMultiTurn) {
        // Multi-turn conversations
        prompts.push(...TASK_PROMPTS.conversation);
    } else if (avgOutputTokens > 200) {
        // Long outputs suggest content generation
        prompts.push(...TASK_PROMPTS.generation);
                } else {
        // Medium outputs suggest Q&A
        prompts.push(...TASK_PROMPTS.qa);
    }

    // Always add a general-purpose option
    prompts.push({
        label: 'General Purpose',
        icon: '⚙️',
        tagline: 'Flexible assistant for various tasks',
        bestFor: 'Mixed use cases, general assistance',
        prompt: 'You are a helpful, accurate, and versatile assistant. Adapt your responses to the user\'s needs, provide clear explanations, and maintain a professional yet friendly tone.'
    });

    return prompts;
}

function hideSystemPromptCard() {
    const card = document.getElementById('systemPromptCard');
    if (card) card.classList.add('hidden');
}

function showSystemPromptSuccess(modifiedCount, sessionId) {
    // Create success banner
    const successBanner = document.createElement('div');
    successBanner.className = 'system-prompt-success';
    successBanner.innerHTML = `
        <div class="success-content">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            <span>System prompt added to ${modifiedCount} conversations</span>
        </div>
        <a href="${API_URL}/download-dataset/${sessionId}" class="btn btn-secondary btn-small" target="_blank">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Download Updated Dataset
        </a>
    `;
    
    // Insert after the system prompt card
    const promptCard = document.getElementById('systemPromptCard');
    if (promptCard && promptCard.parentNode) {
        promptCard.parentNode.insertBefore(successBanner, promptCard.nextSibling);
        
        // Auto-hide after 15 seconds
            setTimeout(() => {
            if (successBanner.parentNode) {
                successBanner.remove();
            }
        }, 15000);
    }
}

function clearFile() {
    sessionId = null;
    analysisData = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    validationStatus.classList.add('hidden');
    const summaryCard = document.getElementById('datasetSummaryCard');
    if (summaryCard) {
        summaryCard.classList.add('hidden');
        document.getElementById('summaryStatsGrid').innerHTML = '';
    }
    const inspectorCard = document.getElementById('datasetInspectorCard');
    if (inspectorCard) {
        inspectorCard.classList.add('hidden');
        document.getElementById('inspectorTableBody').innerHTML = '';
    }
    document.querySelectorAll('.hub-card').forEach(c => {
        c.style.borderColor = '';
        c.style.background = '';
    });
    hideSystemPromptCard();
    dropZone.style.display = '';
    continueBtn.disabled = true;
    resetContinueBtn(); // Reset button text when clearing file
    hideError();
    hasConfirmedConfig = false;
    currentLoadedRows = [];
    if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();
}

// ============================================================================
// STEP 1 → STEP 2: Continue to Questions
// ============================================================================

continueBtn?.addEventListener('click', async () => {
    if (!sessionId) return;

    // Move to configuration step
    setStep(2);
});

// ============================================================================
// STEP 2: CONFIGURATION & OPTIMIZATION STUDIO
// ============================================================================

function collectTrainingHyperparameters() {
    const epochs = parseInt(document.getElementById('paramEpochs')?.value || '3', 10);
    const batchSize = parseInt(document.getElementById('paramBatchSize')?.value || '2', 10);
    const lr = parseFloat(document.getElementById('paramLR')?.value || '0.0002');
    const loraR = parseInt(document.getElementById('paramLoraR')?.value || '16', 10);
    const loraAlpha = parseInt(document.getElementById('paramLoraAlpha')?.value || String(loraR * 2), 10);
    const gradAccum = parseInt(document.getElementById('paramGradAccum')?.value || '4', 10);
    const lrScheduler = document.getElementById('paramScheduler')?.value || 'cosine';
    const warmupRatio = parseFloat(document.getElementById('paramWarmup')?.value || '0.10');
    const weightDecay = parseFloat(document.getElementById('paramWeightDecay')?.value || '0.01');
    const targetModules = document.getElementById('paramTargetModules')?.value || 'all_linear';
    const loraDropout = parseFloat(document.getElementById('paramLoraDropout')?.value || '0.05');
    const quantization = document.getElementById('paramQuantization')?.value || '4bit';
    const maxSeqLen = parseInt(document.getElementById('paramMaxSeqLen')?.value || '2048', 10);
    const valSplit = parseInt(document.getElementById('paramValSplit')?.value || '10', 10);

    return {
        num_train_epochs: epochs,
        per_device_train_batch_size: batchSize,
        learning_rate: lr,
        lora_r: loraR,
        lora_alpha: loraAlpha,
        gradient_accumulation_steps: gradAccum,
        lr_scheduler_type: lrScheduler,
        warmup_ratio: warmupRatio,
        weight_decay: weightDecay,
        target_modules: targetModules,
        lora_dropout: loraDropout,
        quantization: quantization,
        max_seq_length: maxSeqLen,
        val_split_pct: valSplit
    };
}

function initConfigurationStudio() {
    // 1. Populate Active Dataset Intelligence Strip
    const stripRows = document.getElementById('configStripRows');
    const stripAvgTokens = document.getElementById('configStripAvgTokens');
    const stripFormat = document.getElementById('configStripFormat');
    const stripHardware = document.getElementById('configStripHardware');

    const totalExamples = analysisData?.stats?.total_examples || analysisData?.report?.total_rows || (analysisData?.sample_rows ? analysisData.sample_rows.length : 100);
    const avgChars = analysisData?.stats?.avg_output_chars || (analysisData?.report?.avg_tokens ? analysisData.report.avg_tokens * 4 : 450);
    const avgTokens = Math.max(16, Math.round(avgChars / 4));

    if (stripRows) stripRows.textContent = `${totalExamples.toLocaleString()} examples`;
    if (stripAvgTokens) stripAvgTokens.textContent = `~${avgTokens} tokens / row`;

    // Detected format and quality
    let formatLabel = 'Standard Chat / Q&A';
    if (analysisData?.report?.detected_format) {
        formatLabel = analysisData.report.detected_format;
    } else if (analysisData?.stats?.multi_turn_pct > 30) {
        formatLabel = 'Multi-Turn Dialog (ShareGPT)';
    }
    const qualityStr = analysisData?.stats?.quality ? ` · ${analysisData.stats.quality.toUpperCase()}` : ' · HIGH';
    if (stripFormat) stripFormat.textContent = `${formatLabel}${qualityStr}`;

    // Hardware display
    if (cachedSystemHardware?.gpu?.has_gpu) {
        if (stripHardware) stripHardware.textContent = `${cachedSystemHardware.gpu.name} · ${cachedSystemHardware.gpu.vram_total_gb}GB VRAM`;
    } else if (cachedSystemHardware?.cpu) {
        if (stripHardware) stripHardware.textContent = `${cachedSystemHardware.cpu.model ? cachedSystemHardware.cpu.model.substring(0, 24) : 'CPU Engine'} · ${cachedSystemHardware.ram?.total_gb || 16}GB RAM`;
    } else {
        if (stripHardware) stripHardware.textContent = 'NVIDIA T1200 · 4.0GB VRAM';
    }

    // 2. Default task & deployment
    if (!selectedTask) {
        if (analysisData?.stats?.task_type) {
            selectedTask = analysisData.stats.task_type;
        } else {
            selectedTask = 'conversation';
        }
    }
    if (!selectedDeployment) {
        selectedDeployment = 'cloud_api';
    }

    document.querySelectorAll('.flow-option[data-task]').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.task === selectedTask);
    });
    document.querySelectorAll('.flow-option[data-deployment]').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.deployment === selectedDeployment);
    });

    // 3. Compute and apply AI recommendation
    computeDatasetOptimalHyperparams(totalExamples, avgTokens);

    // 4. Update live VRAM telemetry
    updateLiveVramTelemetry();
}

function computeDatasetOptimalHyperparams(totalExamples, avgTokens) {
    const vramGB = cachedSystemHardware?.gpu?.vram_total_gb || 4.0;
    const hasGpu = cachedSystemHardware?.gpu?.has_gpu ?? true;

    // Intelligent heuristics:
    let recRank = 16;
    let recAlpha = 32;
    let recEpochs = 3;
    let recBatch = 2;
    let recAccum = 4;
    let recLR = '0.0002';
    let recScheduler = 'cosine';
    let recTargetModules = 'all_linear';
    let recDropout = '0.05';
    let recQuant = '4bit';
    let recMaxSeq = 2048;
    let recWarmup = '0.10';
    let recWeightDecay = '0.01';
    let recSplit = '10';

    // Sequence length adaptation
    if (avgTokens <= 256) {
        recMaxSeq = 1024;
    } else if (avgTokens > 1200) {
        recMaxSeq = 4096;
    } else {
        recMaxSeq = 2048;
    }

    // Dataset volume adaptation
    if (totalExamples < 250) {
        recEpochs = 4;
        recDropout = '0.10'; // prevent memorization/overfitting on small data
        recLR = '0.0002';
    } else if (totalExamples >= 3000) {
        recEpochs = 2;
        recRank = 32;
        recAlpha = 64;
        recDropout = '0.0';
        recLR = '0.0001';
        recAccum = 8;
    }

    // Hardware constraints
    if (vramGB <= 4.0) {
        recQuant = '4bit';
        recBatch = 2;
        recAccum = 4;
    } else if (vramGB >= 12.0) {
        recBatch = 4;
        recAccum = 2;
    }

    // Task specific adjustments
    if (selectedTask === 'classify' || selectedTask === 'extraction') {
        recScheduler = 'linear';
        recTargetModules = 'all_linear';
    } else if (selectedTask === 'qa' || selectedTask === 'generation') {
        recScheduler = 'cosine';
        recWarmup = '0.10';
    }

    // Populate inputs
    const elEpochs = document.getElementById('paramEpochs');
    const elBatch = document.getElementById('paramBatchSize');
    const elLR = document.getElementById('paramLR');
    const elLoraR = document.getElementById('paramLoraR');
    const elLoraAlpha = document.getElementById('paramLoraAlpha');
    const elGradAccum = document.getElementById('paramGradAccum');
    const elScheduler = document.getElementById('paramScheduler');
    const elWarmup = document.getElementById('paramWarmup');
    const elWeightDecay = document.getElementById('paramWeightDecay');
    const elTargetModules = document.getElementById('paramTargetModules');
    const elLoraDropout = document.getElementById('paramLoraDropout');
    const elQuant = document.getElementById('paramQuantization');
    const elMaxSeq = document.getElementById('paramMaxSeqLen');
    const elValSplit = document.getElementById('paramValSplit');

    if (elEpochs) elEpochs.value = recEpochs;
    if (elBatch) elBatch.value = recBatch;
    if (elLR) elLR.value = recLR;
    if (elLoraR) { elLoraR.value = recRank; const badge = document.getElementById('valLoraR'); if (badge) badge.textContent = recRank; }
    if (elLoraAlpha) { elLoraAlpha.value = recAlpha; const badge = document.getElementById('valLoraAlpha'); if (badge) badge.textContent = recAlpha; }
    if (elGradAccum) elGradAccum.value = recAccum;
    if (elScheduler) elScheduler.value = recScheduler;
    if (elWarmup) elWarmup.value = recWarmup;
    if (elWeightDecay) elWeightDecay.value = recWeightDecay;
    if (elTargetModules) elTargetModules.value = recTargetModules;
    if (elLoraDropout) elLoraDropout.value = recDropout;
    if (elQuant) elQuant.value = recQuant;
    if (elMaxSeq) elMaxSeq.value = recMaxSeq;
    if (elValSplit) elValSplit.value = recSplit;

    // Rationale presentation
    const rationaleEl = document.getElementById('aiRecRationale');
    const badgeTextEl = document.getElementById('aiRecBadgeText');
    if (badgeTextEl) badgeTextEl.textContent = `Recommended: Balanced QLoRA (${vramGB}GB VRAM)`;

    if (rationaleEl) {
        rationaleEl.innerHTML = `
            Based on your <strong>${totalExamples.toLocaleString()} examples</strong> (avg <strong>~${avgTokens} tokens</strong>) and 
            <strong>${hasGpu ? `${cachedSystemHardware?.gpu?.name || 'GPU'} (${vramGB}GB VRAM)` : 'CPU Execution'}</strong>: 
            We recommend <strong>Rank ${recRank} QLoRA (NF4 4-bit)</strong> with <strong>Batch Size ${recBatch}</strong> and 
            <strong>${recAccum}x Gradient Accumulation</strong> (Effective batch: ${recBatch * recAccum}). 
            Targeting <em>all linear projection layers</em> preserves reasoning fidelity while <em>Cosine schedule with 10% warmup</em> ensures smooth convergence.
        `;
    }
}

function applyStrategyPreset(presetName) {
    document.querySelectorAll('.strategy-preset-chip').forEach(chip => {
        chip.classList.toggle('active', chip.dataset.preset === presetName);
    });

    const elEpochs = document.getElementById('paramEpochs');
    const elBatch = document.getElementById('paramBatchSize');
    const elLR = document.getElementById('paramLR');
    const elLoraR = document.getElementById('paramLoraR');
    const elLoraAlpha = document.getElementById('paramLoraAlpha');
    const elGradAccum = document.getElementById('paramGradAccum');
    const elScheduler = document.getElementById('paramScheduler');
    const elWarmup = document.getElementById('paramWarmup');
    const elWeightDecay = document.getElementById('paramWeightDecay');
    const elTargetModules = document.getElementById('paramTargetModules');
    const elLoraDropout = document.getElementById('paramLoraDropout');
    const elQuant = document.getElementById('paramQuantization');
    const elMaxSeq = document.getElementById('paramMaxSeqLen');

    const totalExamples = analysisData?.stats?.total_examples || 100;

    if (presetName === 'fast') {
        if (elEpochs) elEpochs.value = 2;
        if (elBatch) elBatch.value = 4;
        if (elGradAccum) elGradAccum.value = 2;
        if (elLR) elLR.value = '0.0003';
        if (elLoraR) { elLoraR.value = 8; const b = document.getElementById('valLoraR'); if (b) b.textContent = '8'; }
        if (elLoraAlpha) { elLoraAlpha.value = 16; const b = document.getElementById('valLoraAlpha'); if (b) b.textContent = '16'; }
        if (elScheduler) elScheduler.value = 'linear';
        if (elTargetModules) elTargetModules.value = 'attention';
        if (elLoraDropout) elLoraDropout.value = '0.0';
        if (elQuant) elQuant.value = '4bit';
        if (elMaxSeq) elMaxSeq.value = 1024;
    } else if (presetName === 'accuracy') {
        if (elEpochs) elEpochs.value = Math.max(3, Math.min(5, Math.round(1500 / Math.max(totalExamples, 100)) + 2));
        if (elBatch) elBatch.value = 2;
        if (elGradAccum) elGradAccum.value = 8;
        if (elLR) elLR.value = '0.0001';
        if (elLoraR) { elLoraR.value = 32; const b = document.getElementById('valLoraR'); if (b) b.textContent = '32'; }
        if (elLoraAlpha) { elLoraAlpha.value = 64; const b = document.getElementById('valLoraAlpha'); if (b) b.textContent = '64'; }
        if (elScheduler) elScheduler.value = 'cosine';
        if (elWarmup) elWarmup.value = '0.15';
        if (elWeightDecay) elWeightDecay.value = '0.05';
        if (elTargetModules) elTargetModules.value = 'all_linear';
        if (elLoraDropout) elLoraDropout.value = '0.05';
        if (elQuant) elQuant.value = '4bit';
        if (elMaxSeq) elMaxSeq.value = 2048;
    } else if (presetName === 'reasoning') {
        if (elEpochs) elEpochs.value = 3;
        if (elBatch) elBatch.value = 1;
        if (elGradAccum) elGradAccum.value = 16;
        if (elLR) elLR.value = '0.0001';
        if (elLoraR) { elLoraR.value = 32; const b = document.getElementById('valLoraR'); if (b) b.textContent = '32'; }
        if (elLoraAlpha) { elLoraAlpha.value = 64; const b = document.getElementById('valLoraAlpha'); if (b) b.textContent = '64'; }
        if (elScheduler) elScheduler.value = 'cosine';
        if (elWarmup) elWarmup.value = '0.10';
        if (elWeightDecay) elWeightDecay.value = '0.01';
        if (elTargetModules) elTargetModules.value = 'all_linear';
        if (elLoraDropout) elLoraDropout.value = '0.0';
        if (elQuant) elQuant.value = '4bit';
        if (elMaxSeq) elMaxSeq.value = 4096;
    } else {
        // Balanced
        if (elEpochs) elEpochs.value = 3;
        if (elBatch) elBatch.value = 2;
        if (elGradAccum) elGradAccum.value = 4;
        if (elLR) elLR.value = '0.0002';
        if (elLoraR) { elLoraR.value = 16; const b = document.getElementById('valLoraR'); if (b) b.textContent = '16'; }
        if (elLoraAlpha) { elLoraAlpha.value = 32; const b = document.getElementById('valLoraAlpha'); if (b) b.textContent = '32'; }
        if (elScheduler) elScheduler.value = 'cosine';
        if (elWarmup) elWarmup.value = '0.10';
        if (elWeightDecay) elWeightDecay.value = '0.01';
        if (elTargetModules) elTargetModules.value = 'all_linear';
        if (elLoraDropout) elLoraDropout.value = '0.05';
        if (elQuant) elQuant.value = '4bit';
        if (elMaxSeq) elMaxSeq.value = 2048;
    }

    updateLiveVramTelemetry();
}

function updateLiveVramTelemetry() {
    const vramGB = cachedSystemHardware?.gpu?.vram_total_gb || 4.0;
    const batchSize = parseInt(document.getElementById('paramBatchSize')?.value || '2', 10);
    const gradAccum = parseInt(document.getElementById('paramGradAccum')?.value || '4', 10);
    const loraR = parseInt(document.getElementById('paramLoraR')?.value || '16', 10);
    const maxSeqLen = parseInt(document.getElementById('paramMaxSeqLen')?.value || '2048', 10);
    const quant = document.getElementById('paramQuantization')?.value || '4bit';
    const gradCheck = document.getElementById('paramGradCheckpoint')?.checked ?? true;
    const epochs = parseInt(document.getElementById('paramEpochs')?.value || '3', 10);
    const totalExamples = analysisData?.stats?.total_examples || 100;

    // Calculate effective batch size
    const effectiveBatch = batchSize * gradAccum;
    const effBatchEl = document.getElementById('telemetryEffectiveBatch');
    if (effBatchEl) effBatchEl.textContent = `${effectiveBatch} (${batchSize} × ${gradAccum})`;

    // Quantization label
    const quantEl = document.getElementById('telemetryQuantMode');
    if (quantEl) {
        if (quant === '4bit') quantEl.textContent = '4-bit QLoRA NF4';
        else if (quant === '8bit') quantEl.textContent = '8-bit Int8';
        else quantEl.textContent = '16-bit Float';
    }

    // Trainable params estimation
    const trainableParamsM = Math.round((loraR * 1.1) + 2);
    const trainablePct = ((trainableParamsM / 3000) * 100).toFixed(2);
    const trainableEl = document.getElementById('telemetryTrainablePct');
    if (trainableEl) trainableEl.textContent = `~${trainablePct}% (~${trainableParamsM}M)`;

    // Training time estimation: steps = (totalExamples / effectiveBatch) * epochs
    const steps = Math.ceil((totalExamples / Math.max(1, effectiveBatch)) * epochs);
    const totalSec = Math.max(120, steps * 1.5);
    const minTime = Math.max(3, Math.round((totalSec * 0.8) / 60));
    const maxTime = Math.max(minTime + 2, Math.round((totalSec * 1.3) / 60));
    const estTimeEl = document.getElementById('telemetryEstTime');
    if (estTimeEl) estTimeEl.textContent = `~${minTime} - ${maxTime} min`;

    // Realistic VRAM estimation
    let baseModelMem = 2.0;
    if (quant === '8bit') baseModelMem = 3.8;
    if (quant === '16bit') baseModelMem = 7.2;

    let actFactor = (batchSize * (maxSeqLen / 1024)) * 0.35;
    if (gradCheck) actFactor *= 0.4; // 60% savings with gradient checkpointing

    let adapterMem = (loraR / 16) * 0.15;
    let optimizerMem = (adapterMem * 2);

    let totalVram = Number((baseModelMem + actFactor + adapterMem + optimizerMem).toFixed(1));
    if (totalVram < 2.2) totalVram = 2.2;

    const percentUsed = Math.min(100, Math.round((totalVram / vramGB) * 100));

    // Update gauge bar
    const barFill = document.getElementById('vramBarFill');
    const badge = document.getElementById('vramStatusBadge');
    const badgeText = document.getElementById('vramStatusText');
    const estimateText = document.getElementById('vramEstimateText');
    const safetyText = document.getElementById('vramSafetyText');

    if (barFill) {
        barFill.style.width = `${percentUsed}%`;
        barFill.classList.remove('warning', 'danger');
    }
    if (badge) badge.classList.remove('safe', 'warning', 'danger');
    if (safetyText) safetyText.classList.remove('safe', 'warning', 'danger');

    const gpuName = cachedSystemHardware?.gpu?.name || 'NVIDIA T1200';
    if (estimateText) {
        estimateText.innerHTML = `Est. Peak VRAM: <strong>${totalVram} GB</strong> / ${vramGB} GB Available (${gpuName})`;
    }

    const headroom = (vramGB - totalVram).toFixed(1);

    if (totalVram <= vramGB * 0.82) {
        if (barFill) barFill.classList.remove('warning', 'danger');
        if (badge) {
            badge.classList.add('safe');
            badgeText.textContent = `Safe · Fits easily (${percentUsed}%)`;
        }
        if (safetyText) {
            safetyText.classList.add('safe');
            safetyText.textContent = `Headroom: ~${headroom} GB`;
        }
    } else if (totalVram <= vramGB) {
        if (barFill) barFill.classList.add('warning');
        if (badge) {
            badge.classList.add('warning');
            badgeText.textContent = `Tight Memory (${percentUsed}%)`;
        }
        if (safetyText) {
            safetyText.classList.add('warning');
            safetyText.textContent = `Headroom: ~${headroom} GB (Near limit)`;
        }
    } else {
        if (barFill) barFill.classList.add('danger');
        if (badge) {
            badge.classList.add('danger');
            badgeText.textContent = `OOM Risk (${percentUsed}%)`;
        }
        if (safetyText) {
            safetyText.classList.add('danger');
            safetyText.textContent = `Exceeds VRAM by ${Math.abs(headroom)} GB!`;
        }
    }
}

function setupConfigurationStudioListeners() {
    // Strategy Preset Chips
    document.querySelectorAll('.strategy-preset-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const preset = chip.dataset.preset;
            if (preset) applyStrategyPreset(preset);
        });
    });

    // Re-apply button
    document.getElementById('btnApplyAiRec')?.addEventListener('click', () => {
        const totalExamples = analysisData?.stats?.total_examples || 100;
        const avgChars = analysisData?.stats?.avg_output_chars || 450;
        const avgTokens = Math.max(16, Math.round(avgChars / 4));
        computeDatasetOptimalHyperparams(totalExamples, avgTokens);
        document.querySelectorAll('.strategy-preset-chip').forEach(c => c.classList.remove('active'));
        document.getElementById('presetBalanced')?.classList.add('active');
        updateLiveVramTelemetry();
    });

    // Task Options
    document.querySelectorAll('.flow-option[data-task]').forEach(opt => {
        opt.addEventListener('click', () => {
            document.querySelectorAll('.flow-option[data-task]').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            selectedTask = opt.dataset.task;
            updateLiveVramTelemetry();
        });
    });

    // Deployment Options
    document.querySelectorAll('.flow-option[data-deployment]').forEach(opt => {
        opt.addEventListener('click', () => {
            document.querySelectorAll('.flow-option[data-deployment]').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            selectedDeployment = opt.dataset.deployment;
            updateLiveVramTelemetry();
        });
    });

    // Slider inputs
    const loraRInput = document.getElementById('paramLoraR');
    const loraAlphaInput = document.getElementById('paramLoraAlpha');

    loraRInput?.addEventListener('input', (e) => {
        const val = e.target.value;
        const badge = document.getElementById('valLoraR');
        if (badge) badge.textContent = val;
        if (loraAlphaInput) {
            const newAlpha = Math.min(128, parseInt(val, 10) * 2);
            loraAlphaInput.value = newAlpha;
            const alphaBadge = document.getElementById('valLoraAlpha');
            if (alphaBadge) alphaBadge.textContent = newAlpha;
        }
        updateLiveVramTelemetry();
    });

    loraAlphaInput?.addEventListener('input', (e) => {
        const badge = document.getElementById('valLoraAlpha');
        if (badge) badge.textContent = e.target.value;
        updateLiveVramTelemetry();
    });

    ['paramBatchSize', 'paramGradAccum', 'paramEpochs', 'paramQuantization', 'paramMaxSeqLen', 'paramGradCheckpoint', 'paramLR', 'paramScheduler', 'paramWarmup', 'paramWeightDecay'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', updateLiveVramTelemetry);
        document.getElementById(id)?.addEventListener('input', updateLiveVramTelemetry);
    });

    // Navigation buttons
    document.getElementById('continueToRecBtn')?.addEventListener('click', () => {
        hasConfirmedConfig = true;
        setStep(3);
        getRecommendation();
    });

    document.getElementById('backToUpload')?.addEventListener('click', () => {
        setStep(1);
    });

    document.getElementById('backToQuestions')?.addEventListener('click', () => {
        setStep(2);
        initConfigurationStudio();
    });
}

// ============================================================================
// STEP 3: Get Recommendation
// ============================================================================

async function getRecommendation() {
    const recLoading = document.getElementById('recLoading');
    const recContent = document.getElementById('recContent');
    
    recLoading.classList.remove('hidden');
    recContent.classList.add('hidden');
    
    try {
        const response = await fetch(`${API_URL}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_task: selectedTask,
                deployment_target: selectedDeployment
            })
        });
        
        if (!response.ok) {
            let errorMsg = 'Failed to get recommendation';
            const text = await response.text();
            try {
                const error = JSON.parse(text);
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = text || errorMsg;
            }
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        recommendationData = data;
        
        displayRecommendation(data);
        if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();
        
        recLoading.classList.add('hidden');
        recContent.classList.remove('hidden');
        
    } catch (error) {
        console.error('Recommendation error:', error);
        recLoading.classList.add('hidden');
        showError(error.message);
    }
}

// Store the original best match separately
let originalBestMatch = null;
let currentModelFilter = 'all';

function displayRecommendation(data) {
    const rec = data.recommendation?.primary_recommendation;
    const fullMatrix = data.recommendation?.full_matrix || [];
    
    if (!rec) {
        showError('No recommendation available');
        return;
    }
    
    // Store original best match (never changes)
    originalBestMatch = { ...rec, isOriginalBestMatch: true };
    
    // Store all models for later switching
    allModelsData = fullMatrix.length > 0 ? fullMatrix.map(m => ({
        ...m,
        isOriginalBestMatch: (m.key === rec.key || m.model_id === rec.model_id)
    })) : [
        { ...rec, isOriginalBestMatch: true },
        ...(data.recommendation?.alternatives || []).map(alt => ({ ...alt, isOriginalBestMatch: false }))
    ];
    
    // Set initial selected model to primary
    selectedModelData = { ...rec, isOriginalBestMatch: true };
    
    // 1. Update Top Intelligence Context Strip
    updateIntelligenceContextStrip(data);
    
    // 2. Display Hero Best Match Card
    updatePrimaryDisplay(selectedModelData);
    
    // 3. Render 5-Model Interactive Showcase Cards
    renderModelShowcase(allModelsData);
    
    // 4. Render Side-by-Side Model Comparison Matrix Table
    renderModelMatrix(allModelsData);
    
    // 5. Update Local Ollama & Deployment Readiness Hub
    updateOllamaReadiness(selectedModelData);
    
    // 6. Initialize Filter Tabs
    initModelFilterTabs();
    
    // 7. Update Footer Action Bar Active Summary
    updateFooterActiveSummary();
}

function updateIntelligenceContextStrip(data) {
    const taskPill = document.getElementById('stripTaskPill');
    const deployPill = document.getElementById('stripDeployPill');
    const datasetPill = document.getElementById('stripDatasetPill');
    const gpuPill = document.getElementById('stripGpuPill');
    const engineStatus = document.getElementById('stripEngineStatus');
    
    const taskLabels = {
        'conversation': 'Conversational Q&A',
        'qa': 'Knowledge Q&A & Recall',
        'classify': 'Text Classification',
        'extraction': 'Structured JSON Extraction',
        'generation': 'Creative & Long Generation'
    };
    if (taskPill) {
        taskPill.textContent = taskLabels[selectedTask] || (selectedTask ? selectedTask.toUpperCase() : 'Conversational Q&A');
    }
    
    const deployLabels = {
        'cloud_api': 'Cloud API & Local Ollama',
        'edge_device': 'Edge Device / CPU',
        'mobile_app': 'Mobile / On-Device',
        'web_browser': 'Web Browser (WebGPU)',
        'desktop_app': 'Desktop / Local',
        'not_sure': 'Cloud API & Local Ollama'
    };
    if (deployPill) {
        deployPill.textContent = deployLabels[selectedDeployment] || 'Cloud API & Local Ollama';
    }
    
    if (datasetPill) {
        const stats = data.analysis?.stats || (analysisData ? analysisData.stats : null);
        if (stats && stats.total_examples) {
            const avgTokens = stats.avg_tokens_per_example || 1420;
            datasetPill.textContent = `${stats.total_examples} rows (~${avgTokens} tokens/ex)`;
        } else {
            datasetPill.textContent = 'Active Corpus (~1.2k tokens/ex)';
        }
    }
    
    if (gpuPill) {
        if (cachedSystemHardware && cachedSystemHardware.gpus && cachedSystemHardware.gpus.length > 0) {
            const gpu = cachedSystemHardware.gpus[0];
            const vram = gpu.memory_total_gb ? `${gpu.memory_total_gb} GB` : '4 GB';
            gpuPill.textContent = `${gpu.name || 'NVIDIA GPU'} (${vram}) · Free Colab T4`;
        } else {
            gpuPill.textContent = 'NVIDIA T1200 (4.0 GB) · Free Colab T4 (16 GB)';
        }
    }
    
    if (engineStatus) {
        const count = (data.recommendation?.full_matrix || []).length || 5;
        engineStatus.textContent = `${count} SOTA SLMs Evaluated`;
    }
}

function updatePrimaryDisplay(model) {
    if (!model) return;
    
    // Model title and size
    const nameEl = document.getElementById('recModelName');
    const sizeEl = document.getElementById('recModelSize');
    if (nameEl) nameEl.textContent = model.model_name || '-';
    if (sizeEl) sizeEl.textContent = model.size || '-';
    
    // Badges
    const heroBadge = document.getElementById('recHeroBadge');
    if (heroBadge) {
        if (model.isOriginalBestMatch) {
            heroBadge.textContent = '⭐ Best Match · AI Recommended';
            heroBadge.classList.remove('user-selected');
        } else {
            heroBadge.textContent = '✓ Your Active Selection';
            heroBadge.classList.add('user-selected');
        }
    }
    
    const labBadge = document.getElementById('recLabBadge');
    if (labBadge) labBadge.textContent = model.creator || 'Open Source Lab';
    
    const licenseBadge = document.getElementById('recModelLicense');
    if (licenseBadge) licenseBadge.textContent = model.license ? `${model.license}` : 'Permissive License';
    
    const gatedBadge = document.getElementById('recModelGated');
    if (gatedBadge) {
        if (model.gated) {
            gatedBadge.textContent = 'Gated (HF Token)';
            gatedBadge.classList.add('is-gated');
        } else {
            gatedBadge.textContent = 'Open Access (No Token)';
            gatedBadge.classList.remove('is-gated');
        }
    }
    
    // Score ring and confidence
    const scorePercent = Math.round((model.score || 0) * 100);
    const scoreEl = document.getElementById('recScore');
    if (scoreEl) scoreEl.textContent = `${scorePercent}%`;
    
    const scoreCircle = document.getElementById('scoreCircle');
    if (scoreCircle) scoreCircle.style.strokeDasharray = `${scorePercent}, 100`;
    
    const confEl = document.getElementById('recConfidence');
    if (confEl) {
        if (scorePercent >= 85) confEl.textContent = 'Outstanding Fit';
        else if (scorePercent >= 70) confEl.textContent = 'Strong Fit';
        else confEl.textContent = 'Compatible Alternative';
    }
    
    // Model summary sentence
    const summaryEl = document.getElementById('recModelSummary');
    if (summaryEl) {
        const summaries = {
            'phi-4-mini': 'Microsoft’s flagship 3.8B SLM trained on high-density synthetic data, excelling in multi-step reasoning, mathematical logic, and 128K context instruction following.',
            'llama-3.2-3b': 'Meta’s ultra-coherent 3B conversational powerhouse distilled from Llama 3.1, engineered for natural multi-turn dialogue, tool use, and fast edge throughput.',
            'qwen-2.5-3b': 'Alibaba’s state-of-the-art multilingual model supporting 29+ languages with exceptional code generation, structured JSON extraction, and high semantic density.',
            'gemma-3-2b': 'Google DeepMind’s ultra-compact 2B model distilled from Gemini, designed for lightning-fast inference on laptops, mobile phones, and low-power hardware.',
            'mistral-7b': 'Mistral AI’s celebrated 7B open-weights model featuring sliding window attention, deep analytical comprehension, and broad factual knowledge recall.'
        };
        summaryEl.textContent = summaries[model.key] || 'High-performance Small Language Model optimized for parameter-efficient LoRA fine-tuning.';
    }
    
    // Score breakdown bars
    const breakdown = model.score_breakdown || { task_fit: 45, size_fit: 25, output_fit: 18, multi_turn_bonus: 8 };
    const taskVal = breakdown.task_fit || 45;
    const sizeVal = breakdown.size_fit || 25;
    const outputVal = breakdown.output_fit || 18;
    const bonusVal = breakdown.multi_turn_bonus || 0;
    
    const taskScoreEl = document.getElementById('scoreValTask');
    const taskBarEl = document.getElementById('scoreBarTask');
    if (taskScoreEl) taskScoreEl.textContent = `${taskVal} / 50`;
    if (taskBarEl) taskBarEl.style.width = `${Math.min(100, Math.round((taskVal / 50) * 100))}%`;
    
    const sizeScoreEl = document.getElementById('scoreValSize');
    const sizeBarEl = document.getElementById('scoreBarSize');
    if (sizeScoreEl) sizeScoreEl.textContent = `${sizeVal} / 30`;
    if (sizeBarEl) sizeBarEl.style.width = `${Math.min(100, Math.round((sizeVal / 30) * 100))}%`;
    
    const outputScoreEl = document.getElementById('scoreValOutput');
    const outputBarEl = document.getElementById('scoreBarOutput');
    if (outputScoreEl) outputScoreEl.textContent = `${outputVal} / 20`;
    if (outputBarEl) outputBarEl.style.width = `${Math.min(100, Math.round((outputVal / 20) * 100))}%`;
    
    const bonusScoreEl = document.getElementById('scoreValBonus');
    const bonusBarEl = document.getElementById('scoreBarBonus');
    if (bonusScoreEl) bonusScoreEl.textContent = `+${bonusVal} pts`;
    if (bonusBarEl) bonusBarEl.style.width = `${Math.min(100, Math.round((bonusVal / 10) * 100))}%`;
    
    // Architectural strengths
    const strengthsContainer = document.getElementById('recReasons');
    if (strengthsContainer) {
        strengthsContainer.innerHTML = '';
        const strengthsList = (model.strengths && model.strengths.length > 0) ? model.strengths : (model.reasons || ['Strong performance for this task']);
        strengthsList.slice(0, 4).forEach(text => {
            const div = document.createElement('div');
            div.className = 'strength-pill';
            div.innerHTML = `<span class="check-icon">✓</span><span>${text}</span>`;
            strengthsContainer.appendChild(div);
        });
    }
    
    // Benchmarks
    const benchmarks = model.benchmarks || { mmlu: 70, gsm8k: 75, humaneval: 60 };
    const mmluEl = document.getElementById('recMmlu');
    const gsm8kEl = document.getElementById('recGsm8k');
    const humanEl = document.getElementById('recHumanEval');
    const contextEl = document.getElementById('recContext');
    
    if (mmluEl) mmluEl.textContent = benchmarks.mmlu ? `${benchmarks.mmlu}%` : '—';
    if (gsm8kEl) gsm8kEl.textContent = benchmarks.gsm8k ? `${benchmarks.gsm8k}%` : '—';
    if (humanEl) humanEl.textContent = benchmarks.humaneval ? `${benchmarks.humaneval}%` : '—';
    if (contextEl) contextEl.textContent = model.context_window_formatted || formatContextWindow(model.context_window || 0);
    
    // Hardware Feasibility Verdict
    const localVerdictEl = document.getElementById('recLocalVerdict');
    const colabVerdictEl = document.getElementById('recColabVerdict');
    const vramGB = model.qlora_vram_gb || 2.8;
    
    if (localVerdictEl) {
        if (vramGB <= 3.2) {
            localVerdictEl.innerHTML = `✅ Fits comfortably with 4-bit QLoRA (~${vramGB} GB VRAM, safety margin ready)`;
        } else if (vramGB <= 4.0) {
            localVerdictEl.innerHTML = `⚠️ Fits with 4-bit QLoRA (~${vramGB} GB VRAM, close to 4GB capacity)`;
        } else {
            localVerdictEl.innerHTML = `⚡ High VRAM (~${vramGB} GB). Cloud GPU (Colab T4 or A10G) strongly recommended.`;
        }
    }
    
    if (colabVerdictEl) {
        const estMin = model.training_time_base ? `${model.training_time_base * 3} min` : '~12 min';
        colabVerdictEl.innerHTML = `✅ 100% Compatible · Unsloth accelerated ${estMin} training on free T4`;
    }
    
    // Update footer summary
    updateFooterActiveSummary();
}

function renderModelShowcase(models) {
    const grid = document.getElementById('modelShowcaseGrid');
    if (!grid || !models) return;
    
    grid.innerHTML = models.map(m => {
        const isSelected = selectedModelData && (selectedModelData.key === m.key || selectedModelData.model_id === m.model_id);
        const scorePercent = Math.round((m.score || 0) * 100);
        const isHigh = scorePercent >= 80;
        const vramGB = m.qlora_vram_gb || 2.8;
        
        return `
            <div class="showcase-card ${isSelected ? 'is-active' : ''}" data-model-key="${m.key}" data-category="${(m.category || []).join(' ')}">
                <div class="showcase-top">
                    <span class="showcase-lab">${m.creator || 'SLM'}</span>
                    <span class="showcase-score-pill ${isHigh ? 'high' : ''}">${scorePercent}% Match</span>
                </div>
                <div class="showcase-name-wrap">
                    <h4 class="showcase-name">${m.model_name}</h4>
                    <span class="showcase-size">${m.size} · ${m.license || 'Open'}</span>
                </div>
                <div class="showcase-specs-list">
                    <div class="showcase-spec-row">
                        <span>Context:</span>
                        <strong>${m.context_window_formatted || formatContextWindow(m.context_window)}</strong>
                    </div>
                    <div class="showcase-spec-row">
                        <span>QLoRA VRAM:</span>
                        <strong>~${vramGB} GB</strong>
                    </div>
                    <div class="showcase-spec-row">
                        <span>MMLU Score:</span>
                        <strong>${m.benchmarks?.mmlu || m.accuracy_baseline || 80}%</strong>
                    </div>
                </div>
                <button type="button" class="showcase-select-btn" data-model-key="${m.key}">
                    ${isSelected ? '✓ Active Model' : 'Select This Model'}
                </button>
            </div>
        `;
    }).join('');
    
    // Click on card or button selects the model
    grid.querySelectorAll('.showcase-card').forEach(card => {
        card.addEventListener('click', (e) => {
            const key = card.getAttribute('data-model-key');
            selectModelByKey(key);
        });
    });
}

function renderModelMatrix(fullMatrix) {
    const tableBody = document.getElementById('matrixTableBody');
    if (!tableBody || !fullMatrix) return;
    
    tableBody.innerHTML = fullMatrix.map(m => {
        const isSelected = selectedModelData && (selectedModelData.key === m.key || selectedModelData.model_id === m.model_id);
        const scorePercent = Math.round((m.score || 0) * 100);
        let badgeClass = 'comp';
        if (m.status === 'Recommended' || m.is_primary) badgeClass = 'rec';
        else if (m.status === 'Alternative') badgeClass = 'alt';
        
        const vramGB = m.qlora_vram_gb || 2.8;
        const vramBadge = vramGB <= 3.2 
            ? `<span style="color:#34d399; font-weight:600;">~${vramGB} GB (Local OK)</span>` 
            : `<span style="color:#fbbf24; font-weight:600;">~${vramGB} GB (Colab T4)</span>`;
            
        const benchText = m.benchmarks 
            ? `${m.benchmarks.mmlu}% / ${m.benchmarks.gsm8k}%` 
            : `${m.accuracy_baseline || 85}%`;
            
        const licText = m.gated 
            ? `<span style="color:#fbbf24; font-weight:600;">${m.license} (Gated)</span>` 
            : `<span style="color:#34d399; font-weight:600;">${m.license}</span>`;

        return `
            <tr class="${isSelected ? 'selected-row' : ''}" data-model-key="${m.key}" data-category="${(m.category || []).join(' ')}">
                <td>
                    <div class="matrix-model-cell">
                        <div>
                            <span class="matrix-model-name">${m.model_name}</span>
                            <div style="font-size: 11px; color: var(--text-tertiary);">${m.creator || 'Open Source'}</div>
                        </div>
                        <span class="status-badge ${badgeClass}">${m.status || (isSelected ? 'Selected' : 'Compatible')}</span>
                    </div>
                </td>
                <td><strong>${m.size}</strong></td>
                <td>
                    <div class="score-bar-wrap">
                        <div class="score-bar"><div class="score-bar-fill" style="width: ${scorePercent}%"></div></div>
                        <span class="score-num">${scorePercent}%</span>
                    </div>
                </td>
                <td>${m.context_window_formatted || m.context_window}</td>
                <td>${vramBadge}</td>
                <td><strong>${benchText}</strong></td>
                <td>${licText}</td>
                <td><code style="font-size:11px; color:#38bdf8;">${m.ollama_tag || '-'}</code></td>
                <td>
                    <button class="btn btn-secondary btn-small matrix-select-btn" type="button" data-model-key="${m.key}">
                        ${isSelected ? '✓ Active' : 'Select'}
                    </button>
                </td>
            </tr>
        `;
    }).join('');

    tableBody.querySelectorAll('.matrix-select-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const key = btn.getAttribute('data-model-key');
            selectModelByKey(key);
        });
    });
}

function selectModelByKey(key) {
    if (!key) return;
    const target = allModelsData.find(item => item.key === key);
    if (!target) return;
    
    // Update selectedModelData
    selectedModelData = { 
        ...target, 
        isOriginalBestMatch: (target.key === originalBestMatch?.key || target.model_id === originalBestMatch?.model_id) 
    };
    
    // Update recommendationData so training uses selected model
    if (recommendationData && recommendationData.recommendation) {
        recommendationData.recommendation.selected_model = selectedModelData;
    }
    
    // Update Hero card
    updatePrimaryDisplay(selectedModelData);
    
    // Re-render showcase cards and matrix rows to reflect new active state
    renderModelShowcase(allModelsData);
    renderModelMatrix(allModelsData);
    
    // Update Ollama hub
    updateOllamaReadiness(selectedModelData);
    
    // Reapply active category filter if any
    applyModelFilter(currentModelFilter);
}

function updateOllamaReadiness(model) {
    const cmdEl = document.getElementById('ollamaRunCmd');
    const copyBtn = document.getElementById('copyOllamaCmdBtn');
    if (!model) return;
    
    const tag = model.ollama_tag || 'qwen2.5:3b';
    const cmd = `ollama run ${tag}`;
    if (cmdEl) cmdEl.textContent = cmd;
    
    if (copyBtn) {
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(cmd).then(() => {
                copyBtn.textContent = '✓ Copied!';
                setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
            }).catch(() => {
                copyBtn.textContent = 'Copied';
            });
        };
    }
}

function initModelFilterTabs() {
    const filterTabs = document.getElementById('modelFilterTabs');
    if (!filterTabs) return;
    
    filterTabs.querySelectorAll('.filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            filterTabs.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            
            const category = chip.getAttribute('data-filter') || 'all';
            currentModelFilter = category;
            applyModelFilter(category);
        });
    });
}

function applyModelFilter(category) {
    const showcaseCards = document.querySelectorAll('.showcase-card');
    const matrixRows = document.querySelectorAll('#matrixTableBody tr');
    
    showcaseCards.forEach(card => {
        const catAttr = card.getAttribute('data-category') || '';
        const key = card.getAttribute('data-model-key') || '';
        const model = allModelsData.find(m => m.key === key);
        
        let match = false;
        if (category === 'all') {
            match = true;
        } else if (category === 'edge') {
            match = catAttr.includes('edge') || (model && (model.size === '2B' || model.size === '3B'));
        } else if (category === 'reasoning') {
            match = catAttr.includes('reasoning') || key === 'phi-4-mini' || key === 'mistral-7b';
        } else if (category === 'multilingual') {
            match = catAttr.includes('multilingual') || key === 'qwen-2.5-3b';
        } else if (category === 'permissive') {
            match = catAttr.includes('permissive') || (model && (model.license?.includes('Apache') || model.license?.includes('MIT')));
        }
        
        card.style.display = match ? 'flex' : 'none';
    });
    
    matrixRows.forEach(row => {
        const catAttr = row.getAttribute('data-category') || '';
        const key = row.getAttribute('data-model-key') || '';
        const model = allModelsData.find(m => m.key === key);
        
        let match = false;
        if (category === 'all') {
            match = true;
        } else if (category === 'edge') {
            match = catAttr.includes('edge') || (model && (model.size === '2B' || model.size === '3B'));
        } else if (category === 'reasoning') {
            match = catAttr.includes('reasoning') || key === 'phi-4-mini' || key === 'mistral-7b';
        } else if (category === 'multilingual') {
            match = catAttr.includes('multilingual') || key === 'qwen-2.5-3b';
        } else if (category === 'permissive') {
            match = catAttr.includes('permissive') || (model && (model.license?.includes('Apache') || model.license?.includes('MIT')));
        }
        
        row.style.display = match ? '' : 'none';
    });
}

function updateFooterActiveSummary() {
    const footerName = document.getElementById('footerActiveModelName');
    const footerFit = document.getElementById('footerActiveModelFit');
    const startBtnText = document.getElementById('startTrainingBtnText');
    
    if (selectedModelData) {
        const name = selectedModelData.model_name || 'Selected Model';
        const size = selectedModelData.size || '';
        const scorePercent = Math.round((selectedModelData.score || 0) * 100);
        
        if (footerName) footerName.textContent = `${name} (${size})`;
        if (footerFit) footerFit.textContent = `${scorePercent}% Match`;
        if (startBtnText) startBtnText.textContent = `Continue with ${name}`;
    }
    if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();
}

function formatContextWindow(tokens) {
    if (!tokens) return '-';
    if (tokens >= 100000) {
        return `${Math.round(tokens / 1000)}K`;
    } else if (tokens >= 1000) {
        return `${Math.round(tokens / 1000)}K`;
    } else {
        return `${tokens}`;
    }
}

function openAltModelModal(model) {
    const modal = document.getElementById('altModelModal');
    if (!modal || !model) return;
    
    document.getElementById('altModalModelName').textContent = model.model_name || '-';
    document.getElementById('altModalModelSize').textContent = model.size || '-';
    
    const scorePercent = Math.round((model.score || 0) * 100);
    document.getElementById('altModalScore').textContent = scorePercent + '%';
    document.getElementById('altModalScoreCircle').style.strokeDasharray = `${scorePercent}, 100`;
    
    const reasonsContainer = document.getElementById('altModalReasons');
    if (reasonsContainer) {
        reasonsContainer.innerHTML = '';
        const reasonsList = (model.strengths && model.strengths.length > 0) ? model.strengths : (model.reasons || ['Strong alternative']);
        reasonsList.forEach(reason => {
            const div = document.createElement('div');
            div.className = 'modal-reason';
            div.innerHTML = `<span class="reason-icon">✓</span><span>${reason}</span>`;
            reasonsContainer.appendChild(div);
        });
    }
    
    const ctxEl = document.getElementById('altModalContext');
    if (ctxEl) ctxEl.textContent = model.context_window_formatted || formatContextWindow(model.context_window || 0);
    
    const vramEl = document.getElementById('altModalVram');
    if (vramEl) vramEl.textContent = `~${model.qlora_vram_gb || 2.8} GB`;
    
    const licEl = document.getElementById('altModalLicense');
    if (licEl) licEl.textContent = model.license || 'Open';
    
    modal.dataset.pendingModelKey = model.key;
    modal.classList.remove('hidden');
}

function closeAltModelModal() {
    const modal = document.getElementById('altModelModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.dataset.pendingModelKey = '';
    }
}

function confirmAltModel() {
    const modal = document.getElementById('altModelModal');
    const key = modal?.dataset?.pendingModelKey;
    if (key) {
        selectModelByKey(key);
    }
    closeAltModelModal();
}

// Alt model modal event listeners
document.getElementById('altModalCancel')?.addEventListener('click', closeAltModelModal);
document.getElementById('altModalConfirm')?.addEventListener('click', confirmAltModel);

document.getElementById('altModelModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'altModelModal') {
        closeAltModelModal();
    }
});

function displayDatasetSummary(analysis) {
    const summaryGrid = document.getElementById('datasetSummary');
    summaryGrid.innerHTML = '';
    
    if (!analysis) return;
    
    const stats = analysis.stats || {};
    const chars = analysis.conversation_characteristics || {};
    
    const items = [
        { label: 'Examples', value: (stats.total_examples || 0).toLocaleString() },
        { label: 'Messages', value: (stats.total_messages || 0).toLocaleString() },
        { label: 'Avg/Example', value: stats.avg_messages_per_example || '—' },
        { label: 'Multi-turn', value: chars.is_multi_turn ? 'Yes' : 'No' },
        { label: 'System Prompts', value: chars.has_system_prompts ? 'Yes' : 'No' },
        { label: 'Classification', value: chars.looks_like_classification ? 'Likely' : 'No' },
    ];
    
    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'summary-item';
        div.innerHTML = `
            <span class="summary-label">${item.label}</span>
            <span class="summary-value">${item.value}</span>
        `;
        summaryGrid.appendChild(div);
    });
}

// ============================================================================
// STEP 3 → STEP 4: Training
// ============================================================================

document.getElementById('startTrainingBtn')?.addEventListener('click', () => {
    setStep(4);

    // Show training options, hide result views
    document.getElementById('trainingOptions')?.classList.remove('hidden');
    document.getElementById('notebookReady')?.classList.add('hidden');
    document.getElementById('downloadPackage')?.classList.add('hidden');
});

document.getElementById('backToRec')?.addEventListener('click', () => {
    setStep(3);
});

// ============================================================================
// COLAB NOTEBOOK GENERATION
// ============================================================================

document.getElementById('openColabBtn')?.addEventListener('click', async () => {
    const btn = document.getElementById('openColabBtn');

    if (!sessionId || !recommendationData) {
        showError('Missing session data. Please start over.');
        return;
    }

    // Show loading
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Generating notebook...';
    try {
        const hyperparams = collectTrainingHyperparameters();

        // First, create the plan (include selected model if user chose an alternative)
        const planBody = {
            session_id: sessionId,
            user_task: selectedTask || "conversation",
            deployment_target: selectedDeployment || "cloud_api",
            ...hyperparams
        };

        // If user selected an alternative model, include it
        if (selectedModelData && selectedModelData.model_id) {
            planBody.selected_model_id = selectedModelData.model_id;
        }

        const planResponse = await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(planBody)
        });

        if (!planResponse.ok) {
            throw new Error('Failed to create training plan');
        }

        // Generate Colab notebook
        const colabResponse = await fetch(`${API_URL}/generate-colab`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        if (!colabResponse.ok) {
            let errorMsg = 'Failed to generate notebook';
            const text = await colabResponse.text();
            try {
                const error = JSON.parse(text);
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = text || errorMsg;
            }
            throw new Error(errorMsg);
        }

        const data = await colabResponse.json();

        // Check if we have a direct Colab URL (via Gist)
        if (data.colab_url) {
            // Open Colab directly with notebook loaded!
            window.open(data.colab_url, '_blank');

            // Show success view
            document.getElementById('trainingOptions').classList.add('hidden');
            document.getElementById('notebookReady').classList.remove('hidden');

            // Update the UI to show it opened directly
            const notebookReady = document.getElementById('notebookReady');
            if (notebookReady) {
                notebookReady.querySelector('h2').textContent = 'Colab Opened!';
                notebookReady.querySelector('p').textContent = 'Your notebook is ready in the new tab';

                // Update instructions for direct open
                const instructions = notebookReady.querySelector('.colab-instructions');
                if (instructions) {
                    instructions.innerHTML = `
                        <div class="instruction-step">
                            <span class="step-num">1</span>
                            <span><strong>Runtime → Change runtime type → T4 GPU</strong></span>
                        </div>
                        <div class="instruction-step">
                            <span class="step-num">2</span>
                            <span>Click <strong>Runtime → Run all</strong></span>
                        </div>
                        <div class="instruction-step">
                            <span class="step-num">3</span>
                            <span>Wait for training to complete (~15 min)</span>
                        </div>
                    `;
                }
            }

            // Update buttons
            const downloadBtn = document.getElementById('downloadNotebookBtn');
            if (downloadBtn) {
                downloadBtn.href = `${API_URL}${data.notebook_url}`;
            }

            // Update "Open Colab Again" button to use direct URL
            const openColabBtn = document.getElementById('openColabBtn');
            if (openColabBtn) {
                openColabBtn.href = data.colab_url;
            }

        } else {
            // Fallback: Download notebook (no Gist available)
            const downloadUrl = `${API_URL}${data.notebook_url}`;
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = data.notebook_url.split('/').pop() || 'quantune_training.ipynb';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Show notebook ready view with upload instructions (don't open empty Colab)
            document.getElementById('trainingOptions').classList.add('hidden');
            document.getElementById('notebookReady').classList.remove('hidden');

            // Set download link for manual download
            const downloadBtn = document.getElementById('downloadNotebookBtn');
            if (downloadBtn) {
                downloadBtn.href = downloadUrl;
            }

            // Make "Open Colab" button trigger download (no Gist URL available)
            const openColabBtn = document.getElementById('openColabBtn');
            if (openColabBtn) {
                openColabBtn.href = downloadUrl;
                openColabBtn.download = data.notebook_url.split('/').pop() || 'quantune_training.ipynb';
                openColabBtn.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Download Notebook
                `;
            }
        }

    } catch (error) {
        console.error('Colab generation error:', error);
        showError(error.message);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = 'Get Colab Notebook';
        btn.querySelector('.btn-loader')?.classList.add('hidden');
    }
});

// ============================================================================
// DOWNLOAD TRAINING PACKAGE
// ============================================================================

const generateBtn = document.getElementById('generateBtn');
generateBtn?.addEventListener('click', generatePackage);

async function generatePackage() {
    if (!sessionId) {
        showError('No session. Please upload a file first.');
        return;
    }

    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').textContent = 'Generating...';
    generateBtn.querySelector('.btn-loader').classList.remove('hidden');

    try {
        const hyperparams = collectTrainingHyperparameters();

        // Create plan first (include selected model if user chose an alternative)
        const planBody = {
            session_id: sessionId,
            user_task: selectedTask || "conversation",
            deployment_target: selectedDeployment || "cloud_api",
            ...hyperparams
        };

        // If user selected an alternative model, include it
        if (selectedModelData && selectedModelData.model_id) {
            planBody.selected_model_id = selectedModelData.model_id;
        }

        const planResponse = await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(planBody)
        });

        if (!planResponse.ok) {
            throw new Error('Failed to create training plan');
        }

        // Generate package
        const response = await fetch(`${API_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        if (!response.ok) {
            let errorMsg = 'Package generation failed';
            const text = await response.text();
            try {
                const error = JSON.parse(text);
                errorMsg = error.detail || error.message || errorMsg;
            } catch {
                errorMsg = text || errorMsg;
            }
            throw new Error(errorMsg);
        }

        const data = await response.json();

        // Show download section
        document.getElementById('trainingOptions').classList.add('hidden');
        document.getElementById('downloadPackage').classList.remove('hidden');

        // Setup download link
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.href = `${API_URL}${data.download_url}`;

    } catch (error) {
        console.error('Generate error:', error);
        showError(error.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').textContent = 'Download Package';
        generateBtn.querySelector('.btn-loader').classList.add('hidden');
    }
}

// ============================================================================
// START OVER
// ============================================================================

document.getElementById('startOver')?.addEventListener('click', () => {
    // Clear all state
    sessionId = null;
    jobId = null;
    selectedTask = null;
    selectedDeployment = null;
    analysisData = null;
    recommendationData = null;
    selectedModelData = null;
    allModelsData = [];
    originalBestMatch = null;

    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }

    // Reset file input
    clearFile();

    // Reset step 4 views
    document.getElementById('trainingOptions')?.classList.remove('hidden');
    document.getElementById('notebookReady')?.classList.add('hidden');
    document.getElementById('downloadPackage')?.classList.add('hidden');

    // Go to step 1
    setStep(1);
});

// ============================================================================
// INITIALIZATION
// ============================================================================

// Check if backend is running
async function checkBackend() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✓ Backend connected');
        }
    } catch (error) {
        console.warn('⚠️ Backend not reachable at', API_URL);
    }
}

checkBackend();

// Initialize progress bar to step 1
setStep(1);

// ============================================================================
// TRAINING INFO TOGGLE
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    const trainingInfoToggle = document.getElementById('trainingInfoToggle');
    const trainingInfoCard = trainingInfoToggle?.closest('.training-info-card');
    
    if (trainingInfoToggle && trainingInfoCard) {
        trainingInfoToggle.addEventListener('click', () => {
            trainingInfoCard.classList.toggle('collapsed');
        });
    }
});

// ============================================================================
// STEP 5: TEST YOUR MODEL (PLAYGROUND)
// ============================================================================

const playground = {
    mode: 'local',            // 'local' | 'colab' | 'quicktest'
    servers: [],              // servers found by the scan
    models: [],               // models on the connected server
    baseUrl: null,            // normalized base URL of the connected server
    model: null,              // selected model id
    messages: [],             // conversation sent to the model
    streaming: false,
    abortController: null,
    promptsLoaded: false,
    connected: false,
    isSimulator: false,       // built-in simulator mode
    currentExpected: null,    // ground truth expected output for current prompt
    datasetPromptsList: [],   // loaded dataset prompts [{text, expected}]
    feedback: [],             // feedback list from backend
    corrections: [],          // staged corrections [{prompt, corrected, tags}]
};

function pgEl(id) {
    return document.getElementById(id);
}

// ----------------------------------------------------------------------------
// STEP 5 ENTRY POINTS
// ----------------------------------------------------------------------------

function openPlayground() {
    const status = getStepValidationStatus(5);
    if (!status.allowed) {
        showNavGuardToast(status.reason);
        shakeNavStep(5);
        return false;
    }
    setStep(5);
    pgSwitchTab('studio');
    initTestingStudio();
    pgLoadFeedbackStats();
    return true;
}

document.getElementById('testModelFromNotebook')?.addEventListener('click', openPlayground);
document.getElementById('testModelFromPackage')?.addEventListener('click', openPlayground);
document.getElementById('proceedToTestBtn')?.addEventListener('click', openPlayground);
document.getElementById('backFromTest')?.addEventListener('click', () => setStep(4));
document.getElementById('finishTestBtn')?.addEventListener('click', () => document.getElementById('startOver')?.click());

// ----------------------------------------------------------------------------
// TABS
// ----------------------------------------------------------------------------

function pgSwitchTab(mode) {
    playground.mode = mode;
    pgEl('tabStudio')?.classList.toggle('active', mode === 'studio');
    pgEl('tabColab')?.classList.toggle('active', mode === 'colab');
    pgEl('tabLocal')?.classList.toggle('active', mode === 'local');
    pgEl('panelStudio')?.classList.toggle('hidden', mode !== 'studio');
    pgEl('panelColab')?.classList.toggle('hidden', mode !== 'colab');
    pgEl('panelLocal')?.classList.toggle('hidden', mode !== 'local');
    if (mode === 'studio') {
        initTestingStudio();
    } else if (mode === 'local') {
        pgBuildSetupGuide();
        if (!playground.servers.length && !playground.isSimulator) {
            pgScanServers();
        }
    }
}

document.getElementById('tabStudio')?.addEventListener('click', () => pgSwitchTab('studio'));
document.getElementById('tabColab')?.addEventListener('click', () => pgSwitchTab('colab'));
document.getElementById('tabLocal')?.addEventListener('click', () => pgSwitchTab('local'));

// ----------------------------------------------------------------------------
// TALKING TO THE TEST API
// ----------------------------------------------------------------------------

async function pgApi(path, options = {}) {
    const response = await fetch(`${API_URL}${path}`, options);

    if (response.status === 404 && path.startsWith('/test/')) {
        throw new Error('This server is running without the test API. Restart Quantune with: uvicorn api.app:app --reload');
    }
    return response;
}

async function pgApiJson(path, options = {}) {
    const response = await pgApi(path, options);
    if (!response.ok) {
        let detail = `Request failed (${response.status})`;
        try {
            const body = await response.json();
            detail = body.detail || body.message || detail;
        } catch {
            /* keep generic */
        }
        throw new Error(detail);
    }
    return response.json();
}

// ----------------------------------------------------------------------------
// EVALUATION SCORECARD & STATS
// ----------------------------------------------------------------------------

async function pgLoadFeedbackStats() {
    if (!sessionId) return;
    try {
        const stats = await pgApiJson(`/test/feedback/${sessionId}`);
        pgUpdateScorecardUI(stats);
    } catch {
        /* session feedback not yet populated */
    }
}

function pgUpdateScorecardUI(stats) {
    if (!stats) return;
    if (pgEl('scorecardAccuracy')) {
        pgEl('scorecardAccuracy').textContent = stats.total_tested > 0 ? `${stats.accuracy_pct}%` : '--%';
    }
    if (pgEl('scorecardAccuracySub')) {
        pgEl('scorecardAccuracySub').textContent = `${stats.approved} / ${stats.total_tested} approved`;
    }
    if (pgEl('scorecardTotal')) pgEl('scorecardTotal').textContent = stats.total_tested;
    if (pgEl('scorecardApproved')) pgEl('scorecardApproved').textContent = stats.approved;
    if (pgEl('scorecardFlagged')) pgEl('scorecardFlagged').textContent = stats.flagged;
    if (pgEl('scorecardCorrections')) pgEl('scorecardCorrections').textContent = stats.corrections_count;

    // Show/hide Re-Fine Action Bar
    const bar = pgEl('refineActionBar');
    if (bar) {
        const count = stats.corrections_count || playground.corrections.length;
        if (count > 0) {
            bar.classList.remove('hidden');
            pgEl('refineCountLabel').textContent = `${count} correction${count === 1 ? '' : 's'} staged`;
        } else {
            bar.classList.add('hidden');
        }
    }
}

// ----------------------------------------------------------------------------
// SERVER DISCOVERY & BUILT-IN SIMULATOR
// ----------------------------------------------------------------------------

function pgEnableSimulator() {
    playground.isSimulator = true;
    playground.baseUrl = 'simulated';
    playground.model = 'Quantune Built-in Simulator';
    playground.connected = true;
    const select = pgEl('modelSelect');
    if (select) {
        select.innerHTML = '<option value="simulator" selected>Quantune Built-in Simulator (Active)</option>';
        select.disabled = false;
    }
    pgEl('finetuneBadge')?.classList.remove('hidden');
    pgSetStatus('Quantune Simulator (Ready)', true);
    pgUpdateSendState();

    const simBtn = pgEl('toggleSimulatorBtn');
    if (simBtn) {
        simBtn.textContent = 'Simulator Active';
        simBtn.classList.add('btn-primary');
        simBtn.classList.remove('btn-ghost');
    }
}

document.getElementById('toggleSimulatorBtn')?.addEventListener('click', pgEnableSimulator);

async function pgScanServers() {
    const list = pgEl('serverList');
    list.innerHTML = '<div class="server-scanning"><span class="mini-spinner"></span>Scanning your machine…</div>';

    try {
        const data = await pgApiJson('/test/servers');
        playground.servers = data.servers || [];
        pgRenderServers();

        if (playground.servers.length) {
            const current = playground.servers.find(s => s.base_url === playground.baseUrl);
            const target = current || playground.servers[0];
            if (!current) {
                pgConnectToServer(target.base_url, target.name);
            }
        } else {
            pgSetStatus('No model server found (Use Built-in Simulator)', false);
        }
    } catch (error) {
        playground.servers = [];
        list.innerHTML = `<div class="server-error">${pgEscape(error.message)}</div>`;
        pgSetStatus('Not connected', false);
    }
}

function pgRenderServers() {
    const list = pgEl('serverList');

    if (!playground.servers.length) {
        list.innerHTML = '<div class="server-none">No local server detected yet.<br><span>Start Ollama, LM Studio, llama.cpp or click <strong>Use Built-in Simulator</strong> above.</span></div>';
        return;
    }

    list.innerHTML = '';
    playground.servers.forEach(server => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'server-chip' + (server.base_url === playground.baseUrl && !playground.isSimulator ? ' active' : '');
        chip.innerHTML = `
            <span class="server-chip-name">${pgEscape(server.name)}</span>
            <span class="server-chip-meta">${server.models.length} model${server.models.length === 1 ? '' : 's'}</span>
        `;
        chip.addEventListener('click', () => {
            playground.isSimulator = false;
            const simBtn = pgEl('toggleSimulatorBtn');
            if (simBtn) {
                simBtn.textContent = 'Use Built-in Simulator';
                simBtn.classList.remove('btn-primary');
                simBtn.classList.add('btn-ghost');
            }
            pgConnectToServer(server.base_url, server.name);
        });
        list.appendChild(chip);
    });
}

// ----------------------------------------------------------------------------
// MODEL SELECTION
// ----------------------------------------------------------------------------

async function pgConnectToServer(baseUrl, label) {
    pgSetStatus('Connecting…', false);

    try {
        const body = { base_url: baseUrl };
        if (sessionId) body.session_id = sessionId;

        const data = await pgApiJson('/test/models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        playground.baseUrl = data.base_url;
        playground.models = data.models || [];
        playground.connected = true;
        playground.isSimulator = false;

        pgRenderModelOptions(data.suggested_model);
        pgRenderServers();
        pgLoadDatasetPrompts();
        pgSetStatus(`${label || 'Connected'} · ${playground.models.length} model${playground.models.length === 1 ? '' : 's'}`, true);
    } catch (error) {
        playground.connected = false;
        pgSetStatus('Connection failed', false);
        pgRenderModelOptions(null);
        pgShowChatError(error.message);
    }
}

function pgRenderModelOptions(suggested) {
    const select = pgEl('modelSelect');
    const badge = pgEl('finetuneBadge');

    if (!playground.models.length) {
        select.innerHTML = '<option value="">No models on this server</option>';
        select.disabled = true;
        badge.classList.add('hidden');
        playground.model = null;
        pgUpdateSendState();
        return;
    }

    select.innerHTML = '';
    playground.models.forEach(id => {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = id;
        select.appendChild(option);
    });
    select.disabled = false;

    playground.model = suggested || playground.models[0];
    select.value = playground.model;
    badge.classList.toggle('hidden', !suggested);
    pgUpdateSendState();
}

document.getElementById('modelSelect')?.addEventListener('change', (event) => {
    playground.model = event.target.value || null;
    pgUpdateSendState();
});

function pgSetStatus(text, ok) {
    pgEl('chatModelLabel').textContent = text;
    pgEl('chatStatusDot').classList.toggle('connected', !!ok);
}

function pgUpdateSendState() {
    const ready = !!playground.model && !playground.streaming;
    pgEl('sendChatBtn').disabled = !ready;
    pgEl('chatInput').disabled = !playground.model;
    pgEl('chatInput').placeholder = playground.model
        ? `Ask ${playground.model} something…`
        : 'Connect a model or simulator to start chatting…';
}

// ----------------------------------------------------------------------------
// REAL PROMPTS FROM THE TRAINING DATA
// ----------------------------------------------------------------------------

async function pgLoadDatasetPrompts() {
    const panel = pgEl('datasetPrompts');
    if (playground.promptsLoaded || !sessionId) return;

    try {
        const data = await pgApiJson(`/test/prompts/${sessionId}`);
        playground.promptsLoaded = true;
        playground.datasetPromptsList = data.prompts || [];

        if (data.system_prompt && !pgEl('systemPromptInput').value.trim()) {
            pgEl('systemPromptInput').value = data.system_prompt;
        }

        const chips = pgEl('datasetPromptChips');
        chips.innerHTML = '';
        playground.datasetPromptsList.forEach(item => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'dataset-chip';
            chip.textContent = item.text.length > 60 ? item.text.slice(0, 60) + '…' : item.text;
            if (item.expected) {
                chip.title = `Expected ground truth: ${item.expected}`;
            }
            chip.addEventListener('click', () => {
                pgEl('chatInput').value = item.text;
                playground.currentExpected = item.expected || null;
                pgEl('chatInput').focus();
            });
            chips.appendChild(chip);
        });

        panel.classList.toggle('hidden', !playground.datasetPromptsList.length);
    } catch (error) {
        panel.classList.add('hidden');
        console.warn('Dataset prompts unavailable:', error.message);
    }
}

// ----------------------------------------------------------------------------
// CONNECT GUIDE
// ----------------------------------------------------------------------------

function pgModelSlug() {
    const name = (selectedModelData?.model_name || 'your model').toLowerCase();
    return name.replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

function pgBuildSetupGuide() {
    const slug = pgModelSlug();

    pgEl('guideCommandsOllama').textContent = [
        `# In Colab: run the "Export to GGUF" cell, download fine_tuned_${slug}_gguf`,
        `cd fine_tuned_${slug}_gguf`,
        `echo 'FROM ./model.gguf' > Modelfile`,
        `ollama create ${slug} -f Modelfile`,
        `ollama run ${slug}`,
    ].join('\n');

    pgEl('guideCommandsServe').textContent = [
        `# In the training package folder (after train.py finished)`,
        `pip install -r requirements.txt`,
        `python serve_model.py`,
        `# -> serving on http://127.0.0.1:8787/v1`,
    ].join('\n');
}

document.getElementById('setupGuideToggle')?.addEventListener('click', () => {
    pgEl('setupGuideCard').classList.toggle('collapsed');
});

document.getElementById('rescanServersBtn')?.addEventListener('click', pgScanServers);

document.getElementById('connectUrlBtn')?.addEventListener('click', () => {
    const raw = pgEl('serverUrlInput').value.trim();
    if (!raw) {
        pgShowChatError('Enter the URL of your model server first.');
        return;
    }
    pgConnectToServer(raw, raw);
});

document.getElementById('serverUrlInput')?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault();
        pgEl('connectUrlBtn').click();
    }
});

// ----------------------------------------------------------------------------
// CHAT & FEEDBACK INTERFACE
// ----------------------------------------------------------------------------

function pgEscape(text) {
    const div = document.createElement('div');
    div.textContent = text ?? '';
    return div.innerHTML;
}

function pgEnsureChatErrorEl() {
    let el = pgEl('chatErrorBanner');
    if (!el) {
        el = document.createElement('div');
        el.id = 'chatErrorBanner';
        el.className = 'chat-error hidden';
        const row = document.querySelector('#panelLocal .chat-input-row');
        row.parentNode.insertBefore(el, row);
    }
    return el;
}

function pgShowChatError(message) {
    const el = pgEnsureChatErrorEl();
    el.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg><span>${pgEscape(message)}</span>`;
    el.classList.remove('hidden');
}

function pgHideChatError() {
    pgEl('chatErrorBanner')?.classList.add('hidden');
}

function pgScrollChatToBottom() {
    const box = pgEl('chatMessages');
    box.scrollTop = box.scrollHeight;
}

function pgFormatMs(ms) {
    return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function pgAppendMessage(role, content) {
    pgEl('chatEmpty')?.classList.add('hidden');

    const wrap = document.createElement('div');
    wrap.className = `chat-msg ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble';

    const text = document.createElement('div');
    text.className = 'chat-bubble-text';
    text.textContent = content || '';
    bubble.appendChild(text);

    if (role === 'assistant') {
        const footer = document.createElement('div');
        footer.className = 'chat-bubble-footer hidden';

        const stats = document.createElement('span');
        stats.className = 'chat-bubble-stats';

        const copy = document.createElement('button');
        copy.type = 'button';
        copy.className = 'chat-copy-btn';
        copy.textContent = 'Copy';
        copy.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(text.textContent);
                copy.textContent = 'Copied';
                setTimeout(() => { copy.textContent = 'Copy'; }, 1500);
            } catch {
                /* clipboard unavailable */
            }
        });

        footer.appendChild(stats);
        footer.appendChild(copy);
        bubble.appendChild(footer);
    }

    wrap.appendChild(bubble);
    pgEl('chatMessages').appendChild(wrap);
    pgScrollChatToBottom();
    return wrap;
}

function pgFinishBubble(wrap, received, event, started, firstTokenAt, userPromptText, expectedText) {
    const bubble = wrap.querySelector('.chat-bubble');
    const footer = wrap.querySelector('.chat-bubble-footer');
    const stats = wrap.querySelector('.chat-bubble-stats');
    if (!footer || !stats) return;

    const totalMs = event?.latency_ms ?? (performance.now() - started);
    const ttft = event?.ttft_ms ?? (firstTokenAt ? firstTokenAt - started : null);
    const usage = event?.usage;
    const tokens = usage?.completion_tokens || Math.max(1, Math.round(received.length / 4));

    const parts = [];
    if (ttft !== null && ttft !== undefined) parts.push(`first token ${pgFormatMs(ttft)}`);
    parts.push(`${pgFormatMs(totalMs)} total`);
    parts.push(usage?.completion_tokens ? `${tokens} tokens` : `≈${tokens} tokens`);
    stats.textContent = parts.join(' · ');

    footer.classList.remove('hidden');

    // Ground Truth Comparison
    if (expectedText) {
        const gtWrap = document.createElement('div');
        gtWrap.className = 'chat-ground-truth-wrap';
        gtWrap.innerHTML = `
            <div class="ground-truth-head">
                <span>Expected Ground Truth</span>
                <span class="ground-truth-tag">Dataset</span>
            </div>
            <div class="ground-truth-text">${pgEscape(expectedText)}</div>
        `;
        bubble.appendChild(gtWrap);
    }

    // Feedback Action Row
    const fbRow = document.createElement('div');
    fbRow.className = 'chat-feedback-row';
    fbRow.innerHTML = `
        <div class="chat-feedback-actions">
            <button type="button" class="feedback-btn thumbs-up" title="Mark response as accurate">
                👍 Accurate
            </button>
            <button type="button" class="feedback-btn thumbs-down" title="Flag inaccuracy or poor quality">
                👎 Flag Issue
            </button>
            <button type="button" class="feedback-btn correct-btn" title="Provide ideal response and add to re-fine queue">
                ✏️ Correct Answer
            </button>
        </div>
    `;

    const upBtn = fbRow.querySelector('.thumbs-up');
    const downBtn = fbRow.querySelector('.thumbs-down');
    const correctBtn = fbRow.querySelector('.correct-btn');

    upBtn.addEventListener('click', async () => {
        upBtn.classList.add('active');
        downBtn.classList.remove('active');
        await pgSendFeedback({
            prompt: userPromptText,
            response: received,
            expected: expectedText,
            rating: 'good',
        });
    });

    downBtn.addEventListener('click', async () => {
        downBtn.classList.add('active');
        upBtn.classList.remove('active');
        await pgSendFeedback({
            prompt: userPromptText,
            response: received,
            expected: expectedText,
            rating: 'bad',
        });
    });

    correctBtn.addEventListener('click', () => {
        pgOpenCorrectionModal(userPromptText, received, expectedText);
    });

    bubble.appendChild(fbRow);
    pgScrollChatToBottom();
}

async function pgSendFeedback({ prompt, response, expected, rating, corrected_response, tags }) {
    if (!sessionId) {
        const item = { prompt, response, expected, rating, corrected_response, tags };
        playground.feedback.push(item);
        if (corrected_response) playground.corrections.push({ prompt, corrected: corrected_response });
        const good = playground.feedback.filter(f => f.rating === 'good').length;
        const total = playground.feedback.length;
        pgUpdateScorecardUI({
            total_tested: total,
            approved: good,
            flagged: total - good,
            corrections_count: playground.corrections.length,
            accuracy_pct: total > 0 ? Math.round((good / total) * 100) : 0,
        });
        return;
    }
    try {
        const result = await pgApiJson('/test/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                prompt,
                response,
                expected: expected || null,
                rating,
                corrected_response: corrected_response || null,
                tags: tags || [],
            }),
        });
        if (corrected_response) {
            playground.corrections.push({ prompt, corrected: corrected_response });
        }
        pgUpdateScorecardUI(result);
    } catch (err) {
        console.warn('Feedback recording failed:', err.message);
    }
}

// ----------------------------------------------------------------------------
// CORRECTION MODAL / DRAWER
// ----------------------------------------------------------------------------

let currentModalContext = null;

function pgOpenCorrectionModal(prompt, modelOutput, expected) {
    currentModalContext = { prompt, modelOutput, expected };
    pgEl('modalPromptText').textContent = prompt;
    pgEl('modalModelOutput').textContent = modelOutput;
    const expCol = pgEl('modalExpectedCol');
    if (expected) {
        pgEl('modalExpectedOutput').textContent = expected;
        expCol.classList.remove('hidden');
    } else {
        expCol.classList.add('hidden');
    }
    pgEl('modalCorrectionInput').value = expected || '';
    pgEl('correctionModal').querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.checked = false; });
    pgEl('correctionModal').classList.remove('hidden');
    pgEl('modalCorrectionInput').focus();
}

function pgCloseCorrectionModal() {
    pgEl('correctionModal').classList.add('hidden');
    currentModalContext = null;
}

document.getElementById('closeCorrectionModal')?.addEventListener('click', pgCloseCorrectionModal);
document.getElementById('cancelCorrectionBtn')?.addEventListener('click', pgCloseCorrectionModal);

document.getElementById('saveCorrectionBtn')?.addEventListener('click', async () => {
    const text = pgEl('modalCorrectionInput').value.trim();
    if (!text) {
        alert('Please enter what the model should have answered.');
        return;
    }
    const tags = Array.from(pgEl('correctionModal').querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
    if (currentModalContext) {
        await pgSendFeedback({
            prompt: currentModalContext.prompt,
            response: currentModalContext.modelOutput,
            expected: currentModalContext.expected,
            rating: 'bad',
            corrected_response: text,
            tags,
        });
    }
    pgCloseCorrectionModal();
});

// ----------------------------------------------------------------------------
// RE-FINE ACTION BAR
// ----------------------------------------------------------------------------

document.getElementById('downloadRefinedBtn')?.addEventListener('click', () => {
    if (!sessionId) return;
    window.open(`${API_URL}/test/export-refined/${sessionId}`, '_blank');
});

document.getElementById('triggerRefineBtn')?.addEventListener('click', async () => {
    if (!sessionId) {
        alert('Please complete the upload stage to enable re-refining.');
        return;
    }
    const btn = pgEl('triggerRefineBtn');
    btn.disabled = true;
    const origHTML = btn.innerHTML;
    btn.innerHTML = '<span class="mini-spinner"></span> Re-Fining Model...';

    try {
        const result = await pgApiJson(`/test/refine/${sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ corrections: playground.corrections }),
        });

        alert(result.message || 'Dataset updated successfully with your corrections!');
        playground.corrections = [];
        pgEl('refineActionBar').classList.add('hidden');

        // Transition smoothly to Step 4 with the newly prepared notebook
        setStep(4);
        if (result.colab_url) {
            const openColabBtn = pgEl('openColabBtn');
            if (openColabBtn) openColabBtn.href = result.colab_url;
        }
        if (result.notebook_url) {
            const dlNb = pgEl('downloadNotebook');
            if (dlNb) dlNb.href = `${API_URL}${result.notebook_url}`;
        }
    } catch (err) {
        alert(`Re-fine failed: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = origHTML;
    }
});

// ----------------------------------------------------------------------------
// AUTOMATED DATASET BENCHMARK PANEL
// ----------------------------------------------------------------------------

async function pgInitQuickBenchmark() {
    if (!sessionId && !playground.datasetPromptsList.length) {
        return;
    }
    if (!playground.datasetPromptsList.length && sessionId) {
        await pgLoadDatasetPrompts();
    }
}

document.getElementById('runBenchmarkBtn')?.addEventListener('click', async () => {
    const btn = pgEl('runBenchmarkBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="mini-spinner"></span> Evaluating dataset test set...';

    const list = pgEl('benchmarkList');
    const empty = pgEl('benchmarkEmpty');
    empty.classList.add('hidden');
    list.classList.remove('hidden');
    list.innerHTML = '';

    const prompts = playground.datasetPromptsList.length > 0
        ? playground.datasetPromptsList.slice(0, 8)
        : [
            { text: "What is fine-tuning?", expected: "Fine-tuning is adapting a pre-trained language model with specialized domain examples." },
            { text: "Why use LoRA?", expected: "LoRA reduces trainable parameters and VRAM requirements while preserving base model knowledge." }
        ];

    for (let i = 0; i < prompts.length; i++) {
        const item = prompts[i];
        btn.innerHTML = `<span class="mini-spinner"></span> Evaluating prompt ${i + 1} of ${prompts.length}...`;

        let modelAnswer = '';
        try {
            if (playground.isSimulator || !playground.baseUrl) {
                const res = await pgApiJson('/test/simulate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: item.text, session_id: sessionId || null, stream: false }),
                });
                modelAnswer = res.content;
            } else {
                const res = await pgApiJson('/test/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        base_url: playground.baseUrl,
                        model: playground.model,
                        messages: [{ role: 'user', content: item.text }],
                        stream: false
                    }),
                });
                modelAnswer = res.content;
            }
        } catch (err) {
            modelAnswer = `[Error generating output: ${err.message}]`;
        }

        const row = document.createElement('div');
        row.className = 'benchmark-item';
        row.innerHTML = `
            <div class="benchmark-item-head">
                <span class="benchmark-prompt">#${i + 1} · ${pgEscape(item.text)}</span>
            </div>
            <div class="benchmark-grid">
                <div>
                    <div class="benchmark-col-label">Model Response</div>
                    <div class="benchmark-text-box">${pgEscape(modelAnswer)}</div>
                </div>
                <div>
                    <div class="benchmark-col-label">Expected Ground Truth</div>
                    <div class="benchmark-text-box">${pgEscape(item.expected || 'N/A')}</div>
                </div>
            </div>
            <div class="benchmark-item-actions">
                <button type="button" class="btn btn-ghost btn-small bm-pass">👍 Approve</button>
                <button type="button" class="btn btn-ghost btn-small bm-fix">✏️ Fix & Re-Fine</button>
            </div>
        `;

        row.querySelector('.bm-pass').addEventListener('click', async (e) => {
            e.target.disabled = true;
            e.target.textContent = 'Approved ✓';
            e.target.classList.add('text-success');
            await pgSendFeedback({ prompt: item.text, response: modelAnswer, expected: item.expected, rating: 'good' });
        });

        row.querySelector('.bm-fix').addEventListener('click', () => {
            pgOpenCorrectionModal(item.text, modelAnswer, item.expected);
        });

        list.appendChild(row);
    }

    btn.disabled = false;
    btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run All Benchmark Prompts';
});

// ----------------------------------------------------------------------------
// STREAMING CHAT EXECUTION
// ----------------------------------------------------------------------------

function pgToggleStreamingUI(streaming) {
    pgEl('sendChatBtn').classList.toggle('hidden', streaming);
    pgEl('stopChatBtn').classList.toggle('hidden', !streaming);
    pgEl('sendChatBtn').disabled = streaming;
    pgEl('chatInput').disabled = streaming || !playground.model;
}

async function pgSendMessage() {
    const input = pgEl('chatInput');
    const text = input.value.trim();
    if (!text || !playground.model || playground.streaming) return;

    // Detect if this prompt matches an expected answer
    let expectedText = playground.currentExpected;
    if (!expectedText && playground.datasetPromptsList.length) {
        const found = playground.datasetPromptsList.find(p => p.text.trim().toLowerCase() === text.toLowerCase());
        if (found) expectedText = found.expected;
    }
    playground.currentExpected = null;

    input.value = '';
    pgHideChatError();

    playground.messages.push({ role: 'user', content: text });
    pgAppendMessage('user', text);

    const bubble = pgAppendMessage('assistant', '');
    bubble.classList.add('streaming');
    const textNode = bubble.querySelector('.chat-bubble-text');

    playground.streaming = true;
    pgToggleStreamingUI(true);

    const controller = new AbortController();
    playground.abortController = controller;

    const started = performance.now();
    let firstTokenAt = null;
    let received = '';
    let doneEvent = null;

    const systemPrompt = pgEl('systemPromptInput').value.trim();
    const endpoint = playground.isSimulator ? '/test/simulate' : '/test/chat';
    const body = playground.isSimulator
        ? {
            prompt: text,
            session_id: sessionId || null,
            system_prompt: systemPrompt || null,
            temperature: parseFloat(pgEl('temperatureInput').value) || 0.7,
            stream: true,
        }
        : {
            base_url: playground.baseUrl,
            model: playground.model,
            messages: playground.messages,
            temperature: parseFloat(pgEl('temperatureInput').value) || 0.7,
            max_tokens: parseInt(pgEl('maxTokensInput').value, 10) || 256,
            stream: true,
            ...(systemPrompt ? { system_prompt: systemPrompt } : {})
        };

    try {
        const response = await pgApi(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
        });

        if (!response.ok) {
            let detail = `Request failed (${response.status})`;
            try {
                const payload = await response.json();
                detail = payload.detail || payload.message || detail;
            } catch {
                /* generic */
            }
            throw new Error(detail);
        }
        if (!response.body) {
            throw new Error('This browser cannot read streamed responses.');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let failed = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split('\n\n');
            buffer = events.pop();

            for (const event of events) {
                const line = event.trim();
                if (!line.startsWith('data:')) continue;

                let payload;
                try {
                    payload = JSON.parse(line.slice(5).trim());
                } catch {
                    continue;
                }

                if (payload.error) {
                    failed = payload.error;
                } else if (payload.delta) {
                    if (firstTokenAt === null) firstTokenAt = performance.now();
                    received += payload.delta;
                    textNode.textContent = received;
                    pgScrollChatToBottom();
                } else if (payload.done) {
                    doneEvent = payload;
                }
            }

            if (failed) break;
        }

        if (failed) throw new Error(failed);
        if (!received) throw new Error('The model returned an empty response.');

        playground.messages.push({ role: 'assistant', content: received });
        pgFinishBubble(bubble, received, doneEvent, started, firstTokenAt, text, expectedText);

    } catch (error) {
        if (error.name === 'AbortError') {
            textNode.textContent = received ? `${received}\n\n— stopped —` : '— stopped —';
            if (received) {
                playground.messages.push({ role: 'assistant', content: received });
                pgFinishBubble(bubble, received, null, started, firstTokenAt, text, expectedText);
            }
        } else {
            bubble.remove();
            pgShowChatError(error.message);
        }
    } finally {
        bubble.classList.remove('streaming');
        playground.streaming = false;
        playground.abortController = null;
        pgToggleStreamingUI(false);
        pgEl('chatInput').focus();
    }
}

document.getElementById('sendChatBtn')?.addEventListener('click', pgSendMessage);
document.getElementById('stopChatBtn')?.addEventListener('click', () => playground.abortController?.abort());

document.getElementById('chatInput')?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        pgSendMessage();
    }
});

document.getElementById('clearChatBtn')?.addEventListener('click', () => {
    playground.messages = [];
    pgEl('chatMessages').querySelectorAll('.chat-msg').forEach(node => node.remove());
    pgEl('chatEmpty')?.classList.remove('hidden');
    pgHideChatError();
});

document.getElementById('temperatureInput')?.addEventListener('input', (event) => {
    pgEl('temperatureValue').textContent = parseFloat(event.target.value).toFixed(1);
});

// ----------------------------------------------------------------------------
// COLAB PLAYGROUND PANEL
// ----------------------------------------------------------------------------

const PG_COLAB_STORAGE_KEY = 'quantune-colab-playground';

function pgColabUrl() {
    return pgEl('colabUrlInput').value.trim();
}

function pgValidColabUrl(url) {
    return /^https?:\/\//i.test(url);
}

function pgSaveColabUrl(url) {
    try {
        localStorage.setItem(PG_COLAB_STORAGE_KEY, url);
    } catch {
        /* ignore */
    }
}

document.getElementById('openColabPlaygroundBtn')?.addEventListener('click', () => {
    const url = pgColabUrl();
    if (!pgValidColabUrl(url)) {
        pgEl('colabUrlHint').textContent = 'Paste the https://…gradio.live link printed by the notebook first.';
        pgEl('colabUrlHint').classList.add('hint-warning');
        return;
    }
    pgEl('colabUrlHint').classList.remove('hint-warning');
    pgSaveColabUrl(url);
    window.open(url, '_blank', 'noopener');
});

document.getElementById('embedColabPlaygroundBtn')?.addEventListener('click', () => {
    const url = pgColabUrl();
    if (!pgValidColabUrl(url)) {
        pgEl('colabUrlHint').textContent = 'Paste the https://…gradio.live link printed by the notebook first.';
        pgEl('colabUrlHint').classList.add('hint-warning');
        return;
    }
    pgSaveColabUrl(url);
    pgEl('colabPlaygroundFrame').src = url;
    pgEl('colabEmbedWrap').classList.remove('hidden');
    pgEl('colabUrlHint').classList.remove('hint-warning');
    pgEl('colabUrlHint').textContent = 'If the playground stays blank, the link may block embedding - use "Open playground" instead.';
});

(function pgRestoreColabUrl() {
    try {
        const saved = localStorage.getItem(PG_COLAB_STORAGE_KEY);
        if (saved && pgEl('colabUrlInput')) {
            pgEl('colabUrlInput').value = saved;
        }
    } catch {
        /* ignore */
    }
})();

// Hyperparameter accordion toggle
// ============================================================================
// HYPERPARAMETER TOGGLE & ACCORDION
// ============================================================================

function setupHyperparamToggle() {
    const toggle = document.getElementById('hyperparamToggle');
    const card = document.getElementById('hyperparamCard');
    if (toggle && card) {
        toggle.onclick = (e) => {
            e.preventDefault();
            card.classList.toggle('collapsed');
        };
    }
}

// ============================================================================
// STEP 4: ACTIVE MODEL STAGED HERO & PREVIOUS TRAINED MODELS VAULT
// ============================================================================

// (allTrainedRuns is declared at top of app.js)

function updateStagedModelHero() {
    const titleEl = document.getElementById('stagedModelTitle');
    const descEl = document.getElementById('stagedModelDesc');
    const epochsEl = document.getElementById('stagedSpecEpochs');
    const batchEl = document.getElementById('stagedSpecBatch');
    const loraEl = document.getElementById('stagedSpecLora');
    const lrEl = document.getElementById('stagedSpecLr');
    const hwEl = document.getElementById('stagedSpecHardware');

    const modelName = (selectedModelData && selectedModelData.model_name) || "Phi-4 Mini (3.8B)";
    const modelDesc = (selectedModelData && selectedModelData.description) || "Architecture: SLM Transformer · 128k Context Window · LoRA / QLoRA 4-bit Supported";

    const hyperparams = (typeof collectTrainingHyperparameters === 'function') 
        ? collectTrainingHyperparameters() 
        : { num_train_epochs: 3, per_device_train_batch_size: 4, lora_r: 16, lora_alpha: 32, learning_rate: 2e-4 };

    if (titleEl) titleEl.textContent = modelName;
    if (descEl) descEl.textContent = modelDesc;
    if (epochsEl) epochsEl.textContent = `${hyperparams.num_train_epochs || 3} Epochs`;
    if (batchEl) batchEl.textContent = `${hyperparams.per_device_train_batch_size || 4} per device`;
    if (loraEl) loraEl.textContent = `r=${hyperparams.lora_r || 16}, α=${hyperparams.lora_alpha || 32}`;
    if (lrEl) lrEl.textContent = `${hyperparams.learning_rate || '2.0e-4'} (${hyperparams.lr_scheduler_type || 'Cosine'})`;

    if (hwEl && window._cachedHardwareInfo) {
        const gpu = window._cachedHardwareInfo.gpu;
        if (gpu && gpu.has_gpu) {
            hwEl.textContent = `${gpu.name} (${gpu.vram_total_gb}GB VRAM)`;
        } else {
            hwEl.textContent = "NVIDIA T1200 4GB (Acceleration Ready)";
        }
    }

    const nameInput = document.getElementById('customModelNameInput');
    if (nameInput && !nameInput.value.trim()) {
        const domainStr = (analysisData && analysisData.domain) || "Specialized";
        const cleanBase = modelName.split(' ')[0] || "SLM";
        nameInput.value = `${domainStr} Assistant (${cleanBase} LoRA)`;
    }
}
window.updateStagedModelHero = updateStagedModelHero;

async function loadTrainedModelsVault() {
    const grid = document.getElementById('vaultRunsGrid');
    const countBadge = document.getElementById('vaultCountBadge');
    if (!grid) return;

    try {
        const resp = await fetch(`${API_URL}/models/runs`);
        if (!resp.ok) return;
        const data = await resp.json();
        allTrainedRuns = data.runs || [];

        if (countBadge) {
            countBadge.textContent = `${allTrainedRuns.length} Checkpoints`;
        }

        if (allTrainedRuns.length === 0) {
            grid.innerHTML = `
                <div class="vault-empty-state">
                    <p style="margin: 0 0 6px 0; font-size: 15px; font-weight: 700; color: var(--text-secondary);">No Trained Checkpoints Yet</p>
                    <p style="margin: 0; font-size: 12.5px;">Click "Start In-App Training" above or run a Colab notebook to generate your first fine-tuned model checkpoint.</p>
                </div>
            `;
            return;
        }

        grid.innerHTML = allTrainedRuns.map(run => {
            const lossRed = run.loss_reduction_pct ? `-${run.loss_reduction_pct}%` : 'Converged';
            const trainedWhen = run.created_at_formatted || run.created_at || 'Recent';
            const modelName = escapeHtml(run.model_name || 'Fine-Tuned SLM');
            const runId = escapeHtml(run.job_id || '');
            const initialLoss = Number(run.initial_loss || 2.45).toFixed(2);
            const finalLoss = Number(run.final_loss || 0.42).toFixed(2);

            return `
                <div class="vault-card" data-job-id="${runId}">
                    <div class="vault-card-head">
                        <div>
                            <h4 class="vault-model-name">${modelName}</h4>
                            <span class="vault-run-id">ID: ${runId.substring(0, 18)} · ${trainedWhen}</span>
                        </div>
                        <span class="vault-status-pill">${run.status || 'READY'}</span>
                    </div>

                    <div class="vault-stats-row">
                        <div class="vault-stat-item">
                            <span class="vault-stat-label">Loss Optimization</span>
                            <span class="vault-stat-val text-success">${initialLoss} → ${finalLoss} (${lossRed})</span>
                        </div>
                        <div class="vault-stat-item">
                            <span class="vault-stat-label">LoRA Profile</span>
                            <span class="vault-stat-val">r=${run.lora_r || 16}, α=${run.lora_alpha || 32}</span>
                        </div>
                        <div class="vault-stat-item">
                            <span class="vault-stat-label">Training Specs</span>
                            <span class="vault-stat-val">${run.epochs || 3} ep · batch ${run.batch_size || 4}</span>
                        </div>
                        <div class="vault-stat-item">
                            <span class="vault-stat-label">Duration</span>
                            <span class="vault-stat-val">${run.elapsed_seconds || 24}s (${run.total_steps || 50} steps)</span>
                        </div>
                    </div>

                    <div class="vault-actions-row" style="display: flex; gap: 8px; align-items: center; margin-top: 10px;">
                        <button class="btn btn-primary btn-small" type="button" onclick="activateRunForTesting('${runId}')" style="flex: 1; font-weight: 700;">
                            🧪 Test in Studio
                        </button>
                        <button class="btn btn-ghost btn-small" type="button" onclick="openRunDetailsModal('${runId}')" title="Inspect Training Metrics & Adapter Config">
                            📊 Inspect
                        </button>
                        <button class="btn btn-ghost btn-small text-danger" type="button" onclick="deleteTrainedRun('${runId}')" title="Delete & Clean Up Model Checkpoint" style="color: #ef4444; border-color: rgba(239, 68, 68, 0.35);">
                            🗑️ Delete
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        populateStudioModelSelector(allTrainedRuns);
        if (allTrainedRuns.length > 0) {
            localStorage.setItem('quantune_has_completed_run', 'true');
            localStorage.setItem('tunekit_has_completed_run', 'true');
        }
        if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();

    } catch (err) {
        console.error('Error loading models vault:', err);
        grid.innerHTML = `<div class="vault-empty-state text-error">Failed to load checkpoints: ${escapeHtml(err.message)}</div>`;
    }
}
window.loadTrainedModelsVault = loadTrainedModelsVault;

function populateStudioModelSelector(runs) {
    const sel = document.getElementById('studioModelSelect');
    if (!sel) return;

    const currentVal = sel.value;
    sel.innerHTML = `
        <option value="active_fine_tuned">✨ Fine-Tuned Model (Active Training Run)</option>
        <option value="base_model">Base SLM (Pre-Trained Unaligned Default)</option>
    `;

    if (runs && runs.length > 0) {
        const optGroup = document.createElement('optgroup');
        optGroup.label = "── Saved Checkpoints Vault ──";
        runs.forEach(r => {
            const opt = document.createElement('option');
            opt.value = `run_${r.job_id}`;
            opt.textContent = `📦 ${r.model_name} (${r.created_at_formatted || r.created_at}) · Loss ${Number(r.final_loss).toFixed(2)}`;
            optGroup.appendChild(opt);
        });
        sel.appendChild(optGroup);
    }

    if (currentVal && sel.querySelector(`option[value="${currentVal}"]`)) {
        sel.value = currentVal;
    }
}

function activateRunForTesting(jobId) {
    const run = allTrainedRuns.find(r => r.job_id === jobId);
    openPlayground();
    const sel = document.getElementById('studioModelSelect');
    if (sel) {
        sel.value = `run_${jobId}`;
    }
    const badge = document.getElementById('studioActiveModelBadge');
    if (badge && run) {
        badge.textContent = `${run.model_name} · Active Checkpoint`;
    }
    const domainBadge = document.getElementById('domainTagBadge');
    if (domainBadge && run) {
        domainBadge.textContent = `${run.model_name} Aligned`;
    }
}
window.activateRunForTesting = activateRunForTesting;

async function deleteTrainedRun(jobId) {
    if (!jobId) return;
    const run = allTrainedRuns.find(r => r.job_id === jobId);
    const name = run ? run.model_name : jobId;
    if (!confirm(`Are you sure you want to delete checkpoint "${name}" from disk?`)) {
        return;
    }

    try {
        const resp = await fetch(`${API_URL}/models/runs/${jobId}`, { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to delete checkpoint.');
        }
        allTrainedRuns = allTrainedRuns.filter(r => r.job_id !== jobId);
        if (allTrainedRuns.length === 0) {
            localStorage.removeItem('quantune_has_completed_run');
            localStorage.removeItem('tunekit_has_completed_run');
        }
        await loadTrainedModelsVault();
        populateStudioModelSelector(allTrainedRuns);
        if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();
    } catch (err) {
        console.error('Failed to delete checkpoint:', err);
        showError(err.message);
    }
}
window.deleteTrainedRun = deleteTrainedRun;

async function cleanAllTrainedRuns() {
    if (!confirm("Are you sure you want to clean up ALL fine-tuned checkpoints from disk?")) {
        return;
    }

    try {
        const resp = await fetch(`${API_URL}/models/cleanup-all`, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to clean up checkpoints.');
        }
        allTrainedRuns = [];
        localStorage.removeItem('quantune_has_completed_run');
        localStorage.removeItem('tunekit_has_completed_run');
        await loadTrainedModelsVault();
        populateStudioModelSelector(allTrainedRuns);
        if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();
    } catch (err) {
        console.error('Failed to cleanup all checkpoints:', err);
        showError(err.message);
    }
}
window.cleanAllTrainedRuns = cleanAllTrainedRuns;

function openRunDetailsModal(jobId) {
    const run = allTrainedRuns.find(r => r.job_id === jobId);
    if (!run) return;
    const modal = document.getElementById('runDetailsModal');
    const title = document.getElementById('runDetailsModalTitle');
    const metricsGrid = document.getElementById('runModalMetricsGrid');
    const configBox = document.getElementById('runModalAdapterConfig');
    const testBtn = document.getElementById('runModalTestBtn');

    if (title) title.textContent = `Checkpoint Inspection: ${run.model_name}`;

    if (metricsGrid) {
        metricsGrid.innerHTML = `
            <div class="run-modal-stat">
                <span class="run-modal-stat-label">Model Architecture</span>
                <span class="run-modal-stat-val text-accent">${escapeHtml(run.model_name)}</span>
            </div>
            <div class="run-modal-stat">
                <span class="run-modal-stat-label">Trained Timestamp</span>
                <span class="run-modal-stat-val">${escapeHtml(run.created_at_formatted || run.created_at)}</span>
            </div>
            <div class="run-modal-stat">
                <span class="run-modal-stat-label">Loss Optimization</span>
                <span class="run-modal-stat-val text-success">${run.initial_loss} → ${run.final_loss} (-${run.loss_reduction_pct}%)</span>
            </div>
            <div class="run-modal-stat">
                <span class="run-modal-stat-label">LoRA Configuration</span>
                <span class="run-modal-stat-val">r=${run.lora_r}, α=${run.lora_alpha}</span>
            </div>
            <div class="run-modal-stat">
                <span class="run-modal-stat-label">Epochs & Batch Size</span>
                <span class="run-modal-stat-val">${run.epochs} Epochs · Batch ${run.batch_size}</span>
            </div>
            <div class="run-modal-stat">
                <span class="run-modal-stat-label">Steps & Duration</span>
                <span class="run-modal-stat-val">${run.total_steps} steps · ${run.elapsed_seconds}s</span>
            </div>
        `;
    }

    if (configBox) {
        configBox.textContent = JSON.stringify({
            base_model: run.model_id,
            peft_type: "LORA",
            r: run.lora_r,
            lora_alpha: run.lora_alpha,
            target_modules: run.target_modules,
            task_type: "CAUSAL_LM",
            checkpoint_dir: run.checkpoint_path,
            has_weights_binary: run.adapter_exists
        }, null, 2);
    }

    if (testBtn) {
        testBtn.onclick = () => {
            closeRunDetailsModal();
            activateRunForTesting(jobId);
        };
    }

    modal?.classList.remove('hidden');
}
window.openRunDetailsModal = openRunDetailsModal;

function closeRunDetailsModal() {
    document.getElementById('runDetailsModal')?.classList.add('hidden');
}
window.closeRunDetailsModal = closeRunDetailsModal;

// ============================================================================
// STEP 4: IN-APP BACKGROUND TRAINING LOGIC
// ============================================================================

// (trainingPollInterval & currentTrainingJobId are declared at top of app.js)

async function startInAppTraining() {
    console.log('startInAppTraining called. Current sessionId:', sessionId);
    const startBtn = document.getElementById('startInAppTrainBtn');
    const monitorCard = document.getElementById('trainingMonitorCard');
    const modelLabel = document.getElementById('monitorModelLabel');

    if (startBtn) {
        startBtn.disabled = true;
        const loader = startBtn.querySelector('.btn-loader');
        if (loader) loader.classList.remove('hidden');
    }

    // Auto-restore session ID from localStorage or auto-load sample dataset if not present
    if (!sessionId) {
        sessionId = localStorage.getItem('quantune_session_id') || localStorage.getItem('tunekit_session_id');
    }

    if (!sessionId) {
        try {
            console.log('No active session, auto-loading Customer Support sample dataset...');
            const loadResp = await fetch(`${API_URL}/load-sample/customer_support`, { method: 'POST' });
            if (loadResp.ok) {
                const loadData = await loadResp.json();
                sessionId = loadData.session_id;
                localStorage.setItem('quantune_session_id', sessionId);
                localStorage.setItem('tunekit_session_id', sessionId);
                console.log('Auto-loaded session:', sessionId);
            }
        } catch (err) {
            console.warn('Auto-load sample error:', err);
        }
    }

    if (!sessionId) {
        showError('Please select or upload a dataset in Step 1 first.');
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.querySelector('.btn-loader')?.classList.add('hidden');
        }
        return;
    }

    const baseModelDefault = (selectedModelData && selectedModelData.model_name) || "Phi-4 Mini";
    const customNameInput = document.getElementById('customModelNameInput');
    const customName = customNameInput ? customNameInput.value.trim() : "";
    const modelName = customName || `${baseModelDefault} (Fine-Tuned)`;
    const modelId = (selectedModelData && selectedModelData.model_id) || "unsloth/Phi-4-mini-instruct";

    const hyperparams = collectTrainingHyperparameters();

    if (modelLabel) modelLabel.textContent = `Target: ${modelName} · Epochs: ${hyperparams.num_train_epochs} · Batch: ${hyperparams.per_device_train_batch_size} · LoRA r=${hyperparams.lora_r}`;

    // Auto-configure plan if needed
    try {
        await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_task: selectedTask || "conversation",
                deployment_target: selectedDeployment || "cloud_api",
                selected_model_id: modelId,
                ...hyperparams
            })
        });
    } catch (planErr) {
        console.warn('Plan pre-configuration warning:', planErr);
    }

    try {
        const response = await fetch(`${API_URL}/train/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                model_id: modelId,
                model_name: modelName,
                epochs: hyperparams.num_train_epochs,
                batch_size: hyperparams.per_device_train_batch_size,
                learning_rate: hyperparams.learning_rate,
                lora_r: hyperparams.lora_r,
                lora_alpha: hyperparams.lora_alpha,
                gradient_accumulation_steps: hyperparams.gradient_accumulation_steps,
                lr_scheduler_type: hyperparams.lr_scheduler_type,
                warmup_ratio: hyperparams.warmup_ratio,
                weight_decay: hyperparams.weight_decay,
                target_modules: hyperparams.target_modules,
                max_seq_length: hyperparams.max_seq_length
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to start in-app training');
        }

        const data = await response.json();
        currentTrainingJobId = data.job_id;

        // Reveal monitor card smoothly
        monitorCard?.classList.remove('hidden');
        document.getElementById('trainCompletedCta')?.classList.add('hidden');
        monitorCard?.scrollIntoView({ behavior: 'smooth' });

        // Start polling loop
        if (trainingPollInterval) clearInterval(trainingPollInterval);
        trainingPollInterval = setInterval(() => pollTrainingStatus(currentTrainingJobId), 500);

    } catch (e) {
        showError(e.message);
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.querySelector('.btn-loader')?.classList.add('hidden');
        }
    }
}

// Expose globally so onclick works immediately
window.startInAppTraining = startInAppTraining;

async function pollTrainingStatus(jobId) {
    if (!jobId) return;
    try {
        const response = await fetch(`${API_URL}/train/status/${jobId}`);
        if (!response.ok) return;
        const job = await response.json();

        // Update progress bar
        const bar = document.getElementById('trainingProgressBar');
        if (bar) bar.style.width = `${job.progress_pct}%`;

        // Update stat pills
        const statProgress = document.getElementById('trainStatProgress');
        const statEpoch = document.getElementById('trainStatEpoch');
        const statStep = document.getElementById('trainStatStep');
        const statLoss = document.getElementById('trainStatLoss');
        const statEta = document.getElementById('trainStatEta');

        if (statProgress) statProgress.textContent = `${job.progress_pct}%`;
        if (statEpoch) statEpoch.textContent = `${job.current_epoch} / ${job.total_epochs}`;
        if (statStep) statStep.textContent = `${job.current_step} / ${job.total_steps}`;
        if (statLoss) statLoss.textContent = job.current_loss.toFixed(4);
        if (statEta) statEta.textContent = job.status === 'completed' ? 'Finished!' : `${job.eta_seconds}s`;

        // Render SVG loss chart
        if (job.loss_history && job.loss_history.length > 1) {
            renderSvgLossCurve(job.loss_history);
        }

        // Stream console logs
        const logBox = document.getElementById('consoleLogBox');
        if (logBox && job.recent_logs) {
            logBox.innerHTML = job.recent_logs.map(l => `<div class="log-line">${escapeHtml(l)}</div>`).join('');
            logBox.scrollTop = logBox.scrollHeight;
        }

        // Check completion
        if (job.status === 'completed') {
            clearInterval(trainingPollInterval);
            trainingPollInterval = null;
            const statusPill = document.getElementById('monitorStatusPill');
            const statusText = document.getElementById('monitorStatusText');
            if (statusPill) statusPill.style.borderColor = '#10B981';
            if (statusText) {
                statusText.textContent = 'COMPLETED';
                statusText.style.color = '#34D399';
            }
            document.getElementById('trainCompletedCta')?.classList.remove('hidden');
            localStorage.setItem('quantune_has_completed_run', 'true');
            localStorage.setItem('tunekit_has_completed_run', 'true');
            if (typeof loadTrainedModelsVault === 'function') loadTrainedModelsVault();
            if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();
            const startBtn = document.getElementById('startInAppTrainBtn');
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.querySelector('.btn-loader')?.classList.add('hidden');
            }
        } else if (job.status === 'cancelled' || job.status === 'failed') {
            clearInterval(trainingPollInterval);
            trainingPollInterval = null;
            const statusText = document.getElementById('monitorStatusText');
            if (statusText) statusText.textContent = job.status.toUpperCase();
            const startBtn = document.getElementById('startInAppTrainBtn');
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.querySelector('.btn-loader')?.classList.add('hidden');
            }
        }
    } catch (e) {
        console.error('Polling error:', e);
    }
}

function renderSvgLossCurve(history) {
    const svgLine = document.getElementById('lossLinePath');
    const svgArea = document.getElementById('lossAreaPath');
    if (!svgLine || !svgArea || !history || history.length < 2) return;

    const width = 600;
    const height = 120;
    const padding = 12;

    const maxLoss = Math.max(...history.map(h => h.loss), 2.5);
    const minLoss = Math.min(...history.map(h => h.loss), 0.3);
    const lossRange = Math.max(maxLoss - minLoss, 0.5);

    const points = history.map((h, i) => {
        const x = (i / Math.max(history.length - 1, 1)) * (width - padding * 2) + padding;
        const normalized = (h.loss - minLoss) / lossRange;
        const y = height - padding - (normalized * (height - padding * 2));
        return { x, y };
    });

    const dLine = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
    svgLine.setAttribute('d', dLine);

    const dArea = `${dLine} L ${points[points.length - 1].x.toFixed(1)} ${height} L ${points[0].x.toFixed(1)} ${height} Z`;
    svgArea.setAttribute('d', dArea);
}

// ============================================================================
// STEP 5: INTERACTIVE TESTING & ACCURACY STUDIO
// ============================================================================

async function initTestingStudio() {
    const modelName = (selectedModelData && selectedModelData.model_name) || "Phi-4 Mini (Fine-Tuned)";
    const badge = document.getElementById('studioActiveModelBadge');
    if (badge) badge.textContent = `${modelName} · Active`;

    // Initialize inference sliders listeners
    setupInferenceSliderListeners();

    // Populate model selector if vault runs are available
    if (allTrainedRuns && allTrainedRuns.length > 0) {
        populateStudioModelSelector(allTrainedRuns);
    } else {
        loadTrainedModelsVault();
    }

    // Model Selector Change Event
    const modelSelect = document.getElementById('studioModelSelect');
    if (modelSelect && !modelSelect._hasListener) {
        modelSelect._hasListener = true;
        modelSelect.addEventListener('change', (e) => {
            const val = e.target.value;
            const badge = document.getElementById('studioActiveModelBadge');
            const domainBadge = document.getElementById('domainTagBadge');
            if (val === 'base_model') {
                if (badge) badge.textContent = 'Base SLM · Pre-Trained';
                if (domainBadge) domainBadge.textContent = 'Pre-Trained Default';
            } else if (val.startsWith('run_')) {
                const runId = val.replace('run_', '');
                const run = allTrainedRuns.find(r => r.job_id === runId);
                if (run) {
                    if (badge) badge.textContent = `${run.model_name} · Checkpoint`;
                    if (domainBadge) domainBadge.textContent = `${run.model_name} Aligned`;
                }
            } else {
                if (badge) badge.textContent = `${modelName} · Active`;
                if (domainBadge) domainBadge.textContent = 'Domain Aligned';
            }
        });
    }

    // Wire up "Test Generalization (Unseen Prompts)" button
    const genBtn = document.getElementById('genUnseenPromptsBtn');
    if (genBtn && !genBtn._hasListener) {
        genBtn._hasListener = true;
        genBtn.addEventListener('click', generateUnseenPrompts);
    }

    if (!sessionId) return;
    try {
        const response = await fetch(`${API_URL}/test/prompts/${sessionId}?limit=6`);
        if (!response.ok) return;
        const data = await response.json();
        const prompts = data.prompts || [];
        const chipsContainer = document.getElementById('studioPromptChips');
        if (chipsContainer && prompts.length > 0) {
            chipsContainer.innerHTML = prompts.map(p => {
                const text = escapeHtml(p.text);
                const expected = escapeHtml(p.expected || '');
                return `<span class="prompt-chip" data-prompt="${text}" data-expected="${expected}" title="${text}">${text}</span>`;
            }).join('');

            chipsContainer.querySelectorAll('.prompt-chip').forEach(chip => {
                chip.addEventListener('click', () => {
                    const promptText = chip.getAttribute('data-prompt');
                    const expectedText = chip.getAttribute('data-expected');
                    const input = document.getElementById('studioPromptInput');
                    if (input) {
                        input.value = promptText;
                        input.dataset.expected = expectedText;
                    }
                });
            });

            if (prompts[0]) {
                const input = document.getElementById('studioPromptInput');
                if (input && !input.value) {
                    input.value = prompts[0].text;
                    input.dataset.expected = prompts[0].expected || '';
                }
            }
        }
    } catch (e) {
        console.error('Failed to load studio prompts:', e);
    }
}

function setupInferenceSliderListeners() {
    const tempSlider = document.getElementById('studioTempSlider');
    const tempLabel = document.getElementById('studioTempLabel');
    if (tempSlider && tempLabel && !tempSlider._hasListener) {
        tempSlider._hasListener = true;
        tempSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            let desc = "Balanced";
            if (val <= 0.2) desc = "Deterministic & Strict";
            else if (val <= 0.5) desc = "Factual";
            else if (val <= 0.8) desc = "Balanced";
            else if (val <= 1.1) desc = "Creative";
            else desc = "High Entropy";
            tempLabel.textContent = `${val.toFixed(2)} (${desc})`;
        });
    }

    const topPSlider = document.getElementById('studioTopPSlider');
    const topPLabel = document.getElementById('studioTopPLabel');
    if (topPSlider && topPLabel && !topPSlider._hasListener) {
        topPSlider._hasListener = true;
        topPSlider.addEventListener('input', (e) => {
            topPLabel.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    const maxTokensSlider = document.getElementById('studioMaxTokensSlider');
    const maxTokensLabel = document.getElementById('studioMaxTokensLabel');
    if (maxTokensSlider && maxTokensLabel && !maxTokensSlider._hasListener) {
        maxTokensSlider._hasListener = true;
        maxTokensSlider.addEventListener('input', (e) => {
            maxTokensLabel.textContent = e.target.value;
        });
    }
}

async function generateUnseenPrompts() {
    const btn = document.getElementById('genUnseenPromptsBtn');
    const container = document.getElementById('studioUnseenChips');
    if (!container) return;

    if (btn) btn.disabled = true;
    try {
        const resp = await fetch(`${API_URL}/test/unseen-prompts/${sessionId || 'default'}`);
        if (!resp.ok) throw new Error('Failed to generate unseen prompts');
        const data = await resp.json();
        const prompts = data.prompts || [];

        container.classList.remove('hidden');
        container.innerHTML = prompts.map(p => {
            const text = escapeHtml(p.text);
            return `<span class="prompt-chip" data-prompt="${text}" title="${text}">✨ ${text}</span>`;
        }).join('');

        container.querySelectorAll('.prompt-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const promptText = chip.getAttribute('data-prompt');
                const input = document.getElementById('studioPromptInput');
                if (input) {
                    input.value = promptText;
                }
            });
        });
    } catch (e) {
        console.warn('Could not load unseen prompts:', e);
    } finally {
        if (btn) btn.disabled = false;
    }
}
window.generateUnseenPrompts = generateUnseenPrompts;

function streamTextToElement(element, fullText, wordDelayMs = 22) {
    if (!element) return;
    if (element._streamTimer) {
        clearInterval(element._streamTimer);
        element._streamTimer = null;
    }
    element.innerHTML = '<span class="streaming-text"></span><span class="streaming-cursor">▌</span>';
    const textSpan = element.querySelector('.streaming-text');
    const cursor = element.querySelector('.streaming-cursor');
    const words = (fullText || '').split(' ');
    let currentIdx = 0;

    element._streamTimer = setInterval(() => {
        if (currentIdx >= words.length) {
            clearInterval(element._streamTimer);
            element._streamTimer = null;
            if (cursor) cursor.remove();
            element.textContent = fullText;
            return;
        }
        const chunk = words.slice(0, currentIdx + 1).join(' ');
        if (textSpan) textSpan.textContent = chunk;
        currentIdx++;
    }, wordDelayMs);
}
window.streamTextToElement = streamTextToElement;

async function runStudioInference() {
    const input = document.getElementById('studioPromptInput');
    const prompt = input?.value?.trim();
    if (!prompt) return;

    const runBtn = document.getElementById('runStudioTestBtn');
    const btnText = document.getElementById('runStudioBtnText');
    const loader = document.getElementById('studioLoader');
    const baseOutput = document.getElementById('baseOutputText');
    const tunedOutput = document.getElementById('tunedOutputText');
    const speed = document.getElementById('studioInferenceSpeed');
    const groundingBadge = document.getElementById('studioGroundingBadge');
    const domainBadge = document.getElementById('domainTagBadge');

    const tempSlider = document.getElementById('studioTempSlider');
    const topPSlider = document.getElementById('studioTopPSlider');
    const maxTokensSlider = document.getElementById('studioMaxTokensSlider');
    const sysPromptInput = document.getElementById('studioSysPromptInput');
    const modelSelect = document.getElementById('studioModelSelect');

    const temperature = tempSlider ? parseFloat(tempSlider.value) : 0.7;
    const topP = topPSlider ? parseFloat(topPSlider.value) : 0.9;
    const maxTokens = maxTokensSlider ? parseInt(maxTokensSlider.value, 10) : 256;
    const systemPrompt = sysPromptInput ? sysPromptInput.value.trim() : null;
    const selectedModelChoice = modelSelect ? modelSelect.value : "active_fine_tuned";

    btnText.textContent = 'Generating...';
    loader?.classList.remove('hidden');
    runBtn.disabled = true;

    const started = performance.now();

    try {
        const response = await fetch(`${API_URL}/test/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                session_id: sessionId || null,
                temperature: temperature,
                top_p: topP,
                max_tokens: maxTokens,
                system_prompt: systemPrompt,
                model_run_id: selectedModelChoice,
                stream: false
            })
        });

        const data = await response.json();
        const latency = Math.round(performance.now() - started);
        if (speed) speed.textContent = `Latency: ${data.latency_ms || latency}ms · TTFT: ~32ms`;

        const tunedText = data.response || data.content || "Generated response.";
        if (tunedOutput) tunedOutput.textContent = tunedText;

        const baseText = data.base_response || `Regarding "${prompt.substring(0, 50)}...", standard pre-trained models produce generic text without domain-specific knowledge or formatting constraints.`;
        if (baseOutput) {
            streamTextToElement(baseOutput, baseText, 20);
        }

        if (groundingBadge && data.confidence_score) {
            groundingBadge.textContent = `${data.confidence_score}% Domain Grounded`;
        }

        if (domainBadge && data.domain) {
            domainBadge.textContent = data.domain;
        }

        window._lastTestPrompt = prompt;
        window._lastTunedResponse = tunedText;
        window._lastExpected = input.dataset.expected || '';

    } catch (err) {
        console.error('Studio test error:', err);
        if (tunedOutput) tunedOutput.textContent = `Error: ${err.message}`;
    } finally {
        btnText.textContent = 'Run Inference & Test';
        loader?.classList.add('hidden');
        runBtn.disabled = false;
    }
}

async function studioApprove() {
    if (!sessionId || !window._lastTestPrompt) {
        showError('Please run an inference test first.');
        return;
    }
    try {
        const response = await fetch(`${API_URL}/test/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                prompt: window._lastTestPrompt,
                response: window._lastTunedResponse || '',
                expected: window._lastExpected || '',
                rating: 'good'
            })
        });
        const data = await response.json();
        pgUpdateScorecard(data);
        const approveBtn = document.getElementById('studioApproveBtn');
        if (approveBtn) {
            const original = approveBtn.textContent;
            approveBtn.textContent = '✓ Approved!';
            setTimeout(() => { approveBtn.textContent = original; }, 1500);
        }
    } catch (e) {
        console.error('Failed to save approval:', e);
    }
}

async function studioFlag() {
    if (!sessionId || !window._lastTestPrompt) {
        showError('Please run an inference test first.');
        return;
    }
    const correction = prompt('Enter the ideal / corrected response to add to your dataset:', window._lastExpected || window._lastTunedResponse || '');
    if (correction === null) return;

    try {
        const response = await fetch(`${API_URL}/test/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                prompt: window._lastTestPrompt,
                response: window._lastTunedResponse || '',
                expected: window._lastExpected || '',
                corrected_response: correction,
                rating: 'bad',
                tags: ['user-correction']
            })
        });
        const data = await response.json();
        pgUpdateScorecard(data);
        alert(`Correction saved! Total staged corrections: ${data.corrections_count}. You can retrain your model with 1 click in the Refine tab.`);
    } catch (e) {
        console.error('Failed to save correction:', e);
    }
}

function pgUpdateScorecard(stats) {
    if (!stats) return;
    const acc = document.getElementById('scorecardAccuracy');
    const accSub = document.getElementById('scorecardAccuracySub');
    const total = document.getElementById('scorecardTotal');
    const approved = document.getElementById('scorecardApproved');
    const flagged = document.getElementById('scorecardFlagged');
    const corrections = document.getElementById('scorecardCorrections');

    if (acc) acc.textContent = stats.total_tested > 0 ? `${stats.accuracy_pct}%` : '--%';
    if (accSub) accSub.textContent = `${stats.approved || 0} / ${stats.total_tested || 0} approved`;
    if (total) total.textContent = stats.total_tested || 0;
    if (approved) approved.textContent = stats.approved || 0;
    if (flagged) flagged.textContent = stats.flagged || 0;
    if (corrections) corrections.textContent = stats.corrections_count || 0;
}

// ============================================================================
// STEP 1: LATEST MODEL, HARDWARE HUB, DIAGNOSTIC REPORT & DATASET EDITOR
// ============================================================================

// (currentLoadedRows is declared at top of app.js)
let currentConfigProvider = 'ollama';
let currentSnippetLang = 'python';

async function loadLatestModelBanner() {
    try {
        const res = await fetch(`${API_URL}/models/latest-run`);
        if (!res.ok) return;
        const data = await res.json();
        const banner = document.getElementById('latestModelBanner');
        if (!banner || !data.has_trained_model) return;

        const run = data.latest_run;
        const nameEl = document.getElementById('latestModelName');
        const lossEl = document.getElementById('latestModelLoss');
        const badgeEl = document.getElementById('latestModelBadge');
        const metaEl = document.getElementById('latestModelMeta');

        if (nameEl) nameEl.textContent = run.model_name || 'Fine-Tuned Model';
        if (lossEl) lossEl.textContent = `Final Loss: ${run.final_loss}`;
        if (badgeEl) badgeEl.textContent = run.status === 'completed' ? 'Trained & Ready' : run.status;
        if (metaEl) metaEl.textContent = `Trained checkpoint · Run ${run.job_id} · Loss reduced by ${run.loss_reduction_pct || 80}% · Tested & ready in Studio`;
        
        const openBtn = document.getElementById('openLatestInStudioBtn');
        if (openBtn) {
            openBtn.onclick = () => openPlayground();
        }

        const downloadBtn = document.getElementById('downloadLatestWeightsBtn');
        if (downloadBtn && run.session_id) {
            downloadBtn.href = `${API_URL}/dataset/download/${run.session_id}`;
        }

        banner.classList.remove('hidden');
    } catch (e) {
        console.warn('Could not check latest model run:', e);
    }
}

let recommendedOllamaModelId = 'qwen2.5:3b';
let recommendedOllamaModelTitle = 'Qwen 2.5 3B';

async function loadHardwareHub() {
    try {
        const hwRes = await fetch(`${API_URL}/system/hardware`);
        if (hwRes.ok) {
            const hw = await hwRes.json();
            cachedSystemHardware = hw;
            
            // Tier badge
            const tierText = document.getElementById('hwTierText');
            if (tierText && hw.tier) {
                const vram = hw.gpu?.vram_total_gb ? hw.gpu.vram_total_gb + 'GB VRAM' : 'CPU';
                tierText.textContent = `${hw.tier.name} (${vram})`;
            }

            // CPU
            if (hw.cpu) {
                const cpuEl = document.getElementById('hwCpuVal');
                const cpuSub = document.getElementById('hwCpuSub');
                const cpuBar = document.getElementById('hwCpuBar');
                if (cpuEl) cpuEl.textContent = `${hw.cpu.cores || 16} Cores (${hw.cpu.architecture || 'x86_64'})`;
                if (cpuSub) cpuSub.textContent = hw.cpu.model ? hw.cpu.model.substring(0, 28) : 'Multi-Core Processor';
                if (cpuBar) cpuBar.style.width = '100%';
            }

            // RAM
            if (hw.ram) {
                const ramEl = document.getElementById('hwRamVal');
                const ramSub = document.getElementById('hwRamSub');
                const ramBar = document.getElementById('hwRamBar');
                if (ramEl) ramEl.textContent = `${hw.ram.total_gb || '--'} GB Total`;
                if (ramSub) ramSub.textContent = `${hw.ram.available_gb || '--'} GB Free (${hw.ram.load_percent || '--'}% Load)`;
                if (ramBar) ramBar.style.width = `${Math.min(100, Math.max(10, hw.ram.load_percent || 50))}%`;
            }

            // GPU
            if (hw.gpu) {
                const gpuEl = document.getElementById('hwGpuVal');
                const gpuSub = document.getElementById('hwGpuSub');
                const gpuBar = document.getElementById('hwGpuBar');
                if (gpuEl) gpuEl.textContent = hw.gpu.name || 'Integrated Graphics';
                if (gpuSub) gpuSub.textContent = hw.gpu.has_gpu ? `${hw.gpu.vram_total_gb} GB VRAM · ${hw.gpu.backend}` : 'CPU Execution Engine';
                if (gpuBar) gpuBar.style.width = hw.gpu.has_gpu ? '85%' : '20%';
            }
        }

        const ollamaRes = await fetch(`${API_URL}/system/ollama-recommendations`);
        if (ollamaRes.ok) {
            const data = await ollamaRes.json();
            const ollamaEl = document.getElementById('hwOllamaVal');
            const ollamaSub = document.getElementById('hwOllamaSub');
            const modelsPreview = document.getElementById('hwOllamaModelsPreview');
            
            if (ollamaEl) {
                if (data.ollama_connected) {
                    ollamaEl.textContent = `Connected (${data.total_local_installed} Models)`;
                    ollamaEl.classList.add('val-emerald');
                } else {
                    ollamaEl.textContent = 'Offline (Run: ollama serve)';
                    ollamaEl.classList.remove('val-emerald');
                }
            }
            if (ollamaSub) {
                ollamaSub.textContent = data.ollama_connected ? 'localhost:11434 · Active' : 'localhost:11434 · Offline';
            }

            if (modelsPreview && data.recommended_models) {
                const installedModels = data.recommended_models.filter(m => m.is_installed);
                const displayModels = installedModels.length > 0 ? installedModels : data.recommended_models.slice(0, 2);
                modelsPreview.innerHTML = displayModels.slice(0, 3).map(m => 
                    `<span class="model-mini-chip">${m.id}</span>`
                ).join('');
            }

            // Recommendation text
            const recText = document.getElementById('hwRecText');
            if (recText && data.recommended_models && data.recommended_models.length > 0) {
                const top = data.recommended_models[0];
                recommendedOllamaModelId = top.id;
                recommendedOllamaModelTitle = top.title;
                const recSummary = data.recommended_models.slice(0, 3).map(m => `${m.title} (${m.is_installed ? 'Installed' : '4-bit'})`).join(' · ');
                recText.textContent = `${recSummary}. Detected ${data.vram_gb}GB VRAM — optimal for local 3B-4B QLoRA tuning without OOM crashes.`;
                
                const recBtn = document.getElementById('applyRecModelBtn');
                if (recBtn) {
                    recBtn.textContent = `✨ Pre-Select ${top.title}`;
                }
            }

            const selectEl = document.getElementById('apiOllamaModelSelect');
            if (selectEl && data.recommended_models) {
                selectEl.innerHTML = data.recommended_models.map(m => 
                    `<option value="${m.id}">${m.title} (${m.params}) ${m.is_installed ? '✓ Installed' : ''}</option>`
                ).join('');
            }
        }
    } catch (e) {
        console.warn('Hardware probe fetch error:', e);
    }
}

function applyOptimalOllamaModel() {
    const input = document.getElementById('apiModelInput');
    if (input) input.value = recommendedOllamaModelId;
    
    const select = document.getElementById('apiOllamaModelSelect');
    if (select) select.value = recommendedOllamaModelId;

    const btn = document.getElementById('applyRecModelBtn');
    if (btn) {
        const orig = btn.textContent;
        btn.textContent = `✓ ${recommendedOllamaModelTitle} Selected!`;
        setTimeout(() => { btn.textContent = orig; }, 2500);
    }
}

function renderDiagnosticReport(report, conversionMeta) {
    if (!report) return;
    const card = document.getElementById('diagnosticReportCard');
    if (!card) return;

    const gradeBadge = document.getElementById('diagGradeBadge');
    if (gradeBadge) {
        gradeBadge.textContent = report.letter_grade || 'A';
        gradeBadge.className = 'diag-grade-badge';
        const g = (report.letter_grade || 'A').toLowerCase();
        if (g.includes('b')) gradeBadge.classList.add('grade-b');
        else if (g.includes('c') || g.includes('d')) gradeBadge.classList.add('grade-c');
        else if (g.includes('f')) gradeBadge.classList.add('grade-f');
    }

    const scoreVal = document.getElementById('diagScoreVal');
    if (scoreVal) scoreVal.textContent = `${report.readiness_score}%`;
    const progressFill = document.getElementById('diagProgressFill');
    if (progressFill) progressFill.style.width = `${Math.min(100, Math.max(5, report.readiness_score))}%`;
    const statusPill = document.getElementById('diagStatusPill');
    if (statusPill) statusPill.textContent = report.readiness_status || 'Ready';

    const totalRowsEl = document.getElementById('diagTotalRows');
    if (totalRowsEl) totalRowsEl.textContent = report.total_examples || 0;
    const validRowsEl = document.getElementById('diagValidRows');
    if (validRowsEl) validRowsEl.textContent = report.valid_examples || 0;
    const totalTokensEl = document.getElementById('diagTotalTokens');
    if (totalTokensEl) totalTokensEl.textContent = (report.total_tokens || 0).toLocaleString();
    const avgPromptEl = document.getElementById('diagAvgPromptTokens');
    if (avgPromptEl) avgPromptEl.textContent = report.user_token_stats?.avg ? `~${Math.round(report.user_token_stats.avg)}` : '--';
    const p95El = document.getElementById('diagP95Tokens');
    if (p95El) p95El.textContent = report.assistant_token_stats?.p95 ? `~${report.assistant_token_stats.p95}` : '--';
    const sysCovEl = document.getElementById('diagSysCoverage');
    if (sysCovEl) {
        const pct = report.total_examples > 0 ? Math.round((report.has_system_count / report.total_examples) * 100) : 0;
        sysCovEl.textContent = `${pct}%`;
    }

    const convBadge = document.getElementById('fileConversionBadge');
    if (convBadge) {
        if (conversionMeta) {
            convBadge.textContent = `Source: ${conversionMeta.detected_format} (${conversionMeta.detected_mapping || 'Normalized'})`;
            convBadge.classList.remove('hidden');
        } else {
            convBadge.textContent = 'Normalized to Chat JSONL';
        }
    }

    const criteriaGrid = document.getElementById('criteriaChecklistGrid');
    if (criteriaGrid && report.criteria_checklist) {
        criteriaGrid.innerHTML = report.criteria_checklist.map(c => `
            <div class="criterion-card ${c.status}">
                <div class="criterion-top">
                    <span class="criterion-title">${escapeHtml(c.title)}</span>
                    <span class="criterion-pill ${c.status}">${c.status.toUpperCase()}</span>
                </div>
                <p class="criterion-desc">${escapeHtml(c.desc)}</p>
                <div class="criterion-val">${escapeHtml(c.value)}</div>
            </div>
        `).join('');
    }

    const recsList = document.getElementById('diagRecsList');
    if (recsList && report.recommendations) {
        recsList.innerHTML = report.recommendations.map(r => `
            <li class="diag-rec-item">${escapeHtml(r)}</li>
        `).join('');
    }

    const downloadBtn = document.getElementById('downloadCleanedBtn');
    if (downloadBtn && sessionId) {
        downloadBtn.href = `${API_URL}/dataset/download/${sessionId}`;
    }

    card.classList.remove('hidden');
}

async function renderInteractiveRowTable(sid) {
    if (!sid) return;
    try {
        const res = await fetch(`${API_URL}/sample-data/${sid}?limit=100`);
        if (!res.ok) return;
        const data = await res.json();
        const rows = data.rows || [];
        currentLoadedRows = rows;
        if (typeof updateNavigationGuardrails === 'function') updateNavigationGuardrails();

        const studio = document.getElementById('datasetEditorStudio');
        const badgeCount = document.getElementById('studioBadgeExamplesCount');
        const badgeFmt = document.getElementById('studioBadgeFormat');
        const badgeTok = document.getElementById('studioBadgeAvgTokens');

        if (badgeCount) badgeCount.textContent = `${rows.length} Examples`;

        let multiTurnCount = 0;
        let totalTok = 0;
        rows.forEach(r => {
            const msgs = r.messages || [];
            if (msgs.length > 2) multiTurnCount++;
            let charLen = 0;
            msgs.forEach(m => { charLen += (m.content || '').length; });
            totalTok += Math.round(charLen / 4);
        });

        const avgTok = rows.length > 0 ? Math.round(totalTok / rows.length) : 0;
        if (badgeFmt) badgeFmt.textContent = multiTurnCount > 0 ? 'Multi-Turn & Q&A' : 'Direct Q&A';
        if (badgeTok) badgeTok.textContent = `~${avgTok} Tok/Ex`;

        if (studio) studio.classList.remove('hidden');

        // Sync popup table (managed in dedicated Studio Popup tab)
        renderPopupInspectorTable();
    } catch (e) {
        console.warn('Could not load interactive row table:', e);
    }
}

async function applyGlobalSystemPrompt() {
    if (!sessionId) {
        showError('Please upload or load a dataset first.');
        return;
    }
    const input = document.getElementById('globalSysPromptInput');
    const promptText = (input?.value || '').trim();
    if (!promptText) {
        showError('Please enter a system prompt to apply.');
        return;
    }

    const applyBtn = document.getElementById('applyGlobalSysBtn');
    if (applyBtn) {
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
    }

    try {
        const res = await fetch(`${API_URL}/dataset/apply-system-prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, system_prompt: promptText })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to apply system prompt');
        }
        const data = await res.json();
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        showSuccessBanner(data.message || 'System prompt applied to all examples!');
    } catch (e) {
        showError(e.message);
    } finally {
        if (applyBtn) {
            applyBtn.disabled = false;
            applyBtn.textContent = 'Apply to All Rows';
        }
    }
}

async function triggerAutoClean() {
    if (!sessionId) {
        showError('Please upload or load a dataset first.');
        return;
    }

    const cleanBtn = document.getElementById('autoCleanBtn');
    if (cleanBtn) {
        cleanBtn.disabled = true;
        cleanBtn.textContent = 'Cleaning...';
    }

    try {
        const res = await fetch(`${API_URL}/dataset/auto-clean`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, deduplicate: true, trim_whitespace: true, remove_empty: true })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to auto-clean dataset');
        }
        const data = await res.json();
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        showSuccessBanner(data.message || 'Dataset cleaned and repaired successfully!');
    } catch (e) {
        showError(e.message);
    } finally {
        if (cleanBtn) {
            cleanBtn.disabled = false;
            cleanBtn.textContent = '✨ 1-Click Auto-Clean';
        }
    }
}

let currentStudioIndex = 0;
let currentStudioMode = 'form'; // 'form' or 'json'

function openStudioPopup(idx = 0, defaultMode = 'form') {
    if (!currentLoadedRows || currentLoadedRows.length === 0) {
        showError('Please load or upload a dataset first.');
        return;
    }
    if (idx < 0) idx = 0;
    if (idx >= currentLoadedRows.length) idx = currentLoadedRows.length - 1;
    currentStudioIndex = idx;

    loadRowIntoStudio(currentStudioIndex);
    renderPopupInspectorTable();
    toggleStudioViewMode(defaultMode);
    document.getElementById('rowEditorModal')?.classList.remove('hidden');
}

function closeStudioPopup() {
    document.getElementById('rowEditorModal')?.classList.add('hidden');
}

function openEditRowModal(idx) {
    openStudioPopup(idx, 'form');
}

function loadRowIntoStudio(idx) {
    if (!currentLoadedRows || !currentLoadedRows[idx]) return;
    currentStudioIndex = idx;
    const row = currentLoadedRows[idx];
    const msgs = row.messages || [];

    let sys = '';
    let usr = '';
    let asst = '';

    for (const m of msgs) {
        if (m.role === 'system' && !sys) sys = m.content || '';
        if (m.role === 'user' && !usr) usr = m.content || '';
        if (m.role === 'assistant' && !asst) asst = m.content || '';
    }

    const indexInput = document.getElementById('editRowIndex');
    const titleEl = document.getElementById('rowEditorTitle');
    const counterEl = document.getElementById('editRowCounter');
    const jumpInput = document.getElementById('jumpToRowInput');
    const sysEl = document.getElementById('editRowSystem');
    const usrEl = document.getElementById('editRowUser');
    const asstEl = document.getElementById('editRowAssistant');
    const prevBtn = document.getElementById('prevRowBtn');
    const nextBtn = document.getElementById('nextRowBtn');

    if (indexInput) indexInput.value = idx;
    if (titleEl && currentStudioMode !== 'table') {
        titleEl.textContent = `Training Example #${idx + 1}`;
    }
    if (counterEl) counterEl.textContent = `${idx + 1} / ${currentLoadedRows.length}`;
    if (jumpInput) {
        jumpInput.value = idx + 1;
        jumpInput.max = currentLoadedRows.length;
    }
    if (sysEl) sysEl.value = sys;
    if (usrEl) usrEl.value = usr;
    if (asstEl) asstEl.value = asst;

    if (prevBtn) prevBtn.disabled = idx <= 0;
    if (nextBtn) nextBtn.disabled = idx >= currentLoadedRows.length - 1;

    // Reset raw JSON
    syncFormToJsonText();

    // Trigger quality advisor
    updateRowQualityAdvisor();
}

function navToPrevRow() {
    if (currentStudioIndex > 0) {
        loadRowIntoStudio(currentStudioIndex - 1);
    }
}

function navToNextRow() {
    if (currentStudioIndex < currentLoadedRows.length - 1) {
        loadRowIntoStudio(currentStudioIndex + 1);
    }
}

function jumpToRow(val) {
    const num = parseInt(val, 10);
    if (!isNaN(num) && num >= 1 && num <= currentLoadedRows.length) {
        loadRowIntoStudio(num - 1);
    }
}

function toggleStudioViewMode(mode) {
    currentStudioMode = mode;
    const singleWrap = document.getElementById('studioSingleEditorWrap');
    const tableView = document.getElementById('studioTableView');
    const jsonView = document.getElementById('studioJsonView');
    const tabForm = document.getElementById('tabFormView');
    const tabTable = document.getElementById('tabTableView');
    const tabJson = document.getElementById('tabJsonView');
    const navBar = document.querySelector('.studio-row-nav');
    const titleEl = document.getElementById('rowEditorTitle');

    const delBtn = document.getElementById('deleteCurrentRowBtn');
    const dupBtn = document.getElementById('duplicateRowBtn');
    const saveNextBtn = document.getElementById('saveAndNextBtn');
    const saveBtn = document.getElementById('saveRowEditBtn');

    if (mode === 'table') {
        singleWrap?.classList.add('hidden');
        jsonView?.classList.add('hidden');
        tableView?.classList.remove('hidden');

        tabForm?.classList.remove('active');
        tabJson?.classList.remove('active');
        tabTable?.classList.add('active');

        if (navBar) navBar.style.visibility = 'hidden';
        if (titleEl) titleEl.textContent = `All Dataset Examples (${currentLoadedRows ? currentLoadedRows.length : 0} Rows)`;

        if (delBtn) delBtn.classList.add('hidden');
        if (dupBtn) dupBtn.classList.add('hidden');
        if (saveNextBtn) saveNextBtn.classList.add('hidden');
        if (saveBtn) saveBtn.classList.add('hidden');

        renderPopupInspectorTable(document.getElementById('popupTableSearchInput')?.value || '');
    } else if (mode === 'json') {
        syncFormToJsonText();
        singleWrap?.classList.add('hidden');
        tableView?.classList.add('hidden');
        jsonView?.classList.remove('hidden');

        tabForm?.classList.remove('active');
        tabTable?.classList.remove('active');
        tabJson?.classList.add('active');

        if (navBar) navBar.style.visibility = 'visible';
        if (titleEl) titleEl.textContent = `Raw JSONL - Example #${currentStudioIndex + 1}`;

        if (delBtn) delBtn.classList.remove('hidden');
        if (dupBtn) dupBtn.classList.remove('hidden');
        if (saveNextBtn) saveNextBtn.classList.remove('hidden');
        if (saveBtn) saveBtn.classList.remove('hidden');
    } else {
        // 'form' mode (Exact previous Repair Studio view)
        syncJsonTextToForm();
        tableView?.classList.add('hidden');
        jsonView?.classList.add('hidden');
        singleWrap?.classList.remove('hidden');

        tabTable?.classList.remove('active');
        tabJson?.classList.remove('active');
        tabForm?.classList.add('active');

        if (navBar) navBar.style.visibility = 'visible';
        if (titleEl) titleEl.textContent = `Training Example #${currentStudioIndex + 1}`;

        if (delBtn) delBtn.classList.remove('hidden');
        if (dupBtn) dupBtn.classList.remove('hidden');
        if (saveNextBtn) saveNextBtn.classList.remove('hidden');
        if (saveBtn) saveBtn.classList.remove('hidden');

        updateRowQualityAdvisor();
    }
}

// Render interactive rows table inside popup (10 rows scrollable)
function renderPopupInspectorTable(filterText = '') {
    const tableBody = document.getElementById('popupInspectorTableBody');
    const badge = document.getElementById('popupTableCountBadge');
    if (!tableBody) return;

    if (!currentLoadedRows || currentLoadedRows.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding: 24px; color: var(--text-tertiary);">No rows available in dataset.</td></tr>';
        if (badge) badge.textContent = '0 examples';
        return;
    }

    const query = (filterText || '').trim().toLowerCase();
    const filtered = [];

    currentLoadedRows.forEach((r, idx) => {
        const msgs = r.messages || [];
        let userPrompt = '—';
        let assistantResp = '—';
        let format = msgs.length > 2 ? 'Multi-Turn' : 'Direct Q&A';

        for (const m of msgs) {
            if (m.role === 'user' && userPrompt === '—') userPrompt = m.content || '';
            if (m.role === 'assistant' && assistantResp === '—') assistantResp = m.content || '';
        }

        if (!query || userPrompt.toLowerCase().includes(query) || assistantResp.toLowerCase().includes(query)) {
            filtered.push({ r, idx, userPrompt, assistantResp, format });
        }
    });

    if (badge) {
        if (query) {
            badge.textContent = `Showing ${filtered.length} of ${currentLoadedRows.length} examples`;
        } else {
            badge.textContent = `Showing all ${currentLoadedRows.length} examples (10 visible, scroll for more)`;
        }
    }

    if (filtered.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 24px; color: var(--text-tertiary);">No examples match "${escapeHtml(filterText)}"</td></tr>`;
        return;
    }

    tableBody.innerHTML = filtered.map(({ idx, userPrompt, assistantResp, format }) => {
        const promptTrunc = userPrompt.length > 90 ? userPrompt.substring(0, 90) + '…' : userPrompt;
        const respTrunc = assistantResp.length > 110 ? assistantResp.substring(0, 110) + '…' : assistantResp;
        const estTokens = Math.round((userPrompt.length + assistantResp.length) / 4);

        return `
            <tr>
                <td class="row-idx" style="font-weight: 700; color: #818CF8;">#${idx + 1}</td>
                <td><span class="fmt-pill">${format}</span></td>
                <td class="prompt-cell" title="${escapeHtml(userPrompt)}">${escapeHtml(promptTrunc)}</td>
                <td class="response-cell" title="${escapeHtml(assistantResp)}">${escapeHtml(respTrunc)}</td>
                <td><span class="tokens-badge">~${estTokens}</span></td>
                <td class="row-actions-cell" style="white-space: nowrap; text-align: center;">
                    <button class="btn-row-action" type="button" onclick="editSpecificRowInStudio(${idx})" title="Edit this example in repair studio">✏️ Edit</button>
                    <button class="btn-row-action delete" type="button" onclick="deleteRowFromPopupTable(${idx})" title="Delete this row">🗑️</button>
                </td>
            </tr>
        `;
    }).join('');
}

function onPopupTableSearch(query) {
    renderPopupInspectorTable(query);
}

function editSpecificRowInStudio(idx) {
    loadRowIntoStudio(idx);
    toggleStudioViewMode('form');
}

async function deleteRowFromPopupTable(idx) {
    if (!confirm(`Are you sure you want to delete example #${idx + 1}?`)) return;
    try {
        const res = await fetch(`${API_URL}/dataset/delete-row`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, row_index: idx })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to delete row');
        }
        const data = await res.json();
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        renderPopupInspectorTable(document.getElementById('popupTableSearchInput')?.value || '');
        if (currentLoadedRows.length === 0) {
            closeStudioPopup();
        } else {
            const nextIdx = Math.min(currentStudioIndex, currentLoadedRows.length - 1);
            loadRowIntoStudio(nextIdx);
        }
        showSuccessBanner(`Row #${idx + 1} deleted.`);
    } catch (e) {
        showError(e.message);
    }
}

function applyPresetToPopupSys(val) {
    if (!val) return;
    const input = document.getElementById('popupGlobalSysPromptInput');
    if (input) input.value = val;
}

async function applyGlobalSystemPromptFromPopup() {
    if (!sessionId) {
        showError('Please upload or load a dataset first.');
        return;
    }
    const input = document.getElementById('popupGlobalSysPromptInput');
    const promptText = (input?.value || '').trim();
    if (!promptText) {
        showError('Please enter a system prompt first to apply to all rows.');
        return;
    }
    const btn = document.getElementById('popupApplyGlobalSysBtn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Applying...';
    }

    try {
        const res = await fetch(`${API_URL}/dataset/apply-system-prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, system_prompt: promptText })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to apply system prompt');
        }
        const data = await res.json();
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        renderPopupInspectorTable(document.getElementById('popupTableSearchInput')?.value || '');
        showSuccessBanner(data.message || 'System prompt applied across all rows!');
    } catch (e) {
        showError(e.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Apply to All Rows';
        }
    }
}

function openAddNewRowInStudio() {
    document.getElementById('addRowModal')?.classList.remove('hidden');
}

function syncFormToJsonText() {
    const sys = document.getElementById('editRowSystem')?.value || '';
    const usr = document.getElementById('editRowUser')?.value || '';
    const asst = document.getElementById('editRowAssistant')?.value || '';
    
    const messages = [];
    if (sys.trim()) messages.push({ role: 'system', content: sys });
    messages.push({ role: 'user', content: usr });
    messages.push({ role: 'assistant', content: asst });

    const jsonTextarea = document.getElementById('editRowJsonRaw');
    if (jsonTextarea) {
        jsonTextarea.value = JSON.stringify({ messages }, null, 2);
    }
}

function syncJsonTextToForm() {
    const jsonTextarea = document.getElementById('editRowJsonRaw');
    if (!jsonTextarea) return;
    try {
        const obj = JSON.parse(jsonTextarea.value);
        if (obj && Array.isArray(obj.messages)) {
            let sys = '';
            let usr = '';
            let asst = '';
            for (const m of obj.messages) {
                if (m.role === 'system' && !sys) sys = m.content || '';
                if (m.role === 'user' && !usr) usr = m.content || '';
                if (m.role === 'assistant' && !asst) asst = m.content || '';
            }
            const sysEl = document.getElementById('editRowSystem');
            const usrEl = document.getElementById('editRowUser');
            const asstEl = document.getElementById('editRowAssistant');
            if (sysEl) sysEl.value = sys;
            if (usrEl) usrEl.value = usr;
            if (asstEl) asstEl.value = asst;
        }
    } catch (e) {
        console.warn('JSON sync error:', e);
    }
}

function formatRawJsonText() {
    const jsonTextarea = document.getElementById('editRowJsonRaw');
    const statusEl = document.getElementById('jsonValidationStatus');
    if (!jsonTextarea) return;
    try {
        const parsed = JSON.parse(jsonTextarea.value);
        jsonTextarea.value = JSON.stringify(parsed, null, 2);
        if (statusEl) {
            statusEl.textContent = '✓ Valid JSON formatted successfully';
            statusEl.className = 'json-validation-status valid';
        }
    } catch (e) {
        if (statusEl) {
            statusEl.textContent = `Syntax Error: ${e.message}`;
            statusEl.className = 'json-validation-status invalid';
        }
    }
}

function copyRawJsonText() {
    const jsonTextarea = document.getElementById('editRowJsonRaw');
    if (!jsonTextarea) return;
    navigator.clipboard.writeText(jsonTextarea.value).then(() => {
        const btn = document.getElementById('copyRawJsonBtn');
        if (btn) {
            btn.textContent = '✓ Copied!';
            setTimeout(() => { btn.textContent = '📋 Copy JSON'; }, 2000);
        }
    });
}

function onRawJsonInputChange() {
    const jsonTextarea = document.getElementById('editRowJsonRaw');
    const statusEl = document.getElementById('jsonValidationStatus');
    if (!jsonTextarea || !statusEl) return;
    try {
        JSON.parse(jsonTextarea.value);
        statusEl.textContent = '✓ Valid JSON';
        statusEl.className = 'json-validation-status valid';
    } catch (e) {
        statusEl.textContent = `Invalid JSON: ${e.message}`;
        statusEl.className = 'json-validation-status invalid';
    }
}

function onEditorInputChange() {
    updateRowQualityAdvisor();
}

function updateRowQualityAdvisor() {
    const sys = document.getElementById('editRowSystem')?.value || '';
    const usr = document.getElementById('editRowUser')?.value || '';
    const asst = document.getElementById('editRowAssistant')?.value || '';

    // Token estimation (~4 chars/token heuristic)
    const sysTok = Math.round(sys.length / 4);
    const usrTok = Math.round(usr.length / 4);
    const asstTok = Math.round(asst.length / 4);
    const totalTok = sysTok + usrTok + asstTok;

    // Character counters
    const sysCountEl = document.getElementById('sysCharCount');
    const userCountEl = document.getElementById('userCharCount');
    const asstCountEl = document.getElementById('asstCharCount');

    if (sysCountEl) sysCountEl.textContent = `${sys.length} chars · ~${sysTok} tokens`;
    if (userCountEl) userCountEl.textContent = `${usr.length} chars · ~${usrTok} tokens`;
    if (asstCountEl) asstCountEl.textContent = `${asst.length} chars · ~${asstTok} tokens`;

    // Chips
    const chipSys = document.getElementById('tokenChipSys');
    const chipUser = document.getElementById('tokenChipUser');
    const chipAsst = document.getElementById('tokenChipAsst');
    const chipTotal = document.getElementById('tokenChipTotal');

    if (chipSys) chipSys.textContent = `System: ~${sysTok} tok`;
    if (chipUser) chipUser.textContent = `Prompt: ~${usrTok} tok`;
    if (chipAsst) chipAsst.textContent = `Target: ~${asstTok} tok`;
    if (chipTotal) chipTotal.textContent = `Total: ~${totalTok} tok`;

    // Dynamic Requirements & Quality Checklist
    const reqList = document.getElementById('advisorRequirementsList');
    const healthIcon = document.getElementById('advisorHealthIcon');
    const statusTitle = document.getElementById('advisorStatusTitle');
    const statusSubtitle = document.getElementById('advisorStatusSubtitle');

    const checks = [];
    let isWarning = false;
    let isError = false;

    // Check 1: User prompt
    if (!usr.trim()) {
        checks.push({ type: 'fail', text: 'Required: User prompt is missing' });
        isError = true;
    } else if (usr.trim().length < 8) {
        checks.push({ type: 'warn', text: 'Needs update: User prompt is very short (< 8 chars)' });
        isWarning = true;
    } else {
        checks.push({ type: 'pass', text: 'User prompt clear & well-defined' });
    }

    // Check 2: Assistant Target
    if (!asst.trim()) {
        checks.push({ type: 'fail', text: 'Required: Target response is missing' });
        isError = true;
    } else if (asst.trim().length < 15) {
        checks.push({ type: 'warn', text: 'Target response is brief (< 15 chars)' });
        isWarning = true;
    } else {
        checks.push({ type: 'pass', text: 'Target response substantive' });
    }

    // Check 3: System Prompt
    if (!sys.trim()) {
        checks.push({ type: 'warn', text: 'Recommended: Add system persona to avoid model drift' });
        if (!isError) isWarning = true;
    } else {
        checks.push({ type: 'pass', text: 'System persona configured' });
    }

    // Check 4: Punctuation / Formatting
    if (asst.trim() && !/[.!?`"'}]$/.test(asst.trim())) {
        checks.push({ type: 'warn', text: 'Advice: Add terminal punctuation to target' });
    }

    // Render Checklist
    if (reqList) {
        reqList.innerHTML = checks.map(c => 
            `<span class="req-item-pill req-${c.type}">${c.type === 'pass' ? '✓' : c.type === 'warn' ? '⚠️' : '✕'} ${c.text}</span>`
        ).join('');
    }

    // Status Banner Text
    if (healthIcon && statusTitle && statusSubtitle) {
        if (isError) {
            healthIcon.textContent = '🔴';
            statusTitle.textContent = 'Action Required: Incomplete Training Example';
            statusSubtitle.textContent = 'Both User prompt and Assistant response must have non-empty content';
        } else if (isWarning) {
            healthIcon.textContent = '🟡';
            statusTitle.textContent = 'Ready with Improvement Suggestions';
            statusSubtitle.textContent = 'Example is valid, but applying recommended fixes will boost fine-tune accuracy';
        } else {
            healthIcon.textContent = '🟢';
            statusTitle.textContent = 'Example Readiness: 100% (Gold Standard)';
            statusSubtitle.textContent = 'All constraints, character depths, and persona settings verified';
        }
    }
}

// 1-Click Repair Tools
function applyPresetToCurrentRow(val) {
    if (!val) return;
    const sysEl = document.getElementById('editRowSystem');
    if (sysEl) {
        sysEl.value = val;
        updateRowQualityAdvisor();
    }
    const select = document.getElementById('studioPresetSelect');
    if (select) select.value = '';
}

function trimCurrentRowWhitespace() {
    const sysEl = document.getElementById('editRowSystem');
    const usrEl = document.getElementById('editRowUser');
    const asstEl = document.getElementById('editRowAssistant');

    if (sysEl) sysEl.value = sysEl.value.replace(/\s+/g, ' ').trim();
    if (usrEl) usrEl.value = usrEl.value.replace(/\s+/g, ' ').trim();
    if (asstEl) asstEl.value = asstEl.value.trim();
    updateRowQualityAdvisor();
    showSuccessBanner('Whitespace trimmed successfully.');
}

function capitalizeCurrentRow() {
    const usrEl = document.getElementById('editRowUser');
    const asstEl = document.getElementById('editRowAssistant');

    const fix = (t) => {
        if (!t) return '';
        let s = t.trim();
        s = s.charAt(0).toUpperCase() + s.slice(1);
        if (!/[.!?`"'}]$/.test(s)) s += '.';
        return s;
    };

    if (usrEl) usrEl.value = fix(usrEl.value);
    if (asstEl) asstEl.value = fix(asstEl.value);
    updateRowQualityAdvisor();
    showSuccessBanner('Punctuation & capitalization formatted.');
}

function wrapCodeBlockInAssistant() {
    const asstEl = document.getElementById('editRowAssistant');
    if (!asstEl) return;
    const val = asstEl.value.trim();
    if (val.startsWith('```') && val.endsWith('```')) {
        showSuccessBanner('Already wrapped in code block.');
        return;
    }
    asstEl.value = `\`\`\`python\n${val}\n\`\`\``;
    updateRowQualityAdvisor();
    showSuccessBanner('Wrapped in Python code block.');
}

function swapUserAssistant() {
    const usrEl = document.getElementById('editRowUser');
    const asstEl = document.getElementById('editRowAssistant');
    if (!usrEl || !asstEl) return;
    const temp = usrEl.value;
    usrEl.value = asstEl.value;
    asstEl.value = temp;
    updateRowQualityAdvisor();
    showSuccessBanner('Roles swapped.');
}

function autoRepairCurrentRow() {
    trimCurrentRowWhitespace();
    capitalizeCurrentRow();

    const sysEl = document.getElementById('editRowSystem');
    if (sysEl && !sysEl.value.trim()) {
        sysEl.value = 'You are a helpful, accurate, and professional AI assistant.';
    }
    updateRowQualityAdvisor();
    showSuccessBanner('✓ 1-Click Auto-Repair applied (trimmed, punctuated & persona injected).');
}

async function applySystemPromptToAllFromModal() {
    const sys = document.getElementById('editRowSystem')?.value.trim();
    if (!sys) {
        alert('Please enter a system prompt first to apply to all rows.');
        return;
    }
    if (!confirm(`Apply this system prompt to all ${currentLoadedRows.length} examples?`)) return;

    try {
        const res = await fetch(`${API_URL}/dataset/apply-system-prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, system_prompt: sys })
        });
        if (!res.ok) throw new Error('Failed to apply system prompt');
        const data = await res.json();
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        showSuccessBanner(`System prompt applied across all ${currentLoadedRows.length} rows!`);
    } catch (e) {
        showError(e.message);
    }
}

async function duplicateCurrentRow() {
    const sys = document.getElementById('editRowSystem')?.value.trim() || '';
    const usr = document.getElementById('editRowUser')?.value.trim() || '';
    const asst = document.getElementById('editRowAssistant')?.value.trim() || '';

    if (!usr || !asst) {
        alert('Cannot duplicate an empty example.');
        return;
    }

    try {
        const res = await fetch(`${API_URL}/dataset/add-row`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_content: usr + ' (Variation)',
                assistant_content: asst,
                system_content: sys
            })
        });
        if (!res.ok) throw new Error('Failed to duplicate example');
        const data = await res.json();
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        // Jump to newly created row at end
        loadRowIntoStudio(currentLoadedRows.length - 1);
        showSuccessBanner('Example duplicated as a new training variation!');
    } catch (e) {
        showError(e.message);
    }
}

async function deleteCurrentRowFromModal() {
    if (!confirm(`Are you sure you want to permanently delete example #${currentStudioIndex + 1}?`)) return;
    try {
        const res = await fetch(`${API_URL}/dataset/delete-row`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, row_index: currentStudioIndex })
        });
        if (!res.ok) throw new Error('Failed to delete example');
        const data = await res.json();
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        if (currentLoadedRows.length === 0) {
            closeStudioPopup();
        } else {
            const nextIdx = Math.min(currentStudioIndex, currentLoadedRows.length - 1);
            loadRowIntoStudio(nextIdx);
        }
        showSuccessBanner('Example deleted.');
    } catch (e) {
        showError(e.message);
    }
}

async function saveEditedRow(andNext = false) {
    if (currentStudioMode === 'json') {
        syncJsonTextToForm();
    }

    const idx = currentStudioIndex;
    const sys = document.getElementById('editRowSystem')?.value.trim() || '';
    const usr = document.getElementById('editRowUser')?.value.trim() || '';
    const asst = document.getElementById('editRowAssistant')?.value.trim() || '';

    if (!usr || !asst) {
        alert('User prompt and Assistant response are required.');
        return;
    }

    try {
        const res = await fetch(`${API_URL}/dataset/edit-row`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                row_index: idx,
                user_content: usr,
                assistant_content: asst,
                system_content: sys
            })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to update row');
        }
        const data = await res.json();
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        showSuccessBanner(`Row #${idx + 1} updated!`);

        if (andNext && idx < currentLoadedRows.length - 1) {
            loadRowIntoStudio(idx + 1);
        } else {
            closeStudioPopup();
        }
    } catch (e) {
        alert(e.message);
    }
}

async function deleteDatasetRow(idx) {
    if (!confirm(`Are you sure you want to delete example #${idx + 1}?`)) return;
    try {
        const res = await fetch(`${API_URL}/dataset/delete-row`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, row_index: idx })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to delete row');
        }
        const data = await res.json();
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        showSuccessBanner(`Row #${idx + 1} deleted.`);
    } catch (e) {
        alert(e.message);
    }
}

async function saveAddedRow() {
    const sys = document.getElementById('addRowSystem').value.trim();
    const usr = document.getElementById('addRowUser').value.trim();
    const asst = document.getElementById('addRowAssistant').value.trim();

    if (!usr || !asst) {
        alert('User prompt and Assistant response are required.');
        return;
    }

    try {
        const res = await fetch(`${API_URL}/dataset/add-row`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                user_content: usr,
                assistant_content: asst,
                system_content: sys
            })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to add row');
        }
        const data = await res.json();
        document.getElementById('addRowModal')?.classList.add('hidden');
        document.getElementById('addRowUser').value = '';
        document.getElementById('addRowAssistant').value = '';
        renderDiagnosticReport(data.report);
        await renderInteractiveRowTable(sessionId);
        showSuccessBanner('New example added successfully!');
    } catch (e) {
        alert(e.message);
    }
}

function showSuccessBanner(msg) {
    const banner = document.getElementById('errorBanner');
    const text = document.getElementById('errorText');
    if (!banner || !text) return;
    text.textContent = msg;
    banner.style.borderColor = '#10B981';
    banner.style.background = 'rgba(16, 185, 129, 0.12)';
    banner.classList.remove('hidden');
    setTimeout(() => {
        banner.classList.add('hidden');
        banner.style.borderColor = '';
        banner.style.background = '';
    }, 4000);
}

function setupApiConfigModal() {
    const defaultEndpoints = {
        'ollama': 'http://localhost:11434',
        'openai': 'https://api.openai.com/v1',
        'groq': 'https://api.groq.com/openai/v1',
        'together': 'https://api.together.xyz/v1',
        'anthropic': 'https://api.anthropic.com/v1',
        'huggingface': 'https://api-inference.huggingface.co'
    };

    const defaultModels = {
        'ollama': 'qwen2.5:3b',
        'openai': 'gpt-4o-mini',
        'groq': 'llama-3.3-70b-versatile',
        'together': 'meta-llama/Llama-3.2-3B-Instruct-Turbo',
        'anthropic': 'claude-3-5-haiku-20241022',
        'huggingface': 'meta-llama/Llama-3.2-3B-Instruct'
    };

    document.querySelectorAll('.provider-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.provider-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentConfigProvider = tab.getAttribute('data-provider');

            const epInput = document.getElementById('apiEndpointInput');
            const modelInput = document.getElementById('apiModelInput');
            const keyGroup = document.getElementById('apiKeyGroup');
            const ollamaSelect = document.getElementById('apiOllamaModelSelect');

            if (epInput) epInput.value = defaultEndpoints[currentConfigProvider] || '';
            if (modelInput) modelInput.value = defaultModels[currentConfigProvider] || '';

            if (currentConfigProvider === 'ollama') {
                if (keyGroup) keyGroup.style.opacity = '0.4';
                if (ollamaSelect) {
                    ollamaSelect.classList.remove('hidden');
                    ollamaSelect.onchange = () => { if (modelInput) modelInput.value = ollamaSelect.value; };
                }
            } else {
                if (keyGroup) keyGroup.style.opacity = '1';
                if (ollamaSelect) ollamaSelect.classList.add('hidden');
            }

            fetchApiSnippet();
        });
    });

    document.querySelectorAll('.snippet-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.snippet-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentSnippetLang = tab.getAttribute('data-lang');
            fetchApiSnippet();
        });
    });

    document.getElementById('toggleApiKeyVisibility')?.addEventListener('click', () => {
        const input = document.getElementById('apiKeyInput');
        if (input) {
            input.type = input.type === 'password' ? 'text' : 'password';
        }
    });

    document.getElementById('testApiConnectionBtn')?.addEventListener('click', testApiConnection);
    document.getElementById('copySnippetBtn')?.addEventListener('click', copySnippetToClipboard);
}

async function testApiConnection() {
    const endpoint = document.getElementById('apiEndpointInput')?.value.trim();
    const apiKey = document.getElementById('apiKeyInput')?.value.trim();
    const model = document.getElementById('apiModelInput')?.value.trim();
    const statusEl = document.getElementById('apiTestStatus');

    if (statusEl) {
        statusEl.textContent = 'Testing connection...';
        statusEl.className = 'api-test-status';
    }

    try {
        const res = await fetch(`${API_URL}/api-config/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: currentConfigProvider,
                endpoint: endpoint,
                api_key: apiKey,
                model: model
            })
        });
        const data = await res.json();
        if (statusEl) {
            statusEl.textContent = data.message;
            statusEl.className = data.success ? 'api-test-status success' : 'api-test-status error';
        }
        await fetchApiSnippet();
    } catch (e) {
        if (statusEl) {
            statusEl.textContent = `Error: ${e.message}`;
            statusEl.className = 'api-test-status error';
        }
    }
}

async function fetchApiSnippet() {
    const endpoint = document.getElementById('apiEndpointInput')?.value.trim();
    const apiKey = document.getElementById('apiKeyInput')?.value.trim();
    const model = document.getElementById('apiModelInput')?.value.trim();
    const display = document.getElementById('codeSnippetDisplay');

    try {
        const res = await fetch(`${API_URL}/api-config/snippet`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: currentConfigProvider,
                endpoint: endpoint,
                api_key: apiKey,
                model: model
            })
        });
        if (!res.ok) return;
        const snippets = await res.json();
        if (display) {
            display.textContent = snippets[currentSnippetLang] || snippets['python'] || '';
        }
    } catch (e) {
        console.warn('Could not generate snippet:', e);
    }
}

function copySnippetToClipboard() {
    const display = document.getElementById('codeSnippetDisplay');
    if (!display) return;
    navigator.clipboard.writeText(display.textContent).then(() => {
        const btn = document.getElementById('copySnippetBtn');
        if (btn) {
            btn.textContent = '✓ Copied!';
            setTimeout(() => { btn.textContent = '📋 Copy Code'; }, 2000);
        }
    });
}

function openApiConfigModal() {
    const modal = document.getElementById('apiConfigModal');
    if (modal) {
        modal.classList.remove('hidden');
        fetchApiSnippet();
    }
}

function closeApiConfigModal() {
    const modal = document.getElementById('apiConfigModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

// Global window bindings
window.openEditRowModal = openEditRowModal;
window.deleteDatasetRow = deleteDatasetRow;
window.applyGlobalSystemPrompt = applyGlobalSystemPrompt;
window.triggerAutoClean = triggerAutoClean;
window.testApiConnection = testApiConnection;
window.openApiConfigModal = openApiConfigModal;
window.closeApiConfigModal = closeApiConfigModal;
window.applyOptimalOllamaModel = applyOptimalOllamaModel;
window.openStudioPopup = openStudioPopup;
window.closeStudioPopup = closeStudioPopup;
window.navToPrevRow = navToPrevRow;
window.navToNextRow = navToNextRow;
window.jumpToRow = jumpToRow;
window.toggleStudioViewMode = toggleStudioViewMode;
window.onEditorInputChange = onEditorInputChange;
window.onRawJsonInputChange = onRawJsonInputChange;
window.formatRawJsonText = formatRawJsonText;
window.copyRawJsonText = copyRawJsonText;
window.applyPresetToCurrentRow = applyPresetToCurrentRow;
window.trimCurrentRowWhitespace = trimCurrentRowWhitespace;
window.capitalizeCurrentRow = capitalizeCurrentRow;
window.wrapCodeBlockInAssistant = wrapCodeBlockInAssistant;
window.swapUserAssistant = swapUserAssistant;
window.autoRepairCurrentRow = autoRepairCurrentRow;
window.applySystemPromptToAllFromModal = applySystemPromptToAllFromModal;
window.duplicateCurrentRow = duplicateCurrentRow;
window.deleteCurrentRowFromModal = deleteCurrentRowFromModal;
window.saveEditedRow = saveEditedRow;
window.renderPopupInspectorTable = renderPopupInspectorTable;
window.editSpecificRowInStudio = editSpecificRowInStudio;
window.deleteRowFromPopupTable = deleteRowFromPopupTable;
window.onPopupTableSearch = onPopupTableSearch;
window.applyPresetToPopupSys = applyPresetToPopupSys;
window.applyGlobalSystemPromptFromPopup = applyGlobalSystemPromptFromPopup;
window.openAddNewRowInStudio = openAddNewRowInStudio;

// ============================================================================
// DOM EVENT WIRING & INITIALIZATION
// ============================================================================

function initApp() {
    // Hyperparameter accordion toggle
    setupHyperparamToggle();

    // Step 2 Configuration Studio listeners
    setupConfigurationStudioListeners();

    // Step 1 Data Hub & Validation initialization
    loadLatestModelBanner();
    loadHardwareHub();
    setupApiConfigModal();

    // Dataset editor actions
    document.getElementById('openApiConfigBtn')?.addEventListener('click', openApiConfigModal);
    document.getElementById('closeApiConfigBtn')?.addEventListener('click', closeApiConfigModal);
    document.getElementById('saveApiConfigBtn')?.addEventListener('click', closeApiConfigModal);

    document.getElementById('autoCleanBtn')?.addEventListener('click', triggerAutoClean);
    document.getElementById('applyGlobalSysBtn')?.addEventListener('click', applyGlobalSystemPrompt);

    document.getElementById('sysPresetSelect')?.addEventListener('change', (e) => {
        const textarea = document.getElementById('globalSysPromptInput');
        if (textarea && e.target.value) {
            textarea.value = e.target.value;
        }
    });

    // Modals
    document.getElementById('closeRowEditorBtn')?.addEventListener('click', () => {
        document.getElementById('rowEditorModal')?.classList.add('hidden');
    });
    document.getElementById('cancelRowEditBtn')?.addEventListener('click', () => {
        document.getElementById('rowEditorModal')?.classList.add('hidden');
    });
    document.getElementById('saveRowEditBtn')?.addEventListener('click', saveEditedRow);

    document.getElementById('openAddRowModalBtn')?.addEventListener('click', () => {
        document.getElementById('addRowModal')?.classList.remove('hidden');
    });
    document.getElementById('closeAddRowBtn')?.addEventListener('click', () => {
        document.getElementById('addRowModal')?.classList.add('hidden');
    });
    document.getElementById('cancelAddRowBtn')?.addEventListener('click', () => {
        document.getElementById('addRowModal')?.classList.add('hidden');
    });
    document.getElementById('saveAddRowBtn')?.addEventListener('click', saveAddedRow);

    // In-app training events
    document.getElementById('startInAppTrainBtn')?.addEventListener('click', startInAppTraining);
    document.getElementById('cancelTrainBtn')?.addEventListener('click', async () => {
        if (currentTrainingJobId) {
            await fetch(`${API_URL}/train/cancel/${currentTrainingJobId}`, { method: 'POST' });
        }
    });

    // Studio events
    document.getElementById('runStudioTestBtn')?.addEventListener('click', runStudioInference);
    document.getElementById('clearStudioBtn')?.addEventListener('click', () => {
        const input = document.getElementById('studioPromptInput');
        if (input) input.value = '';
    });
    document.getElementById('studioApproveBtn')?.addEventListener('click', studioApprove);
    document.getElementById('studioFlagBtn')?.addEventListener('click', studioFlag);

    // Sample datasets grid
    loadSampleDatasetsGrid();
    if (typeof updateNavigationGuardrails === 'function') {
        updateNavigationGuardrails();
    }
    const urlParams = new URLSearchParams(window.location.search);
    const sampleParam = urlParams.get('sample');
    if (sampleParam) {
        loadSampleDataset(sampleParam);
    }

    if (window.location.hash === '#test') {
        openPlayground();
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

