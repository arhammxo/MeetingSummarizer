document.addEventListener('DOMContentLoaded', () => {
    // Global variables to store state
    let currentJobId = null;
    let currentTranscript = null;
    let currentResult = null;
    let pollInterval = null;

    // Form submission handlers
    document.getElementById('audioForm').addEventListener('submit', handleAudioFormSubmit);
    document.getElementById('pasteTextForm').addEventListener('submit', handlePasteTextFormSubmit);
    document.getElementById('uploadTextForm').addEventListener('submit', handleUploadTextFormSubmit);
    
    // Button event listeners
    document.getElementById('summarizeAudioBtn').addEventListener('click', handleSummarizeAudioClick);
    document.getElementById('summarizeTextBtn').addEventListener('click', handleSummarizeTextClick);
    document.getElementById('downloadJsonBtn').addEventListener('click', handleDownloadJson);
    document.getElementById('downloadTextBtn').addEventListener('click', handleDownloadText);

    /**
     * Handle audio form submission
     */
    async function handleAudioFormSubmit(event) {
        event.preventDefault();
        
        const formData = new FormData(event.target);
        const audioFile = formData.get('file');
        
        if (!audioFile || audioFile.size === 0) {
            showAlert('Please select an audio file to upload', 'danger');
            return;
        }
        
        try {
            // Show processing status
            document.getElementById('audioProcessingStatus').classList.remove('d-none');
            updateAudioProgress(0, 'Starting audio processing...');
            
            // Submit the form
            const response = await fetch('/api/upload-audio', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${await response.text()}`);
            }
            
            const data = await response.json();
            currentJobId = data.job_id;
            
            // Start polling for status updates
            startAudioJobPolling(currentJobId);
            
        } catch (error) {
            console.error('Error uploading audio:', error);
            updateAudioProgress(0, `Error: ${error.message}`, true);
        }
    }
    
    /**
     * Poll for audio processing job status
     */
    function startAudioJobPolling(jobId) {
        clearInterval(pollInterval);
        
        pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/job/${jobId}`);
                
                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}`);
                }
                
                const data = await response.json();
                
                // Update progress
                updateAudioProgress(data.progress || 0, data.message || 'Processing...');
                
                // Check if job is complete
                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    
                    // Store transcript data
                    if (data.result && data.result.transcript) {
                        currentTranscript = data.result;
                        
                        // Show transcript preview
                        const previewText = document.getElementById('transcriptPreviewText');
                        const formattedTranscript = data.result.formatted_transcript.join('\n');
                        previewText.textContent = formattedTranscript.length > 1000 
                            ? formattedTranscript.substring(0, 1000) + '...' 
                            : formattedTranscript;
                        
                        document.getElementById('transcriptPreview').classList.remove('d-none');
                    }
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    updateAudioProgress(0, `Error: ${data.message}`, true);
                }
                
            } catch (error) {
                console.error('Error polling job status:', error);
                clearInterval(pollInterval);
                updateAudioProgress(0, `Error checking job status: ${error.message}`, true);
            }
        }, 1000);
    }
    
    /**
     * Update audio processing progress
     */
    function updateAudioProgress(progress, message, isError = false) {
        const progressBar = document.getElementById('audioProgressBar');
        const statusText = document.getElementById('audioStatusText');
        
        progressBar.style.width = `${progress}%`;
        statusText.textContent = message;
        
        if (isError) {
            progressBar.classList.remove('bg-primary');
            progressBar.classList.add('bg-danger');
        } else {
            progressBar.classList.remove('bg-danger');
            progressBar.classList.add('bg-primary');
        }
    }
    
    /**
     * Handle summarize audio button click
     */
    async function handleSummarizeAudioClick() {
        if (!currentTranscript) {
            showAlert('No transcript available to summarize', 'warning');
            return;
        }
        
        try {
            // Show processing status
            document.getElementById('summaryProcessingStatus').classList.remove('d-none');
            document.getElementById('resultsSection').classList.add('d-none');
            updateSummaryProgress(10, 'Starting summarization...');
            
            // Extract participants (speakers) from transcript
            const speakers = new Set();
            currentTranscript.transcript.forEach(segment => {
                speakers.add(`Speaker ${segment.speaker}`);
            });
            
            // Prepare request data
            const requestData = {
                transcript: currentTranscript.formatted_transcript.join('\n'),
                participants: Array.from(speakers),
                language: currentTranscript.language,
                is_long_recording: document.getElementById('isLongRecording').checked
            };
            
            // Submit the request
            const response = await fetch('/api/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${await response.text()}`);
            }
            
            const data = await response.json();
            currentJobId = data.job_id;
            
            // Start polling for status updates
            startSummaryJobPolling(currentJobId);
            
        } catch (error) {
            console.error('Error starting summarization:', error);
            updateSummaryProgress(0, `Error: ${error.message}`, true);
        }
    }
    
    /**
     * Handle paste text form submission
     */
    async function handlePasteTextFormSubmit(event) {
        event.preventDefault();
        
        const formData = new FormData(event.target);
        const transcript = formData.get('transcript');
        
        if (!transcript) {
            showAlert('Please enter a transcript', 'danger');
            return;
        }
        
        try {
            // First detect participants if not provided
            let participants = formData.get('participants');
            
            if (!participants) {
                const participantsResponse = await fetch('/api/extract-participants', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ transcript })
                });
                
                if (!participantsResponse.ok) {
                    throw new Error(`Server responded with ${participantsResponse.status}`);
                }
                
                participants = await participantsResponse.json();
            } else {
                participants = participants.split(',').map(p => p.trim()).filter(p => p);
            }
            
            // Show processing status
            document.getElementById('summaryProcessingStatus').classList.remove('d-none');
            document.getElementById('resultsSection').classList.add('d-none');
            updateSummaryProgress(10, 'Starting summarization...');
            
            // Prepare request data
            const requestData = {
                transcript: transcript,
                participants: participants,
                language: formData.get('language') || null,
                is_long_recording: false
            };
            
            // Store transcript for download
            currentTranscript = {
                formatted_transcript: transcript.split('\n')
            };
            
            // Submit the request
            const response = await fetch('/api/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${await response.text()}`);
            }
            
            const data = await response.json();
            currentJobId = data.job_id;
            
            // Start polling for status updates
            startSummaryJobPolling(currentJobId);
            
        } catch (error) {
            console.error('Error processing text:', error);
            updateSummaryProgress(0, `Error: ${error.message}`, true);
        }
    }
    
    /**
     * Handle upload text form submission
     */
    async function handleUploadTextFormSubmit(event) {
        event.preventDefault();
        
        const formData = new FormData(event.target);
        const textFile = formData.get('file');
        
        if (!textFile || textFile.size === 0) {
            showAlert('Please select a text file to upload', 'danger');
            return;
        }
        
        try {
            // Upload the text file
            const uploadResponse = await fetch('/api/upload-text', {
                method: 'POST',
                body: formData
            });
            
            if (!uploadResponse.ok) {
                throw new Error(`Server responded with ${uploadResponse.status}: ${await uploadResponse.text()}`);
            }
            
            const uploadData = await uploadResponse.json();
            const transcript = uploadData.transcript;
            
            // Display preview
            const previewText = document.getElementById('textPreviewContent');
            previewText.textContent = transcript.length > 1000 
                ? transcript.substring(0, 1000) + '...' 
                : transcript;
            
            document.getElementById('textPreview').classList.remove('d-none');
            
            // Show detected participants
            const participantsEl = document.getElementById('detectedParticipants');
            if (uploadData.participants && uploadData.participants.length > 0) {
                participantsEl.textContent = `✅ Detected participants: ${uploadData.participants.join(', ')}`;
            } else {
                participantsEl.textContent = '';
            }
            
            // Store transcript for later use
            currentTranscript = {
                formatted_transcript: transcript.split('\n')
            };
            
            // Store participants
            document.getElementById('uploadParticipants').value = uploadData.participants.join(', ');
            
        } catch (error) {
            console.error('Error uploading text file:', error);
            showAlert(`Error uploading text file: ${error.message}`, 'danger');
        }
    }
    
    /**
     * Handle summarize text button click
     */
    async function handleSummarizeTextClick() {
        try {
            // Get participants
            let participants = document.getElementById('uploadParticipants').value;
            
            if (!participants) {
                showAlert('Please enter participants', 'warning');
                return;
            }
            
            participants = participants.split(',').map(p => p.trim()).filter(p => p);
            
            // Show processing status
            document.getElementById('summaryProcessingStatus').classList.remove('d-none');
            document.getElementById('resultsSection').classList.add('d-none');
            updateSummaryProgress(10, 'Starting summarization...');
            
            // Get the transcript
            const transcript = currentTranscript.formatted_transcript.join('\n');
            
            // Prepare request data
            const requestData = {
                transcript: transcript,
                participants: participants,
                language: document.getElementById('uploadLanguage').value || null,
                is_long_recording: false
            };
            
            // Submit the request
            const response = await fetch('/api/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${await response.text()}`);
            }
            
            const data = await response.json();
            currentJobId = data.job_id;
            
            // Start polling for status updates
            startSummaryJobPolling(currentJobId);
            
        } catch (error) {
            console.error('Error starting summarization:', error);
            updateSummaryProgress(0, `Error: ${error.message}`, true);
        }
    }
    
    /**
     * Poll for summary job status
     */
    function startSummaryJobPolling(jobId) {
        clearInterval(pollInterval);
        
        pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/job/${jobId}`);
                
                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}`);
                }
                
                const data = await response.json();
                
                // Update progress
                updateSummaryProgress(data.progress || 0, data.message || 'Processing...');
                
                // Check if job is complete
                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    
                    // Display results
                    if (data.result) {
                        currentResult = data.result;
                        displayResults(data.result);
                    }
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    updateSummaryProgress(0, `Error: ${data.message}`, true);
                }
                
            } catch (error) {
                console.error('Error polling job status:', error);
                clearInterval(pollInterval);
                updateSummaryProgress(0, `Error checking job status: ${error.message}`, true);
            }
        }, 1000);
    }
    
    /**
     * Update summary processing progress
     */
    function updateSummaryProgress(progress, message, isError = false) {
        const progressBar = document.getElementById('summaryProgressBar');
        const statusText = document.getElementById('summaryStatusText');
        
        progressBar.style.width = `${progress}%`;
        statusText.textContent = message;
        
        if (isError) {
            progressBar.classList.remove('bg-primary');
            progressBar.classList.add('bg-danger');
        } else {
            progressBar.classList.remove('bg-danger');
            progressBar.classList.add('bg-primary');
        }
    }
    
    /**
     * Display results in the UI
     */
    function displayResults(result) {
        // Hide processing status and show results section
        document.getElementById('summaryProcessingStatus').classList.add('d-none');
        document.getElementById('resultsSection').classList.remove('d-none');
        
        // Meeting summary
        const meetingSummary = document.getElementById('meetingSummary');
        meetingSummary.textContent = result.meeting_summary.summary;
        
        // Key points
        const keyPoints = document.getElementById('keyPoints');
        keyPoints.innerHTML = '';
        result.meeting_summary.key_points.forEach(point => {
            const li = document.createElement('li');
            li.textContent = point;
            keyPoints.appendChild(li);
        });
        
        // Decisions
        const decisions = document.getElementById('decisions');
        decisions.innerHTML = '';
        result.meeting_summary.decisions.forEach(decision => {
            const li = document.createElement('li');
            li.textContent = decision;
            decisions.appendChild(li);
        });
        
        // Action items
        const actionItems = document.getElementById('actionItems');
        actionItems.innerHTML = '';
        result.action_items.forEach((item, index) => {
            const actionId = `action-${index}`;
            
            // Determine priority color
            let priorityBadge;
            switch(item.priority.toLowerCase()) {
                case 'high':
                    priorityBadge = '<span class="badge bg-danger ms-2">High</span>';
                    break;
                case 'medium':
                    priorityBadge = '<span class="badge bg-warning text-dark ms-2">Medium</span>';
                    break;
                default:
                    priorityBadge = '<span class="badge bg-success ms-2">Low</span>';
            }
            
            const actionItem = `
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading-${actionId}">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-${actionId}" aria-expanded="false" aria-controls="collapse-${actionId}">
                            <strong>${item.action}</strong>${priorityBadge}
                        </button>
                    </h2>
                    <div id="collapse-${actionId}" class="accordion-collapse collapse" aria-labelledby="heading-${actionId}" data-bs-parent="#actionItems">
                        <div class="accordion-body">
                            <p><strong>Assignee:</strong> ${item.assignee}</p>
                            <p><strong>Due Date:</strong> ${item.due_date}</p>
                        </div>
                    </div>
                </div>
            `;
            actionItems.innerHTML += actionItem;
        });
        
        // Metadata
        const metadataTable = document.getElementById('metadataTable');
        metadataTable.innerHTML = '';
        
        if (result.metadata) {
            const metadata = result.metadata;
            
            // Add rows to the table
            addMetadataRow(metadataTable, 'Language', metadata.language || 'Auto-detected');
            
            if (metadata.total_duration_minutes) {
                addMetadataRow(metadataTable, 'Duration', `${metadata.total_duration_minutes} minutes`);
            }
            
            if (metadata.participant_count) {
                addMetadataRow(metadataTable, 'Participants', metadata.participant_count);
            }
            
            if (metadata.chunks_analyzed) {
                addMetadataRow(metadataTable, 'Chunks Analyzed', metadata.chunks_analyzed);
            }
        }
        
        // Speaker summaries
        const speakerSummaries = document.getElementById('speakerSummaries');
        speakerSummaries.innerHTML = '';
        
        if (result.speaker_summaries) {
            Object.entries(result.speaker_summaries).forEach(([speaker, summary], index) => {
                const speakerId = `speaker-${index}`;
                
                let contributionsList = '';
                if (summary.key_contributions && summary.key_contributions.length > 0) {
                    contributionsList = '<h6>Key Contributions:</h6><ul>' + 
                        summary.key_contributions.map(c => `<li>${c}</li>`).join('') + 
                        '</ul>';
                }
                
                let actionsList = '';
                if (summary.action_items && summary.action_items.length > 0) {
                    actionsList = '<h6>Action Items:</h6><ul>' + 
                        summary.action_items.map(a => `<li>${a}</li>`).join('') + 
                        '</ul>';
                }
                
                let questionsList = '';
                if (summary.questions_raised && summary.questions_raised.length > 0) {
                    questionsList = '<h6>Questions Raised:</h6><ul>' + 
                        summary.questions_raised.map(q => `<li>${q}</li>`).join('') + 
                        '</ul>';
                }
                
                const speakerItem = `
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="heading-${speakerId}">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-${speakerId}" aria-expanded="false" aria-controls="collapse-${speakerId}">
                                <strong>${speaker}</strong>
                            </button>
                        </h2>
                        <div id="collapse-${speakerId}" class="accordion-collapse collapse" aria-labelledby="heading-${speakerId}" data-bs-parent="#speakerSummaries">
                            <div class="accordion-body">
                                <p><strong>Summary:</strong> ${summary.brief_summary}</p>
                                ${contributionsList}
                                ${actionsList}
                                ${questionsList}
                            </div>
                        </div>
                    </div>
                `;
                speakerSummaries.innerHTML += speakerItem;
            });
        }
    }
    
    /**
     * Add a row to the metadata table
     */
    function addMetadataRow(table, label, value) {
        const row = table.insertRow();
        const labelCell = row.insertCell(0);
        const valueCell = row.insertCell(1);
        
        labelCell.innerHTML = `<strong>${label}:</strong>`;
        valueCell.textContent = value;
    }
    
    /**
     * Handle download JSON button click
     */
    function handleDownloadJson() {
        if (!currentResult) {
            showAlert('No results available to download', 'warning');
            return;
        }
        
        // Create a Blob with the JSON data
        const jsonBlob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
        
        // Create a download link
        const url = URL.createObjectURL(jsonBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'meeting_summary.json';
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 0);
    }
    
    /**
     * Handle download text button click
     */
    function handleDownloadText() {
        if (!currentTranscript || !currentTranscript.formatted_transcript) {
            showAlert('No transcript available to download', 'warning');
            return;
        }
        
        // Create a Blob with the text data
        const textBlob = new Blob([currentTranscript.formatted_transcript.join('\n')], { type: 'text/plain' });
        
        // Create a download link
        const url = URL.createObjectURL(textBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'meeting_transcript.txt';
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 0);
    }
    
    /**
     * Show an alert message
     */
    function showAlert(message, type = 'info') {
        // Create alert element
        const alertEl = document.createElement('div');
        alertEl.className = `alert alert-${type} alert-dismissible fade show`;
        alertEl.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert at the top of the container
        const container = document.querySelector('.container');
        container.insertBefore(alertEl, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertEl.parentNode) {
                alertEl.classList.remove('show');
                setTimeout(() => alertEl.remove(), 150);
            }
        }, 5000);
    }
});