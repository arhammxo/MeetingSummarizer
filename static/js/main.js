document.addEventListener('DOMContentLoaded', () => {
    // Global variables to store state
    let currentJobId = null;
    let currentTranscript = null;
    let currentResult = null;
    let pollInterval = null;

    // Add global context section function
    window.addMeetingContextSection = function() {
        console.log("ADDING MEETING CONTEXT SECTION - Starting...");
        
        // Remove any existing context section to avoid duplicates
        const existingSection = document.getElementById('meetingContextSection');
        if (existingSection) {
            existingSection.remove();
            console.log("Removed existing context section");
        }
        
        // Create a very visible context input section
        const contextSection = document.createElement('div');
        contextSection.id = 'meetingContextSection';
        contextSection.className = 'card mt-3 mb-3';
        contextSection.style.border = '2px solid #007bff';  // Blue border to make it stand out
        contextSection.style.boxShadow = '0 0 10px rgba(0,123,255,0.5)';  // Add glow effect
        
        contextSection.innerHTML = `
            <div class="card-header bg-primary text-white">
                <h4>Additional Meeting Context (Optional)</h4>
                <p class="mb-0">Provide context to improve summary quality</p>
            </div>
            <div class="card-body">
                <textarea id="meetingContext" class="form-control" rows="4" 
                    placeholder="Example: This is a weekly team meeting for the marketing department. The main goals were to review Q2 campaign results and plan for Q3."></textarea>
                <div class="form-text text-muted mt-2">
                    Adding context helps our AI create a more accurate and relevant summary.
                </div>
            </div>
        `;
        
        console.log("Context section created, now adding to page...");
        
        // Try multiple insertion methods to ensure it appears
        
        // Method 1: Insert before summarize button
        const summarizeBtn = document.getElementById('summarizeAudioBtn');
        if (summarizeBtn && summarizeBtn.parentNode) {
            summarizeBtn.parentNode.insertBefore(contextSection, summarizeBtn);
            console.log("SUCCESS: Added context before summarize button");
            return true;
        }
        
        // Method 2: Insert after transcript preview
        const transcriptPreview = document.getElementById('transcriptPreview');
        if (transcriptPreview) {
            transcriptPreview.after(contextSection);
            console.log("SUCCESS: Added context after transcript preview");
            return true;
        }
        
        // Method 3: Insert at end of audio form
        const audioForm = document.getElementById('audioForm');
        if (audioForm) {
            audioForm.appendChild(contextSection);
            console.log("SUCCESS: Added context at end of audio form");
            return true;
        }
        
        // Method 4: Last resort - add to main container
        const container = document.querySelector('.container');
        if (container) {
            container.appendChild(contextSection);
            console.log("SUCCESS: Added context to main container");
            return true;
        }
        
        console.error("FAILED: Could not find any suitable location to add context section");
        return false;
    };

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
                console.log(`Polling job status for job ID: ${jobId}`);
                const response = await fetch(`/api/job/${jobId}`);
                
                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}`);
                }
                
                const data = await response.json();
                console.log("Received job status data:", data);
                
                // Add debug logging for confidence metrics
                if (data.result) {
                    console.log("Result data received:", data.result);
                    if (data.result.confidence_metrics) {
                        console.log("Confidence metrics found:", data.result.confidence_metrics);
                    } else {
                        console.log("No confidence metrics in result data");
                    }
                }
                
                // Update progress
                updateAudioProgress(data.progress || 0, data.message || 'Processing...');
                
                // Check if job is complete
                if (data.status === 'completed') {
                    console.log("Job completed successfully");
                    clearInterval(pollInterval);
                    
                    // Store transcript data
                    if (data.result && data.result.transcript) {
                        console.log("Transcript data received");
                        currentTranscript = data.result;
                        
                        // Show detected language in UI
                        if (data.result.language) {
                            console.log("Detected language:", data.result.language);
                            const detectedLang = data.result.language;
                            const languageDisplay = getLanguageDisplayName(detectedLang);
                            
                            // Update message with clear language detection info
                            let message = `Processing complete. `;
                            
                            // If language was auto-detected, make it clearer
                            const selectedLanguage = document.getElementById('audioLanguage').value;
                            if (selectedLanguage === 'auto') {
                                message += `Language automatically detected as: ${languageDisplay} (${detectedLang})`;
                            } else {
                                message += `Using selected language: ${languageDisplay}`;
                            }
                            
                            // Add confidence information if available
                            if (data.result.confidence_metrics && data.result.confidence_metrics.average) {
                                const avgConfidence = data.result.confidence_metrics.average;
                                message += `<br>Average transcription confidence: <strong>${avgConfidence}%</strong>`;
                                
                                // Add warning for low confidence
                                if (data.result.confidence_metrics.low_confidence_percentage > 20) {
                                    message += `<br><span class="text-warning">⚠️ ${data.result.confidence_metrics.low_confidence_percentage}% of segments have low confidence</span>`;
                                }
                            }
                            
                            updateAudioProgress(100, message);
                            
                            // Add confidence stats to the UI only once
                            if (data.result.confidence_metrics) {
                                console.log("Creating confidence notice box with metrics:", data.result.confidence_metrics);
                                
                                // Remove existing notice if present
                                const existingNotice = document.getElementById('confidenceNotice');
                                if (existingNotice) {
                                    console.log("Removing existing confidence notice");
                                    existingNotice.remove();
                                }
                                
                                const notice = document.createElement('div');
                                notice.id = 'confidenceNotice';
                                notice.className = 'alert alert-info mt-2';
                                
                                // Log the metrics being used
                                const metrics = data.result.confidence_metrics;
                                console.log("Using confidence metrics:", {
                                    average: metrics.average,
                                    min: metrics.min,
                                    max: metrics.max
                                });
                                
                                notice.innerHTML = `
                                    <h5>Transcription Confidence</h5>
                                    <div class="row">
                                        <div class="col-md-6">
                                            <p><strong>Average:</strong> ${metrics.average}%</p>
                                            <p><strong>Range:</strong> ${metrics.min}% - ${metrics.max}%</p>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="progress mb-2" style="height: 20px;">
                                                <div class="progress-bar bg-success" role="progressbar" 
                                                    style="width: ${metrics.average}%" 
                                                    aria-valuenow="${metrics.average}" 
                                                    aria-valuemin="0" aria-valuemax="100">
                                                    ${metrics.average}%
                                                </div>
                                            </div>
                                            <small class="text-muted">
                                                <span class="badge bg-success me-1">✓</span> High confidence (90%+)<br>
                                                <span class="badge bg-warning text-dark me-1">~</span> Medium confidence (70-89%)<br>
                                                <span class="badge bg-danger me-1">?</span> Low confidence (<70%)
                                            </small>
                                        </div>
                                    </div>
                                `;
                                
                                const transcriptPreview = document.getElementById('transcriptPreview');
                                if (transcriptPreview) {
                                    console.log("Appending confidence notice to transcript preview");
                                    transcriptPreview.appendChild(notice);
                                } else {
                                    console.error("Could not find transcriptPreview element");
                                }
                            } else {
                                console.log("No confidence metrics available in result data");
                            }
                        }
                        
                        // Show transcript preview with confidence indicators
                        const previewText = document.getElementById('transcriptPreviewText');
                        const formattedTranscript = data.result.formatted_transcript;
                        
                        // Convert to HTML with color-coding by confidence
                        const htmlContent = formattedTranscript.map(line => {
                            // Color-code based on confidence indicators
                            if (line.includes("✓ Speaker")) {
                                return `<div class="text-success">${line}</div>`;
                            } else if (line.includes("~ Speaker")) {
                                return `<div class="text-warning">${line}</div>`;
                            } else if (line.includes("? Speaker")) {
                                return `<div class="text-danger">${line}</div>`;
                            } else {
                                return `<div>${line}</div>`;
                            }
                        }).join('');
                        
                        // Use innerHTML to render the HTML formatting
                        previewText.innerHTML = htmlContent.length > 10000 
                            ? htmlContent.substring(0, 10000) + '...' 
                            : htmlContent;
                        
                        document.getElementById('transcriptPreview').classList.remove('d-none');
                        
                        // Add context section after showing transcript
                        console.log("Transcript preview displayed, adding context section...");
                        setTimeout(window.addMeetingContextSection, 100);
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
    
    function addConfidenceIndicators(text, confidence) {
        if (!confidence) return text;
        
        let indicator = "";
        let badgeClass = "";
        
        if (confidence >= 90) {
            indicator = "✓";
            badgeClass = "bg-success";
        } else if (confidence >= 70) {
            indicator = "~";
            badgeClass = "bg-warning text-dark";
        } else {
            indicator = "?";
            badgeClass = "bg-danger";
        }
        
        return `<span class="badge ${badgeClass} me-1" title="${confidence}% confidence">${indicator}</span> ${text}`;
    }

    /**
     * Update audio processing progress
     */
    function updateAudioProgress(progress, message, isError = false) {
        const progressBar = document.getElementById('audioProgressBar');
        const statusText = document.getElementById('audioStatusText');
        
        progressBar.style.width = `${progress}%`;
        statusText.innerHTML = message;
        
        if (isError) {
            progressBar.classList.remove('bg-primary');
            progressBar.classList.add('bg-danger');
        } else {
            progressBar.classList.remove('bg-danger');
            progressBar.classList.add('bg-primary');
        }

        // Add context input after transcript is shown
        if (progress === 100 && !isError) {
            setTimeout(addContextInput, 500); // Short delay to ensure DOM is updated
        }
    }
    
    /**
     * Add context input section to the UI
     */
    function addContextInput() {
        // Check if we already added the context input
        if (document.getElementById('meetingContextSection')) {
            return;
        }
        
        // Create the context input section
        const contextSection = document.createElement('div');
        contextSection.id = 'meetingContextSection';
        contextSection.className = 'card mt-3 mb-3';
        contextSection.innerHTML = `
            <div class="card-header">
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="enableContextInput">
                    <label class="form-check-label" for="enableContextInput">
                        <strong>Add Meeting Context</strong> (Optional)
                    </label>
                </div>
            </div>
            <div class="card-body" id="contextInputBody" style="display: none;">
                <p class="text-muted">Provide additional context about this meeting to improve summary quality:</p>
                <textarea id="meetingContext" class="form-control" rows="3" 
                    placeholder="Example: This is a weekly team meeting for the marketing department. The main goals were to review Q2 campaign results and plan for Q3."></textarea>
            </div>
        `;
        
        // Insert before the summarize button
        const audioProcessingStatus = document.getElementById('audioProcessingStatus');
        audioProcessingStatus.parentNode.insertBefore(contextSection, 
            document.getElementById('summarizeAudioBtn').parentNode);
        
        // Add toggle functionality
        document.getElementById('enableContextInput').addEventListener('change', function() {
            document.getElementById('contextInputBody').style.display = 
                this.checked ? 'block' : 'none';
        });
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
            
            // Get additional context if provided
            let additionalContext = null;
            const contextInput = document.getElementById('meetingContext');
            
            if (contextInput) {
                additionalContext = contextInput.value.trim();
                console.log("Including additional context:", additionalContext);
            }
            
            // Prepare request data
            const requestData = {
                transcript: currentTranscript.formatted_transcript.join('\n'),
                participants: Array.from(speakers),
                language: currentTranscript.language,
                is_long_recording: document.getElementById('isLongRecording').checked,
                additional_context: additionalContext  // Add the context to the request
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
        
        // Add confidence information to metadata if available
        if (currentTranscript && currentTranscript.confidence_metrics) {
            const metrics = currentTranscript.confidence_metrics;
            const confidenceData = document.createElement('div');
            confidenceData.className = 'small text-muted mt-2';
            confidenceData.innerHTML = `
                <strong>Transcription Confidence:</strong> ${metrics.average}% average
                ${metrics.low_confidence_percentage > 10 ? 
                `<span class="text-warning ms-2">⚠️ ${metrics.low_confidence_percentage}% low confidence segments</span>` : ''}
            `;
            meetingSummary.appendChild(confidenceData);
        }

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
            if (metadata.language_name) {
                // Show the full language name rather than just the code
                addMetadataRow(metadataTable, 'Language', metadata.language_name);
            } else {
                addMetadataRow(metadataTable, 'Language', metadata.language || 'Auto-detected');
            }
            
            // Add confidence metrics if available
            if (currentTranscript && currentTranscript.confidence_metrics) {
                const metrics = currentTranscript.confidence_metrics;
                addMetadataRow(metadataTable, 'Transcription Confidence', 
                              `${metrics.average}% (range: ${metrics.min}%-${metrics.max}%)`);
                              
                // Only show low confidence warning if significant
                if (metrics.low_confidence_percentage > 10) {
                    addMetadataRow(metadataTable, 'Low Confidence Segments', 
                                  `${metrics.low_confidence_count} segments (${metrics.low_confidence_percentage}%)`);
                }
            }
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
                
                // Add confidence information if available
                let confidenceDisplay = '';
                if (currentTranscript) {
                    // First try to use backend-calculated metrics if available
                    if (currentTranscript.speaker_confidence_metrics && 
                        currentTranscript.speaker_confidence_metrics[speaker.replace("Speaker ", "")]) {
                        
                        const speakerMetrics = currentTranscript.speaker_confidence_metrics[speaker.replace("Speaker ", "")];
                        const avgConfidence = speakerMetrics.average_confidence;
                        const confidenceClass = avgConfidence >= 90 ? 'text-success' : 
                                              (avgConfidence >= 70 ? 'text-warning' : 'text-danger');
                        
                        confidenceDisplay = `<span class="${confidenceClass} ms-2">(${avgConfidence.toFixed(1)}% confidence)</span>`;
                    }
                    // Fall back to calculation on the frontend if backend metrics aren't available
                    else {
                        // Find confidence scores for this speaker
                        const speakerSegments = currentTranscript.transcript.filter(s => `Speaker ${s.speaker}` === speaker);
                        if (speakerSegments.length > 0) {
                            // Calculate average confidence for this speaker
                            const confidences = speakerSegments
                                .filter(s => s.confidence !== undefined && s.confidence !== null)
                                .map(s => s.confidence);
                            
                            if (confidences.length > 0) {
                                const avgConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;
                                const confidenceClass = avgConfidence >= 90 ? 'text-success' : 
                                                      (avgConfidence >= 70 ? 'text-warning' : 'text-danger');
                                
                                confidenceDisplay = `<span class="${confidenceClass} ms-2">(${avgConfidence.toFixed(1)}% confidence)</span>`;
                            }
                        }
                    }
                }
                
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
                                <strong>${speaker}</strong>${confidenceDisplay}
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

    // Add this function somewhere in your main.js
    function getLanguageDisplayName(languageCode) {
        const languageMap = {
            "en": "English",
            "hi": "Hindi", 
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ja": "Japanese",
            "ru": "Russian",
            "ar": "Arabic",
            "auto": "Auto-detected"
        };
        
        return languageMap[languageCode] || languageCode;
    }
});