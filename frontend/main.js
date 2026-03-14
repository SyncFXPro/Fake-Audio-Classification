var API_BASE_URL = 'http://localhost:5000';

console.log('API Base URL:', API_BASE_URL);
console.log('Testing connection to backend...');

var uploadModeBtn = document.getElementById('uploadModeBtn');
var recordModeBtn = document.getElementById('recordModeBtn');
var uploadMode = document.getElementById('uploadMode');
var recordMode = document.getElementById('recordMode');
var dropZone = document.getElementById('dropZone');
var fileInput = document.getElementById('fileInput');
var browseBtn = document.getElementById('browseBtn');
var recordBtn = document.getElementById('recordBtn');
var loadingIndicator = document.getElementById('loadingIndicator');
var resultContainer = document.getElementById('resultContainer');
var analyzeAgainBtn = document.getElementById('analyzeAgainBtn');

var isRecording = false;
var mediaRecorder = null;
var audioChunks = [];

uploadModeBtn.addEventListener('click', function() {
    uploadModeBtn.classList.add('active');
    recordModeBtn.classList.remove('active');
    uploadMode.classList.add('active');
    recordMode.classList.remove('active');
    hideResults();
});

recordModeBtn.addEventListener('click', function() {
    recordModeBtn.classList.add('active');
    uploadModeBtn.classList.remove('active');
    recordMode.classList.add('active');
    uploadMode.classList.remove('active');
    hideResults();
});

dropZone.addEventListener('click', function() {
    fileInput.click();
});

browseBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    fileInput.click();
});

dropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', function() {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    var files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

fileInput.addEventListener('change', function(e) {
    var file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
});

recordBtn.addEventListener('click', function() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

analyzeAgainBtn.addEventListener('click', function() {
    hideResults();
    fileInput.value = '';
});

function handleFileUpload(file) {
    var validTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp3'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|ogg)$/i)) {
        alert('Please upload a valid audio file (MP3, WAV, or OGG)');
        return;
    }
    
    showLoading();
    
    var formData = new FormData();
    formData.append('file', file);
    
    console.log('Uploading to:', API_BASE_URL + '/api/predict/upload');
    
    fetch(API_BASE_URL + '/api/predict/upload', {
        method: 'POST',
        body: formData,
        mode: 'cors'
    })
    .then(function(response) {
        console.log('Response status:', response.status);
        if (!response.ok) {
            throw new Error('Server returned ' + response.status);
        }
        return response.json();
    })
    .then(function(data) {
        console.log('Response data:', data);
        hideLoading();
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            displayResults(data);
            console.log(data.prediction);
        }
    })
    .catch(function(error) {
        hideLoading();
        console.error('Fetch error:', error);
        alert('Error connecting to server. Make sure the backend is running on http://localhost:5000\n\nError: ' + error.message);
    });
}

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            var options = { mimeType: 'audio/webm' };
            try {
                mediaRecorder = new MediaRecorder(stream, options);
            } catch (e) {
                mediaRecorder = new MediaRecorder(stream);
            }
            audioChunks = [];
            
            mediaRecorder.addEventListener('dataavailable', function(event) {
                audioChunks.push(event.data);
            });
            
            mediaRecorder.addEventListener('stop', function() {
                var audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                handleRecordedAudio(audioBlob);
                
                stream.getTracks().forEach(function(track) {
                    track.stop();
                });
            });
            
            mediaRecorder.start();
            isRecording = true;
            recordBtn.classList.add('recording');
            recordBtn.querySelector('.record-text').textContent = 'Stop Recording';
            
            visualizeAudio(stream);
        })
        .catch(function(error) {
            alert('Error accessing microphone: ' + error.message);
            console.error('Error:', error);
        });
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.record-text').textContent = 'Start Recording';
    }
}

function visualizeAudio(stream) {
    var canvas = document.getElementById('waveformCanvas');
    var canvasContext = canvas.getContext('2d');
    var audioContext = new AudioContext();
    var analyser = audioContext.createAnalyser();
    var source = audioContext.createMediaStreamSource(stream);
    
    source.connect(analyser);
    analyser.fftSize = 2048;
    
    var bufferLength = analyser.frequencyBinCount;
    var dataArray = new Uint8Array(bufferLength);
    
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    function draw() {
        if (!isRecording) {
            canvasContext.clearRect(0, 0, canvas.width, canvas.height);
            return;
        }
        
        requestAnimationFrame(draw);
        
        analyser.getByteTimeDomainData(dataArray);
        
        canvasContext.fillStyle = '#f8f9fa';
        canvasContext.fillRect(0, 0, canvas.width, canvas.height);
        
        canvasContext.lineWidth = 2;
        canvasContext.strokeStyle = '#667eea';
        canvasContext.beginPath();
        
        var sliceWidth = canvas.width / bufferLength;
        var x = 0;
        
        for (var i = 0; i < bufferLength; i++) {
            var v = dataArray[i] / 128.0;
            var y = v * canvas.height / 2;
            
            if (i === 0) {
                canvasContext.moveTo(x, y);
            } else {
                canvasContext.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        canvasContext.lineTo(canvas.width, canvas.height / 2);
        canvasContext.stroke();
    }
    
    draw();
}

function handleRecordedAudio(audioBlob) {
    showLoading();
    
    var reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = function() {
        var base64Audio = reader.result;
        
        console.log('Sending recorded audio to:', API_BASE_URL + '/api/predict/realtime');
        
        fetch(API_BASE_URL + '/api/predict/realtime', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ audio: base64Audio }),
            mode: 'cors'
        })
        .then(function(response) {
            console.log('Response status:', response.status);
            if (!response.ok) {
                throw new Error('Server returned ' + response.status);
            }
            return response.json();
        })
        .then(function(data) {
            console.log('Response data:', data);
            hideLoading();
            if (data.error) {
                alert('Error: ' + data.error);
            } else {
                displayResults(data);
            }
        })
        .catch(function(error) {
            hideLoading();
            console.error('Fetch error:', error);
            alert('Error connecting to server. Make sure the backend is running on http://localhost:5000\n\nError: ' + error.message);
        });
    };
}

function showLoading() {
    uploadMode.classList.remove('active');
    recordMode.classList.remove('active');
    resultContainer.classList.add('hidden');
    loadingIndicator.classList.remove('hidden');
}

function hideLoading() {
    loadingIndicator.classList.add('hidden');
}

function hideResults() {
    resultContainer.classList.add('hidden');
    if (uploadModeBtn.classList.contains('active')) {
        uploadMode.classList.add('active');
    } else {
        recordMode.classList.add('active');
    }
}

function displayResults(data) {
    var fakenessScore = data.fakeness_score;
    var prediction = data.prediction;
    var confidence = data.confidence;
    
    var resultBadge = document.getElementById('resultBadge');
    resultBadge.textContent = prediction.toUpperCase();
    resultBadge.className = 'result-badge ' + prediction;
    
    var gaugeText = document.getElementById('gaugeText');
    gaugeText.textContent = Math.round(fakenessScore * 100) + '%';
    
    var gaugeFill = document.getElementById('gaugeFill');
    var pathLength = 251.2;
    var fillLength = pathLength * fakenessScore;
    gaugeFill.style.strokeDasharray = fillLength + ' ' + pathLength;
    
    if (fakenessScore < 0.3) {
        gaugeFill.style.stroke = '#27ae60';
    } else if (fakenessScore < 0.7) {
        gaugeFill.style.stroke = '#f39c12';
    } else {
        gaugeFill.style.stroke = '#e74c3c';
    }
    
    var predictionText = document.getElementById('predictionText');
    predictionText.textContent = prediction.toUpperCase();
    predictionText.style.color = prediction === 'real' ? '#27ae60' : '#e74c3c';
    
    var confidenceText = document.getElementById('confidenceText');
    confidenceText.textContent = Math.round(confidence * 100) + '%';
    
    resultContainer.classList.remove('hidden');
}

var RESULTS_IMAGE_NAMES = [
    'training_history',
    'confusion_matrix',
    'roc_curve',
    'probability_distribution',
    'precision_recall_curve'
];

function getResultsImageUrl(graphName) {
    return API_BASE_URL + '/api/graph/' + encodeURIComponent(graphName);
}

function buildResultsGallery() {
    var track = document.getElementById('resultsGalleryTrack');
    if (!track) return;
    track.innerHTML = '';
    var i, name;
    for (i = 0; i < RESULTS_IMAGE_NAMES.length; i++) {
        name = RESULTS_IMAGE_NAMES[i];
        track.appendChild(createGalleryItem(name));
    }
    for (i = 0; i < RESULTS_IMAGE_NAMES.length; i++) {
        name = RESULTS_IMAGE_NAMES[i];
        track.appendChild(createGalleryItem(name));
    }
}

function createGalleryItem(graphName) {
    var div = document.createElement('div');
    div.className = 'results-gallery-item';
    var img = document.createElement('img');
    img.src = getResultsImageUrl(graphName);
    img.alt = graphName.replace(/_/g, ' ');
    img.loading = 'lazy';
    img.onerror = function() {
        div.style.display = 'none';
    };
    div.appendChild(img);
    return div;
}

buildResultsGallery();

console.log('Deepfake Audio Detection loaded');
