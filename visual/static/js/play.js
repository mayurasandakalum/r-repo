// Sparkle animation
const card = document.querySelector('.card');
for (let i = 0; i < 10; i++) {
const sparkle = document.createElement('div');
sparkle.classList.add('sparkle');
sparkle.style.top = Math.random() * 100 + '%';
sparkle.style.left = Math.random() * 100 + '%';
sparkle.style.animationDelay = Math.random() * 5 + 's';
sparkle.style.width = Math.random() * 6 + 4 + 'px';
sparkle.style.height = sparkle.style.width;
card.appendChild(sparkle);
}

// Toggle sub-question content
function toggleSubQuestion(index) {
const content = document.getElementById(`subQuestion${index}`);
const header = content.previousElementSibling;

// Close all other active questions
document.querySelectorAll('.sub-question-content').forEach(item => {
    if (item !== content && item.classList.contains('active')) {
    item.classList.remove('active');
    item.previousElementSibling.classList.remove('active');
    }
});

// Toggle current question
content.classList.toggle('active');
header.classList.toggle('active');
}

// Toggle hint visibility
function toggleHint(index) {
const hint = document.getElementById(`hint${index}`);
hint.style.display = hint.style.display === 'none' ? 'block' : 'none';
}

// Webcam functionality
function initWebcam(webcamId) {
const video = document.getElementById(webcamId);
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function (error) {
        console.error("Error accessing webcam:", error);
    });
}
}

// Modified capture image function
function captureImage(webcamId) {
console.log(`Capturing from ${webcamId}`);
const video = document.getElementById(webcamId);
const canvas = document.createElement('canvas');
const subQuestionId = webcamId.replace('webcam', '');

// Set canvas dimensions to match video
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

// Draw video frame to canvas
canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

// Get base64 image data
const imageData = canvas.toDataURL('image/jpeg');

// Add hidden input field with the captured image data
const form = video.closest('form');
const hiddenInput = document.createElement('input');
hiddenInput.type = 'hidden';
hiddenInput.name = `captured_image_${subQuestionId}`;
hiddenInput.value = imageData;
form.appendChild(hiddenInput);

// Show thumbnail of captured image
const thumbnailContainer = document.createElement('div');
thumbnailContainer.classList.add('capture-thumbnail-container');
thumbnailContainer.innerHTML = `
    <div class="capture-thumbnail">
    <img src="${imageData}" alt="Captured image">
    <div class="capture-success">
        <i class="fa fa-check-circle"></i>
        <span>ඡායාරූපය ගනු ලැබීය</span>
    </div>
    </div>
`;

// Add the thumbnail after the video
const webcamContainer = video.closest('.webcam-container');
webcamContainer.appendChild(thumbnailContainer);

// Disable the capture button
const captureBtn = document.getElementById('capture' + subQuestionId);
captureBtn.disabled = true;
captureBtn.innerHTML = '<i class="fa fa-check"></i> ඡායාරූපය ලබා ගන්නා ලදී';

// Optional: Show alert
alert('ඡායාරූපය ගනු ලැබීය! පිළිතුර යැවීමට ඉදිරියට යන්න.');
}

// Function to show modal with correct/wrong styling
function showResultModal(isCorrect) {
const modal = document.getElementById('resultModal');
const modalContainer = document.getElementById('resultModalContainer');
const modalIcon = document.getElementById('modalIcon');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const userAnswer = document.getElementById('userAnswer');
const correctAnswer = document.getElementById('correctAnswer');

if (isCorrect) {
    // Reset classes and add correct class
    modalContainer.className = 'result-modal correct-modal';

    // Set correct icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
    </svg>
    `;

    // Set correct title with emoji
    modalTitle.innerHTML = '🌟 නිවැරදියි! 🌟';

    // Set correct message with emojis
    modalMessage.innerHTML = `
    <span class="emoji">🎉</span>
    <span>ඔබේ පිළිතුර නිවැරදියි!</span>
    <span class="emoji">🎊</span>
    `;

    // Example answer values
    userAnswer.textContent = '6408';
    correctAnswer.textContent = '6408';
} else {
    // Reset classes and add wrong class
    modalContainer.className = 'result-modal wrong-modal';

    // Set incorrect icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
    </svg>
    `;

    // Set incorrect title with emoji
    modalTitle.innerHTML = '😕 අපොයි! 😕';

    // Set incorrect message with emojis
    modalMessage.innerHTML = `
    <span class="emoji">🤔</span>
    <span>ඔබේ පිළිතුර වැරදියි. අපි නැවත වරක් උත්සාහ කරමු!</span>
    <span class="emoji">📚</span>
    `;

    // Example answer values with different values
    userAnswer.textContent = '6409';
    correctAnswer.textContent = '6408';
}

// Show modal
modal.classList.add('active');
}

// Function to close modal
function closeModal() {
const modal = document.getElementById('resultModal');
modal.classList.remove('active');

// Clear any confetti
document.querySelectorAll('.confetti').forEach(el => el.remove());
}

// Function to create confetti celebration
function createConfetti() {
const colors = ['#4F46E5', '#A855F7', '#EC4899', '#F59E0B', '#10B981'];
const confettiCount = 100;

for (let i = 0; i < confettiCount; i++) {
    const confetti = document.createElement('div');
    confetti.classList.add('confetti');

    const color = colors[Math.floor(Math.random() * colors.length)];
    const size = Math.random() * 10 + 5;
    const shakeOffset = (Math.random() * 50) - 25;
    const fallDuration = (Math.random() * 3) + 2 + 's';
    const shakeDuration = (Math.random() * 0.5) + 0.5 + 's';

    confetti.style.setProperty('--color', color);
    confetti.style.setProperty('--shake-offset', `${shakeOffset}px`);
    confetti.style.setProperty('--fall-duration', fallDuration);
    confetti.style.setProperty('--shake-duration', shakeDuration);

    confetti.style.width = `${size}px`;
    confetti.style.height = `${size}px`;
    confetti.style.backgroundColor = color;
    confetti.style.left = `${Math.random() * 100}%`;

    document.body.appendChild(confetti);

    // Remove confetti after animation completes
    setTimeout(() => {
    if (confetti.parentNode === document.body) {
        document.body.removeChild(confetti);
    }
    }, parseFloat(fallDuration) * 1000);
}
}

// Add function to get available cameras
async function getAvailableCameras(selectElement) {
try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    // Clear existing options except the first one
    selectElement.innerHTML = '<option value="">කැමරාව තෝරන්න...</option>';

    videoDevices.forEach((device, index) => {
    const option = document.createElement('option');
    option.value = device.deviceId;
    option.text = device.label || `Camera ${index + 1}`;
    selectElement.appendChild(option);
    });

    selectElement.style.display = 'block';
} catch (error) {
    console.error('Error getting cameras:', error);
}
}

// Modified openCamera function
async function openCamera(webcamId, btnElement) {
const video = document.getElementById(webcamId);
const captureBtn = document.getElementById('capture' + webcamId.replace('webcam', ''));
const cameraSelect = document.getElementById('cameraSelect' + webcamId.replace('webcam', ''));
const index = webcamId.replace('webcam', '');
const mirrorBtn = document.getElementById('mirror' + index);

try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();
    btnElement.style.display = 'none';
    captureBtn.style.display = 'block';
    
    // Make mirror button visible for time-related questions
    const answerMethod = document.querySelector('input[name="answer_method"]').value;
    if (answerMethod === 'analog_clock' || answerMethod === 'digital_clock') {
    mirrorBtn.style.display = 'block';
    }

    // Get available cameras after initial camera is opened
    await getAvailableCameras(cameraSelect);
} catch (error) {
    console.error("Error accessing webcam:", error);
}
}

// Add function to switch camera
async function switchCamera(selectElement, webcamId) {
const video = document.getElementById(webcamId);
const deviceId = selectElement.value;

if (!deviceId) return;

// Stop current stream
if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
}

try {
    const stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: { exact: deviceId } }
    });
    video.srcObject = stream;
    video.play();
} catch (error) {
    console.error('Error switching camera:', error);
}
}

// Initialize webcams - remove this section since we'll open camera on button click
window.addEventListener('DOMContentLoaded', function () {
addPulsingEffect();

// Add math symbol styling to numbers in the question
document.querySelectorAll('.main-question-text strong').forEach(element => {
    if (!isNaN(element.textContent) && element.textContent.length <= 2) {
    element.classList.add('math-symbol');
    }
});
});

// Add pulsing effect to important elements
function addPulsingEffect() {
document.querySelectorAll('.btn-success').forEach(btn => {
    btn.classList.add('pulse-subtle');
});
}

// AJAX form submission
function submitAnswer(form, event) {
event.preventDefault();

// Check if an image has been captured
const formData = new FormData(form);
let imageFound = false;
for (const [key, value] of formData.entries()) {
    if (key.startsWith('captured_image_')) {
    imageFound = true;
    break;
    }
}

if (!imageFound) {
    alert('කරුණාකර පිළිතුර යැවීමට පෙර ඡායාරූපයක් ගන්න.');
    return false;
}

// Show loading overlay
document.getElementById('loadingOverlay').classList.add('active');

// Submit the form via AJAX
fetch(form.action, {
    method: 'POST',
    body: formData,
    headers: {
    'X-Requested-With': 'XMLHttpRequest'
    }
})
    .then(response => response.json())
    .then(data => {
    // Hide loading overlay
    document.getElementById('loadingOverlay').classList.remove('active');

    // Display result in modal
    showResultData(data);
    })
    .catch(error => {
    console.error('Error:', error);
    // Hide loading overlay
    document.getElementById('loadingOverlay').classList.remove('active');
    alert('දෝෂයක් ඇති විය. කරුණාකර යළි උත්සාහ කරන්න.');
    });

return false;
}

// Display result data in modal
function showResultData(data) {
const modal = document.getElementById('resultModal');
const modalContainer = document.getElementById('resultModalContainer');
const modalIcon = document.getElementById('modalIcon');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const userAnswer = document.getElementById('userAnswer');
const correctAnswer = document.getElementById('correctAnswer');
const resultImageContainer = document.getElementById('resultImageContainer');
const resultImage = document.getElementById('resultImage');

// Set whether answer is correct or not
const isCorrect = data.is_correct;

if (isCorrect) {
    // Reset classes and add correct class
    modalContainer.className = 'result-modal correct-modal';

    // Set correct icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
    </svg>
    `;

    // Set correct title with emoji
    modalTitle.innerHTML = '🌟 නිවැරදියි! 🌟';

    // Set correct message with emojis
    modalMessage.innerHTML = `
    <span class="emoji">🎉</span>
    <span>ඔබේ පිළිතුර නිවැරදියි!</span>
    <span class="emoji">🎊</span>
    `;

    // Create confetti celebration
    createConfetti();
} else {
    // Reset classes and add wrong class
    modalContainer.className = 'result-modal wrong-modal';

    // Set incorrect icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
    </svg>
    `;

    // Set incorrect title with emoji
    modalTitle.innerHTML = '😕 අපොයි! 😕';

    // Set incorrect message with emojis
    modalMessage.innerHTML = `
    <span class="emoji">🤔</span>
    <span>ඔබේ පිළිතුර වැරදියි. අපි නැවත වරක් උත්සාහ කරමු!</span>
    <span class="emoji">📚</span>
    `;
}

// Display detected and correct values
userAnswer.textContent = data.detected_value || 'හඳුනාගත නොහැක';
correctAnswer.textContent = data.expected_value;

// Show annotated image if available
if (data.annotated_image_url) {
    resultImageContainer.style.display = 'block';
    resultImage.src = data.annotated_image_url;
} else {
    resultImageContainer.style.display = 'none';
}

// Show modal
modal.classList.add('active');
}

// Close modal and go to next question
function closeModalAndContinue() {
const modal = document.getElementById('resultModal');

// Get the redirect URL from the modal's data attribute
const redirectUrl = modal.getAttribute('data-redirect-url');

modal.classList.remove('active');

// Clear any confetti
document.querySelectorAll('.confetti').forEach(el => el.remove());

// Redirect to the next question if URL is available
if (redirectUrl) {
    window.location.href = redirectUrl;
} else {
    // Fallback to play route if no redirect URL is provided
    window.location.href = '/kinesthetic/play';
}
}

function captureImage(webcamId, subQuestionId, index) {
    console.log(`Capturing from ${webcamId} for sub-question ${subQuestionId}`);
    const video = document.getElementById(webcamId);
    const canvas = document.createElement('canvas');
    const mirrorBtn = document.getElementById('mirror' + index);
    const isMirrored = mirrorBtn && mirrorBtn.getAttribute('data-mirrored') === 'true';

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    
    // If mirrored, flip the image horizontally
    if (isMirrored) {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    
    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get base64 image data
    const imageData = canvas.toDataURL('image/jpeg');

    // Generate a unique filename
    const timestamp = new Date().getTime();
    const filename = `capture_${subQuestionId}_${timestamp}.jpg`;

    // First save the image to the server
    fetch('/kinesthetic/save-captured-image', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': document.querySelector('input[name="csrf_token"]').value
      },
      body: JSON.stringify({
        image_data: imageData,
        filename: filename,
        sub_question_id: subQuestionId
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        console.log("Image saved to server:", data.file_path);
        
        // Add hidden input field with the saved image path
        const form = document.querySelector('.all-answers-form');
        const hiddenInput = document.createElement('input');
        hiddenInput.type = 'hidden';
        hiddenInput.name = `captured_image_${subQuestionId}`;
        hiddenInput.value = data.file_path;  // Use the server-provided path
        form.appendChild(hiddenInput);

        // Show thumbnail of captured image
        const thumbnailContainer = document.createElement('div');
        thumbnailContainer.classList.add('capture-thumbnail-container');
        thumbnailContainer.innerHTML = `
          <div class="capture-thumbnail">
            <img src="${data.file_path}" alt="Captured image">
            <div class="capture-success">
              <i class="fa fa-check-circle"></i>
              <span>ඡායාරූපය ගනු ලැබීය</span>
            </div>
          </div>
        `;

        // Add the thumbnail after the video
        const webcamContainer = video.closest('.webcam-container');
        webcamContainer.appendChild(thumbnailContainer);

        // Disable the capture button
        const captureBtn = document.getElementById('capture' + index);
        captureBtn.disabled = true;
        captureBtn.innerHTML = '<i class="fa fa-check"></i> ඡායාරූපය ලබා ගන්නා ලදී';

        // Update completion indicator
        const completionIndicator = document.getElementById('completionIndicator' + index);
        completionIndicator.innerHTML = '<i class="fa fa-check-circle completion-done"></i>';

        // Update captured answers count and check if all are done
        updateCapturedAnswersCount();
      } else {
        console.error("Error saving image:", data.error);
        alert("ඡායාරූපය සුරැකීමට නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න.");
      }
    })
    .catch(error => {
      console.error("Error:", error);
      alert("ඡායාරූපය සුරැකීමට නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න.");
    });
}

// Function to update captured answers count
function updateCapturedAnswersCount() {
const form = document.querySelector('.all-answers-form');
const subQuestionIds = form.querySelectorAll('input[name="sub_question_ids"]');
const capturedImages = Array.from(form.elements).filter(el => el.name.startsWith('captured_image_'));
const capturedCount = capturedImages.length;
const totalSubQuestions = subQuestionIds.length;

document.getElementById('capturedAnswersCount').textContent = capturedCount;

// Enable/disable the submit all button based on whether all answers are captured
const submitAllBtn = document.getElementById('submitAllBtn');
if (capturedCount >= totalSubQuestions) {
    submitAllBtn.disabled = false;
    submitAllBtn.classList.add('ready');
} else {
    submitAllBtn.disabled = true;
    submitAllBtn.classList.remove('ready');
}
}

// AJAX submission for all answers
function submitAllAnswers(form, event) {
event.preventDefault();

// Check if all sub-questions have been answered
const formData = new FormData(form);
const subQuestionIds = formData.getAll('sub_question_ids');
let allAnswered = true;

for (const subQuestionId of subQuestionIds) {
    let hasImage = false;
    for (const [key, value] of formData.entries()) {
    if (key === `captured_image_${subQuestionId}`) {
        hasImage = true;
        break;
    }
    }

    if (!hasImage) {
    allAnswered = false;
    break;
    }
}

if (!allAnswered) {
    alert('කරුණාකර සියලුම අනු ප්‍රශ්න සඳහා පිංතූර ගන්න.');
    return false;
}

// Show loading overlay
document.getElementById('loadingOverlay').classList.add('active');

// Submit the form via AJAX
fetch(form.action, {
    method: 'POST',
    body: formData,
    headers: {
    'X-Requested-With': 'XMLHttpRequest'
    }
})
    .then(response => {
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    return response.json();
    })
    .then(data => {
    // Hide loading overlay
    document.getElementById('loadingOverlay').classList.remove('active');

    // Display results in modal
    showAllResults(data);
    })
    .catch(error => {
    console.error('Error:', error);
    // Hide loading overlay
    document.getElementById('loadingOverlay').classList.remove('active');
    alert('දෝෂයක් ඇති විය. කරුණාකර යළි උත්සාහ කරන්න.');
    });

return false;
}

// Function to display all results at once
function showAllResults(data) {
const modal = document.getElementById('resultModal');
const modalContainer = document.getElementById('resultModalContainer');
const modalIcon = document.getElementById('modalIcon');
const modalTitle = document.getElementById('modalTitle');
const detailedResults = document.getElementById('detailedResults');
const correctAnswersCount = document.getElementById('correctAnswersCount');
const wrongAnswersCount = document.getElementById('wrongAnswersCount');

// Store the redirect URL for use when closing the modal
modal.setAttribute('data-redirect-url', data.redirect_url || '');

// Count correct and wrong answers
const totalCorrect = data.results.filter(r => r.is_correct).length;
const totalWrong = data.results.length - totalCorrect;

correctAnswersCount.textContent = totalCorrect;
wrongAnswersCount.textContent = totalWrong;

// Set overall status (correct if more than 60% correct)
const overallCorrect = totalCorrect >= Math.ceil(data.results.length * 0.6);

if (overallCorrect) {
    // Reset classes and add correct class
    modalContainer.className = 'result-modal correct-modal';

    // Set correct icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
    </svg>
    `;

    // Set correct title with emoji
    modalTitle.innerHTML = '🌟 සාර්ථකයි! 🌟';

    // Create confetti celebration
    createConfetti();
} else {
    // Reset classes and add wrong class
    modalContainer.className = 'result-modal wrong-modal';

    // Set incorrect icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
    </svg>
    `;

    // Set incorrect title with emoji
    modalTitle.innerHTML = '😕 වැඩිදියුණු විය යුතුයි 😕';
}

// Clear previous results
detailedResults.innerHTML = '';

// Add detailed results for each sub-question
data.results.forEach((result, index) => {
    const resultItem = document.createElement('div');
    resultItem.classList.add('result-item');
    resultItem.classList.add(result.is_correct ? 'correct' : 'wrong');

    // Format the detected value display - for time questions within tolerance
    let detectedValueDisplay = result.detected_value || 'හඳුනාගත නොහැක';
    let additionalInfo = '';
    
    // Check if it's a time-based question (contains ':' in expected or detected value)
    if (result.is_correct && 
        (String(result.expected_value).includes(':') || String(result.detected_value).includes(':'))) {
      if (result.detected_value !== result.expected_value) {
        // Within tolerance case
        additionalInfo = `<div class="time-tolerance-info">(±3 මිනිත්තු ඉවසීම තුළ නිවැරදියි)</div>`;
      }
    }

    resultItem.innerHTML = `
    <div class="result-header">
        <div class="result-icon">
        ${result.is_correct ?
        '<i class="fa fa-check-circle"></i>' :
        '<i class="fa fa-times-circle"></i>'}
        </div>
        <div class="result-title">${result.sub_question_text}</div>
    </div>
    <div class="result-details">
        <div class="result-row">
        <span class="result-label">ඔබ පෙන්වූ පිළිතුර</span>
        <span class="result-value">${detectedValueDisplay}</span>
        ${additionalInfo}
        </div>
        <div class="result-row">
        <span class="result-label">නිවැරදි පිළිතුර</span>
        <span class="result-value">${result.expected_value}</span>
        </div>
        ${result.annotated_image_url ? `
        <div class="result-image-small">
            <img src="${result.annotated_image_url}" alt="Analyzed image">
        </div>
        ` : ''}
    </div>
    `;

    detailedResults.appendChild(resultItem);
});

// Show modal
modal.classList.add('active');
}

// Add function to toggle mirror mode
function toggleMirror(index) {
const video = document.getElementById('webcam' + index);
const mirrorBtn = document.getElementById('mirror' + index);

if (video.classList.contains('mirrored')) {
    video.classList.remove('mirrored');
    mirrorBtn.innerHTML = '<i class="fa fa-exchange"></i> පිංතූරය පෙරළන්න';
    mirrorBtn.setAttribute('data-mirrored', 'false');
} else {
    video.classList.add('mirrored');
    mirrorBtn.innerHTML = '<i class="fa fa-exchange"></i> සාමාන්ය පිංතූරය';
    mirrorBtn.setAttribute('data-mirrored', 'true');
}
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', function () {
// ...existing code...
updateCapturedAnswersCount();
});

// Global variables to track drawing state
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let clockCanvasContext = {};
let clockImage = new Image();
clockImage.src = "/static/images/clock_face.jpg"; // Update with your clock image path
clockImage.crossOrigin = "anonymous";
clockImage.src = "https://visuallearning000123.weebly.com/uploads/1/5/2/4/152446337/c0_orig.jpg"; // Update with your clock image path

// Initialize clock canvas when the page loads
document.querySelectorAll('[id^="clockCanvas"]').forEach(canvas => {
  const index = canvas.id.replace('clockCanvas', '');
  initializeClockCanvas(index);
});

function initializeClockCanvas(index) {
  const canvas = document.getElementById(`clockCanvas${index}`);
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  clockCanvasContext[index] = ctx;
  clockImage.onload = function() {
  // Draw the clock face when the image loads
  clockImage.onload = function() {
    drawClockFace(index);
  };
  
  // If image already loaded, draw it now
  if (clockImage.complete) {
    drawClockFace(index);
  }
  
  // Set up event listeners for drawing
  canvas.addEventListener('mousedown', function(e) {
    startDrawing(e, index);
  });
  
  canvas.addEventListener('mousemove', function(e) {
    draw(e, index);
  });
  
  canvas.addEventListener('mouseup', function() {
    stopDrawing();
  });
  
  canvas.addEventListener('mouseout', function() {
    stopDrawing();
  });
  canvas.addEventListener('touchstart', function(e) {
    // Touch events for mobile
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
    e.preventDefault();
  });
  
  canvas.addEventListener('touchmove', function(e) {
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
    e.preventDefault();
  });
    const mouseEvent = new MouseEvent('mouseup');
  canvas.addEventListener('touchend', function() {
    const mouseEvent = new MouseEvent('mouseup');
    canvas.dispatchEvent(mouseEvent);
  });
}

function drawClockFace(index) {
  const ctx = clockCanvasContext[index];
  const canvas = document.getElementById(`clockCanvas${index}`);
  ctx.drawImage(clockImage, 0, 0, canvas.width, canvas.height);
} try {
    // Draw the background image (clock face)
    ctx.drawImage(clockImage, 0, 0, canvas.width, canvas.height);
  } catch (e) {
    console.error("Failed to draw clock image, drawing fallback clock face", e);
    // Draw a simple clock face as fallback
    drawFallbackClockFace(ctx, canvas.width, canvas.height);
  }
}

// Add a fallback function to draw a basic clock face if the image fails to load
function drawFallbackClockFace(ctx, width, height) {
  // Clear the canvas
  ctx.clearRect(0, 0, width, height);
  
  // Draw clock circle
  ctx.beginPath();
  ctx.arc(width/2, height/2, Math.min(width, height)/2 - 10, 0, 2 * Math.PI);
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 2;
  ctx.stroke();
  
  // Draw clock center
  ctx.beginPath();
  ctx.arc(width/2, height/2, 5, 0, 2 * Math.PI);
  ctx.fillStyle = '#000';
  ctx.fill();
  
  // Draw hour marks
  for (let i = 0; i < 12; i++) {
    const angle = (i * Math.PI / 6);
    const x1 = width/2 + Math.sin(angle) * (Math.min(width, height)/2 - 30);
    const y1 = height/2 - Math.cos(angle) * (Math.min(width, height)/2 - 30);
    const x2 = width/2 + Math.sin(angle) * (Math.min(width, height)/2 - 10);
    const y2 = height/2 - Math.cos(angle) * (Math.min(width, height)/2 - 10);
    
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Add hour numbers
    const numX = width/2 + Math.sin(angle) * (Math.min(width, height)/2 - 50);
    const numY = height/2 - Math.cos(angle) * (Math.min(width, height)/2 - 50);
    
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i === 0 ? '12' : i.toString(), numX, numY);
  }
}

function startDrawing(e, index) {
  isDrawing = true;
  const canvas = document.getElementById(`clockCanvas${index}`);
  const rect = canvas.getBoundingClientRect();
  lastX = e.clientX - rect.left;
  lastY = e.clientY - rect.top;
}

function draw(e, index) {
  if (!isDrawing) return;
  
  const ctx = clockCanvasContext[index];
  const canvas = document.getElementById(`clockCanvas${index}`);
  const rect = canvas.getBoundingClientRect();
  const currentX = e.clientX - rect.left;
  const currentY = e.clientY - rect.top;
  
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#000000';
  
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(currentX, currentY);
  ctx.stroke();
  
  lastX = currentX;
  lastY = currentY;
}

function stopDrawing() {
  isDrawing = false;
}

function clearClockHands(index) {
  const canvas = document.getElementById(`clockCanvas${index}`);
  const ctx = clockCanvasContext[index];
  
  // Clear the entire canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Redraw just the clock face image
  if (clockImage && clockImage.complete && clockImage.naturalWidth > 0) {
    // If the image is already loaded, draw it directly
    ctx.drawImage(clockImage, 0, 0, canvas.width, canvas.height);
  } else {
    // If there's an issue with the image, draw a fallback clock face
    drawFallbackClockFace(ctx, canvas.width, canvas.height);
  }
  
  // Remove any preview that might be showing
  const existingPreview = canvas.parentElement.querySelector('.captured-preview');
  if (existingPreview) {
    existingPreview.remove();
  }
}

function captureClockDrawing(index, subQuestionId) {
  const canvas = document.getElementById(`clockCanvas${index}`);
  
  try {
    // Get the image data
    const imageData = canvas.toDataURL('image/png');
    
    // Generate a unique filename
    const timestamp = new Date().getTime();
    const filename = `clock_drawing_${subQuestionId}_${timestamp}.png`;
    
    // Show loading indicator
    showNotification('පිළිතුර විශ්ලේෂණය කරමින්...', 'info');
    
    // Save image and process with the clock model - Fix the URL to match the route
    fetch('/save-and-process-clock', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': document.querySelector('input[name="csrf_token"]').value
      },
      body: JSON.stringify({
        image_data: imageData,
        filename: filename,
        sub_question_id: subQuestionId
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        console.log("Clock drawing saved and processed:", data);
        
        // Add hidden input field with the saved image path
        const form = document.querySelector('.all-answers-form');
        let hiddenInput = document.getElementById(`answer_${subQuestionId}`);
        if (!hiddenInput) {
          hiddenInput = document.createElement('input');
          hiddenInput.type = 'hidden';
          hiddenInput.name = `captured_image_${subQuestionId}`;
          hiddenInput.id = `answer_${subQuestionId}`;
          form.appendChild(hiddenInput);
        }
        
        hiddenInput.value = data.file_path;  // Use the server-provided path
        
        // Update the completion indicator
        updateCompletionIndicator(index, true);
        
        // Update submission button state
        updateSubmissionButtonState();
        
        // Add a preview of the captured image with detected time
        const previewContainer = document.createElement('div');
        previewContainer.className = 'captured-preview';
        previewContainer.innerHTML = `
          <p class="text-success"><i class="fa fa-check-circle"></i> පිළිතුර ලබා ගන්නා ලදී</p>
          ${data.detected_time ? `<p class="detected-time">Detected time: ${data.detected_time}</p>` : ''}
          <img src="${data.annotated_image_url || data.file_path}" alt="Clock drawing" class="img-thumbnail" style="max-width: 150px; margin-top: 10px;">
        `;
        
        // Find existing preview and replace, or append new one
        const existingPreview = canvas.parentElement.querySelector('.captured-preview');
        if (existingPreview) {
          canvas.parentElement.replaceChild(previewContainer, existingPreview);
        } else {
          canvas.parentElement.appendChild(previewContainer);
        }
        
        // Show success notification
        showNotification('පිළිතුර සාර්ථකව සුරකින ලදී', 'success');
      } else {
        console.error("Error processing clock drawing:", data.error);
        showNotification('පිළිතුර සුරැකීමට අසමත් විය', 'error');
      }
    })
    .catch(error => {
      console.error("Error:", error);
      showNotification('පිළිතුර සුරැකීමට අසමත් විය', 'error');
    });
    
  } catch (error) {
    console.error("Error capturing clock drawing:", error);
    
    // If there's a CORS issue, try to draw to a new untainted canvas and capture that
    try {
      const newCanvas = document.createElement('canvas');
      newCanvas.width = canvas.width;
      newCanvas.height = canvas.height;
      const newCtx = newCanvas.getContext('2d');
      
      // Draw just the user's drawing (not the background image)
      drawFallbackClockFace(newCtx, canvas.width, canvas.height);
      
      // Copy over the user's drawn lines from the original context
      const userDrawingImageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
      newCtx.putImageData(userDrawingImageData, 0, 0);
      
      // Try to get image data from this new canvas
      const imageData = newCanvas.toDataURL('image/png');
      
      // Try saving the fallback image
      const timestamp = new Date().getTime();
      const filename = `clock_drawing_fallback_${subQuestionId}_${timestamp}.png`;
      
      fetch('/kinesthetic/save-captured-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': document.querySelector('input[name="csrf_token"]').value
        },
        body: JSON.stringify({
          image_data: imageData,
          filename: filename,
          sub_question_id: subQuestionId
        })
      })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          // Similar handling for success case with fallback
          const form = document.querySelector('.all-answers-form');
          let hiddenInput = document.getElementById(`answer_${subQuestionId}`);
          if (!hiddenInput) {
            hiddenInput = document.createElement('input');
            hiddenInput.type = 'hidden';
            hiddenInput.name = `captured_image_${subQuestionId}`;
            hiddenInput.id = `answer_${subQuestionId}`;
            form.appendChild(hiddenInput);
          }
          
          hiddenInput.value = data.file_path;
          updateCompletionIndicator(index, true);
          updateSubmissionButtonState();
          showNotification('පිළිතුර සුරකින ලදී (අඩු තත්වයෙන්)', 'warning');
        }
      })
      .catch(fallbackError => {
        console.error("Fallback error:", fallbackError);
        showNotification('පිළිතුර සුරැකීමට අසමත් විය', 'error');
      });
      
    } catch (fallbackError) {
      showNotification('පිළිතුර සුරැකීමට අසමත් විය', 'error');
      console.error("Fallback error:", fallbackError);
    }
  }
}

// Global variables for decimal drawing
let decimalDrawingState = {};
let decimalCurrentTool = {};
let decimalGridImages = {};

// Initialize decimal canvas when the page loads
document.addEventListener('DOMContentLoaded', function() {
  // Find all decimal canvases and initialize them
  document.querySelectorAll('[id^="decimalCanvas"]').forEach(canvas => {
    const index = canvas.id.replace('decimalCanvas', '');
    initializeDecimalCanvas(index);
  });
});

function initializeDecimalCanvas(index) {
  const canvas = document.getElementById(`decimalCanvas${index}`);
  if (!canvas) return;
  
  // Default tool is shade
  decimalCurrentTool[index] = 'shade';
  
  // Setup drawing state
  decimalDrawingState[index] = {
    isDrawing: false,
    lastX: 0,
    lastY: 0
  };
  
  const ctx = canvas.getContext('2d');
  
  // Load the decimal grid image (10 squares)
  const gridImage = new Image();
  // Add crossOrigin attribute to prevent canvas tainting
  gridImage.crossOrigin = "anonymous";
  gridImage.src = "https://visuallearning000123.weebly.com/uploads/1/5/2/4/152446337/d1.png";
  
  gridImage.onload = function() {
    // Draw the decimal grid
    ctx.drawImage(gridImage, 0, 0, canvas.width, canvas.height);
    
    // Store the grid image for later use
    decimalGridImages[index] = gridImage;
    
    // Set up event listeners for drawing
    setupDecimalDrawingEvents(canvas, index);
  };
  
  // If image loading fails, draw the grid manually
  gridImage.onerror = function() {
    console.error("Failed to load the grid image, falling back to manual grid drawing");
    // Draw 10 squares manually
    drawDecimalGrid(ctx, canvas.width, canvas.height);
    setupDecimalDrawingEvents(canvas, index);
  };
}

function drawDecimalGrid(ctx, width, height) {
  const squareSize = width / 10;
  const squareHeight = height * 0.6; // Make squares not too tall
  const yOffset = (height - squareHeight) / 2;
  
  // Draw 10 squares
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 2;
  
  for(let i = 0; i < 10; i++) {
    const x = i * squareSize;
    ctx.strokeRect(x, yOffset, squareSize, squareHeight);
    
    // Add fraction label below each square
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#333';
    ctx.fillText(`${i+1}/10`, x + squareSize/2, yOffset + squareHeight + 20);
  }
  
  // Add a title
  ctx.font = 'bold 16px Arial';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#333';
  ctx.fillText('දශම සංඛ්‍යාව සෙවීමට අවශ්‍ය කොටස් වර්ණ කරන්න', width/2, yOffset - 20);
}

function setupDecimalDrawingEvents(canvas, index) {
  // Mouse events
  canvas.addEventListener('mousedown', function(e) {
    startDecimalDrawing(e, canvas, index);
  });
  
  canvas.addEventListener('mousemove', function(e) {
    drawOnDecimalCanvas(e, canvas, index);
  });
  
  canvas.addEventListener('mouseup', function() {
    stopDecimalDrawing(index);
  });
  
  canvas.addEventListener('mouseout', function() {
    stopDecimalDrawing(index);
  });
  
  // Touch events for mobile
  canvas.addEventListener('touchstart', function(e) {
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
    e.preventDefault();
  });
  
  canvas.addEventListener('touchmove', function(e) {
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
    e.preventDefault();
  });
  
  canvas.addEventListener('touchend', function() {
    const mouseEvent = new MouseEvent('mouseup');
    canvas.dispatchEvent(mouseEvent);
  });
}

function selectDecimalTool(index, tool) {
  decimalCurrentTool[index] = tool;
  
  // Update UI to show active tool
  const shadeBtn = document.getElementById(`shadeTool${index}`);
  const eraserBtn = document.getElementById(`eraserTool${index}`);
  
  if (tool === 'shade') {
    shadeBtn.classList.add('active');
    eraserBtn.classList.remove('active');
  } else {
    eraserBtn.classList.add('active');
    shadeBtn.classList.remove('active');
  }
}

function startDecimalDrawing(e, canvas, index) {
  const rect = canvas.getBoundingClientRect();
  decimalDrawingState[index].isDrawing = true;
  decimalDrawingState[index].lastX = e.clientX - rect.left;
  decimalDrawingState[index].lastY = e.clientY - rect.top;
}

function drawOnDecimalCanvas(e, canvas, index) {
  if (!decimalDrawingState[index].isDrawing) return;
  
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const currentX = e.clientX - rect.left;
  const currentY = e.clientY - rect.top;
  
  // Set up drawing properties based on current tool
  if (decimalCurrentTool[index] === 'shade') {
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = '#3B82F6'; // Blue color for shading
    ctx.fillStyle = '#93C5FD';   // Lighter blue for filling
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
  } else { // eraser
    ctx.globalCompositeOperation = 'destination-out';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
  }
  
  // Draw line
  ctx.beginPath();
  ctx.moveTo(decimalDrawingState[index].lastX, decimalDrawingState[index].lastY);
  ctx.lineTo(currentX, currentY);
  ctx.stroke();
  
  // Update last position
  decimalDrawingState[index].lastX = currentX;
  decimalDrawingState[index].lastY = currentY;
}

function stopDecimalDrawing(index) {
  decimalDrawingState[index].isDrawing = false;
}

function clearDecimalDrawing(index) {
  const canvas = document.getElementById(`decimalCanvas${index}`);
  const ctx = canvas.getContext('2d');
  
  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Reset composite operation to default
  ctx.globalCompositeOperation = 'source-over';
  
  // Use cached grid image if available
  if (decimalGridImages[index]) {
    ctx.drawImage(decimalGridImages[index], 0, 0, canvas.width, canvas.height);
  } else {
    // Fallback to loading the image again
    const gridImage = new Image();
    gridImage.crossOrigin = "anonymous";
    gridImage.src = "https://visuallearning000123.weebly.com/uploads/1/5/2/4/152446337/d1.png";
    
    gridImage.onload = function() {
      ctx.drawImage(gridImage, 0, 0, canvas.width, canvas.height);
      // Cache the image for future use
      decimalGridImages[index] = gridImage;
    };
    
    gridImage.onerror = function() {
      drawDecimalGrid(ctx, canvas.width, canvas.height);
    };
  }
  
  // Remove any existing preview
  const existingPreview = canvas.parentElement.querySelector('.captured-preview');
  if (existingPreview) {
    existingPreview.remove();
  }
}

function captureDecimalDrawing(index, subQuestionId) {
  const canvas = document.getElementById(`decimalCanvas${index}`);
  
  try {
    // Get the image data
    const imageData = canvas.toDataURL('image/png');
    
    // Generate a unique filename
    const timestamp = new Date().getTime();
    const filename = `decimal_drawing_${subQuestionId}_${timestamp}.png`;
    
    // Save image to server
    fetch('/save-captured-image', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': document.querySelector('input[name="csrf_token"]').value
      },
      body: JSON.stringify({
        image_data: imageData,
        filename: filename,
        sub_question_id: subQuestionId
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        console.log("Decimal drawing saved to server:", data.file_path);
        
        // Add hidden input field with the saved image path
        const form = document.querySelector('.all-answers-form');
        let hiddenInput = document.getElementById(`answer_${subQuestionId}`);
        if (!hiddenInput) {
          hiddenInput = document.createElement('input');
          hiddenInput.type = 'hidden';
          hiddenInput.name = `captured_image_${subQuestionId}`;
          hiddenInput.id = `answer_${subQuestionId}`;
          form.appendChild(hiddenInput);
        }
        
        hiddenInput.value = data.file_path;
        
        // Update the completion indicator
        updateCompletionIndicator(index, true);
        
        // Update submission button state
        updateSubmissionButtonState();
        
        // Add a preview of the captured image
        const previewContainer = document.createElement('div');
        previewContainer.className = 'captured-preview';
        previewContainer.innerHTML = `
          <p class="text-success"><i class="fa fa-check-circle"></i> පිළිතුර ලබා ගන්නා ලදී</p>
          <img src="${data.file_path}" alt="Decimal representation" class="img-thumbnail" style="max-width: 150px; margin-top: 10px;">
        `;
        
        // Find existing preview and replace, or append new one
        const existingPreview = canvas.parentElement.querySelector('.captured-preview');
        if (existingPreview) {
          canvas.parentElement.replaceChild(previewContainer, existingPreview);
        } else {
          canvas.parentElement.appendChild(previewContainer);
        }
        
        // Show success notification
        showNotification('පිළිතුර සාර්ථකව සුරකින ලදී', 'success');
      } else {
        console.error("Error saving decimal drawing:", data.error);
        showNotification('පිළිතුර සුරැකීමට අසමත් විය', 'error');
      }
    })
    .catch(error => {
      console.error("Error:", error);
      showNotification('පිළිතුර සුරැකීමට අසමත් විය', 'error');
    });
  } catch (error) {
    console.error("Error capturing decimal drawing:", error);
    showNotification('පිළිතුර සුරැකීමට අසමත් විය', 'error');
  }
}

// Add the missing function for updating submission button state
function updateSubmissionButtonState() {
  const form = document.querySelector('.all-answers-form');
  if (!form) return;
  
  // Get all sub-question IDs
  const subQuestionIds = Array.from(form.querySelectorAll('input[name="sub_question_ids"]')).map(el => el.value);
  
  // Count how many have answers
  let answeredCount = 0;
  subQuestionIds.forEach(id => {
    // Check if we have either a captured image or a drawn answer
    const hasCapturedImage = form.querySelector(`input[name="captured_image_${id}"]`);
    const hasDrawnAnswer = form.querySelector(`input[name="answer_${id}"]`);
    
    if (hasCapturedImage || hasDrawnAnswer) {
      answeredCount++;
    }
  });
  
  // Enable/disable the submit all button based on whether all are answered
  const submitAllBtn = document.getElementById('submitAllBtn');
  if (submitAllBtn) {
    const allAnswered = (answeredCount >= subQuestionIds.length);
    submitAllBtn.disabled = !allAnswered;
    
    if (allAnswered) {
      submitAllBtn.classList.add('ready');
    } else {
      submitAllBtn.classList.remove('ready');
    }
    
    // Update the counter display
    const counterElement = document.getElementById('capturedAnswersCount');
    if (counterElement) {
      counterElement.textContent = answeredCount;
    }
  }
  
  console.log(`Submission state updated: ${answeredCount}/${subQuestionIds.length} questions answered`);
}

// First, add the showNotification function if it doesn't exist or fix references to it
function showNotification(message, type) {
  // Create notification element
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.innerHTML = `
    <div class="notification-content">
      <i class="fa ${type === 'success' ? 'fa-check-circle' : 
                  type === 'warning' ? 'fa-exclamation-triangle' : 
                  'fa-exclamation-circle'}"></i>
      <span>${message}</span>
    </div>
  `;
  
  // Add to document
  document.body.appendChild(notification);
  
  // Add active class after a small delay (for animation)
  setTimeout(() => {
    notification.classList.add('active');
  }, 10);
  
  // Remove after 3 seconds
  setTimeout(() => {
    notification.classList.remove('active');
    setTimeout(() => {
      if (notification.parentNode) {
        document.body.removeChild(notification);
      }
    }, 300); // Wait for fade out animation
  }, 3000);
}

// Add function to update completion indicator
function updateCompletionIndicator(index, isComplete) {
  const completionIndicator = document.getElementById(`completionIndicator${index}`);
  if (!completionIndicator) return;
  
  if (isComplete) {
    completionIndicator.innerHTML = '<i class="fa fa-check-circle completion-done"></i>';
  } else {
    completionIndicator.innerHTML = '<i class="fa fa-circle-o"></i>';
  }
  
  // Update the count of captured answers
  updateCapturedAnswersCount();
}