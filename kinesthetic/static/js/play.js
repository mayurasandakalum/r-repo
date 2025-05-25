// Sparkle animation
const card = document.querySelector(".card");
for (let i = 0; i < 10; i++) {
  const sparkle = document.createElement("div");
  sparkle.classList.add("sparkle");
  sparkle.style.top = Math.random() * 100 + "%";
  sparkle.style.left = Math.random() * 100 + "%";
  sparkle.style.animationDelay = Math.random() * 5 + "s";
  sparkle.style.width = Math.random() * 6 + 4 + "px";
  sparkle.style.height = sparkle.style.width;
  card.appendChild(sparkle);
}

// Toggle sub-question content
function toggleSubQuestion(index) {
  const content = document.getElementById(`subQuestion${index}`);
  const header = content.previousElementSibling;

  // Close all other active questions
  document.querySelectorAll(".sub-question-content").forEach((item) => {
    if (item !== content && item.classList.contains("active")) {
      item.classList.remove("active");
      item.previousElementSibling.classList.remove("active");
    }
  });

  // Toggle current question
  content.classList.toggle("active");
  header.classList.toggle("active");
}

// Toggle hint visibility
function toggleHint(index) {
  const hint = document.getElementById(`hint${index}`);
  hint.style.display = hint.style.display === "none" ? "block" : "none";
}

// Webcam functionality
function initWebcam(webcamId) {
  const video = document.getElementById(webcamId);
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({ video: true })
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
  const canvas = document.createElement("canvas");
  const subQuestionId = webcamId.replace("webcam", "");

  // Set canvas dimensions to match video
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Draw video frame to canvas
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

  // Get base64 image data
  const imageData = canvas.toDataURL("image/jpeg");

  // Add hidden input field with the captured image data
  const form = video.closest("form");
  const hiddenInput = document.createElement("input");
  hiddenInput.type = "hidden";
  hiddenInput.name = `captured_image_${subQuestionId}`;
  hiddenInput.value = imageData;
  form.appendChild(hiddenInput);

  // Show thumbnail of captured image
  const thumbnailContainer = document.createElement("div");
  thumbnailContainer.classList.add("capture-thumbnail-container");
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
  const webcamContainer = video.closest(".webcam-container");
  webcamContainer.appendChild(thumbnailContainer);

  // Disable the capture button
  const captureBtn = document.getElementById("capture" + subQuestionId);
  captureBtn.disabled = true;
  captureBtn.innerHTML = '<i class="fa fa-check"></i> ඡායාරූපය ලබා ගන්නා ලදී';

  // Optional: Show alert
  alert("ඡායාරූපය ගනු ලැබීය! පිළිතුර යැවීමට ඉදිරියට යන්න.");
}

// Function to show modal with correct/wrong styling
function showResultModal(isCorrect) {
  const modal = document.getElementById("resultModal");
  const modalContainer = document.getElementById("resultModalContainer");
  const modalIcon = document.getElementById("modalIcon");
  const modalTitle = document.getElementById("modalTitle");
  const modalMessage = document.getElementById("modalMessage");
  const userAnswer = document.getElementById("userAnswer");
  const correctAnswer = document.getElementById("correctAnswer");

  if (isCorrect) {
    // Reset classes and add correct class
    modalContainer.className = "result-modal correct-modal";

    // Set correct icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
    </svg>
    `;

    // Set correct title with emoji
    modalTitle.innerHTML = "🌟 නිවැරදියි! 🌟";

    // Set correct message with emojis
    modalMessage.innerHTML = `
    <span class="emoji">🎉</span>
    <span>ඔබේ පිළිතුර නිවැරදියි!</span>
    <span class="emoji">🎊</span>
    `;

    // Example answer values
    userAnswer.textContent = "6408";
    correctAnswer.textContent = "6408";
  } else {
    // Reset classes and add wrong class
    modalContainer.className = "result-modal wrong-modal";

    // Set incorrect icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
    </svg>
    `;

    // Set incorrect title with emoji
    modalTitle.innerHTML = "😕 අපොයි! 😕";

    // Set incorrect message with emojis
    modalMessage.innerHTML = `
    <span class="emoji">🤔</span>
    <span>ඔබේ පිළිතුර වැරදියි. අපි නැවත වරක් උත්සාහ කරමු!</span>
    <span class="emoji">📚</span>
    `;

    // Example answer values with different values
    userAnswer.textContent = "6409";
    correctAnswer.textContent = "6408";
  }

  // Show modal
  modal.classList.add("active");
}

// Function to close modal
function closeModal() {
  const modal = document.getElementById("resultModal");
  modal.classList.remove("active");

  // Clear any confetti
  document.querySelectorAll(".confetti").forEach((el) => el.remove());
}

// Function to create confetti celebration
function createConfetti() {
  const colors = ["#4F46E5", "#A855F7", "#EC4899", "#F59E0B", "#10B981"];
  const confettiCount = 100;

  for (let i = 0; i < confettiCount; i++) {
    const confetti = document.createElement("div");
    confetti.classList.add("confetti");

    const color = colors[Math.floor(Math.random() * colors.length)];
    const size = Math.random() * 10 + 5;
    const shakeOffset = Math.random() * 50 - 25;
    const fallDuration = Math.random() * 3 + 2 + "s";
    const shakeDuration = Math.random() * 0.5 + 0.5 + "s";

    confetti.style.setProperty("--color", color);
    confetti.style.setProperty("--shake-offset", `${shakeOffset}px`);
    confetti.style.setProperty("--fall-duration", fallDuration);
    confetti.style.setProperty("--shake-duration", shakeDuration);

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
    const videoDevices = devices.filter(
      (device) => device.kind === "videoinput"
    );

    // Clear existing options except the first one
    selectElement.innerHTML = '<option value="">කැමරාව තෝරන්න...</option>';

    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      selectElement.appendChild(option);
    });

    selectElement.style.display = "block";
  } catch (error) {
    console.error("Error getting cameras:", error);
  }
}

// Modified openCamera function
async function openCamera(webcamId, btnElement) {
  const video = document.getElementById(webcamId);
  const captureBtn = document.getElementById(
    "capture" + webcamId.replace("webcam", "")
  );
  const cameraSelect = document.getElementById(
    "cameraSelect" + webcamId.replace("webcam", "")
  );
  const index = webcamId.replace("webcam", "");
  const mirrorBtn = document.getElementById("mirror" + index);

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();
    btnElement.style.display = "none";
    captureBtn.style.display = "block";

    // Always show mirror button for all question types
    mirrorBtn.style.display = "block";

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
    video.srcObject.getTracks().forEach((track) => track.stop());
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: deviceId } },
    });
    video.srcObject = stream;
    video.play();
  } catch (error) {
    console.error("Error switching camera:", error);
  }
}

// Initialize webcams - remove this section since we'll open camera on button click
window.addEventListener("DOMContentLoaded", function () {
  addPulsingEffect();

  // Add math symbol styling to numbers in the question
  document.querySelectorAll(".main-question-text strong").forEach((element) => {
    if (!isNaN(element.textContent) && element.textContent.length <= 2) {
      element.classList.add("math-symbol");
    }
  });

  // Initialize performance metrics visualization if present
  initializePerformanceMetrics();
});

// Add pulsing effect to important elements
function addPulsingEffect() {
  document.querySelectorAll(".btn-success").forEach((btn) => {
    btn.classList.add("pulse-subtle");
  });
}

// New function to initialize performance metrics visualization
function initializePerformanceMetrics() {
  const performanceElements = document.querySelectorAll(
    ".subject-performance-meter"
  );

  performanceElements.forEach((element) => {
    const percentage = parseFloat(
      element.getAttribute("data-percentage") || "0"
    );
    const barElement = element.querySelector(".performance-bar");

    if (barElement) {
      barElement.style.width = `${percentage}%`;

      // Color coding based on performance
      if (percentage < 50) {
        barElement.classList.add("bg-danger");
      } else if (percentage < 70) {
        barElement.classList.add("bg-warning");
      } else {
        barElement.classList.add("bg-success");
      }
    }
  });
}

// AJAX form submission
function submitAnswer(form, event) {
  event.preventDefault();

  // Check if an image has been captured
  const formData = new FormData(form);
  let imageFound = false;
  for (const [key, value] of formData.entries()) {
    if (key.startsWith("captured_image_")) {
      imageFound = true;
      break;
    }
  }

  if (!imageFound) {
    alert("කරුණාකර පිළිතුර යැවීමට පෙර ඡායාරූපයක් ගන්න.");
    return false;
  }

  // Show loading overlay
  document.getElementById("loadingOverlay").classList.add("active");

  // Submit the form via AJAX
  fetch(form.action, {
    method: "POST",
    body: formData,
    headers: {
      "X-Requested-With": "XMLHttpRequest",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      // Hide loading overlay
      document.getElementById("loadingOverlay").classList.remove("active");

      // Display result in modal
      showResultData(data);
    })
    .catch((error) => {
      console.error("Error:", error);
      // Hide loading overlay
      document.getElementById("loadingOverlay").classList.remove("active");
      alert("දෝෂයක් ඇති විය. කරුණාකර යළි උත්සාහ කරන්න.");
    });

  return false;
}

// Display result data in modal
function showResultData(data) {
  const modal = document.getElementById("resultModal");
  const modalContainer = document.getElementById("resultModalContainer");
  const modalIcon = document.getElementById("modalIcon");
  const modalTitle = document.getElementById("modalTitle");
  const modalMessage = document.getElementById("modalMessage");
  const userAnswer = document.getElementById("userAnswer");
  const correctAnswer = document.getElementById("correctAnswer");
  const resultImageContainer = document.getElementById("resultImageContainer");
  const resultImage = document.getElementById("resultImage");

  // Set whether answer is correct or not
  const isCorrect = data.is_correct;

  if (isCorrect) {
    // Reset classes and add correct class
    modalContainer.className = "result-modal correct-modal";

    // Set correct icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
    </svg>
    `;

    // Set correct title with emoji
    modalTitle.innerHTML = "🌟 නිවැරදියි! 🌟";

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
    modalContainer.className = "result-modal wrong-modal";

    // Set incorrect icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
    </svg>
    `;

    // Set incorrect title with emoji
    modalTitle.innerHTML = "😕 අපොයි! 😕";

    // Set incorrect message with emojis
    modalMessage.innerHTML = `
    <span class="emoji">🤔</span>
    <span>ඔබේ පිළිතුර වැරදියි. අපි නැවත වරක් උත්සාහ කරමු!</span>
    <span class="emoji">📚</span>
    `;
  }

  // Display detected and correct values
  userAnswer.textContent = data.detected_value || "හඳුනාගත නොහැක";
  correctAnswer.textContent = data.expected_value;

  // Show annotated image if available
  if (data.annotated_image_url) {
    resultImageContainer.style.display = "block";
    resultImage.src = data.annotated_image_url;
  } else {
    resultImageContainer.style.display = "none";
  }

  // Show modal
  modal.classList.add("active");
}

// Close modal and go to next question
function closeModalAndContinue() {
  const modal = document.getElementById("resultModal");

  // Get the redirect URL from the modal's data attribute
  const redirectUrl = modal.getAttribute("data-redirect-url");

  modal.classList.remove("active");

  // Clear any confetti
  document.querySelectorAll(".confetti").forEach((el) => el.remove());

  // Redirect to the next question if URL is available
  if (redirectUrl) {
    window.location.href = redirectUrl;
  } else {
    // Fallback to play route if no redirect URL is provided
    window.location.href = "/kinesthetic/play";
  }
}

function captureImage(webcamId, subQuestionId, index) {
  console.log(`Capturing from ${webcamId} for sub-question ${subQuestionId}`);
  const video = document.getElementById(webcamId);
  const canvas = document.createElement("canvas");
  const mirrorBtn = document.getElementById("mirror" + index);
  const isMirrored =
    mirrorBtn && mirrorBtn.getAttribute("data-mirrored") === "true";

  // Set canvas dimensions to match video
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");

  // If mirrored, flip the image horizontally
  if (isMirrored) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }

  // Draw video frame to canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Get base64 image data
  const imageData = canvas.toDataURL("image/jpeg");

  // Add hidden input field with the captured image data
  const form = document.querySelector(".all-answers-form");
  const hiddenInput = document.createElement("input");
  hiddenInput.type = "hidden";
  hiddenInput.name = `captured_image_${subQuestionId}`;
  hiddenInput.value = imageData;
  hiddenInput.id = `captured_image_input_${subQuestionId}`;
  form.appendChild(hiddenInput);

  // Show thumbnail of captured image
  const thumbnailContainer = document.createElement("div");
  thumbnailContainer.id = `thumbnail_container_${subQuestionId}`;
  thumbnailContainer.classList.add("capture-thumbnail-container");
  thumbnailContainer.innerHTML = `
      <div class="capture-thumbnail">
        <img src="${imageData}" alt="Captured image">
        <div class="capture-success">
          <i class="fa fa-check-circle"></i>
          <span>ඡායාරූපය ගනු ලැබීය</span>
        </div>
      </div>
      <div class="text-center mt-3">
        <button type="button" class="btn btn-warning" onclick="clearCapturedImage('${subQuestionId}', ${index})">
          <i class="fa fa-trash"></i> පිංතූරය මකන්න
        </button>
      </div>
    `;

  // Add the thumbnail after the video
  const webcamContainer = video.closest(".webcam-container");
  webcamContainer.appendChild(thumbnailContainer);

  // Disable the capture button
  const captureBtn = document.getElementById("capture" + index);
  captureBtn.disabled = true;
  captureBtn.innerHTML = '<i class="fa fa-check"></i> ඡායාරූපය ලබා ගන්නා ලදී';

  // Update completion indicator
  const completionIndicator = document.getElementById(
    "completionIndicator" + index
  );
  completionIndicator.innerHTML =
    '<i class="fa fa-check-circle completion-done"></i>';

  // Update captured answers count and check if all are done
  updateCapturedAnswersCount();
}

// New function to clear captured image
function clearCapturedImage(subQuestionId, index) {
  // Remove the hidden input with the image data
  const hiddenInput = document.getElementById(
    `captured_image_input_${subQuestionId}`
  );
  if (hiddenInput) {
    hiddenInput.remove();
  }

  // Remove the thumbnail container
  const thumbnailContainer = document.getElementById(
    `thumbnail_container_${subQuestionId}`
  );
  if (thumbnailContainer) {
    thumbnailContainer.remove();
  }

  // Re-enable the capture button
  const captureBtn = document.getElementById("capture" + index);
  if (captureBtn) {
    captureBtn.disabled = false;
    captureBtn.innerHTML = '<i class="fa fa-camera"></i> පිංතූරය ගන්න';
  }

  // Reset completion indicator
  const completionIndicator = document.getElementById(
    "completionIndicator" + index
  );
  if (completionIndicator) {
    completionIndicator.innerHTML = '<i class="fa fa-circle-o"></i>';
  }

  // Update captured answers count
  updateCapturedAnswersCount();
}

// Function to update captured answers count
function updateCapturedAnswersCount() {
  const form = document.querySelector(".all-answers-form");
  const subQuestionIds = form.querySelectorAll(
    'input[name="sub_question_ids"]'
  );
  const capturedImages = Array.from(form.elements).filter((el) =>
    el.name.startsWith("captured_image_")
  );
  const capturedCount = capturedImages.length;
  const totalSubQuestions = subQuestionIds.length;

  document.getElementById("capturedAnswersCount").textContent = capturedCount;

  // Enable/disable the submit all button based on whether all answers are captured
  const submitAllBtn = document.getElementById("submitAllBtn");
  if (capturedCount >= totalSubQuestions) {
    submitAllBtn.disabled = false;
    submitAllBtn.classList.add("ready");
  } else {
    submitAllBtn.disabled = true;
    submitAllBtn.classList.remove("ready");
  }
}

// AJAX submission for all answers
function submitAllAnswers(form, event) {
  event.preventDefault();

  // Check if all sub-questions have been answered
  const formData = new FormData(form);
  const subQuestionIds = formData.getAll("sub_question_ids");
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
    alert("කරුණාකර සියලුම අනු ප්‍රශ්න සඳහා පිංතූර ගන්න.");
    return false;
  }

  // Show loading overlay
  document.getElementById("loadingOverlay").classList.add("active");

  // Submit the form via AJAX
  fetch(form.action, {
    method: "POST",
    body: formData,
    headers: {
      "X-Requested-With": "XMLHttpRequest",
    },
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    })
    .then((data) => {
      // Hide loading overlay
      document.getElementById("loadingOverlay").classList.remove("active");

      // Display results in modal
      showAllResults(data);
    })
    .catch((error) => {
      console.error("Error:", error);
      // Hide loading overlay
      document.getElementById("loadingOverlay").classList.remove("active");
      alert("දෝෂයක් ඇති විය. කරුණාකර යළි උත්සාහ කරන්න.");
    });

  return false;
}

// Function to display all results at once
function showAllResults(data) {
  const modal = document.getElementById("resultModal");
  const modalContainer = document.getElementById("resultModalContainer");
  const modalIcon = document.getElementById("modalIcon");
  const modalTitle = document.getElementById("modalTitle");
  const detailedResults = document.getElementById("detailedResults");
  const correctAnswersCount = document.getElementById("correctAnswersCount");
  const wrongAnswersCount = document.getElementById("wrongAnswersCount");

  // Store the redirect URL for use when closing the modal
  modal.setAttribute("data-redirect-url", data.redirect_url || "");

  // Count correct and wrong answers
  const totalCorrect = data.results.filter((r) => r.is_correct).length;
  const totalWrong = data.results.length - totalCorrect;

  correctAnswersCount.textContent = totalCorrect;
  wrongAnswersCount.textContent = totalWrong;

  // Set overall status (correct if more than 60% correct)
  const overallCorrect = totalCorrect >= Math.ceil(data.results.length * 0.6);

  if (overallCorrect) {
    // Reset classes and add correct class
    modalContainer.className = "result-modal correct-modal";

    // Set correct icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
    </svg>
    `;

    // Set correct title with emoji
    modalTitle.innerHTML = "🌟 සාර්ථකයි! 🌟";

    // Create confetti celebration
    createConfetti();
  } else {
    // Reset classes and add wrong class
    modalContainer.className = "result-modal wrong-modal";

    // Set incorrect icon
    modalIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
    </svg>
    `;

    // Set incorrect title with emoji
    modalTitle.innerHTML = "😕 වැඩිදියුණු විය යුතුයි 😕";
  }

  // Clear previous results
  detailedResults.innerHTML = "";

  // Add detailed results for each sub-question
  data.results.forEach((result, index) => {
    const resultItem = document.createElement("div");
    resultItem.classList.add("result-item");
    resultItem.classList.add(result.is_correct ? "correct" : "wrong");

    // Format the detected value display - for time questions within tolerance
    let detectedValueDisplay = result.detected_value || "හඳුනාගත නොහැක";
    let additionalInfo = "";

    // Check if it's a time-based question (contains ':' in expected or detected value)
    if (
      result.is_correct &&
      (String(result.expected_value).includes(":") ||
        String(result.detected_value).includes(":"))
    ) {
      if (result.detected_value !== result.expected_value) {
        // Within tolerance case
        additionalInfo = `<div class="time-tolerance-info">(±3 මිනිත්තු ඉවසීම තුළ නිවැරදියි)</div>`;
      }
    }

    resultItem.innerHTML = `
    <div class="result-header">
        <div class="result-icon">
        ${
          result.is_correct
            ? '<i class="fa fa-check-circle"></i>'
            : '<i class="fa fa-times-circle"></i>'
        }
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
        ${
          result.annotated_image_url
            ? `
        <div class="result-image-small">
            <img src="${result.annotated_image_url}" alt="Analyzed image">
        </div>
        `
            : ""
        }
    </div>
    `;

    detailedResults.appendChild(resultItem);
  });

  // Show modal
  modal.classList.add("active");
}

// Add function to toggle mirror mode
function toggleMirror(index) {
  const video = document.getElementById("webcam" + index);
  const mirrorBtn = document.getElementById("mirror" + index);

  if (video.classList.contains("mirrored")) {
    video.classList.remove("mirrored");
    mirrorBtn.innerHTML = '<i class="fa fa-exchange"></i> පිංතූරය පෙරළන්න';
    mirrorBtn.setAttribute("data-mirrored", "false");
  } else {
    video.classList.add("mirrored");
    mirrorBtn.innerHTML = '<i class="fa fa-exchange"></i> සාමාන්ය පිංතූරය';
    mirrorBtn.setAttribute("data-mirrored", "true");
  }
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", function () {
  // ...existing code...
  updateCapturedAnswersCount();
});
