// This script handles the YouTube video player

// Store the video ID from the data attribute
const videoId = document.getElementById("player").getAttribute("data-video-id");

// Initialize YouTube player variables
let player;
let attempts = 0;
const maxAttempts = 3;

// Function to initialize the player
function initializePlayer() {
  if (typeof YT !== "undefined" && YT.Player) {
    // YouTube API is loaded, create the player
    player = new YT.Player("player", {
      videoId: videoId,
      playerVars: {
        autoplay: 1,
        controls: 1,
        rel: 0,
        showinfo: 0,
        modestbranding: 1,
        iv_load_policy: 3,
      },
      events: {
        onReady: onPlayerReady,
        onStateChange: onPlayerStateChange,
        onError: onPlayerError,
      },
    });
  } else if (attempts < maxAttempts) {
    // API not ready yet, try again after a delay
    attempts++;
    console.log(
      `YouTube API not ready, attempt ${attempts} of ${maxAttempts}...`
    );
    setTimeout(initializePlayer, 1000);
  } else {
    // Failed after several attempts, show error message
    console.error("Could not load YouTube API after multiple attempts");
    document.getElementById("player").innerHTML = `<div class="video-error">
         <p>YouTube වීඩියෝව පූරණය කිරීමට නොහැකි විය.</p>
         <p>YouTube වෙත සෘජුව පිවිසීමට පහත ලින්ක් එක ක්ලික් කරන්න:</p>
         <a href="https://www.youtube.com/watch?v=${videoId}" target="_blank" class="btn btn-danger">
           <i class="fa fa-youtube-play"></i> YouTube හි වීඩියෝව බලන්න
         </a>
       </div>`;
  }
}

// Load the YouTube API
function loadYouTubeAPI() {
  const tag = document.createElement("script");
  tag.src = "https://www.youtube.com/iframe_api";
  const firstScriptTag = document.getElementsByTagName("script")[0];
  firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

  // Set a timeout in case the onYouTubeIframeAPIReady doesn't trigger
  setTimeout(() => {
    if (!window.YT) {
      initializePlayer();
    }
  }, 2000);
}

// This function is called when the YouTube API is ready
function onYouTubeIframeAPIReady() {
  console.log("YouTube API ready");
  initializePlayer();
}

// When the player is ready
function onPlayerReady(event) {
  console.log("Player ready");
  // Force play the video (may be blocked by browser if autoplay is disabled)
  event.target.playVideo();

  // Add a play button overlay in case autoplay is blocked
  const playerElement = document.getElementById("player");
  const playButton = document.createElement("div");
  playButton.className = "play-button-overlay";
  playButton.innerHTML = '<i class="fa fa-play-circle"></i>';
  playButton.addEventListener("click", () => {
    event.target.playVideo();
    playButton.style.display = "none";
  });

  playerElement.parentNode.appendChild(playButton);

  // Hide the overlay if video starts playing
  setTimeout(() => {
    if (event.target.getPlayerState() === YT.PlayerState.PLAYING) {
      playButton.style.display = "none";
    }
  }, 500);
}

// When the player state changes
function onPlayerStateChange(event) {
  if (event.data === YT.PlayerState.ENDED) {
    console.log("Video ended");

    // Get subject from URL path
    const pathParts = window.location.pathname.split("/");
    const subject = pathParts[pathParts.length - 1];

    // Send AJAX request to mark video as watched - fix the URL path
    fetch("/api/video-watched/" + subject, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ method: "automatic" }), // Add information about how it was completed
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok: " + response.status);
        }
        return response.json();
      })
      .then((data) => {
        console.log("Video marked as watched:", data);

        // Update the "Finished Watching" button
        const watchedBtn = document.getElementById("mark-watched-btn");
        if (watchedBtn) {
          watchedBtn.innerHTML = '<i class="fa fa-check"></i> සම්පූර්ණයි!';
          watchedBtn.classList.add("btn-watched");
          watchedBtn.disabled = true;
        }
      })
      .catch((error) => {
        console.error("Error marking video as watched:", error);
      });

    // Show a message when video ends
    const playerContainer =
      document.querySelector(".youtube-player").parentNode;
    const completionMessage = document.createElement("div");
    completionMessage.className = "video-completion-message";
    completionMessage.textContent = "වීඩියෝව අවසන් විය. පරීක්ෂණය ආරම්භ කරන්න!";
    playerContainer.appendChild(completionMessage);

    // Scroll to the "Start Quiz" button
    document
      .querySelector(".video-footer")
      .scrollIntoView({ behavior: "smooth" });
  }
}

// Handle any errors
function onPlayerError(event) {
  console.error("YouTube player error:", event.data);
  // Display error message
  document.getElementById("player").innerHTML = `<div class="video-error">
       <p>වීඩියෝව පූරණය කිරීමේදී දෝෂයක් ඇති විය. (Error code: ${event.data})</p>
       <p>YouTube වෙත සෘජුව පිවිසීමට පහත ලින්ක් එක ක්ලික් කරන්න:</p>
       <a href="https://www.youtube.com/watch?v=${videoId}" target="_blank" class="btn btn-danger">
         <i class="fa fa-youtube-play"></i> YouTube හි වීඩියෝව බලන්න
       </a>
     </div>`;
}

// Function to handle the "Finished Watching" button click
function markVideoAsWatched() {
  // Get subject from URL path
  const pathParts = window.location.pathname.split("/");
  const subject = pathParts[pathParts.length - 1];

  // Get the button element
  const watchedBtn = document.getElementById("mark-watched-btn");

  // Disable button and show loading state
  watchedBtn.disabled = true;
  watchedBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Processing...';

  // Send AJAX request to mark video as watched
  fetch("/api/video-watched/" + subject, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ method: "manual" }), // Adding context for how it was marked
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok: " + response.status);
      }
      return response.json();
    })
    .then((data) => {
      console.log("Video marked as watched:", data);

      // Update the button to show success
      watchedBtn.innerHTML = '<i class="fa fa-check"></i> නරඹා ඇත';
      watchedBtn.classList.add("btn-watched");
      watchedBtn.disabled = true;

      // Add the watched badge to the header if it doesn't exist
      const videoHeader = document.querySelector(".video-header");
      let watchedBadge = document.querySelector(".watched-badge");

      if (!watchedBadge) {
        watchedBadge = document.createElement("div");
        watchedBadge.className = "watched-badge";
        watchedBadge.innerHTML = '<i class="fa fa-check-circle"></i> නරඹා ඇත';
        videoHeader.appendChild(watchedBadge);
      }
    })
    .catch((error) => {
      console.error("Error marking video as watched:", error);

      // Restore button state
      watchedBtn.disabled = false;
      watchedBtn.innerHTML =
        '<i class="fa fa-check-circle"></i> වීඩියෝව බලා අවසන්';

      // Show error
      alert("Error marking video as watched. Please try again.");
    });
}

// Start loading the API when the page loads
document.addEventListener("DOMContentLoaded", loadYouTubeAPI);

// Set up global callback for YouTube API
window.onYouTubeIframeAPIReady = onYouTubeIframeAPIReady;
