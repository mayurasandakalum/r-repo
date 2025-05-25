/**
 * Student Dashboard JavaScript
 * This file contains functionality specific to the student dashboard.
 */

document.addEventListener("DOMContentLoaded", function () {
  // Tab navigation functionality
  const navLinks = document.querySelectorAll(".nav-link");
  const sections = document.querySelectorAll(".dashboard-section");

  // Function to activate a specific tab
  function activateTab(tabId) {
    console.log("Activating tab:", tabId);

    // Hide all sections
    sections.forEach((section) => {
      section.classList.remove("active");
    });

    // Remove active class from all nav links
    navLinks.forEach((link) => {
      link.classList.remove("active");
    });

    // Show the selected section
    const targetSection = document.getElementById(tabId);
    if (targetSection) {
      targetSection.classList.add("active");
    }

    // Add active class to the clicked nav link
    const activeLink = document.querySelector(`.nav-link[href="#${tabId}"]`);
    if (activeLink) {
      activeLink.classList.add("active");
    }
  }

  // Add click event listeners to all nav links
  navLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      // Only handle internal links (those starting with #)
      if (this.getAttribute("href").startsWith("#")) {
        e.preventDefault();
        const tabId = this.getAttribute("href").substring(1);
        console.log("Nav link clicked:", tabId);
        activateTab(tabId);

        // Update URL fragment without reloading the page
        history.pushState(null, null, `#${tabId}`);
      }
    });
  });

  // Handle initial load with URL hash
  if (window.location.hash) {
    const tabId = window.location.hash.substring(1);
    console.log("Initial hash detected:", tabId);
    activateTab(tabId);
  }

  // Handle back/forward browser navigation
  window.addEventListener("popstate", function () {
    if (window.location.hash) {
      const tabId = window.location.hash.substring(1);
      activateTab(tabId);
    } else {
      // Default to first tab if no hash
      activateTab("overview");
    }
  });

  // Fetch progress data if user ID exists (moved from inline script)
  const progressStatsElement = document.getElementById("progressStats");
  if (progressStatsElement) {
    const userId = progressStatsElement.getAttribute("data-user-id");
    if (userId) {
      fetchUserProgress(userId);
    }
  }

  // Function to fetch user progress data
  function fetchUserProgress(userId) {
    fetch(`/api/user/${userId}/progress`)
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          document.getElementById("progressStats").innerHTML =
            "<p>Could not load progress data.</p>";
          return;
        }

        let statsHTML = `
                    <div class="stat-item">
                        <span class="stat-label">Total Score:</span>
                        <span class="stat-value">${data.total_score || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Questions Completed:</span>
                        <span class="stat-value">${
                          data.questions_completed || 0
                        }/15</span>
                    </div>
                `;

        document.getElementById("progressStats").innerHTML = statsHTML;
      })
      .catch((error) => {
        console.error("Error fetching progress:", error);
        document.getElementById("progressStats").innerHTML =
          "<p>Could not load progress data.</p>";
      });
  }
});
