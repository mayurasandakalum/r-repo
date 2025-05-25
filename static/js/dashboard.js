// Initialize datepickers
$(function () {
  $(".datepicker").datepicker({
    dateFormat: "yy-mm-dd",
    changeMonth: true,
    changeYear: true,
    yearRange: "-20:+0", // Allow selecting dates from 20 years ago to today
  });
});

// Fetch and display app status
function updateStatus() {
  fetch("/api/status")
    .then((response) => response.json())
    .then((data) => {
      let statusHtml = "<ul>";
      for (const [app, status] of Object.entries(data)) {
        statusHtml += `<li>${app}: <span style="color: ${
          status === "running" ? "green" : "red"
        }">${status}</span></li>`;
      }
      statusHtml += "</ul>";
      document.getElementById("status").innerHTML = statusHtml;
    })
    .catch((error) => {
      document.getElementById("status").innerHTML =
        "Error checking system status";
    });
}

// Check status on page load and every 10 seconds
updateStatus();
setInterval(updateStatus, 10000);

// Navigation functionality - consolidated version
document.addEventListener("DOMContentLoaded", function () {
  const navLinks = document.querySelectorAll(".nav-link");
  const sections = document.querySelectorAll(".dashboard-section");

  // Function to activate a specific tab
  function activateTab(tabId) {
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
        activateTab(tabId);

        // Update URL fragment without reloading the page
        history.pushState(null, null, `#${tabId}`);
      }
    });
  });

  // Handle initial load with URL hash
  if (window.location.hash) {
    const tabId = window.location.hash.substring(1);
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

  // Add Student Modal functionality
  const addStudentModal = document.getElementById("addStudentModal");
  const openAddStudentBtn = document.getElementById("openAddStudentModal");
  const closeAddModalBtn = document.getElementById("closeAddModal");

  // Open Add Student modal when button is clicked
  if (openAddStudentBtn) {
    openAddStudentBtn.addEventListener("click", function () {
      addStudentModal.style.display = "block";
      // Ensure datepicker is initialized for the birthday field
      $("#student_birthday").datepicker("refresh");
    });
  }

  // Close Add Student modal when close button is clicked
  if (closeAddModalBtn) {
    closeAddModalBtn.addEventListener("click", function () {
      addStudentModal.style.display = "none";
    });
  }

  // Modal functionality for Edit Student
  const editModal = document.getElementById("editStudentModal");
  const closeBtn = document.querySelector("#editStudentModal .close-btn");
  const editForm = document.getElementById("editStudentForm");

  // Close the edit modal when clicking the close button
  if (closeBtn) {
    closeBtn.addEventListener("click", function () {
      editModal.style.display = "none";
    });
  }

  // Close modals when clicking outside of them
  window.addEventListener("click", function (event) {
    if (event.target === addStudentModal) {
      addStudentModal.style.display = "none";
    }
    if (event.target === editModal) {
      editModal.style.display = "none";
    }
  });

  // Handle the edit form submission
  if (editForm) {
    editForm.addEventListener("submit", function (e) {
      e.preventDefault();

      const studentId = document.getElementById("edit_student_id").value;
      const studentName = document.getElementById("edit_student_name").value;
      const studentEmail = document.getElementById("edit_student_email").value;
      const studentGender = document.getElementById(
        "edit_student_gender"
      ).value;
      const studentBirthday = document.getElementById(
        "edit_student_birthday"
      ).value;
      const studentGrade = document.getElementById("edit_student_grade").value;
      const csrfToken = document.querySelector(
        'input[name="csrf_token"]'
      ).value;

      // Create FormData object and append all fields, even if empty
      const formData = new FormData();
      formData.append("student_name", studentName);
      formData.append("student_email", studentEmail);
      formData.append("student_gender", studentGender);
      formData.append("student_birthday", studentBirthday);
      formData.append("student_grade", studentGrade);
      formData.append("csrf_token", csrfToken); // Add CSRF token to FormData

      // Debug - optional, remove in production
      console.log("Updating student with:");
      console.log("Gender:", studentGender);
      console.log("Birthday:", studentBirthday);
      console.log("Grade:", studentGrade);

      fetch(`/edit_student/${studentId}`, {
        method: "POST",
        body: formData,
        headers: {
          "X-Requested-With": "XMLHttpRequest", // Add this header for AJAX requests
        },
      })
        .then((response) => {
          if (response.ok) {
            return response.json().then((data) => {
              // Show success message
              alert(data.message || "Student updated successfully");
              // Reload the page to show updated data
              window.location.reload();
            });
          } else {
            return response.json().then((data) => {
              throw new Error(data.error || "Failed to update student");
            });
          }
        })
        .catch((error) => {
          alert("Error: " + error.message);
          console.error("Error:", error);
        });
    });
  }
});

// Function to open the edit modal
function openEditModal(id, name, email, gender, birthday, grade) {
  const modal = document.getElementById("editStudentModal");
  document.getElementById("edit_student_id").value = id;
  document.getElementById("edit_student_name").value = name;
  document.getElementById("edit_student_email").value = email;

  // Fix handling of gender, birthday and grade fields
  const genderField = document.getElementById("edit_student_gender");
  genderField.value = gender || ""; // Set empty string if undefined

  const birthdayField = document.getElementById("edit_student_birthday");
  birthdayField.value = birthday || ""; // Set empty string if undefined

  const gradeField = document.getElementById("edit_student_grade");
  gradeField.value = grade || ""; // Set empty string if undefined

  // Refresh/initialize the datepicker
  if ($(birthdayField).hasClass("hasDatepicker")) {
    $(birthdayField).datepicker("setDate", birthday);
  }

  modal.style.display = "block";
}
