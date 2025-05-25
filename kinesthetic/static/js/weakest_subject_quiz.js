// Additional JavaScript for weakest_subject_quiz.html

// Set the X-Requested-With header for all AJAX requests
$(document).ready(function () {
  $.ajaxSetup({
    beforeSend: function (xhr) {
      xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");
    },
  });
});

// Override the form submission to include the header
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

  // Submit the form via AJAX with the X-Requested-With header
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
