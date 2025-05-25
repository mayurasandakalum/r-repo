document.addEventListener("DOMContentLoaded", function () {
  // Add highlighting effects for second quiz section
  const quizButton = document.querySelector(
    '.btn-warning[href*="weakest_subject_quiz"]'
  );
  if (quizButton) {
    quizButton.addEventListener("mouseenter", function () {
      this.innerHTML =
        '<i class="fa fa-graduation-cap"></i> ප්‍රශ්න 5 සමඟ ආරම්භ කරන්න';
    });

    quizButton.addEventListener("mouseleave", function () {
      this.innerHTML =
        '<i class="fa fa-graduation-cap"></i> ප්‍රශ්නාවලිය ආරම්භ කරන්න';
    });
  }

  // Add badge showing '5 questions' in the comparison card if it exists
  const comparisonCard = document.querySelector(".comparison-stats");
  if (comparisonCard) {
    const secondQuizTitle = comparisonCard.querySelector(
      ".col-md-6:last-child .card-body h5"
    );
    if (secondQuizTitle) {
      secondQuizTitle.innerHTML +=
        ' <span class="badge badge-pill badge-info">ප්‍රශ්න 5</span>';
    }
  }

  // Fix alignment issues in the subject progress section
  const subjectItems = document.querySelectorAll(".subject-item");
  subjectItems.forEach((item) => {
    const progressBar = item.querySelector(".progress-bar");
    if (progressBar) {
      // Ensure proper width calculation
      const percentage = progressBar.style.width;
      if (percentage) {
        progressBar.style.width = percentage;
      }
    }
  });

  // Fix height consistency in comparison cards
  const fixComparisonCardHeights = () => {
    const cards = document.querySelectorAll(".comparison-stats .card");
    if (cards.length >= 2) {
      let maxHeight = 0;
      cards.forEach((card) => {
        card.style.height = "auto";
        maxHeight = Math.max(maxHeight, card.offsetHeight);
      });
      cards.forEach((card) => {
        card.style.height = `${maxHeight}px`;
      });
    }
  };

  // Run once on load and again after a slight delay to ensure content is rendered
  fixComparisonCardHeights();
  setTimeout(fixComparisonCardHeights, 100);

  // Re-run when window is resized
  window.addEventListener("resize", fixComparisonCardHeights);
});
