document.addEventListener("DOMContentLoaded", function () {
  // Show a loading indicator while the page is being rendered
  const loadingIndicator = document.createElement("div");
  loadingIndicator.className = "loading-overlay";
  loadingIndicator.innerHTML =
    '<div class="spinner-border text-primary" role="status"><span class="sr-only">Loading...</span></div>';

  // Add inline styles
  loadingIndicator.style.position = "fixed";
  loadingIndicator.style.top = "0";
  loadingIndicator.style.left = "0";
  loadingIndicator.style.width = "100%";
  loadingIndicator.style.height = "100%";
  loadingIndicator.style.backgroundColor = "rgba(255,255,255,0.7)";
  loadingIndicator.style.display = "flex";
  loadingIndicator.style.justifyContent = "center";
  loadingIndicator.style.alignItems = "center";
  loadingIndicator.style.zIndex = "9999";

  // Lazy load non-critical elements
  const lazyElements = document.querySelectorAll(".lazy-load");
  lazyElements.forEach((element) => {
    element.style.visibility = "hidden";
  });

  // Remove the loading overlay once the page is fully loaded
  window.addEventListener("load", function () {
    document.body.removeChild(loadingIndicator);

    // Show lazy loaded elements
    lazyElements.forEach((element, index) => {
      setTimeout(() => {
        element.style.visibility = "visible";
      }, index * 50); // Stagger the appearance of elements
    });
  });

  // Add it to the body
  document.body.appendChild(loadingIndicator);

  // Optimize progress bar calculations
  const progressBars = document.querySelectorAll(".progress-bar");
  progressBars.forEach((bar) => {
    const width = bar.style.width;
    if (width) {
      // Round the percentage to avoid sub-pixel calculations
      const percentage = parseFloat(width);
      if (!isNaN(percentage)) {
        bar.style.width = `${Math.round(percentage)}%`;
      }
    }
  });
});
