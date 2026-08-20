/**
 * SAVIS Project Page JavaScript
 * Handles tabs, code copying, image modal viewer, theme toggling, and scroll navigation
 */

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initNavScroll();
  initTabs();
  initCodeBlocks();
  initCopyButtons();
  initImageLightbox();
});

/* --------------------------------------------------------------------------
   Theme Toggle (Light / Dark Mode)
   -------------------------------------------------------------------------- */
function initThemeToggle() {
  const toggleBtn = document.getElementById('theme-toggle-btn');
  if (!toggleBtn) return;

  const currentTheme = localStorage.getItem('savis-theme') || 'light';
  document.documentElement.setAttribute('data-theme', currentTheme);
  updateThemeIcon(toggleBtn, currentTheme);

  toggleBtn.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const nextTheme = isDark ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nextTheme);
    localStorage.setItem('savis-theme', nextTheme);
    updateThemeIcon(toggleBtn, nextTheme);
  });
}

function updateThemeIcon(btn, theme) {
  if (theme === 'dark') {
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;
    btn.setAttribute('title', 'Switch to Light Mode');
    btn.setAttribute('aria-label', 'Switch to Light Mode');
  } else {
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
    btn.setAttribute('title', 'Switch to Dark Mode');
    btn.setAttribute('aria-label', 'Switch to Dark Mode');
  }
}

/* --------------------------------------------------------------------------
   Navigation & Active Link Spy
   -------------------------------------------------------------------------- */
function initNavScroll() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a');

  window.addEventListener('scroll', () => {
    let current = '';
    const scrollPos = window.pageYOffset + 120;

    sections.forEach(section => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  }, { passive: true });
}

/* --------------------------------------------------------------------------
   Generic Tabs (Qualitative Results, etc.)
   -------------------------------------------------------------------------- */
function initTabs() {
  const tabContainers = document.querySelectorAll('.tab-container');

  tabContainers.forEach(container => {
    const buttons = container.querySelectorAll('.tab-btn, .qual-tab-btn');
    const panes = container.querySelectorAll('.tab-pane, .qual-pane-content');

    buttons.forEach(button => {
      button.addEventListener('click', () => {
        const targetId = button.getAttribute('data-tab');

        buttons.forEach(btn => btn.classList.remove('active'));
        panes.forEach(pane => pane.classList.remove('active'));

        button.classList.add('active');
        const targetPane = container.querySelector(`#${targetId}`);
        if (targetPane) {
          targetPane.classList.add('active');
        }
      });
    });
  });
}

/* --------------------------------------------------------------------------
   Code Block Tabs
   -------------------------------------------------------------------------- */
function initCodeBlocks() {
  const codeNavs = document.querySelectorAll('.code-nav');

  codeNavs.forEach(nav => {
    const buttons = nav.querySelectorAll('.code-tab-btn');
    const parentContainer = nav.closest('.code-container');
    const blocks = parentContainer.querySelectorAll('.code-block');

    buttons.forEach(button => {
      button.addEventListener('click', () => {
        const targetCodeId = button.getAttribute('data-code');

        buttons.forEach(btn => btn.classList.remove('active'));
        blocks.forEach(block => block.classList.remove('active'));

        button.classList.add('active');
        const targetBlock = parentContainer.querySelector(`#${targetCodeId}`);
        if (targetBlock) {
          targetBlock.classList.add('active');
        }
      });
    });
  });
}

/* --------------------------------------------------------------------------
   Copy Buttons (Code & BibTeX)
   -------------------------------------------------------------------------- */
function initCopyButtons() {
  const copyButtons = document.querySelectorAll('.copy-btn');

  copyButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      let textToCopy = '';
      const targetSelector = btn.getAttribute('data-target');

      if (targetSelector) {
        const targetEl = document.querySelector(targetSelector);
        if (targetEl) {
          textToCopy = targetEl.innerText || targetEl.textContent;
        }
      } else {
        const parentBlock = btn.closest('.code-container');
        if (parentBlock) {
          const activeBlock = parentBlock.querySelector('.code-block.active') || parentBlock.querySelector('.code-block');
          if (activeBlock) {
            textToCopy = activeBlock.innerText || activeBlock.textContent;
          }
        }
      }

      if (!textToCopy) return;

      try {
        await navigator.clipboard.writeText(textToCopy.trim());
        const originalHtml = btn.innerHTML;
        btn.classList.add('copied');
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;

        setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = originalHtml;
        }, 2000);
      } catch (err) {
        console.error('Failed to copy text: ', err);
      }
    });
  });
}

/* --------------------------------------------------------------------------
   Image Lightbox Viewer (Click to Zoom)
   -------------------------------------------------------------------------- */
function initImageLightbox() {
  const zoomableImages = document.querySelectorAll('.zoomable-img, .figure-wrapper img, .arch-figure-img, .qual-hero-img-box img');
  const lightbox = document.getElementById('lightbox-modal');
  const lightboxImg = document.getElementById('lightbox-img');
  const closeBtn = document.getElementById('lightbox-close-btn');

  if (!lightbox || !lightboxImg) return;

  zoomableImages.forEach(img => {
    img.addEventListener('click', () => {
      lightboxImg.src = img.getAttribute('data-zoom-src') || img.src;
      lightboxImg.alt = img.alt || 'Zoomed Figure';
      lightbox.classList.add('active');
      document.body.style.overflow = 'hidden';
    });
  });

  function closeLightbox() {
    lightbox.classList.remove('active');
    document.body.style.overflow = '';
  }

  if (closeBtn) {
    closeBtn.addEventListener('click', closeLightbox);
  }

  lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox || e.target === closeBtn) {
      closeLightbox();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox.classList.contains('active')) {
      closeLightbox();
    }
  });
}
