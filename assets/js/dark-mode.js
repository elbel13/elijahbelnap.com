(function () {
  'use strict';

  var STORAGE_KEY = 'theme';
  var DARK  = 'dark';
  var LIGHT = 'light';

  function getPreferred() {
    var stored = localStorage.getItem(STORAGE_KEY);
    if (stored === DARK || stored === LIGHT) return stored;
    // Fall back to OS preference; CSS handles the rest if no attribute is set
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? DARK : LIGHT;
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
  }

  function toggleTheme() {
    var current = document.documentElement.getAttribute('data-theme');
    var next = current === DARK ? LIGHT : DARK;
    localStorage.setItem(STORAGE_KEY, next);
    applyTheme(next);
  }

  // Apply theme immediately (before paint) to avoid flash
  applyTheme(getPreferred());

  // Wire up toggle button via event delegation so it works regardless of
  // when this script executes relative to DOMContentLoaded.
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('#dm-toggle');
    if (btn) toggleTheme();
  });
})();
