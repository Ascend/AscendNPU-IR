document.addEventListener('DOMContentLoaded', function () {
  try {
    // Local docs layout:
    //   _build/en/...     -> English
    //   _build/zh_cn/...  -> Chinese
    // Always jump to the two home pages from any doc page.
    var enHref = '../en/index.html';
    var zhHref = '../zh_cn/index.html';

    // Determine current language
    var currentPath = window.location.pathname;
    var isEnglish = currentPath.includes('/en/');
    var isChinese = currentPath.includes('/zh_cn/');

    var enLinkClass = isEnglish ? 'active' : '';
    var zhLinkClass = isChinese ? 'active' : '';

    var wrapper = document.createElement('div');
    wrapper.className = 'lang-switcher';

    wrapper.innerHTML =
      '<span class="lang-label">Language</span>' +
      '<span class="lang-links">' +
        '<a href="' + enHref + '" class="' + enLinkClass + '">English</a>' +
        '<a href="' + zhHref + '" class="' + zhLinkClass + '">中文</a>' +
      '</span>';

    document.body.appendChild(wrapper);
  } catch (e) {
    // Best-effort only; ignore errors.
  }
});

