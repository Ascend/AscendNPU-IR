document.addEventListener('DOMContentLoaded', function () {
  try {
    // Local docs layout:
    //   _build/en/...     -> English
    //   _build/zh_cn/...  -> Chinese

    // Determine current language and build corresponding links
    var currentPath = window.location.pathname;
    var isEnglish = currentPath.includes('/en/');
    var isChinese = currentPath.includes('/zh_cn/');

    var enHref, zhHref;

    // Build links to the same document in other language
    if (isEnglish) {
      // Current is English, build link to Chinese version
      enHref = currentPath;
      zhHref = currentPath.replace('/en/', '/zh_cn/');
      zhHref = zhHref.replace('.html', '_zh.html');
    } else if (isChinese) {
      // Current is Chinese, build link to English version
      zhHref = currentPath;
      enHref = currentPath.replace('/zh_cn/', '/en/');
      enHref = enHref.replace('_zh.html', '.html');
    } else {
      // Default to home pages
      enHref = window.location.origin + '/en/index.html';
      zhHref = window.location.origin + '/zh_cn/index.html';
    }

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

