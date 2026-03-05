# Minimal makefile for Sphinx documentation
# Dual source i18n: .md = English, *_zh.md = Chinese (see docs/README.md)
# Default target: clean _build then build English (fresh output every time).
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Default: clean and build English (ensures fresh static output on every make -C docs)
all: clean html
	@echo "Done. English HTML in $(BUILDDIR)/html"

# Remove generated static files under _build/
clean:
	@rm -rf "$(BUILDDIR)"
	@echo "Cleaned $(BUILDDIR)/"

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build English docs (conf.py reads READTHEDOCS_LANGUAGE for exclude_patterns)
html:
	@LANGUAGE=en $(SPHINXBUILD) -b html -D language=en -D root_doc=index "$(SOURCEDIR)" "$(BUILDDIR)/en" $(SPHINXOPTS) $(O)
	@echo "English HTML built in $(BUILDDIR)/html"
	@printf '%s\n' '<!DOCTYPE html>' \
		'<html>' \
		'<head>' \
		'  <meta charset="utf-8"/>' \
		'  <meta http-equiv="refresh" content="0; url=en/index.html"/>' \
		'  <title>AscendNPU IR Docs</title>' \
		'</head>' \
		'<body>' \
		'  <p>Redirecting to <a href="en/index.html">English documentation</a>...</p>' \
		'</body>' \
		'</html>' > "$(BUILDDIR)/index.html"
	@echo "Root $(BUILDDIR)/index.html created (default → English)"

# Build Chinese docs (conf.py writes index.html redirect via build-finished for RTD + local)
html-zh:
	@LANGUAGE=zh_CN $(SPHINXBUILD) -b html -D language=zh_CN -D root_doc=index_zh "$(SOURCEDIR)" "$(BUILDDIR)/zh_cn" $(SPHINXOPTS) $(O)
	@echo "Chinese HTML built in $(BUILDDIR)/zh_cn"

# Build both: English in _build/html, Chinese in _build/html_zh (for local preview)
html-all: clean html html-zh
	@echo "Both languages built. Serve e.g. _build/en and _build/zh_cn."

.PHONY: all clean help Makefile html html-zh html-all

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
