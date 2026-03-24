# Minimal makefile for Sphinx documentation
# Default target: clean _build then build English (fresh output every time).

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
	@LANGUAGE=en $(SPHINXBUILD) -b html -c $(SOURCEDIR) -D language=en "$(SOURCEDIR)/source/en" "$(BUILDDIR)/en" $(SPHINXOPTS) $(O)
	@echo "English HTML built in $(BUILDDIR)/en"
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
	@LANGUAGE=zh_CN $(SPHINXBUILD) -b html -c $(SOURCEDIR) -D language=zh_CN "$(SOURCEDIR)/source/zh_cn" "$(BUILDDIR)/zh_cn" $(SPHINXOPTS) $(O)
	@echo "Chinese HTML built in $(BUILDDIR)/zh_cn"

# Build both: English in _build/en, Chinese in _build/zh_cn (for local preview)
html-all: clean html html-zh
	@echo "Both languages built. Serve e.g. _build/en and _build/zh_cn."

.PHONY: all clean help Makefile html html-zh html-all

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
