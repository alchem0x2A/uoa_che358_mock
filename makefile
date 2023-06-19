.PHONY: all clean

PYTHON := $(shell which python)
PDFS=intro_logistic_regression.pdf \
     scripts/animation.mov scripts/sample_coin_cover.pdf \
     scripts/coin_linear_combined.pdf \
     scripts/sigmoid_funs.pdf  scripts/sigmoid_funs_emphasize_logistic.pdf scripts/logistic_fun_alone.pdf \
     scripts/coin_fit_lr.pdf

all: $(PDFS)



intro_logistic_regression.pdf: intro_logistic_regression.tex *.py images/*.pdf images/*.mov images/*.mp4 scripts/*.mov scripts/*.pdf
	latexmk --shell-escape -pdflatex="lualatex -interaction=nonstopmode" -pdf intro_logistic_regression.tex

scripts/animation.mov: scripts/0_coin_sample.py
	cd "$(CURDIR)/scripts/" && $(PYTHON) $(notdir $<)

scripts/sample_coin_cover.pdf: scripts/0_coin_sample.py
	cd "$(CURDIR)/scripts/" && $(PYTHON) $(notdir $<)

scripts/coin_linear_combined.pdf: scripts/1_linear_fit_coin.py
	cd "$(CURDIR)/scripts/" && $(PYTHON) $(notdir $<)

scripts/sigmoid_funs.pdf: scripts/2_s_shape_functions.py
	cd "$(CURDIR)/scripts/" && $(PYTHON) $(notdir $<)

scripts/sigmoid_funs_emphasize_logistic.pdf: scripts/2_s_shape_functions.py
	cd "$(CURDIR)/scripts/" && $(PYTHON) $(notdir $<)

scripts/logistic_fun_alone.pdf: scripts/2_s_shape_functions.py
	cd "$(CURDIR)/scripts/" && $(PYTHON) $(notdir $<)

scripts/coin_fit_lr.pdf: scripts/5_coin_lr.py
	cd "$(CURDIR)/scripts/" && $(PYTHON) $(notdir $<)

clean:
	rm -f *.log *.aux *.out *.toc *.lof *.lot *.pdf
