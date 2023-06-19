.PHONY: all clean

all: intro_logistic_regression.pdf

intro_logistic_regression.pdf: intro_logistic_regression.tex images/*.pdf images/*.mov images/*.mp4
	latexmk --shell-escape -pdflatex="lualatex -interaction=nonstopmode" -pdf intro_logistic_regression.tex

clean:
	rm -f *.log *.aux *.out *.toc *.lof *.lot *.pdf
