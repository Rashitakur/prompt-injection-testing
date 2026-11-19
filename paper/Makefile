main.pdf: main.tex references.bib $(wildcard generated/*.tex) $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode main.tex
clean:
	latexmk -C
