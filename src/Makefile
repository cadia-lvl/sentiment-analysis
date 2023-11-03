pdf2:
	jupyter nbconvert FinalReport2.ipynb --to pdf --template=hide_header && open FinalReport2.pdf
pdf3:
	pandoc -f markdown-implicit_figures -t pdf FinalReport3.md -o FinalReport3.pdf && open FinalReport3.pdf

tex:
	jupyter nbconvert FinalReport2.ipynb --to latex
