# Expense Analyzer
Python tools to analyze my expenses - *In development*  
Read the article [here](http://sds-dubois.github.io/2015/12/20/Analyzing-Daily-Expenses.html) for an introduction to the project.  

----

The analysis is done in the IPython Notebook `Viz`.  
In the same directory, there must be a file named `config.py` which defines the variables:
- `filenames` : a list of paths to the CSV files containing the data
- `bank_names` : either None or a list of the bank names, in the same order as in `filenames`  

The initial file can be an Excel file as below, where each entry should have an amount (`Montant`), and can have some labels (`Etiquettes`) and comments (`Commentaires`). Labels and comments are analyzed to form keywords clusters.
![Fig0](Figures/file_example.png?raw=true)  

### Some results
- A simple pie chart showing the average monthly expenses by cluster
![Fig1](Figures/pie_chart.png?raw=true)
- Graph showing the links between the keywords, with clusters shown through the nodes' color
![Fig2](Figures/keywords_graph.png?raw=true)

### Dependencies:
- Pandas
- Scikit-Learn
- Networkx
