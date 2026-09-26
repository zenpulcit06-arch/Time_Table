# Time_Table

## Data Extraction:
### Hand out
since handouts are very diffrent and vary from prof to prof so i used local llm qwen to extract data (mainly due to time constrained and amount of thing i need to change)

## How To Run

### for handout extraction:

**pre requsite** : olama  : qwen2.5:7b
**to install other dependecies**
```Bash
    pip install ollama pdfplumber
```
**Run**

```Bash
    python3 src/handout_parse.py
```


## Data stored

data is stored in 
```
    data/output
```
