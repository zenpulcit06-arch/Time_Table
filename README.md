# Time_Table

## Data Extraction:
### Hand out
since handouts are very diffrent and vary from prof to prof so i used local llm qwen to extract data (mainly due to time constrained and amount of thing i need to change)
### timetable 
timetable is in tabular form so i make a look up table for timming and going throw rows to find all lecture ,tut,
lab and putting in json 

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
### for Time table extraction:
```Bash
    python3 src/timetable_parse.py
```


## Data stored

data is stored in 
```
    data/output
    
    #timetable
    data/output/timetable
```
