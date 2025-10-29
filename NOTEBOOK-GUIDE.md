# Jupyter Notebooks Guide - Interactive DSA & ML Coding Practice

## What You Have Now

✅ **dsa-day1-arrays-searching.ipynb** - Fully interactive Day 1!

## How to Create Remaining Notebooks

I've created Day 1 as an example. You can either:

### Option 1: I Create Them (Recommended)
Run this command in your terminal to let me know, and I'll create all remaining notebooks in our next session:

```bash
# You can ask me to create the remaining 6 notebooks
```

### Option 2: Auto-Convert Markdown to Notebooks
You can convert the markdown files to notebooks yourself using this Python script:

```python
# Save this as convert_md_to_nb.py
import nbformat as nbf
import re

def md_to_notebook(md_file, nb_file):
    """Convert markdown file to Jupyter notebook"""

    # Read markdown file
    with open(md_file, 'r') as f:
        content = f.read()

    # Create new notebook
    nb = nbf.v4.new_notebook()

    # Split by code blocks
    parts = re.split(r'```python\n(.*?)\n```', content, flags=re.DOTALL)

    cells = []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Markdown
            if part.strip():
                cells.append(nbf.v4.new_markdown_cell(part.strip()))
        else:  # Code
            cells.append(nbf.v4.new_code_cell(part))

    nb['cells'] = cells

    # Write notebook
    with open(nb_file, 'w') as f:
        nbf.write(nb, f)

    print(f"Created {nb_file}")

# Convert all DSA files
files_to_convert = [
    ('dsa-day2-sorting.md', 'dsa-day2-sorting.ipynb'),
    ('dsa-day3-two-pointers.md', 'dsa-day3-two-pointers.ipynb'),
    ('dsa-day4-recursion.md', 'dsa-day4-recursion.ipynb'),
    ('dsa-day5-hashmaps.md', 'dsa-day5-hashmaps.ipynb'),
    ('dsa-day6-practice.md', 'dsa-day6-practice.ipynb'),
    ('dsa-bonus-patterns.md', 'dsa-bonus-patterns.ipynb'),
    ('ml-coding-problems.md', 'ml-coding-problems.ipynb'),
]

for md_file, nb_file in files_to_convert:
    try:
        md_to_notebook(md_file, nb_file)
    except Exception as e:
        print(f"Error converting {md_file}: {e}")
```

Then run:
```bash
python convert_md_to_nb.py
```

### Option 3: Use VSCode/JupyterLab
1. Open the markdown files in VSCode
2. Copy code blocks into Jupyter notebook cells manually
3. Add markdown cells for explanations

## Recommended Notebooks to Create

### Priority 1 (Must Have):
1. ✅ **dsa-day1-arrays-searching.ipynb** (DONE!)
2. **dsa-day2-sorting.ipynb** - Merge sort is critical!
3. **dsa-day3-two-pointers.ipynb** - Very common pattern
4. **ml-coding-problems.ipynb** - For Sindhuja's round

### Priority 2 (Should Have):
5. **dsa-day5-hashmaps.ipynb** - Frequency problems
6. **dsa-bonus-patterns.ipynb** - Sliding window advanced

### Priority 3 (Nice to Have):
7. **dsa-day4-recursion.ipynb**
8. **dsa-day6-practice.ipynb**

## How to Use Jupyter Notebooks

### Starting Jupyter
```bash
cd /home/spurge/cisco
jupyter notebook
# or
jupyter lab
```

### Workflow
1. **Open notebook** - Click on .ipynb file
2. **Read explanation** - Markdown cells
3. **Try coding** - Fill in TODOs in code cells
4. **Run cell** - Shift+Enter
5. **See solution** - Run solution cell
6. **Experiment** - Modify and re-run

### Keyboard Shortcuts
- **Shift+Enter** - Run cell and move to next
- **Ctrl+Enter** - Run cell and stay
- **A** - Insert cell above
- **B** - Insert cell below
- **DD** - Delete cell
- **M** - Convert to markdown
- **Y** - Convert to code

## What Day 1 Notebook Includes

✅ Interactive code cells
✅ Fill-in-the-blank exercises
✅ Solution cells (run to reveal)
✅ Test cases with expected outputs
✅ Trace/visualization functions
✅ Additional practice exercises

## Benefits of Notebooks

1. **Interactive Learning**
   - Write code directly
   - Run and see results immediately
   - Experiment freely

2. **Immediate Feedback**
   - Test cases run instantly
   - See what works, what doesn't
   - Debug efficiently

3. **Save Your Work**
   - Your solutions are saved
   - Can review later
   - Track your progress

4. **Visual Learning**
   - See algorithm traces
   - Visualize data structures
   - Understand step-by-step

## Sample Workflow for Day 1

### Step 1: Open Notebook
```bash
cd /home/spurge/cisco
jupyter notebook dsa-day1-arrays-searching.ipynb
```

### Step 2: Work Through Problems

**For each problem:**
1. Read the markdown explanation
2. Try to solve in the "Your Turn" cell
3. Run your code (Shift+Enter)
4. Check test cases
5. If stuck, run the solution cell
6. Understand the solution
7. Re-implement without looking

### Step 3: Practice
- Do additional exercises at the end
- Modify problems (change inputs, add features)
- Time yourself

## Integration with Your Study Schedule

### Oct 15 (Today) - If starting now:
```bash
jupyter notebook dsa-day1-arrays-searching.ipynb
```
Work through Problems 1-2 (Linear Search, Binary Search)

### Oct 16 - Day 2:
Create/use Day 2 notebook for Merge Sort

### Oct 17 - Day 3:
Create/use Day 3 notebook for Two Pointers

### Oct 18-19:
Use remaining notebooks as needed

### Oct 20:
Light review only - don't open new notebooks!

## Alternative: Use Google Colab

If you don't have Jupyter installed locally:

1. Go to https://colab.research.google.com
2. Upload the .ipynb file
3. Work in browser (no installation needed!)
4. Automatically saves to Google Drive

## Quick Start Right Now

### If you have Jupyter installed:
```bash
cd /home/spurge/cisco
jupyter notebook dsa-day1-arrays-searching.ipynb
```

### If you don't have Jupyter:
```bash
# Install Jupyter
pip install jupyter notebook

# Or use conda
conda install jupyter

# Then start
jupyter notebook
```

### Using VSCode:
1. Open VSCode
2. Install "Jupyter" extension
3. Open .ipynb file
4. Start coding!

## Next Steps

**RIGHT NOW:**
1. Open `dsa-day1-arrays-searching.ipynb`
2. Work through Problem 2 (Binary Search)
3. Run the trace function to see how it works
4. Practice until you can write binary search from memory

**TODAY:**
- Complete Day 1 notebook
- Focus on Binary Search (most important!)
- Try the additional exercises

**TOMORROW:**
- Ask me to create remaining notebooks, OR
- Use the conversion script above, OR
- Work from markdown files (they're comprehensive too!)

## Files You Have

### Markdown (Complete Text):
- dsa-day1-arrays-searching.md
- dsa-day2-sorting.md
- dsa-day3-two-pointers.md
- dsa-day4-recursion.md
- dsa-day5-hashmaps.md
- dsa-day6-practice.md
- dsa-bonus-patterns.md
- ml-coding-problems.md

### Jupyter Notebooks (Interactive):
- ✅ dsa-day1-arrays-searching.ipynb
- ⏳ Others to be created

Both formats are valuable:
- **Markdown** - Read and understand concepts
- **Notebooks** - Practice and code

## Tips for Maximum Benefit

1. **Don't just read solutions** - Try first!
2. **Run the code** - See it work
3. **Modify examples** - Experiment
4. **Time yourself** - Build speed
5. **Explain out loud** - Practice for interviews
6. **Review regularly** - Spaced repetition

## Support

If you need help:
1. Ask me to create more notebooks
2. Ask about specific problems
3. Request additional examples
4. Get hints without full solutions

You're set up for success! Start with Day 1 notebook now! 🚀
