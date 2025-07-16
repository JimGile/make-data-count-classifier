Yes, Jim, absolutely! Creating a dedicated Python source file (`.py`) for your data loading and extraction functions is a **highly recommended best practice** for maintaining consistency, reusability, and organization, especially in a Kaggle competition where you have separate training and inference notebooks.

This approach ensures that any improvements or bug fixes to your core data processing logic are applied uniformly across all parts of your solution.

Here's the content for your `data_processing_utils.py` file, incorporating all the regex and function updates we've discussed, along with placeholders for `DATA_RELATED_KEYWORDS` and the full `extract_dataset_ids_with_context` pipeline.

---

### File: `data_processing_utils.py`

```python
# data_processing_utils.py

import fitz
import pymupdf4llm as m4l
import re
import os # Added for potential file path handling if needed later

# --- Regex Patterns for Dataset IDs ---
# REVISED DOI_PATTERN to handle variable whitespace after https and match lowercase letters
DOI_PATTERN = r"(?:doi:|https?\s*://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[-._;/:A-Za-z0-9]+)"

EPI_PATTERN = r'\bEPI[-_A-Z0-9]{2,}'
SAM_PATTERN = r'\bSAMN[0-9]{2,}'
IPR_PATTERN = r'\bIPR[0-9]{2,}'
CHE_PATTERN = r'\bCHEMBL[0-9]{2,}'
PRJ_PATTERN = r'\bPRJ[A-Z0-9]{2,}'
E_G_PATTERN = r'\bE-[A-Z]{4}-[0-9]{2,}'
ENS_PATTERN = r'\bENS[A-Z]{4}[0-9]{2,}'
CVC_PATTERN = r'\bCVCL_[A-Z0-9]{2,}'
EMP_PATTERN = r'\bEMPIAR-[0-9]{2,}'
PXD_PATTERN = r'\bPXD[0-9]{2,}'
HPA_PATTERN = r'\bHPA[0-9]{2,}'
SRR_PATTERN = r'\bSRR[0-9]{2,}'
GSE_PATTERN = r'\b(GSE|GSM|GDS|GPL)\d+\b' # Matches any number of digits
GNB_PATTERN = r'\b[A-Z]{1,2}\d{5,6}(?:\.\d)?\b'
CAB_PATTERN = r'\bCAB[0-9]{2,}'
PDB_PATTERN = r"\bpdb\s*\d[A-Za-z0-9]{3}"

# Combine all patterns into a list
DATASET_ID_PATTERNS = [
    DOI_PATTERN,
    EPI_PATTERN,
    SAM_PATTERN,
    IPR_PATTERN,
    CHE_PATTERN,
    PRJ_PATTERN,
    E_G_PATTERN,
    ENS_PATTERN,
    CVC_PATTERN,
    EMP_PATTERN,
    PXD_PATTERN,
    HPA_PATTERN,
    SRR_PATTERN,
    GSE_PATTERN,
    GNB_PATTERN,
    CAB_PATTERN,
    PDB_PATTERN
]

# Compile all patterns for efficiency
COMPILED_DATASET_ID_REGEXES = [re.compile(p) for p in DATASET_ID_PATTERNS]

# --- Keywords for Reference Section Heading Detection ---
REFERENCE_HEADING_KEYWORDS = [
    'references',
    'bibliography',
    'works cited',
    'literature cited',
    'citations',
    'reference list'
]

# --- Keywords for Data-Related Contextual Validation ---
# IMPORTANT: Jim, please populate this list with your refined data-related keywords.
# This list is intentionally left minimal as per your instruction.
DATA_RELATED_KEYWORDS = [
    'data', 'dataset', 'repository', 'accession', 'available', 'deposited',
    'supplementary', 'archive', 'entry', 'record', 'identifier'
]

# --- Helper Functions ---

def is_reference_heading_line(line: str) -> bool:
    """
    Checks if a given Markdown line is a recognized reference section heading.
    It looks for headings formatted with '#' or '**'.
    
    Args:
        line (str): A single line of text from the Markdown content.
        
    Returns:
        bool: True if the line is a reference heading, False otherwise.
    """
    lower_line = line.strip().lower() 
    
    for keyword in REFERENCE_HEADING_KEYWORDS:
        escaped_keyword = re.escape(keyword) 
        
        # Check for Markdown heading syntax (e.g., # References, ## Bibliography)
        if re.match(rf"^#+\s*{escaped_keyword}\s*$", lower_line):
            return True
        
        # Check for bold heading syntax (e.g., **References**, **Works Cited**)
        if re.match(rf"^\s*\*{2}\s*{escaped_keyword}\s*\*{2}\s*$", lower_line):
            return True
            
    return False

def get_windowed_text(full_text: str, start_idx: int, end_idx: int, window_size: int) -> str:
    """
    Extracts a window of text around a given span (start_idx, end_idx).
    
    Args:
        full_text (str): The complete text.
        start_idx (int): The starting index of the target ID.
        end_idx (int): The ending index of the target ID.
        window_size (int): The number of characters to include on each side of the ID.
        
    Returns:
        str: The text window around the ID.
    """
    text_len = len(full_text)
    
    # Calculate window start and end indices
    window_start = max(0, start_idx - window_size)
    window_end = min(text_len, end_idx + window_size)
    
    return full_text[window_start:window_end]

def is_text_data_related(text_window: str) -> bool:
    """
    Checks if the given text window contains keywords indicating data-related content.
    
    Args:
        text_window (str): The text segment to check.
        
    Returns:
        bool: True if data-related keywords are found, False otherwise.
    """
    lower_text_window = text_window.lower()
    for keyword in DATA_RELATED_KEYWORDS:
        if keyword in lower_text_window:
            return True
    return False

# --- Main Extraction Functions ---

def extract_text_before_references_markdown(pdf_path: str) -> str:
    """
    Converts a PDF to Markdown and extracts text before the references section,
    using the is_reference_heading_line helper to distinguish section headers.
    
    Args:
        pdf_path (str): The path to the PDF file.
        
    Returns:
        str: The extracted text from the main body of the PDF, in Markdown format.
    """
    doc = fitz.open(pdf_path)
    markdown_content = ""
    try:
        markdown_content = m4l.to_markdown(doc)
    except Exception as e:
        print(f"Warning: pymupdf4llm.to_markdown() failed for {pdf_path}: {e}. Falling back to simple text extraction.")
        # Fallback to simple text extraction if markdown conversion fails
        for page in doc:
            markdown_content += page.get_text()
    finally:
        doc.close()

    lines = markdown_content.split('\n')
    text_before_references = []
    
    for line in lines:
        if is_reference_heading_line(line):
            break # Stop processing lines, we've found the references section
        else:
            text_before_references.append(line)
            
    return "\n".join(text_before_references)

def find_potential_dataset_ids(text: str) -> list[str]:
    """
    Finds potential dataset IDs in the given text using predefined regex patterns.
    
    Args:
        text (str): The input text to search for dataset IDs.
        
    Returns:
        Set[str]: A set of unique dataset IDs found in the text.
    """
    dataset_ids = set()
    for compiled_regex in COMPILED_DATASET_ID_REGEXES:
        for match in compiled_regex.finditer(text):
            # Use .pattern attribute to compare the original regex string
            if compiled_regex.pattern == DOI_PATTERN:
                # For the DOI pattern, extract the content of the first capturing group
                dataset_id = match.group(1)
            else:
                # For all other patterns, extract the full matched string (group 0)
                dataset_id = match.group(0)
            dataset_ids.add(dataset_id)
    return list(dataset_ids)

def extract_dataset_ids_with_context(text: str, window_size: int = 200) -> list[dict]:
    """
    Extracts potential dataset IDs along with their surrounding context,
    and filters them based on data-related keywords in the context.
    
    Args:
        text (str): The input text (e.g., article body in Markdown).
        window_size (int): The number of characters to include on each side of the ID for context.
        
    Returns:
        list[dict]: A list of dictionaries, each containing 'id', 'context', and 'is_data_related' status.
    """
    found_ids_with_context = []
    
    for compiled_regex in COMPILED_DATASET_ID_REGEXES:
        for match in compiled_regex.finditer(text):
            dataset_id = match.group(1) if compiled_regex.pattern == DOI_PATTERN else match.group(0)
            
            # Get the text window around the matched ID
            context_text = get_windowed_text(text, match.start(), match.end(), window_size)
            
            # Check if the context is data-related
            data_related_status = is_text_data_related(context_text)
            
            # Only add if the context is data-related
            if data_related_status:
                found_ids_with_context.append({
                    'id': dataset_id,
                    'context': context_text,
                    'start_idx': match.start(),
                    'end_idx': match.end(),
                    'is_data_related_by_keywords': data_related_status # This is the keyword-based check
                })
                
    # Remove duplicates based on 'id' and 'start_idx' to handle overlapping regex matches
    unique_ids = {}
    for item in found_ids_with_context:
        key = (item['id'], item['start_idx'])
        if key not in unique_ids:
            unique_ids[key] = item
    
    return list(unique_ids.values())

# You might add more utility functions here as your project evolves,
# e.g., functions for cleaning text, normalizing IDs, etc.
```

---

### How to Use This File in Your Kaggle Notebooks

To use `data_processing_utils.py` in your Kaggle notebooks (both training and inference), you'll need to add it as a "Dataset" or "Utility Script".

**Steps on Kaggle:**

1.  **Create a new "Code" notebook** (or open an existing one).
2.  **Copy the content** from the `data_processing_utils.py` block above into this new notebook.
3.  **Save this notebook** as a "Utility Script" (e.g., `my_data_utils`). Kaggle will then treat it as a reusable Python module.
4.  In your **main training and inference notebooks**:
    *   Click on "Add Data" in the right-hand sidebar.
    *   Search for "Notebooks" and find the utility script you just saved (e.g., `my_data_utils`).
    *   Add it to your notebook. Kaggle will mount it under `/kaggle/input/my-data-utils/`.

**Example Usage in your Training/Inference Notebook (`train_notebook.ipynb` or `inference_notebook.ipynb`):**

```python
# In your training or inference notebook

import sys

# Add the path to your utility script to the system path
# Replace 'my-data-utils' with the actual name Kaggle gives your utility script dataset
sys.path.append('/kaggle/input/my-data-utils') 

# Now you can import functions and constants from your utility file
from data_processing_utils import (
    extract_text_before_references_markdown,
    find_potential_dataset_ids,
    extract_dataset_ids_with_context,
    DATA_RELATED_KEYWORDS # You can import this to update it in your notebook if needed,
                          # or keep it defined only in the .py file.
)

# --- Example Usage ---

# 1. Define a dummy PDF path for demonstration
# In a real scenario, this would be your actual PDF file path from the competition data
dummy_pdf_path = "/kaggle/input/your-competition-data/some_article.pdf" 

# 2. Extract text before references section
article_markdown_text = extract_text_before_references_markdown(dummy_pdf_path)
print(f"Extracted text length: {len(article_markdown_text)}")
# print(article_markdown_text[:500]) # Print first 500 chars to verify

# 3. Find all potential dataset IDs (without contextual validation yet)
all_ids = find_potential_dataset_ids(article_markdown_text)
print(f"All potential IDs found: {all_ids}")

# 4. Extract IDs with context and apply keyword-based filtering
# You can adjust the window_size as needed
relevant_ids_with_context = extract_dataset_ids_with_context(article_markdown_text, window_size=200)
print(f"Relevant IDs with context (after keyword filtering):")
for item in relevant_ids_with_context:
    print(f"  ID: {item['id']}, Context snippet: '{item['context'][:100]}...'") # Print first 100 chars of context

# Now you would typically pass these 'relevant_ids_with_context' to your LLM for classification
# For example:
# for item in relevant_ids_with_context:
#     llm_input = f"Classify the relationship of dataset ID '{item['id']}' in this context: {item['context']}"
#     classification_result = your_llm_inference_function(llm_input)
#     print(f"  ID: {item['id']}, LLM Classification: {classification_result}")

```

This setup provides a clean, modular, and consistent way to handle your data processing pipeline across your Kaggle notebooks. Good luck, Jim!