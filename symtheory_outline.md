That's an excellent plan, Jim! Your approach to pre-processing the article text to extract relevant snippets for a smaller LLM is very sensible, especially given the constraints of a single RTX 3050 GPU. Your `DatasetCitation` and `ArticleData` classes provide a great foundation.

Let's outline a comprehensive Python notebook plan, incorporating your existing classes and addressing the steps for local development, training, and evaluation.

---

### Python Notebook Plan: Kaggle Data Citation Classification

This notebook will guide you through setting up your environment, preprocessing the Kaggle competition data, training a small open-source LLM using `trl` and `KaggleHub`, and then using it for inference and evaluation.

**Goal:** Identify and classify dataset references as Primary or Secondary within scientific literature.

**Environment:** Local Windows machine with CUDA and a single RTX 3050 GPU.

---

#### 1. Setup and Dependencies

First, we'll ensure all necessary libraries are installed. Given your previous work with `lxml`, `PyMuPDF`, and `spaCy`, we'll include those for text extraction and potentially more advanced NLP preprocessing.

```python
# 1.1. Install necessary libraries
# Use !pip install for notebook environment
!pip install transformers trl accelerate bitsandbytes sentencepiece lxml PyMuPDF spacy
!python -m spacy download en_core_web_sm # Download a small spaCy model

# 1.2. Import Libraries
import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Set, List, Optional, Dict, Any

import fitz # PyMuPDF
from lxml import etree # For XML parsing
import spacy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import Dataset, load_metric
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# For KaggleHub integration (assuming it's set up or models are downloaded)
# You might need to install kagglehub if you plan to use it directly for model download
# !pip install kagglehub

# 1.3. Configure CUDA for local GPU
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    torch.cuda.empty_cache() # Clear GPU memory
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# Load spaCy model for sentence segmentation and potentially other NLP tasks
nlp = spacy.load("en_core_web_sm")
```

---

#### 2. Data Classes (Your Provided Classes with Minor Adjustments)

I've made a small but important adjustment to `ArticleData` to store a *list* of `DatasetCitation` objects. This allows us to associate each specific dataset mention and its context with a potential classification, which aligns better with the competition's goal of classifying *each* dataset found. I've also added `citation_type` to `DatasetCitation` for storing the ground truth label during training data preparation.

```python
# 2.1. DatasetCitation Class
@dataclass
class DatasetCitation:
    dataset_ids: Set[str] = field(default_factory=set)  # Set to store unique dataset IDs
    citation_context: str = ""
    citation_type: Optional[str] = None # "Primary" or "Secondary" - for ground truth during training

    def add_dataset_id(self, dataset_id: str):
        self.dataset_ids.add(dataset_id)

    def set_citation_context(self, context: str):
        """Sets the citation context, cleaning it."""
        if context:
            # Replace newlines with spaces, remove brackets, and normalize whitespace
            context = context.replace('\n', ' ').replace('[', '').replace(']', '')
            context = re.sub(r'\s+', ' ', context.strip())
            self.citation_context = context # Assign, don't concatenate

    def has_dataset(self) -> bool:
        """Returns True if there are both dataset IDs and citation context."""
        return bool(self.dataset_ids and self.citation_context.strip())

    def to_dict(self):
        d = asdict(self)
        d["dataset_ids"] = list(self.dataset_ids)
        return d

# 2.2. ArticleData Class
@dataclass
class ArticleData:
    article_id: str = ""
    article_doi: str = ""
    title: str = ""
    author: str = ""
    abstract: str = ""
    # Changed to a list of DatasetCitation objects
    dataset_citations: List[DatasetCitation] = field(default_factory=list)

    def __post_init__(self):
        # Custom initialization
        if self.article_id and not self.article_doi:
            # If article_id is provided but not article_doi, set article_doi
            self.article_doi = self.article_id.replace("_", "/").lower()

    def add_dataset_citation(self, dataset_citation: DatasetCitation):
        """Adds a DatasetCitation object to the article."""
        if dataset_citation.has_dataset():
            self.dataset_citations.append(dataset_citation)
        
    def to_dict(self):
        d = asdict(self)
        # Convert list of DatasetCitation objects to their dict representation
        d["dataset_citations"] = [dc.to_dict() for dc in self.dataset_citations]
        return d

    def to_json(self):
        return json.dumps(self.to_dict(), separators=(',', ':'))

    def has_data(self) -> bool:
        """Returns True if there are any dataset citations."""
        return bool(self.dataset_citations)
```

---

#### 3. Data Loading and Initial Preprocessing

This section will cover how to load the raw competition data (full text articles and labels) and begin structuring it.

```python
# 3.1. Define Data Paths (Adjust these to your Kaggle data location)
TRAIN_DATA_DIR = "path/to/kaggle/train" # Directory containing full text files (XML/PDF/TXT)
TRAIN_LABELS_PATH = "path/to/kaggle/train.json" # JSON file with ground truth labels
TEST_DATA_DIR = "path/to/kaggle/test"
SAMPLE_SUBMISSION_PATH = "path/to/kaggle/sample_submission.csv"

# 3.2. Helper function to extract text from various file types
def extract_text_from_file(filepath: str) -> str:
    """Extracts text from XML, PDF, or TXT files."""
    if filepath.endswith(".xml"):
        try:
            tree = etree.parse(filepath)
            # A common way to get all text from an XML scientific article
            # This might need adjustment based on the specific XML schema
            return " ".join(tree.xpath("//text()")).strip()
        except Exception as e:
            print(f"Error parsing XML {filepath}: {e}")
            return ""
    elif filepath.endswith(".pdf"):
        try:
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()
        except Exception as e:
            print(f"Error parsing PDF {filepath}: {e}")
            return ""
    elif filepath.endswith(".txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

# 3.3. Load Ground Truth Labels
def load_labels(labels_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Loads the ground truth labels from the JSON file."""
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    # Reformat for easier lookup: {article_id: [{span_text, dataset_id, citation_type}, ...]}
    # The competition's JSON structure might be different, adjust as needed.
    # Assuming it's a list of dicts, where each dict is an example.
    # Example: {"article_id": "...", "spans": [{"text": "...", "dataset_id": "...", "type": "Primary"}]}
    formatted_labels = {}
    for entry in labels:
        article_id = entry["article_id"]
        if article_id not in formatted_labels:
            formatted_labels[article_id] = []
        for span_info in entry.get("spans", []): # Assuming 'spans' key
            formatted_labels[article_id].append({
                "span_text": span_info["text"],
                "dataset_id": span_info["dataset_id"],
                "citation_type": span_info["type"]
            })
    return formatted_labels

train_labels = load_labels(TRAIN_LABELS_PATH)
print(f"Loaded {len(train_labels)} articles with ground truth labels.")

# 3.4. Initial Article Data Collection (for training set)
train_articles_data: Dict[str, ArticleData] = {}
for article_file in os.listdir(TRAIN_DATA_DIR):
    article_id = os.path.splitext(article_file)[0]
    filepath = os.path.join(TRAIN_DATA_DIR, article_file)
    
    full_text = extract_text_from_file(filepath)
    
    # Placeholder for extracting title, author, abstract
    # This is highly dependent on the structure of your full text files.
    # For now, we'll use simple regex or assume they are at the beginning.
    # A more robust solution might involve specific parsers for each document type.
    title_match = re.search(r"Title:\s*(.*)", full_text, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "Unknown Title"

    author_match = re.search(r"Author(?:s)?:\s*(.*)", full_text, re.IGNORECASE)
    author = author_match.group(1).strip() if author_match else "Unknown Author"

    abstract_match = re.search(r"Abstract\s*(.*?)(?=\n\n|\Z)", full_text, re.IGNORECASE | re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else "No Abstract"

    article_data = ArticleData(
        article_id=article_id,
        title=title,
        author=author,
        abstract=abstract
    )
    # Store full text temporarily for context extraction in the next step
    article_data.full_text = full_text # Add a temporary attribute
    train_articles_data[article_id] = article_data

print(f"Loaded initial data for {len(train_articles_data)} training articles.")
```

---

#### 4. Advanced Preprocessing: Extracting Dataset Mentions and Context

This is where your strategy of extracting relevant context comes into play. We'll use regex to find potential dataset IDs and then use spaCy to extract surrounding sentences as context.

```python
# 4.1. Define Regex Patterns for Dataset Identifiers
# These are examples; you'll need to refine them based on the competition data.
# Common patterns: DOI, accession numbers (GSE, PDB, E-MTAB, PRJE, SRA, etc.)
DOI_PATTERN = r"(?:doi:|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)"
ACCESSION_PATTERNS = {
    "GSE": r"GSE\d+",
    "PDB": r"pdb\s*\d[A-Za-z0-9]{3}", # Example: pdb 5yfp
    "E-MTAB": r"E-MTAB-\d+",
    "PRJE": r"PRJE\d+",
    "SRA": r"SRR\d+|SRP\d+|SRX\d+|SRS\d+", # SRA accession numbers
    # Add more as you discover them in the dataset
}

# Combine all patterns for initial scanning
ALL_ID_PATTERNS = [DOI_PATTERN] + list(ACCESSION_PATTERNS.values())

# 4.2. Function to extract context around an ID
def extract_context_around_id(text: str, dataset_id: str, window_size_sentences: int = 3) -> str:
    """
    Extracts a window of sentences around a given dataset ID in the text.
    Uses spaCy for sentence segmentation.
    """
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Find all occurrences of the dataset_id (case-insensitive)
    matches = [(i, sent) for i, sent in enumerate(sentences) if dataset_id.lower() in sent.lower()]
    
    if not matches:
        return ""

    # For simplicity, take the context around the first match.
    # You might want to refine this to capture all relevant contexts or the most prominent one.
    first_match_idx = matches[0][0]
    
    start_idx = max(0, first_match_idx - window_size_sentences)
    end_idx = min(len(sentences), first_match_idx + window_size_sentences + 1)
    
    context_sentences = sentences[start_idx:end_idx]
    return " ".join(context_sentences)

# 4.3. Populate ArticleData with DatasetCitation objects and ground truth
training_data_for_llm = [] # This will be a list of dicts for the LLM training dataset

for article_id, article_data in train_articles_data.items():
    full_text = article_data.full_text # Retrieve temporary full text
    
    # Use spaCy for more robust sentence segmentation
    doc = nlp(full_text)
    
    # Iterate through ground truth labels for this article
    # This ensures we only focus on known dataset mentions for training
    if article_id in train_labels:
        for label_info in train_labels[article_id]:
            gt_dataset_id = label_info["dataset_id"]
            gt_citation_type = label_info["citation_type"]
            gt_span_text = label_info["span_text"] # The exact span from the ground truth

            # Find the context for this specific ground truth span/ID
            # Prioritize finding the exact span text, then fall back to ID
            context = extract_context_around_id(full_text, gt_span_text, window_size_sentences=3)
            if not context: # If exact span not found, try with just the ID
                context = extract_context_around_id(full_text, gt_dataset_id, window_size_sentences=3)
            
            if context:
                dataset_citation = DatasetCitation()
                dataset_citation.add_dataset_id(gt_dataset_id)
                dataset_citation.set_citation_context(context)
                dataset_citation.citation_type = gt_citation_type # Set ground truth label
                
                article_data.add_dataset_citation(dataset_citation)

                # Prepare data for LLM training
                training_data_for_llm.append({
                    "article_title": article_data.title,
                    "article_abstract": article_data.abstract,
                    "citation_context": dataset_citation.citation_context,
                    "dataset_id": list(dataset_citation.dataset_ids)[0], # Assuming one ID per citation for simplicity
                    "label": dataset_citation.citation_type
                })
    
    # Clean up temporary full_text attribute
    del article_data.full_text

print(f"Prepared {len(training_data_for_llm)} training examples for the LLM.")

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_list(training_data_for_llm)
train_dataset = train_dataset.shuffle(seed=42) # Shuffle for good measure
# You might want to split into train/validation here
# train_test_split = train_dataset.train_test_split(test_size=0.1)
# train_dataset = train_test_split['train']
# eval_dataset = train_test_split['test']
```

---

#### 5. Model Selection and Configuration

Given your RTX 3050, 4-bit quantization is essential. We'll use a Qwen model, as you're familiar with it.

```python
# 5.1. Choose a Model from KaggleHub
# Example: Qwen/Qwen1.5-0.5B-Chat (or 1.8B-Chat if 0.5B is too small/performs poorly)
# You can find these on KaggleHub or Hugging Face Hub.
# For local use, you'd typically download them or use `AutoModelForCausalLM.from_pretrained`
# which handles downloading.
model_name = "Qwen/Qwen1.5-0.5B-Chat" # Or "Qwen/Qwen1.5-1.8B-Chat"

# 5.2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Qwen uses EOS for padding

# 5.3. Load Model with Quantization (4-bit)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # Or torch.float16 if bfloat16 is not supported by your GPU
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16, # Match compute_dtype
    device_map="auto", # Automatically maps model to available devices
    trust_remote_code=True # Required for some models like Qwen
)

# Prepare model for k-bit training (LoRA compatible)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

print(f"Model {model_name} loaded with 4-bit quantization.")
```

---

#### 6. Dataset Preparation for Training

We'll format the extracted data into instruction-tuning prompts using the ChatML format, which Qwen models are trained on.

```python
# 6.1. Define the formatting function for ChatML
def format_example(example):
    messages = [
        {"role": "system", "content": "You are an expert assistant for classifying research data citations."},
        {"role": "user", "content": (
            f"Given the following article context and a specific data citation, classify if the data was generated as 'Primary' (newly generated for this study) or 'Secondary' (reused from existing records).\n\n"
            f"Article Title: {example['article_title']}\n"
            f"Article Abstract: {example['article_abstract']}\n"
            f"Data Citation Context: {example['citation_context']}\n"
            f"Dataset ID: {example['dataset_id']}\n\n"
            f"Classification:"
        )}
    ]
    # The target output for the model is just "Primary" or "Secondary"
    messages.append({"role": "assistant", "content": example['label']})
    
    # Apply chat template and return
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

# Apply the formatting to the dataset
formatted_train_dataset = train_dataset.map(format_example)

# Print an example to verify
print("\nExample of formatted training data:")
print(formatted_train_dataset[0]["text"])
```

---

#### 7. Model Training (using `trl.SFTTrainer`)

We'll use LoRA for efficient fine-tuning on your single GPU.

```python
# 7.1. Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear", # Adjust based on model architecture if needed
)

# 7.2. Configure Training Arguments
from transformers import TrainingArguments

output_dir = "./results"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2, # Adjust based on your GPU memory (start low)
    gradient_accumulation_steps=4, # Accumulate gradients to simulate larger batch size (2*4=8)
    learning_rate=2e-4,
    num_train_epochs=3, # Start with a few epochs
    logging_steps=10,
    save_steps=500,
    optim="paged_adamw_8bit", # Memory-efficient optimizer
    fp16=False, # Set to True if your GPU supports it and you want to try float16
    bf16=True, # Set to True if your GPU supports it (RTX 3050 might not fully support bfloat16)
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="none", # Or "tensorboard" if you want to monitor
    disable_tqdm=False,
    remove_unused_columns=False, # Keep columns for formatting
)

# 7.3. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_train_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512, # Max input sequence length (adjust based on context size)
    packing=False, # Set to True for more efficient training if your data is short
    dataset_text_field="text",
)

# 7.4. Start Training
print("\nStarting model training...")
trainer.train()
print("Training complete!")

# Save the fine-tuned model (LoRA adapters)
trainer.save_model(os.path.join(output_dir, "final_model"))
print(f"Fine-tuned model saved to {os.path.join(output_dir, 'final_model')}")
```

---

#### 8. Inference and Evaluation

After training, you'll load the best model (or the final one) and apply it to the test data.

```python
# 8.1. Load the Trained Model (or merge LoRA adapters for full model)
# If you saved LoRA adapters, you'll need to load the base model and then the adapters.
# For inference, it's often easier to merge them.
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=nf4_config, # Use the same config as training
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True
# )
# model = PeftModel.from_pretrained(model, os.path.join(output_dir, "final_model"))
# model = model.merge_and_unload() # Merge LoRA adapters into the base model

# For simplicity, if you just want to test the last saved checkpoint:
# You can also load the model directly from the checkpoint if it's a full save
# model = AutoModelForCausalLM.from_pretrained(os.path.join(output_dir, "final_model"), device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "final_model"))

# If you want to load the base model and then the adapters for inference:
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, os.path.join(output_dir, "final_model"))
model.eval() # Set to evaluation mode

print("Model loaded for inference.")

# 8.2. Preprocess Test Data (similar to training data)
test_articles_data: Dict[str, ArticleData] = {}
# Assuming test data structure is similar to train data (full text files)
for article_file in os.listdir(TEST_DATA_DIR):
    article_id = os.path.splitext(article_file)[0]
    filepath = os.path.join(TEST_DATA_DIR, article_file)
    
    full_text = extract_text_from_file(filepath)
    
    # Extract title, author, abstract (same as training)
    title_match = re.search(r"Title:\s*(.*)", full_text, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "Unknown Title"
    author_match = re.search(r"Author(?:s)?:\s*(.*)", full_text, re.IGNORECASE)
    author = author_match.group(1).strip() if author_match else "Unknown Author"
    abstract_match = re.search(r"Abstract\s*(.*?)(?=\n\n|\Z)", full_text, re.IGNORECASE | re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else "No Abstract"

    article_data = ArticleData(
        article_id=article_id,
        title=title,
        author=author,
        abstract=abstract
    )
    # For test data, we need to find *all* potential dataset IDs, not just ground truth
    # This is the "finding datasets" part of your goal.
    
    # Use regex to find all potential dataset IDs in the full text
    found_dataset_mentions = []
    for pattern in ALL_ID_PATTERNS:
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            dataset_id = match.group(1) if pattern == DOI_PATTERN else match.group(0)
            span_text = match.group(0) # The full matched text
            
            context = extract_context_around_id(full_text, span_text, window_size_sentences=3)
            
            if context:
                dc = DatasetCitation()
                dc.add_dataset_id(dataset_id)
                dc.set_citation_context(context)
                found_dataset_mentions.append(dc)
                
    article_data.dataset_citations = found_dataset_mentions # Assign found mentions
    test_articles_data[article_id] = article_data

print(f"Prepared {len(test_articles_data)} test articles for inference.")

# 8.3. Generate Predictions
predictions = []
true_labels = [] # Only if you have a test_labels.json for evaluation

for article_id, article_data in test_articles_data.items():
    for dc in article_data.dataset_citations:
        # Create the prompt for inference
        messages = [
            {"role": "system", "content": "You are an expert assistant for classifying research data citations."},
            {"role": "user", "content": (
                f"Given the following article context and a specific data citation, classify if the data was generated as 'Primary' (newly generated for this study) or 'Secondary' (reused from existing records).\n\n"
                f"Article Title: {article_data.title}\n"
                f"Article Abstract: {article_data.abstract}\n"
                f"Data Citation Context: {dc.citation_context}\n"
                f"Dataset ID: {list(dc.dataset_ids)[0]}\n\n" # Assuming one ID per citation
                f"Classification:"
            )}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=10, # Expecting "Primary" or "Secondary"
                do_sample=False, # Use greedy decoding as per your preference
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Post-process the generated text to get the classification
        predicted_type = "Unknown"
        if "Primary" in generated_text:
            predicted_type = "Primary"
        elif "Secondary" in generated_text:
            predicted_type = "Secondary"
        
        predictions.append({
            "article_id": article_id,
            "dataset_id": list(dc.dataset_ids)[0],
            "predicted_type": predicted_type
        })

        # If you have test labels, you can collect true_labels here for evaluation
        # For Kaggle, you'll typically submit predictions without knowing test labels.

print(f"Generated {len(predictions)} predictions.")

# 8.4. Evaluation (if test labels are available)
# If you have a separate test_labels.json for local evaluation:
# test_labels = load_labels(TEST_LABELS_PATH) # Load test labels
#
# # Match predictions to true labels and calculate metrics
# # This part requires careful matching of dataset_id within article_id
# # and might involve fuzzy matching for context if exact span isn't available.
# # For simplicity, assuming exact match on article_id and dataset_id.
#
# y_true = []
# y_pred = []
#
# for pred_entry in predictions:
#     article_id = pred_entry["article_id"]
#     dataset_id = pred_entry["dataset_id"]
#     predicted_type = pred_entry["predicted_type"]
#
#     # Find the true label for this specific dataset_id in this article
#     found_true_label = False
#     if article_id in test_labels:
#         for gt_info in test_labels[article_id]:
#             if gt_info["dataset_id"] == dataset_id: # Exact match on ID
#                 y_true.append(gt_info["citation_type"])
#                 y_pred.append(predicted_type)
#                 found_true_label = True
#                 break
#     if not found_true_label:
#         # Handle cases where a predicted ID might not be in ground truth
#         # or where the ID extraction was imperfect.
#         # For competition, this means your ID extraction needs to be precise.
#         pass
#
# if y_true and y_pred:
#     from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#     # Map "Primary" to 1, "Secondary" to 0 for sklearn metrics
#     label_map = {"Primary": 1, "Secondary": 0}
#     y_true_mapped = [label_map.get(l, -1) for l in y_true]
#     y_pred_mapped = [label_map.get(l, -1) for l in y_pred]
#
#     # Filter out -1 if there were unknown labels
#     valid_indices = [i for i, val in enumerate(y_true_mapped) if val != -1 and y_pred_mapped[i] != -1]
#     y_true_mapped = [y_true_mapped[i] for i in valid_indices]
#     y_pred_mapped = [y_pred_mapped[i] for i in valid_indices]
#
#     if y_true_mapped:
#         print("\nEvaluation Results:")
#         print(f"Accuracy: {accuracy_score(y_true_mapped, y_pred_mapped):.4f}")
#         print(f"F1 Score (weighted): {f1_score(y_true_mapped, y_pred_mapped, average='weighted'):.4f}")
#         print(f"Precision (weighted): {precision_score(y_true_mapped, y_pred_mapped, average='weighted'):.4f}")
#         print(f"Recall (weighted): {recall_score(y_true_mapped, y_pred_mapped, average='weighted'):.4f}")
#     else:
#         print("No matching true labels found for evaluation.")
# else:
#     print("Not enough data to perform evaluation.")
```

---

#### 9. Submission File Generation (Kaggle Specific)

Finally, format your predictions into the required `submission.csv` file.

```python
# 9.1. Create Submission DataFrame
import pandas as pd

submission_df = pd.DataFrame(predictions)
# Rename columns to match Kaggle's expected format (e.g., 'id', 'class_label')
# This will depend on the exact submission format specified by Kaggle.
# Example:
# submission_df = submission_df.rename(columns={"article_id": "Id", "dataset_id": "DatasetId", "predicted_type": "Type"})
# submission_df["Id"] = submission_df["Id"] + "_" + submission_df["DatasetId"] # If Id is a combination

# Assuming the submission format is a list of dictionaries with 'article_id', 'dataset_id', 'citation_type'
# You might need to adjust this based on the exact competition requirements.
# For example, if it expects a single ID column like "article_id_dataset_id"
final_submission_data = []
for pred in predictions:
    final_submission_data.append({
        "Id": f"{pred['article_id']}_{pred['dataset_id']}", # Example: combine IDs
        "Type": pred['predicted_type']
    })

final_submission_df = pd.DataFrame(final_submission_data)
final_submission_df.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created successfully!")
```

---

### Next Steps and Considerations for Jim:

1.  **Data Paths:** Crucially, update all `path/to/kaggle/` placeholders to your actual data directories.
2.  **Full Text Parsing:** The `extract_text_from_file` and the initial `title`/`author`/`abstract` extraction are placeholders. Real-world scientific literature often has complex XML or PDF structures. You might need more sophisticated parsing (e.g., using `BeautifulSoup` for XML, or more advanced PDF parsers if PyMuPDF isn't sufficient for layout-aware extraction).
3.  **Dataset ID Patterns:** The `ACCESSION_PATTERNS` are examples. You'll need to thoroughly analyze the training data to identify all common dataset identifier formats. This is a critical step for the "finding datasets" part of the competition.
4.  **Context Window Size:** Experiment with `window_size_sentences` in `extract_context_around_id`. Too small, and you lose context; too large, and you might introduce noise or exceed the LLM's `max_seq_length`.
5.  **Training Batch Size:** The `per_device_train_batch_size` (currently 2) is very small. If you encounter CUDA out-of-memory errors, reduce it further. `gradient_accumulation_steps` helps compensate for small batch sizes.
6.  **`bf16` vs `fp16`:** Your RTX 3050 might not fully support `bfloat16`. If you get errors, switch `bnb_4bit_compute_dtype` and `torch_dtype` to `torch.float16` and `fp16=True` in `TrainingArguments`.
7.  **Hyperparameter Tuning:** Experiment with `learning_rate`, `num_train_epochs`, `lora_alpha`, `r`, and `max_seq_length` to optimize performance.
8.  **Evaluation:** While Kaggle provides a test set without labels, having a local validation set (split from your training data) is crucial for iterative development and hyperparameter tuning.

This plan provides a solid framework, Jim. Let me know if you'd like to dive deeper into any specific section, like refining the text extraction or prompt engineering!