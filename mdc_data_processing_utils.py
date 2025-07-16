# mdc_data_processing_utils.py

import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Set, List, Optional, Dict, Any
import fitz
import pymupdf4llm
from lxml import etree # For XML parsing
from spacy.language import Language

# 2.1. DatasetCitation Class
@dataclass
class DatasetCitation:
    dataset_id: str = ""
    citation_context: str = ""
    citation_type: Optional[str] = None # "Primary", "Secondary", or "Missing" - for ground truth during training
    max_contet_len = 400

    def set_citation_context(self, context: str):
        """Sets the citation context, cleaning it and limiting to last 400 characters."""
        if context:
            # Replace newlines with spaces, remove brackets, and normalize whitespace
            context = context.replace('\n', ' ').replace('[', '').replace(']', '')
            context = re.sub(r'\s+', ' ', context.strip())
            self.citation_context = context[-self.max_contet_len:] # Limit to last 400 characters

    def is_doi(self)-> bool:
        return self.dataset_id.startswith("10.")
    
    def has_dataset(self) -> bool:
        """Returns True if there are both dataset IDs and citation context."""
        return bool(self.dataset_id and self.citation_context.strip())

    def to_dict(self):
        return asdict(self)

# 2.2. ArticleData Class
@dataclass
class ArticleData:
    article_id: str = ""
    article_doi: str = ""
    title: str = ""
    author: str = ""
    abstract: str = ""
    dataset_citations: List[DatasetCitation] = field(default_factory=list)

    def __post_init__(self):
        # Custom initialization
        if self.article_id and not self.article_doi:
            # If article_id is provided but not article_doi, set article_doi
            self.article_doi = self.article_id.replace("_", "/").lower()

    def add_dataset_citation(self, dataset_citation: DatasetCitation):
        """Adds a DatasetCitation object to the article."""
        ds_id_lower = dataset_citation.dataset_id.lower()
        if dataset_citation.has_dataset() and ds_id_lower != self.article_doi:
            existing_ids = ",".join({d.dataset_id.lower() for d in self.dataset_citations})
            # Don't include partial dataset_ids
            if ds_id_lower not in existing_ids:
                self.dataset_citations.append(dataset_citation)
        
    def get_data_for_llm(self) -> list[dict[str, str]]:
        data_for_llm: list[dict[str, str]] = []
        for citation in self.dataset_citations:
            # Convert to dict for LLM training data
            data_for_llm.append(
                {
                    "article_id": self.article_id,
                    "article_doi": self.article_doi,
                    "article_abstract": self.abstract,
                    "dataset_id": citation.dataset_id,                    
                    "citation_context": citation.citation_context,
                    "label": citation.citation_type if citation.citation_type else ""
                }
            )
        return data_for_llm

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


@dataclass
class LlmTrainingData:
    article_id: str = ""
    article_doi: str = ""
    article_abstract: str = ""
    citation_context: str = ""
    dataset_id: str = ""
    label: str = ""

    def to_dict(self):
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict(), separators=(',', ':'))


@dataclass
class SubmissionData:
    article_id: str = ""
    dataset_id: str = ""
    type: str = ""
    context: str = ""

    def to_dict(self):
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict(), separators=(',', ':'))
    

class MdcFileTextExtractor():

    DOI_PATTERN = r"(?:doi:|https\s*://(?:dx\.)?doi\.org/)(10\.\s*\d{4,9}/[-._/:A-Za-z0-9]+)"
    EPI_PATTERN = r'\bEPI[-_A-Z0-9]{2,}'
    SAM_PATTERN = r'\bSAMN[0-9]{2,}'          # SAMN07159041
    IPR_PATTERN = r'\bIPR[0-9]{2,}'
    CHE_PATTERN = r'\bCHEMBL[0-9]{2,}'
    PRJ_PATTERN = r'\bPRJ[A-Z0-9]{2,}'
    E_G_PATTERN = r'\bE-[A-Z]{4}-[0-9]{2,}'   # E-GEOD-19722 or E-PROT-100
    ENS_PATTERN = r'\bENS[A-Z]{4}[0-9]{2,}'
    CVC_PATTERN = r'\bCVCL_[A-Z0-9]{2,}'
    EMP_PATTERN = r'\bEMPIAR-[0-9]{2,}'
    PXD_PATTERN = r'\bPXD[0-9]{2,}'
    HPA_PATTERN = r'\bHPA[0-9]{2,}'
    SRR_PATTERN = r'\bSRR[0-9]{2,}'
    GSE_PATTERN = r'\b(GSE|GSM|GDS|GPL)\d{4,6}\b' # Example for GEO accession numbers (e.g., GSE12345, GSM12345)
    GNB_PATTERN = r'\b[A-Z]{1,2}\d{5,6}(?:\.\d)?\b'
    CAB_PATTERN = r'\bCAB[0-9]{2,}'
    PDB_PATTERN = r"\bpdb\s*\d[A-Za-z0-9]{3}" # Example: pdb 5yfp

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

    # Data related keywords to look for in the text
    # These keywords help to ensure that the text is relevant to datasets
    DATA_RELATED_KEYWORDS = ['data release', 'data associated', 'data referring', 'data availability', 'data access', 'data source', 'program data', 'our data', 'the data', 'dataset', 'database', ' segmented by', 'digital elevation model']

    REFERENCE_KEYWORDS = ['references','bibliography','works cited','citations','reference list']

    NON_STD_UNICODE_DASHES = re.compile(r'[\u2010\u2011\u2012\u2013\u2014]')
    NON_STD_UNICODE_TICKS = re.compile(r'[\u201c\u201d]')

    def __init__(self, article_id: str, file_path: str):
        self.article_id = article_id
        self.article_doi = self.article_id.replace("_", "/").lower()
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)  # 'example.pdf'
        self.article_id, self.file_ext = os.path.splitext(self.file_name)
        

    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing non-standard unicode dashes and extra whitespace.
        
        Args:
            text (str): The text to clean.
            
        Returns:
            str: The cleaned text.
        """
        if not text:
            return ""
        # Replace all non-standard unicode dashes with '-' and ticks with "'"
        text = MdcFileTextExtractor.NON_STD_UNICODE_DASHES.sub('-', text)
        text = MdcFileTextExtractor.NON_STD_UNICODE_TICKS.sub("'", text)

        # Fix known formatting issues
        text = text.replace('\u200b', '').replace('-\n', '-').replace('_\n', '_').replace('/\n', '/')
        text = text.replace('dryad.\n', 'dryad.').replace('doi.\norg', 'doi.org')
        text = text.replace('doi.org/10.\n', 'doi.org/10.').replace('http://dx.doi.org/10.', 'https://doi.org/10.')
        # Remove extra whitespace
        return re.sub(r'\s+', ' ', text).strip()


    def is_text_data_related(self, text: str) -> bool:
        if not text:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in MdcFileTextExtractor.DATA_RELATED_KEYWORDS)

    
    def is_reference_heading_line(self, line: str) -> bool:
        """
        Checks if a given Markdown line is a recognized reference section heading.
        It looks for headings formatted with '#' or '**'.
        
        Args:
            line (str): A single line of text from the Markdown content.
            
        Returns:
            bool: True if the line is a reference heading, False otherwise.
        """
        # Normalize line by stripping leading/trailing whitespace and lowercasing
        lower_line = line.strip().lower() 
        
        for keyword in MdcFileTextExtractor.REFERENCE_KEYWORDS:
            # Escape keyword for regex safety, in case it contains special regex characters
            escaped_keyword = re.escape(keyword) 
            
            # 1. Check for Markdown heading syntax (e.g., # References, ## Bibliography)
            #    Pattern: Starts with one or more '#', followed by optional whitespace,
            #    then the keyword, optional whitespace, and end of line.
            if re.match(rf"^#+\s*{escaped_keyword}\s*$", lower_line):
                return True
            
            # 2. Check for bold heading syntax (e.g., **References**, **Works Cited**)
            #    Pattern: Optional leading whitespace, then two asterisks, optional whitespace,
            #    the keyword, optional whitespace, two asterisks, optional trailing whitespace, and end of line.
            if re.match(rf"^\s*\*{2}\s*{escaped_keyword}\s*\*{2}\s*$", lower_line):
                return True
                
        return False # If no matching heading pattern is found after checking all keywords
    
    def find_reference_start_page_index(self, doc: fitz.Document) -> int:
        """
        Finds the 0-based page index where the references section likely starts in a PyMuPDF document.
        This uses a simpler raw text search for speed, to limit the pages passed to pymupdf4llm.
        
        Args:
            doc (fitz.Document): The PyMuPDF document object.
            
        Returns:
            int: The 0-based index of the page *before* which references start,
                 or doc.page_count if no reference section is found.
        """
        # Use the same keywords as REFERENCE_KEYWORDS for consistency
        simple_reference_keywords = MdcFileTextExtractor.REFERENCE_KEYWORDS

        max_page_no = doc.page_count - 1
        page_no = 0
        for page in doc:
            page_text = page.get_textpage().extractTEXT().lower()
            for keyword in simple_reference_keywords:
                # Look for the keyword as a potential heading in raw text.
                # This is a heuristic: check if it's on its own line or at the very start.
                if f"\n{keyword}\n" in page_text or page_text.startswith(f"{keyword}\n") or \
                   f"\n{keyword} " in page_text or page_text.startswith(f"{keyword} "):
                    return min(page_no, max_page_no) # Return the index of the page where references start
            page_no += 1
        return max_page_no # Return total page count if no reference section found

    def extract_text_from_file(self) -> str:
        """Extracts text from XML, PDF, or TXT files."""
        if not os.path.exists(self.file_path):
            print(f"File not found: {self.file_path}")
            return ""
        
        print(f"Extracting md text from file: {self.file_path}")
        if self.file_path.endswith(".xml"):
            parser = etree.XMLParser(resolve_entities=False, no_network=True)
            try:
                tree = etree.parse(self.file_path, parser)
                # A common way to get all text from an XML scientific article
                # This might need adjustment based on the specific XML schema
                return self.clean_text(" ".join(tree.xpath("//text()")).strip())
            except Exception as e:
                print(f"Error parsing XML {self.file_path}: {e}")
                return ""
        elif self.file_path.endswith(".pdf"):
            doc = fitz.open(self.file_path)
            markdown_content = ""
            try:
                # Step 1: Quickly find the page where references likely start using raw text search
                ref_start_page_idx = self.find_reference_start_page_index(doc)
                pages = range(ref_start_page_idx + 1)
                
                # Step 2: Convert only pages before the references section to Markdown
                # pymupdf4llm.to_markdown uses 0-based indexing for pages.
                # 'end_page' parameter is exclusive, so to get up to page N, we set end_page=N+1.
                # If ref_start_page_idx is 5, we want pages 0-4, so end_page=5.
                markdown_content = pymupdf4llm.to_markdown(
                    doc, 
                    pages=pages,
                    ignore_images=True, 
                    ignore_graphics=True, 
                    ignore_code=True
                )
                
                # Step 3: After getting the markdown, perform the precise line-by-line check
                # to ensure we stop exactly at a *Markdown heading* for references.
                lines = markdown_content.split('\n')
                text_before_references = []
                for line in lines:
                    if self.is_reference_heading_line(line):
                        break # Stop processing lines, we've found the references section
                    else:
                        text_before_references.append(line)
                
                return self.clean_text("\n".join(text_before_references))
            except Exception as e:
                print(f"Error parsing PDF {self.file_path}: {e}")
                return ""
            finally:
                doc.close() # Ensure the document is closed even if errors occur


            # try:
            #     text = pymupdf4llm.to_markdown(self.file_path, ignore_images=True, ignore_graphics=True)
            #     lines = text.split('\n')
            #     text_before_references = []
            #     for line in lines:
            #         # Check if the current line is a reference heading
            #         if self.is_reference_heading_line(line):
            #             break # Stop processing lines, we've found the references section
            #         else:
            #             text_before_references.append(line) # Keep adding lines if not a reference heading
            #     return self.clean_text("\n".join(text_before_references))
            # except Exception as e:
            #     print(f"Error parsing PDF {self.file_path}: {e}")
            #     return ""
        elif self.file_path.endswith(".txt"):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return ""

    def extract_first_few_sentences(self, text: str, nlp: Language, num_sentences: int = 5) -> str:
        """
        Extracts the first few sentences from the text.
        
        Args:
            text (str): The input text.
            num_sentences (int): The number of sentences to extract.
            
        Returns:
            str: The first few sentences from the text.
        """
        if not text:
            return ""
        
        # Return the first few sentences as a single string
        doc = nlp(text)
        sentences = list(doc.sents)
        return " ".join([sent.text for sent in sentences[:num_sentences]]).strip()

    def extract_article_data_from_text(self, full_text: str, nlp: Language) -> ArticleData:
        """
        Extracts article data from the full text.
        
        Args:
            full_text (str): The full text of the article.
            article_id (str): The ID of the article.
            
        Returns:
            ArticleData: An instance of ArticleData with extracted information.
        """
        abstract_match = re.search(r"Abstract\s*(.*?)(?=\n\n|\Z)", full_text, re.IGNORECASE | re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else "No Abstract"
        abstract = self.extract_first_few_sentences(abstract[:400], nlp, num_sentences=3)  # Extract first few sentences for the abstract

        return ArticleData(
            article_id=self.article_id,
            abstract=abstract
        )

    # 4.2. Function to extract context around an ID
    def populate_context_around_citation(self, sentences, dataset_citation: DatasetCitation, window_size_sentences: int = 3) -> DatasetCitation:
        """
        Extracts a window of sentences around a given dataset citation in the text.
        Uses spaCy for sentence segmentation.
        """
        dataset_id = dataset_citation.dataset_id
        citation_type = dataset_citation.citation_type
        if not sentences or not dataset_id or dataset_id == "Missing":
            return dataset_citation
            
        # Find all occurrences of the dataset_id (case-insensitive)
        matches = [(i, sent) for i, sent in enumerate(sentences) if dataset_id.lower() in sent.lower()]
        for idx, sentence in matches:
            start_idx = max(0, idx - window_size_sentences)
            end_idx = min(len(sentences), idx + 1)
            context_sentences = sentences[start_idx:end_idx]
            context = " ".join(context_sentences)
            if self.is_text_data_related(context):
                # Set citation context to the first data related match in the text
                dataset_citation.set_citation_context(context)
                return dataset_citation
            elif not dataset_citation.citation_context and citation_type and citation_type != "Missing":
                # Set citation context to the first match in the text
                dataset_citation.set_citation_context(context)

        return dataset_citation

    def extract_article_data_for_training(self, nlp: Language, ground_truth_list: list[dict[str, str]]) -> ArticleData:
        """
        Extracts article data for training set with ground truth.
        
        Args:
            file_paths_df (pd.DataFrame): DataFrame containing file paths and ground truth info.
            
        Returns:
            Dict[str, ArticleData]: Dictionary mapping article IDs to ArticleData objects.
        """
        full_text = self.extract_text_from_file()
        article_data = self.extract_article_data_from_text(full_text, nlp)
        # potential_dataset_ids = self.find_potential_dataset_ids(full_text)
        # ground_truth_list = self.update_ground_truth_list_with_missing_ids(ground_truth_list, potential_dataset_ids)

        doc = nlp(full_text)
        sentences = [sent.text for sent in doc.sents]

        for gt in ground_truth_list:
            gt_id = self.strip_dataset_id(gt['dataset_id'])
            gt_type = gt.get('type', 'Primary')
            citation = DatasetCitation(dataset_id=gt_id, citation_type=gt_type)
            citation = self.populate_context_around_citation(sentences, citation)
            article_data.add_dataset_citation(citation)

        return article_data

    def extract_article_data_for_inference(self, nlp: Language) -> ArticleData:
        full_text = self.extract_text_from_file()
        article_data = self.extract_article_data_from_text(full_text, nlp)
        potential_dataset_ids = self.find_potential_dataset_ids(full_text)
        if potential_dataset_ids:
            doc = nlp(full_text)
            sentences = [sent.text for sent in doc.sents]

            # Populate article_data with potentially valid dataset_citations
            # the set_citation_context and add_dataset_citation methods do all of the appropriate filtering
            for dataset_id in potential_dataset_ids:
                citation = DatasetCitation(dataset_id=dataset_id)
                citation = self.populate_context_around_citation(sentences, citation)
                article_data.add_dataset_citation(citation)

        return article_data

    def find_potential_dataset_ids(self, text: str) -> list[str]:
        """
        Finds potential dataset IDs in the given text using predefined regex patterns.
        
        Args:
            text (str): The input text to search for dataset IDs.
            
        Returns:
            Set[str]: A set of unique dataset IDs found in the text.
        """
        dataset_ids = set()
        for regex in MdcFileTextExtractor.COMPILED_DATASET_ID_REGEXES:
            for match in re.finditer(regex, text):
                if regex.pattern == MdcFileTextExtractor.DOI_PATTERN:
                    dataset_id = match.group(1)
                else:
                    dataset_id = match.group(0)
                dataset_id = dataset_id.strip('.')
                dataset_ids.add(dataset_id)
        return list(dataset_ids)

    def update_ground_truth_list_with_missing_ids(self, ground_truth_list: list[dict], potential_dataset_ids: list[str]) -> list[dict]:
        # Collect all dataset_ids already present
        existing_ids = {self.strip_dataset_id(d['dataset_id']) for d in ground_truth_list if 'dataset_id' in d}
        # Add missing ids from other_ids
        for dataset_id in potential_dataset_ids:
            if dataset_id not in existing_ids:
                ground_truth_list.append({'dataset_id': dataset_id, 'type': 'Missing'})
        return ground_truth_list
    
    def strip_dataset_id(self, dataset_id: str)-> str:
        return dataset_id.replace("https://doi.org/", "").replace("http://dx.doi.org/", "").replace("doi:", "").strip()
