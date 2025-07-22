# mdc_data_processing_utils.py

import os
import re
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple, ClassVar
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import fitz
from fitz import Document
import pymupdf4llm
from lxml import etree # For XML parsing
import pandas as pd
from tqdm.auto import tqdm
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token


# -----------------------------------------
# Data classes:
# -----------------------------------------

@dataclass
class DatasetCitation:
    dataset_id: str = ""
    citation_context: str = ""
    citation_type: Optional[str] = None # "Primary", "Secondary", or "Missing" - for ground truth during training
    max_contet_len: int = 400
    # Renamed to sentence indices
    start_sentence_idx: int = 0
    end_sentence_idx: int = 0
    section_name: Optional[str] = None # Still useful to know the section name

    # Removed section_start_char_idx as raw_text is no longer stored in ArticleData

    def set_citation_context(self, context: str, start_sentence_idx: int, end_sentence_idx: int):
        """
        Sets the citation context, cleaning it and limiting to last 400 characters.
        Note: 'context' is now expected to be the pre-extracted string.
        'start_sentence_idx' and 'end_sentence_idx' are metadata about its location.
        """
        if context:
            # Replace newlines with spaces, remove brackets, and normalize whitespace
            context = context.replace('\n', ' ').replace('[', '').replace(']', '')
            context = re.sub(r'\s+', ' ', context.strip())
            self.citation_context = context[-self.max_contet_len:] # Limit to last 400 characters
            self.start_sentence_idx = start_sentence_idx
            self.end_sentence_idx = end_sentence_idx

    def is_doi(self)-> bool:
        return self.dataset_id.startswith("10.")
    
    def has_dataset(self) -> bool:
        """Returns True if there are both dataset IDs and citation context."""
        return bool(self.dataset_id and self.citation_context.strip())

    def to_dict(self):
        return asdict(self)


# 1. New Data Structure for Article Sections (Refactored)
@dataclass
class ArticleSection:
    name: str
    start_sentence_idx: int # Index in the list of sentences
    # Removed start_char_idx and end_char_idx
    end_sentence_idx: Optional[int] = None # Index in the list of sentences (exclusive)

    # Class-level constant for typical sections
    TYPICAL_ARTICLE_SECTIONS: ClassVar[List[str]] = [
        "Abstract", "Introduction", "Methods", "Data and Methods", "Data Availability", "Data Accessibility",
        "Results", "Discussion", "Conclusions", "Acknowledgments", "References", "Bibliography", "Appendix", 
        "Citations", "Works Cited", "Reference List"
    ]

    @staticmethod
    def normalize_section_name(name: str) -> str:
        """
        Normalizes a potential section heading for robust matching.
        Removes leading/trailing numbers/roman numerals/punctuation and converts to lowercase.
        """
        name = re.sub(r'^\s*(\d+\.?|\w+\.)?\s*', '', name)
        name = re.sub(r'[\s\W_]+$', '', name)
        return name.strip().lower()

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
    sentences: List[str] = field(default_factory=list)
    sentence_char_spans: List[tuple[int, int]] = field(default_factory=list) # Still needed for external context extraction
    sections: List[ArticleSection] = field(default_factory=list)
    dataset_citations: List[DatasetCitation] = field(default_factory=list)

    def __post_init__(self):
        # Custom initialization
        if self.article_id and not self.article_doi:
            # If article_id is provided but not article_doi, set article_doi
            self.article_doi = self.article_id.replace("_", "/").lower()

    def remove_extra_spaces(self, text):
        # Remove extra whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def add_dataset_citation(self, dataset_citation: DatasetCitation):
        """Adds a DatasetCitation object to the article."""
        ds_id_lower = dataset_citation.dataset_id.lower()
        if dataset_citation.has_dataset() and ds_id_lower != self.article_doi:
            existing_ids = ",".join({d.dataset_id.lower() for d in self.dataset_citations})
            # Don't include partial dataset_ids
            if ds_id_lower not in existing_ids:
                self._assign_citation_to_section(dataset_citation)
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
    
    def set_sentences(self, nlp_sentences: List):
        """
        Sets the sentences from externally tokenized data.
        This method should be called after initializing ArticleData.
        
        Args:
            sentence_data: A list of tuples, where each tuple is (sentence_text, start_char_idx, end_char_idx).
        """
        self.sentences = [sent.text.replace(' ||Block break.', '') for sent in nlp_sentences]
        
        
        # After setting sentences, identify sections based on them
        self._identify_sections()
        # print(self.sentences)

    def _identify_sections(self):
        """
        Identifies common research article sections within the text
        and populates the `self.sections` list using sentence indices.
        """
        self.sections = []
        normalized_typical_sections = [
            ArticleSection.normalize_section_name(s)
            for s in ArticleSection.TYPICAL_ARTICLE_SECTIONS
        ]
        
        last_section_idx = -1 # Index of the last identified section in self.sections list

        for i, sentence in enumerate(self.sentences):
            normalized_sentence = ArticleSection.normalize_section_name(sentence)
            section_name = self.get_section_name(normalized_sentence, normalized_typical_sections)
            if section_name:
                # If a previous section was identified, set its end_sentence_idx
                if last_section_idx != -1:
                    # The previous section ends just before the current new section begins
                    self.sections[last_section_idx].end_sentence_idx = i - 1

                # Add the new section
                new_section = ArticleSection(
                    name=section_name,
                    start_sentence_idx=i
                )
                self.sections.append(new_section)
                last_section_idx = len(self.sections) - 1
                if 'Abstract' == section_name or (not self.abstract and 'Introduction' == section_name):
                    self.abstract = self.remove_extra_spaces(" ".join(self.sentences[i:i+4]))
        
        # After iterating through all sentences, set the end_sentence_idx for the very last identified section.
        # It extends to the end of the sentences list.
        if last_section_idx != -1 and self.sentences:
            self.sections[last_section_idx].end_sentence_idx = len(self.sentences) - 1
        elif last_section_idx != -1 and not self.sentences:
            # Edge case: if sections were identified but no sentences (shouldn't happen with current logic)
            self.sections[last_section_idx].end_sentence_idx = None 

    def get_section_name(self, normalized_sentence, normalized_typical_sections) -> str | None:
        section_name = None
        if normalized_sentence in normalized_typical_sections:
            section_name = ArticleSection.TYPICAL_ARTICLE_SECTIONS[normalized_typical_sections.index(normalized_sentence)]

        if not section_name and not self.abstract and normalized_sentence.startswith("abstract"):
            section_name = "Abstract"

        # for norm_sec in normalized_typical_sections:
        #     if normalized_sentence.startswith(norm_sec):
        #         section_name = ArticleSection.TYPICAL_ARTICLE_SECTIONS[normalized_typical_sections.index(norm_sec)]
        #         print(section_name, normalized_sentence)
        #         return section_name
        return section_name

    def find_section_for_sentence_index(self, sentence_idx: int) -> Optional[ArticleSection]:
        """
        Finds the ArticleSection that contains the given sentence index.
        Assumes sections are sorted by start_sentence_idx.
        """
        for section in reversed(self.sections):
            if section.start_sentence_idx <= sentence_idx:
                if section.end_sentence_idx is None or sentence_idx <= section.end_sentence_idx:
                    return section
        return None # Not found in any defined section

    def assign_citations_to_sections(self):
        """
        Iterates through stored DatasetCitations and assigns them to their respective sections
        based on sentence indices.
        """
        for citation in self.dataset_citations:
            # Use the start_sentence_idx of the citation to find its containing section
            containing_section = self.find_section_for_sentence_index(citation.start_sentence_idx)
            if containing_section:
                citation.section_name = containing_section.name
                # No section_start_char_idx to assign here
            else:
                citation.section_name = "Unknown"

    def _assign_citation_to_section(self, citation: DatasetCitation):
        """
        Iterates through stored DatasetCitations and assigns them to their respective sections
        based on sentence indices.
        """
        # Use the start_sentence_idx of the citation to find its containing section
        containing_section = self.find_section_for_sentence_index(citation.end_sentence_idx)
        if containing_section:
            citation.section_name = containing_section.name
        else:
            citation.section_name = "Unknown"

    def to_dict(self):
        """
        Converts the ArticleData object to a dictionary for serialization,
        excluding the 'sentences' list for memory efficiency.
        """
        data = asdict(self)
        data.pop('sentences', None) # Exclude sentences from serialization
        data.pop('sections', None) # Exclude sentences from serialization
        # data['sections'] = [s.to_dict() for s in self.sections]
        data['dataset_citations'] = [c.to_dict() for c in self.dataset_citations]
        return data

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
    

# -----------------------------------------
# Text extractor class:
# -----------------------------------------


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
    NON_STD_UNICODE_TICKS = re.compile(r'[\u201c\u201d\u2019]')

    def __init__(self, file_path: str):
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
        text = text.replace('\u200b', '').replace('-\n', '-').replace('_\n', '_').replace('/\n', '/').replace(', ... ', ', ')
        text = text.replace('dryad.\n', 'dryad.').replace('doi.\norg', 'doi.org')
        text = text.replace('doi.org/10.\n', 'doi.org/10.').replace('http://dx.doi.org/10.', 'https://doi.org/10.')
        # Remove extra whitespace
        # return re.sub(r' {2,}', ' ', text).strip()
        return re.sub(r'\s+', ' ', text).strip()
    
    def is_text_data_related(self, text: str) -> bool:
        if not text:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in MdcFileTextExtractor.DATA_RELATED_KEYWORDS)

    def extract_article_data_from_text(self, full_text: str, nlp: Language) -> ArticleData:
        """
        Extracts article data from the full text.
        
        Args:
            full_text (str): The full text of the article.
            article_id (str): The ID of the article.
            
        Returns:
            ArticleData: An instance of ArticleData with extracted information.
        """
        doc = nlp(full_text)
        article_data = ArticleData(article_id=self.article_id)
        article_data.set_sentences(list(doc.sents))
        return article_data

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
                dataset_citation.set_citation_context(context, start_idx, end_idx)
                return dataset_citation
            elif not dataset_citation.citation_context and citation_type and citation_type != "Missing":
                # Set citation context to the first match in the text
                dataset_citation.set_citation_context(context, start_idx, end_idx)

        return dataset_citation

    def extract_article_data_for_training(self, full_text: str, nlp: Language, ground_truth_list: list[dict[str, str]]) -> ArticleData:
        """
        Extracts article data for training set with ground truth.
        
        Args:
            file_paths_df (pd.DataFrame): DataFrame containing file paths and ground truth info.
            
        Returns:
            Dict[str, ArticleData]: Dictionary mapping article IDs to ArticleData objects.
        """
        full_text = self.clean_text(full_text)
        article_data = self.extract_article_data_from_text(full_text, nlp)
        for gt in ground_truth_list:
            gt_id = self.strip_dataset_id(gt['dataset_id'])
            gt_type = gt.get('type', 'Primary')
            citation = DatasetCitation(dataset_id=gt_id, citation_type=gt_type)
            citation = self.populate_context_around_citation(article_data.sentences, citation)
            article_data.add_dataset_citation(citation)

        article_data.assign_citations_to_sections()
        return article_data

    def extract_article_data_for_inference(self, full_text: str, nlp: Language) -> ArticleData:
        full_text = self.clean_text(full_text)
        article_data = self.extract_article_data_from_text(full_text, nlp)
        potential_dataset_ids = self.find_potential_dataset_ids(full_text)
        if potential_dataset_ids:
            # Populate article_data with potentially valid dataset_citations
            # the set_citation_context and add_dataset_citation methods do all of the appropriate filtering
            for dataset_id in potential_dataset_ids:
                citation = DatasetCitation(dataset_id=dataset_id)
                citation = self.populate_context_around_citation(article_data.sentences, citation)
                article_data.add_dataset_citation(citation)

        article_data.assign_citations_to_sections()
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
    
    def strip_dataset_id(self, dataset_id: str)-> str:
        return dataset_id.replace("https://doi.org/", "").replace("http://dx.doi.org/", "").replace("doi:", "").strip()


# -----------------------------------------
# File processing classes and functions:
# -----------------------------------------

# --- Top Level File Reading Functions ---
def _read_pdf_to_markdown(pdf_filepath):
    result = pymupdf4llm.to_markdown(
        pdf_filepath,
        ignore_images=True,
        ignore_graphics=True,
        ignore_code=True
    )
    return result

def _read_pdf_plain_text(pdf_filepath):
    plain_text = ""
    with(fitz.open(pdf_filepath)) as doc:
        for page in doc:
            # page_text = page.get_textpage().extractTEXT()
            blocks = page.get_textpage().extractBLOCKS()
            blocks.sort(key=lambda b: (b[0], b[1]))
            for block in blocks:
                plain_text += block[4] + "||Block break. "
    return plain_text

# --- New Mock XML Processing Function ---
def _read_xml_file(xml_filepath):
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    try:
        tree = etree.parse(xml_filepath, parser)
        return " ".join(tree.xpath("//text()")).strip()
    except Exception as e:
        print(f"Error parsing XML {xml_filepath}: {e}")
        return ""

# --- File Specific Processing Task Functions ---

def _process_pdf_task(pdf_filepath: str, markdown_timeout_seconds: float = 3.0) -> tuple[str, str]:
    """
    Encapsulates the PDF processing logic (Markdown with timeout, fallback to plain text).
    Returns content and metadata about the processing.
    """
    base_name = os.path.basename(pdf_filepath)
    content = None
    status = "success"

    # with ThreadPoolExecutor(max_workers=1) as markdown_executor:
    #     markdown_future = markdown_executor.submit(_read_pdf_to_markdown, pdf_filepath)
    #     try:
    #         content = markdown_future.result(timeout=markdown_timeout_seconds)
    #         status = "success"
    #     except TimeoutError:
    #         status = "timeout_fallback"
    #     except Exception as e:
    #         status = "error_fallback"

    if content is None:
        try:
            content = _read_pdf_plain_text(pdf_filepath)
            status = "plain_text"
        except Exception as e:
            content = f"Error reading file {base_name}: {e}."
            status = "error"

    return content, status

def _process_xml_task(xml_filepath: str) -> tuple[str, str]:
    """
    Encapsulates the XML processing logic.
    Returns content and metadata about the processing.
    """
    base_name = os.path.basename(xml_filepath)
    try:
        content = _read_xml_file(xml_filepath)
        status = "success"
    except Exception as e:
        content = f"Error reading file {base_name}: {e}."
        status = "error"

    return content, status

# --- Generic Worker Function (submitted to ThreadPoolExecutor) ---
def _generic_file_worker(filepath: str, output_dir: str, nlp: Language, ground_truth_list: list[dict] | None = None, **logic_kwargs) -> ArticleData | None:
    """
    Generic worker function that calls a specific processing logic function
    and handles saving the results.
    """
    base_name: str = os.path.basename(filepath)
    processing_task_func = _process_pdf_task
    if base_name.lower().endswith('xml'):
        processing_task_func = _process_xml_task
    print(f"Processing {base_name}...")

    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Call the specific file type processing function
        full_text, status = processing_task_func(filepath, **logic_kwargs)
        if "error" == status:
            print(full_text)
        text_extractor = MdcFileTextExtractor(filepath)
        if ground_truth_list:
            article_data = text_extractor.extract_article_data_for_training(full_text, nlp, ground_truth_list)
        else:
            article_data = text_extractor.extract_article_data_for_inference(full_text, nlp)
        
        # Save the result
        output_filepath = os.path.join(output_dir, base_name + ".json")
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(article_data.to_dict(), f_out, indent=2)
        print(f"Saved article_data for {base_name}.")
        return article_data
    except Exception as e:
        print(f"  [Generic Worker] Error processing or saving result for {base_name}: {e}")
        return None

# --- Concurrent File Processor Class ---
class ConcurrentFileProcessor:
    def __init__(self, nlp: Language, output_dir="processed_files", max_workers=None):
        self.nlp = nlp
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.max_workers = max_workers 

    def process_files_for_inference(self, filepaths: list[str], **logic_kwargs) -> list[ArticleData]:
        """
        Processes files concurrently using ThreadPoolExecutor.
        Uses a specified processing_logic_func for each file.
        """
        infr_out_dir = os.path.join(self.output_dir, 'infr')
        os.makedirs(infr_out_dir, exist_ok=True)

        print("\n--- Starting Concurrent File Processing For Inference Data ---")
        start_time = time.time()
        article_data_list: list[ArticleData] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_generic_file_worker, filepath, infr_out_dir, self.nlp, **logic_kwargs): filepath
                for filepath in tqdm(filepaths, total=len(filepaths))
            }
            
            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    article_data = future.result()
                    if article_data: 
                        article_data_list.append(article_data)
                except Exception as exc:
                    print(f'{filepath} generated an unhandled exception: {exc}')
        
        end_time = time.time()
        print(f"Inference Data processing finished in {end_time - start_time:.2f} seconds.")
        return article_data_list

    def process_files_for_training(self, filepaths: list[str], ground_truth_list_of_lists: list[list[dict]], **logic_kwargs):
        """
        Processes files concurrently using ThreadPoolExecutor.
        Uses a specified processing_logic_func for each file.
        """
        train_out_dir = os.path.join(self.output_dir, 'train')
        os.makedirs(train_out_dir, exist_ok=True)

        print("\n--- Starting Concurrent File Processing For Training Data ---")
        start_time = time.time()
        training_data_for_llm: list[dict[str, str]] = [] # This will be a list of LlmTrainingData from the training dataset
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_generic_file_worker, filepath, train_out_dir, self.nlp, ground_truth_list_of_lists[i], **logic_kwargs): (i, filepath)
                for i, filepath in tqdm(enumerate(filepaths), total=len(filepaths))
            }
            
            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    article_data = future.result()
                    if article_data:
                        training_data_for_llm.extend(article_data.get_data_for_llm())
                except Exception as exc:
                    print(f'{filepath} generated an unhandled exception: {exc}')
        
        end_time = time.time()
        print(f"Training Data processing finished in {end_time - start_time:.2f} seconds.")
        return training_data_for_llm
