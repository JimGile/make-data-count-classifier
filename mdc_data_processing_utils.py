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
import pymupdf
import pymupdf4llm
from lxml import etree # For XML parsing
import pandas as pd
from tqdm.auto import tqdm
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

# Import torch for device handling in inference functions
import torch 


# -----------------------------------------
# Data classes:
# -----------------------------------------

@dataclass
class DatasetCitation:
    dataset_id: str = ""
    citation_sentence: str = ""
    citation_context: str = ""
    citation_type: Optional[str] = None # "Primary", "Secondary", or "Missing" - for ground truth during training
    max_contet_len: int = 400
    start_sentence_idx: int = 0
    end_sentence_idx: int = 0
    section_name: str = "Unknown"
    primary_score: int = 0
    secondary_score: int = 0
    data_score: int = 0
    non_data_score: int = 0

    # Class-level constant for typical sections
    PRIMARY_SCORE_WORDS: ClassVar[List[str]] = [" we ", " our ", "the author", "created", "generated", "deposited", "presented", "made available", "archived", "submitted", "uploaded", "sequenced", "segmented", "vetted", "openly available", "freely available", "dryad", "zenodo"]
    SECONDARY_SCORE_WORDS: ClassVar[List[str]] = ["accessed", "retrieved", "downloaded", "obtained", "associated", "provided by", "data from", "data used", "publicly available", "available at", "referring", "supplementa", "supporting"]
    DATA_SCORE_WORDS: ClassVar[List[str]] = ["data avail", "data access", "dataset", "database", "segment", "sequence", "repositor", "archive", "accession", "program", "digital", "model", " dems", "file", "author", "data",]
    # DATA_SCORE_WORDS: ClassVar[List[str]] = ["dataset", "database", "segment", "sequence", "repositor", "archive", "available", "access", "program", "associated", "referring", "supplementa", "supporting", "digital", "model", "file", "author", "data",]
    NON_DATA_SCORE_WORDS: ClassVar[List[str]] = ["bulletin", "journal", "proceedings", "10.1029"]
    # DATA_RELATED_KEYWORDS = ['data release', 'data associated', 'data referring', 'data availability', 'data access', 'data source', 'program data', 'our data', 'the data', 'dataset', 'database', ' segmented by', 'digital elevation model']

    def set_citation_context(self, sentence: str, context: str, start_sentence_idx: int, end_sentence_idx: int):
        """
        Sets the citation context, cleaning it and limiting to last 400 characters.
        Note: 'context' is now expected to be the pre-extracted string.
        'start_sentence_idx' and 'end_sentence_idx' are metadata about its location.
        """
        if sentence and context:
            self.citation_sentence = sentence
            self.citation_context = context[-self.max_contet_len:] # Limit to last 400 characters
            self.start_sentence_idx = start_sentence_idx
            self.end_sentence_idx = end_sentence_idx

    def score_citation(self):
            self.primary_score = self._calculate_score(DatasetCitation.PRIMARY_SCORE_WORDS)
            self.secondary_score = self._calculate_score(DatasetCitation.SECONDARY_SCORE_WORDS)
            self.data_score = self._calculate_score(DatasetCitation.DATA_SCORE_WORDS)
            self.non_data_score = self._calculate_score(DatasetCitation.NON_DATA_SCORE_WORDS)

    def _calculate_score(self, score_words: list[str]) -> int:
        score = 0
        context_lower = self.citation_context.lower()
        for keyword in score_words:
            score += context_lower.count(keyword)

        if score_words == DatasetCitation.SECONDARY_SCORE_WORDS:
            ref_sec_score = 1 if self.is_in_reference_section() else 0
            score += ref_sec_score

        # Check for pattern that looks like page numbers e.g. 758-792
        if score_words == DatasetCitation.NON_DATA_SCORE_WORDS:
            ref_sec_score = 3 if self.is_in_reference_section() else 0
            page_numbers_found = len(re.findall(r',\s+\d{2,5}-\d{2,5}', context_lower))
            if page_numbers_found > 0:
                score += page_numbers_found + ref_sec_score
                # print(f"dataset_id={self.dataset_id}, score={score}, ref_sec_score={ref_sec_score}, page_numbers_found={page_numbers_found}")
        return score
    
    def get_total_score(self) -> int:
        return self.primary_score + self.secondary_score
    
    def get_total_data_score(self) -> int:
        return self.data_score - self.non_data_score
    
    def is_primary(self)-> bool:
        return self.is_data and self.primary_score >= self.secondary_score
    
    def is_secondary(self)-> bool:
        return self.is_data and self.secondary_score > self.primary_score
    
    def is_data(self)-> bool:
        return self.primary_score >= self.non_data_score
    
    def is_doi(self)-> bool:
        return self.dataset_id.startswith("10.")
    
    def is_accession_number(self)-> bool:
        return not self.is_doi()

    def is_in_reference_section(self)-> bool:
        return self.section_name in ArticleSection.REFERENCE_SECTIONS
    
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
        "Abstract", "Introduction", "Methods", "Data and Methods", 
        "Data Availability", "Data Availability Statement", 
        "Data Accessibility", "Data Accessibility Statement",
        "Results", "Discussion", "Conclusions", "Acknowledgments", 
        "References", "Bibliography", "Appendix", 
        "Citations", "Works Cited", "Reference List"
    ]

    REFERENCE_SECTIONS: ClassVar[List[str]] = [
        "References", "Bibliography", "Appendix", 
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
    sections: List[ArticleSection] = field(default_factory=list)
    dataset_citations: List[DatasetCitation] = field(default_factory=list)

    PRIMARY_DATA_SOURCES: ClassVar[List[str]] = ["dryad", "zenodo"]

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
                self._assign_citation_to_section_and_score(dataset_citation)
                self._update_citation_primary_score(dataset_citation)
                self.dataset_citations.append(dataset_citation)
        
    def remove_extraneous_citations(self):
        self.dataset_citations = [item for item in self.dataset_citations if item.is_data()]
        doi_citations = self.get_doi_citations()
        for doi in doi_citations:
            self.remove_accession_numbers_for_primary_doi(doi)

    def get_doi_citations(self) -> list[DatasetCitation]:
        return [item for item in self.dataset_citations if item.is_doi()]

    def remove_accession_numbers_for_primary_doi(self, doi: DatasetCitation):
        if doi.is_primary():
            self.dataset_citations = self.get_doi_citations()

    def deduplicate_citations_by_section_and_score(self):
        """
        Deduplicates dataset citations within the ArticleData object.
        For citations found in the same ArticleSection, only the one with the
        highest primary_score is retained. If scores are tied, the citation
        that was encountered first for that section is preferred.
        """
        best_citations_per_section: Dict[str, DatasetCitation] = {}

        for citation in self.dataset_citations:
            section_name = citation.section_name
            if section_name is None:
                continue

            if section_name not in best_citations_per_section:
                # This is the first citation encountered for this section, so it's the best so far.
                best_citations_per_section[section_name] = citation
            else:
                # Compare with the current best citation for this section
                current_best = best_citations_per_section[section_name]
                if citation.primary_score > current_best.primary_score:
                    # If the new citation has a strictly higher primary_score, it becomes the new best.
                    best_citations_per_section[section_name] = citation

        # Update the article's dataset_citations list with the deduplicated results
        self.dataset_citations = list(best_citations_per_section.values())
        
    def get_data_for_llm(self) -> list[dict[str, str | int]]:
        data_for_llm: list[dict[str, str | int]] = []
        for citation in self.dataset_citations:
            # Convert to dict for LLM training data
            data_for_llm.append(
                {
                    "article_id": self.article_id,
                    "article_author": self.author,
                    "article_abstract": self.abstract,
                    "dataset_id": citation.dataset_id,                    
                    "citation_context": citation.citation_context,
                    "article_section": citation.section_name,
                    "primary_score": citation.primary_score,
                    "secondary_score": citation.secondary_score,
                    "non_data_score": citation.non_data_score,
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
        sentences = [sent.text.replace('This is a block break.', '').replace('||PAGE||', '').strip() for sent in nlp_sentences]
        self.sentences = [s for s in sentences if s != '']
                
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
                    self.abstract = self.remove_extra_spaces(" ".join(self.sentences[i:i+3]))
        
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

    def _assign_citation_to_section_and_score(self, citation: DatasetCitation):
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
        citation.score_citation()

    def _update_citation_primary_score(self, citation: DatasetCitation):
        data_source_score = len([ds for ds in ArticleData.PRIMARY_DATA_SOURCES if ds in citation.dataset_id])
        citation.primary_score += data_source_score
        if self.author:
            id_idx = citation.citation_context.find(citation.dataset_id)
            author_names = self.author.split()
            for name in author_names:
                name_idx = citation.citation_context.find(name)
                if name_idx > 0 and name_idx < id_idx:
                    citation.primary_score += 2

    def clean_up(self):
        """
        Clears large in-memory components (like raw sentences) after
        all necessary processing (sectioning, citation context) is done.
        This reduces the memory footprint of the ArticleData object
        when stored in a list for subsequent steps like inference.
        """
        self.sentences = [] # Clear the list of sentence strings
        self.sections = []  # Clear the list of sections

    def to_dict(self):
        """
        Converts the ArticleData object to a dictionary for serialization,
        excluding the 'sentences' list for memory efficiency.
        """
        data = asdict(self)
        data.pop('sentences', None) # Exclude sentences from serialization
        data.pop('sections', None) # Exclude sections from serialization
        data['sections'] = [s.to_dict() for s in self.sections]
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
    EPI_PATTERN = r'\bEPI[_A-Z0-9]{5,}'
    SAM_PATTERN = r'\bSAMN[0-9]{6,}'          # SAMN07159041
    IPR_PATTERN = r'\bIPR[0-9]{6,}'
    CHE_PATTERN = r'\bCHEMBL[0-9]{2,}'
    PRJ_PATTERN = r'\bPRJ[A-Z0-9]{4,}'
    E_G_PATTERN = r'\bE-[A-Z]{4}-[0-9]{2,}'   # E-GEOD-19722 or E-PROT-100
    ENS_PATTERN = r'\bENS[A-Z]{4}[0-9]{10,}'
    CVC_PATTERN = r'\bCVCL_[A-Z0-9]{4,}'
    EMP_PATTERN = r'\bEMPIAR-[0-9]{5,}'
    PXD_PATTERN = r'\bPXD[0-9]{6,}'
    HPA_PATTERN = r'\bHPA[0-9]{6,}'
    SRR_PATTERN = r'\bSRR[0-9]{6,}'
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
    DATA_RELATED_KEYWORDS = ['data release', 'data associated', 'data referring', 'data availability', 'data access', 'data source', 'program data', 'our data', 'the data', 'dataset', 'database', ' segmented by', 'digital elevation model', ' dems', 'dna sequences', 'bam files']

    NON_STD_UNICODE_DASHES = re.compile(r'[\u2010\u2011\u2012\u2013\u2014]')
    NON_STD_UNICODE_TICKS = re.compile(r'[\u201c\u201d\u2019]')
    NON_STD_ZERO_WIDTH_SPACE = re.compile(r'\u200b')

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

        # Remove extra whitespace
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
        article_data = ArticleData(article_id=self.article_id)
        article_data.author = self.extract_author_name(full_text=full_text, nlp=nlp)
        full_text = full_text.replace('||PAGE||', '')
        doc = nlp(full_text)
        article_data.set_sentences(list(doc.sents))
        return article_data
    
    def extract_author_name(self, full_text: str, nlp: Language) -> str:
        """
        Extracts potential primary author name from the beginning of a research article's text
        using spaCy's Named Entity Recognition. It attempts to isolate the author section
        and applies heuristics to filter out non-author entities.

        Args:
            full_text (str): The complete text content of the research article,
                            typically extracted from a PDF.

        Returns:
            List[str]: A list of unique strings, each representing a potential author name,
                    sorted alphabetically. Returns an empty list if no authors are found.
        """
        if not full_text or not full_text.strip():
            return ""

        author_section_text = full_text.split('||PAGE||')[0]
        author_section_text = author_section_text.replace('1\n,', ',').replace('1,', ',').replace('\u2019', "'")

        # 2. Process the isolated author section with spaCy
        doc = nlp(author_section_text)

        # 3. Extract PERSON entities and apply initial filtering
        potential_authors: list[str] = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                # Basic filtering to reduce false positives:
                # - Exclude very short strings (e.g., single letters, common conjunctions)
                # - Exclude common stop words (e.g., "The", "And")
                # - Exclude all-uppercase strings that might be acronyms (e.g., "WHO", "NASA")
                # - Ensure it contains at least one space (e.g., "John Doe") or is a capitalized
                #   single word that's longer than 2 characters (e.g., "Smith").
                if (len(name) > 1 and
                    name.lower() not in nlp.Defaults.stop_words and
                    not name.isupper() and
                    (' ' in name or (name[0].isupper() and len(name) > 2))):
                    
                    potential_authors.append(name)

        # 4. Apply more advanced heuristics to filter out non-author names
        # This step is crucial for accuracy and often requires tuning.
        for author in potential_authors:
            author = re.sub(r'[,\d[]', ' ', author)
            # Heuristic 1: Filter out names that contain common affiliation keywords.
            # This is a simple check; more robust solutions might use spaCy's dependency
            # parsing to check if a PERSON entity is part of an ORG entity.
            affiliation_keywords = ["univ", "observ", "institute", "department", "center", "lab",
                                    "hospital", "college", "school", "inc.", "ltd.", "company",
                                    "corp.", "group", "foundation", "research", "table", "figure"]
            if any(keyword in author.lower() for keyword in affiliation_keywords):
                continue # Skip if it looks like an affiliation

            # Heuristic 2: Filter out names that contain email patterns or ORCID patterns.
            if '@' in author or re.search(r'\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b', author):
                continue # Skip if it contains an email or ORCID

            # Heuristic 3: Filter out names that are likely just initials or very short.
            # This is partially covered by initial filtering, but can be refined.
            # E.g., "J. D." might be an author, but "J." alone is unlikely.
            if len(author.split()) == 1 and len(author) <= 2 and author.isupper():
                continue # Skip single-letter or two-letter uppercase (e.g., "JD")

            name = ""
            words = author.split()
            for word in words:
                if len(word) >=2 and not word.isupper():
                    name += word + " "
            return name.strip()

        # Convert to list and sort for consistent output
        return ""
    
    # 4.2. Function to extract context around an ID
    def populate_context_around_citation(self, article_data: ArticleData, dataset_citation: DatasetCitation, window_size_sentences: int = 3) -> DatasetCitation:
        """
        Extracts a window of sentences around a given dataset citation in the text.
        Uses spaCy for sentence segmentation.
        """
        dataset_id = dataset_citation.dataset_id
        citation_type = dataset_citation.citation_type
        sentences = article_data.sentences
        if not sentences or not dataset_id or dataset_id == "Missing":
            return dataset_citation
            
        # Find all occurrences of the dataset_id (case-insensitive)
        matches = [(i, sent) for i, sent in enumerate(sentences) if dataset_id.lower() in sent.lower()]
        for idx, sentence in matches:
            sentence = sentences[idx]
            section = article_data.find_section_for_sentence_index(idx)
            floor = section.start_sentence_idx if section else 0
            start_idx = max(floor, idx - window_size_sentences)
            end_idx = min(len(sentences), idx + 1)
            context_sentences = sentences[start_idx:end_idx]
            context = " ".join(context_sentences)
            if self.is_text_data_related(context):
                # Set citation context to the first data related match in the text
                dataset_citation.set_citation_context(sentence, context, start_idx, end_idx)
                return dataset_citation
            elif not dataset_citation.citation_context and citation_type and citation_type != "Missing":
                # Set citation context to the first match in the text
                dataset_citation.set_citation_context(sentence, context, start_idx, end_idx)

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
            citation = self.populate_context_around_citation(article_data, citation)
            article_data.add_dataset_citation(citation)

        article_data.clean_up()
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
                citation = self.populate_context_around_citation(article_data, citation)
                article_data.add_dataset_citation(citation)

        # article_data.remove_extraneous_citations()
        article_data.clean_up()
        return article_data

    def find_potential_dataset_ids(self, text: str) -> list[str]:
        """
        Finds potential dataset IDs in the given text using predefined regex patterns.
        
        Args:
            text (str): The input text to search for dataset IDs.
            
        Returns:
            Set[str]: A set of unique dataset IDs found in the text.
        """
        ds_id_list = []
        ds_id_set = set()
        for regex in MdcFileTextExtractor.COMPILED_DATASET_ID_REGEXES:
            for match in re.finditer(regex, text):
                if regex.pattern == MdcFileTextExtractor.DOI_PATTERN:
                    dataset_id = match.group(1)
                else:
                    dataset_id = match.group(0)
                dataset_id = dataset_id.strip('.')
                ds_id_set.add(dataset_id)
                if len(ds_id_set) > len(ds_id_list):
                    ds_id_list.append(dataset_id)
        return ds_id_list
    
    def strip_dataset_id(self, dataset_id: str)-> str:
        return dataset_id.replace("https://doi.org/", "").replace("http://dx.doi.org/", "").replace("doi:", "").strip()


# -----------------------------------------
# File processing classes and functions:
# -----------------------------------------

# --- Top Level File Reading Functions ---
def _read_pdf_plain_text(pdf_filepath, footer_margin=50, header_margin=50):
    flags = pymupdf.TEXTFLAGS_BLOCKS | pymupdf.TEXT_DEHYPHENATE
    plain_text = ""
    with(fitz.open(pdf_filepath)) as doc:
        for page in doc:
            clip = +page.rect
            clip.y1 -= footer_margin  # Remove footer area
            clip.y0 += header_margin  # Remove header area
            blocks = page.get_textpage(clip=clip, flags=flags).extractBLOCKS()
            # Sort by horizontal direction then by ascending vertical to handle multi column layouts.
            blocks.sort(key=lambda b: (int(b[0]), int(b[1])))
            for block in blocks:
                text =str(block[4]).replace(', ... ', ', ')
                # Replace non-standard unicode characters
                text = MdcFileTextExtractor.NON_STD_ZERO_WIDTH_SPACE.sub('', text)
                text = MdcFileTextExtractor.NON_STD_UNICODE_DASHES.sub('-', text)
                text = MdcFileTextExtractor.NON_STD_UNICODE_TICKS.sub("'", text)
                text = re.sub(r'(\.)(\d{1,2}\s+)([A-Z])', r'\g<1> \g<3>', text)
                text = text.replace('-\n', '-').replace('_\n', '_').replace('/\n', '/').replace(' and\n', ' and ')
                text = text.replace('//doi.\n', '//doi.').replace('//doi.org/10.\n', '//doi.org/10.')
                text = text.replace('/dryad.\n', '/dryad.').replace('/zenodo.\n', '/zenodo.')
                if text.endswith('\n'):
                    plain_text += text.replace('\n', ' ') + "This is a block break.\n"
                else:
                    plain_text += text.replace('\n', ' ')
            plain_text += "||PAGE||"

    plain_text = plain_text.replace('http://dx.doi.org/10.', 'https://doi.org/10.').replace(' This is a block break.\n/', '/')
    # print(plain_text)
    return plain_text

# --- XML Processing Function ---
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
    try:
        content = _read_pdf_plain_text(pdf_filepath)
        status = "success"
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
def _generic_file_worker(filepath: str, output_dir: str, nlp: Language, ground_truth_list: list[dict] | None = None) -> ArticleData | None:
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
        full_text, status = processing_task_func(filepath)
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

# --- New Inference-specific functions (moved from notebook) ---
def format_citation_prompt_for_inference(tokenizer: Any, article_data: ArticleData, dataset_citation: DatasetCitation):
    """
    Formats a single citation into a ChatML prompt for inference.
    """
    messages = [
        {"role": "system", "content": "You are an expert assistant for classifying research article data citations. /no_think"},
        {"role": "user", "content": (
            f"""
Given the following article information and a specific data citation ('Dataset ID' and 'Data Citation Context' combination) within that article, classify the data citation as **one** of the following: 
* **'Primary'**: If the data citation refers to raw or processed **data created/generated as part of the paper**, specifically for this study. (Hint: use the 'Primary Score' and 'Article Author' as a guide). 
* **'Secondary'**: If the data citation refers to raw or processed **data derived/reused from existing records** or previously published data. (Hint: use the 'Secondary Score' and 'Article Section' as a guide). 
* **'Missing'**: If the data citation refers to another **article/paper/journal**, a **figure, software, or other non-data entity**.  (Hint: use the 'Missing Score' and 'Article Section' as a guide).\n\n"""
            f"Now, classify the following:\n\n" 
            f"Article Author: {article_data.author}\n" 
            f"Article Abstract: {article_data.abstract}\n" 
            f"Article Section: {dataset_citation.section_name}\n" 
            f"Dataset ID: {dataset_citation.dataset_id}\n"                
            f"Data Citation Context: {dataset_citation.citation_context}\n\n"
            f"Primary Score: {dataset_citation.primary_score}\n\n"
            f"Secondary Score: {dataset_citation.secondary_score}\n\n"
            f"Missing Score: {dataset_citation.non_data_score}\n\n"
            f"Classification:"
        )}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    return inputs

def invoke_model_for_inference(tokenizer: Any, model: Any, article_data: ArticleData, device: Any) -> list[SubmissionData]:
    """
    Invokes the LLM for inference on all dataset citations within a single ArticleData object.
    """
    submission_data_list = []
    article_id = article_data.article_id
    dataset_citations = article_data.dataset_citations
    
    # If no citations are found after initial processing and filtering, add a "Missing" entry
    if not dataset_citations:
        submission_data_list.append(SubmissionData(article_id, dataset_id="Missing", type="Missing", context=""))
        return submission_data_list

    for dc in dataset_citations:
        inputs = format_citation_prompt_for_inference(tokenizer, article_data, dc)
        # Move inputs to the correct device here, as each thread will handle its own inputs
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Ensure model is in evaluation mode (important for inference)
        model.eval() 

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        predicted_type = "Missing"
        if "Primary" in generated_text:
            predicted_type = "Primary"
        elif "Secondary" in generated_text:
            predicted_type = "Secondary"
        
        submission_data_list.append(SubmissionData(article_id, dataset_id=dc.dataset_id, type=predicted_type, context=dc.citation_context))

    return submission_data_list

# --- New worker function for concurrent model inference ---
def _inference_worker_task(article_data: ArticleData, tokenizer: Any, model: Any, device: Any) -> list[SubmissionData]:
    """
    Worker function for concurrent model inference on a single ArticleData object.
    This function is designed to be submitted to a ThreadPoolExecutor.
    """
    try:
        return invoke_model_for_inference(tokenizer, model, article_data, device)
    except Exception as e:
        print(f"Error during inference for {article_data.article_id}: {e}")
        # Return a SubmissionData with "Missing" type for error cases
        return [SubmissionData(article_data.article_id, dataset_id="Error", type="Missing", context=f"Inference error: {e}")]


# --- Concurrent File Processor Class ---
class ConcurrentFileProcessor:
    def __init__(self, nlp: Language, output_dir="processed_files", max_workers=3):
        self.nlp = nlp
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.max_workers = max_workers 

    def process_files_for_inference(self, filepaths: list[str]) -> list[ArticleData]:
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
                executor.submit(_generic_file_worker, filepath, infr_out_dir, self.nlp): filepath
                for filepath in filepaths
            }
            
            for future in tqdm(as_completed(futures), total=len(filepaths)):
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

    def process_files_for_training(self, filepaths: list[str], ground_truth_list_of_lists: list[list[dict]]) -> list[dict[str, str | int]]:
        """
        Processes files concurrently using ThreadPoolExecutor.
        Uses a specified processing_logic_func for each file.
        """
        train_out_dir = os.path.join(self.output_dir, 'train')
        os.makedirs(train_out_dir, exist_ok=True)

        print("\n--- Starting Concurrent File Processing For Training Data ---")
        start_time = time.time()
        training_data_for_llm: list[dict[str, str | int]] = [] # This will be a list of LlmTrainingData from the training dataset
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_generic_file_worker, filepath, train_out_dir, self.nlp, ground_truth_list_of_lists[i]): (i, filepath)
                for i, filepath in enumerate(filepaths)
            }
            
            for future in tqdm(as_completed(futures), total=len(filepaths)):
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

    def run_inference_concurrently(self, article_data_list: list[ArticleData], tokenizer: Any, model: Any, device: Any) -> list[SubmissionData]:
        """
        Runs model inference concurrently on a list of pre-processed ArticleData objects.
        """
        print("\n--- Starting Concurrent Model Inference ---")
        start_time = time.time()
        all_submission_data: list[SubmissionData] = []

        # Use a new ThreadPoolExecutor for inference
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_inference_worker_task, ad, tokenizer, model, device): ad.article_id
                for ad in article_data_list
            }

            for future in tqdm(as_completed(futures), total=len(article_data_list), desc="Running Inference"):
                article_id = futures[future]
                try:
                    submission_data_for_article = future.result()
                    all_submission_data.extend(submission_data_for_article)
                except Exception as exc:
                    print(f'Inference for {article_id} generated an unhandled exception: {exc}')
                    # Add a "Missing" entry for this article if an error occurs
                    all_submission_data.append(SubmissionData(article_id, dataset_id="Error", type="Missing", context=f"Unhandled inference error: {exc}"))
        
        end_time = time.time()
        print(f"Concurrent model inference finished in {end_time - start_time:.2f} seconds.")
        return all_submission_data
