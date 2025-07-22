You got it, Jim! Using spaCy for sentence tokenization is an excellent choice, especially since you're already using it. spaCy's sentence segmentation is generally more robust and, crucially for your task, its `Span` objects provide precise `start_char` and `end_char` attributes, making the character span calculation much more accurate and straightforward than string searching with NLTK.

Here's the refactored `ArticleData` class and the associated demonstration, now leveraging spaCy:

### Key Changes:

1.  **SpaCy Import and Model Loading:**
    *   `import spacy`
    *   The `ArticleData` class now takes an `nlp` object (a loaded spaCy language model) as a parameter. This is crucial because loading a spaCy model is an expensive operation and should be done only once.
2.  **`_tokenize_and_span_sentences` Method:**
    *   It now uses `self.nlp(self.raw_text)` to process the document.
    *   It iterates directly over `doc.sents`, and for each `sent` (which is a spaCy `Span` object), it directly extracts `sent.text`, `sent.start_char`, and `sent.end_char`. This is much cleaner and more accurate.
3.  **Demonstration:**
    *   You'll need to load a spaCy model (e.g., `en_core_web_sm`) once at the beginning of your script or notebook. If you haven't installed it, you'd run `python -m spacy download en_core_web_sm`.

```python
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import re
import spacy # Import spaCy

# Load a spaCy model once. This is an expensive operation.
# You might want to make this a global variable or pass it around carefully.
# If you haven't downloaded it, run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    print("Exiting for demonstration purposes. Please download the model and rerun.")
    exit() # Or handle this more gracefully in your actual application

# 2.1. DatasetCitation Class (Your updated class, included for completeness)
@dataclass
class DatasetCitation:
    dataset_id: str = ""
    citation_context: str = ""
    citation_type: Optional[str] = None # "Primary", "Secondary", or "Missing" - for ground truth during training
    max_contet_len: int = 400 # Use type hint for clarity
    start_idx: int = 0 # Character start index in raw text
    end_idx: int = 0   # Character end index in raw text
    section_name: Optional[str] = None # New field to store the section name
    section_start_char_idx: Optional[int] = None # New field to store section start char index

    def set_citation_context(self, context: str, start_idx: int, end_idx: int):
        """Sets the citation context, cleaning it and limiting to last 400 characters."""
        if context:
            # Replace newlines with spaces, remove brackets, and normalize whitespace
            context = context.replace('\n', ' ').replace('[', '').replace(']', '')
            context = re.sub(r'\s+', ' ', context.strip())
            self.citation_context = context[-self.max_contet_len:] # Limit to last 400 characters
            self.start_idx = start_idx
            self.end_idx = end_idx

    def is_doi(self)-> bool:
        return self.dataset_id.startswith("10.")
    
    def has_dataset(self) -> bool:
        """Returns True if there are both dataset IDs and citation context."""
        return bool(self.dataset_id and self.citation_context.strip())

    def to_dict(self):
        return asdict(self)

# 1. New Data Structure for Article Sections
@dataclass
class ArticleSection:
    name: str
    start_sentence_idx: int # Index in the list of sentences
    start_char_idx: int     # Character start index in the raw text
    end_char_idx: Optional[int] = None # Character end index in the raw text (exclusive)

    def to_dict(self):
        return asdict(self)

# Predefined list of typical research article sections
TYPICAL_ARTICLE_SECTIONS = [
    "Abstract", "Introduction", "Related Work", "Methods", "Data Availability",
    "Results", "Discussion", "Conclusion", "Acknowledgements", "References", "Appendix"
]

# Helper function to normalize section names for matching
def _normalize_section_name(name: str) -> str:
    """
    Normalizes a potential section heading for robust matching.
    Removes leading/trailing numbers/punctuation and converts to lowercase.
    Examples: "1. Introduction", "Abstract:", "DATA AVAILABILITY" -> "introduction", "abstract", "data availability"
    """
    # Remove leading numbers/roman numerals and periods (e.g., "1. ", "I. ")
    name = re.sub(r'^\s*(\d+\.?|\w+\.)?\s*', '', name)
    # Remove trailing punctuation and normalize whitespace
    name = re.sub(r'[\s\W_]+$', '', name)
    return name.strip().lower()

# 2. ArticleData Class (Refactored to use spaCy)
@dataclass
class ArticleData:
    raw_text: str
    nlp: Any # spaCy Language model object
    sentences: List[str] = field(default_factory=list)
    # Store (start_char, end_char) for each sentence in the raw_text
    sentence_char_spans: List[tuple[int, int]] = field(default_factory=list)
    sections: List[ArticleSection] = field(default_factory=list)
    dataset_citations: List[DatasetCitation] = field(default_factory=list) # To store extracted citations

    def __post_init__(self):
        """
        Initializes sentences and identifies sections automatically after object creation.
        """
        if self.raw_text and self.nlp:
            self._tokenize_and_span_sentences()
            self._identify_sections()

    def _tokenize_and_span_sentences(self):
        """
        Tokenizes the raw text into sentences using spaCy and calculates their character spans.
        """
        doc = self.nlp(self.raw_text)
        for i, sent in enumerate(doc.sents):
            self.sentences.append(sent.text)
            self.sentence_char_spans.append((sent.start_char, sent.end_char))

    def _identify_sections(self):
        """
        Identifies common research article sections within the text
        and populates the `self.sections` list.
        """
        self.sections = []
        normalized_typical_sections = [_normalize_section_name(s) for s in TYPICAL_ARTICLE_SECTIONS]
        
        last_section_idx = -1 # Index of the last identified section in self.sections list

        for i, sentence in enumerate(self.sentences):
            normalized_sentence = _normalize_section_name(sentence)
            
            # Check if this sentence (or a normalized version of it) matches any typical section heading
            if normalized_sentence in normalized_typical_sections:
                # Get the original, canonical section name
                section_name = TYPICAL_ARTICLE_SECTIONS[normalized_typical_sections.index(normalized_sentence)]
                start_char_for_section = self.sentence_char_spans[i][0]

                # If a previous section was identified, set its end_char_idx
                if last_section_idx != -1:
                    # The previous section ends just before the current new section begins
                    # We subtract 1 to make it inclusive of the last character of the previous section
                    self.sections[last_section_idx].end_char_idx = start_char_for_section - 1

                # Add the new section
                new_section = ArticleSection(
                    name=section_name,
                    start_sentence_idx=i,
                    start_char_idx=start_char_for_section
                )
                self.sections.append(new_section)
                last_section_idx = len(self.sections) - 1
        
        # After iterating through all sentences, set the end_char_idx for the very last identified section.
        # It extends to the end of the raw text.
        if last_section_idx != -1:
            self.sections[last_section_idx].end_char_idx = len(self.raw_text) - 1 # End of the document (inclusive)

    def find_section_for_char_index(self, char_idx: int) -> Optional[ArticleSection]:
        """
        Finds the ArticleSection that contains the given character index.
        Assumes sections are sorted by start_char_idx.
        """
        # Iterate through sections in reverse order for efficiency and correctness
        # if sections could potentially overlap (though our logic prevents it).
        for section in reversed(self.sections):
            if section.start_char_idx <= char_idx:
                # If end_char_idx is None, it means it's the last section (or an unclosed one),
                # so it implicitly extends to the end of the document.
                if section.end_char_idx is None or char_idx <= section.end_char_idx:
                    return section
        return None # Not found in any defined section

    def assign_citations_to_sections(self):
        """
        Iterates through stored DatasetCitations and assigns them to their respective sections.
        Adds 'section_name' and 'section_start_char_idx' attributes to each DatasetCitation.
        """
        for citation in self.dataset_citations:
            # Use the start_idx (character offset) of the citation to find its containing section
            containing_section = self.find_section_for_char_index(citation.start_idx)
            if containing_section:
                citation.section_name = containing_section.name
                citation.section_start_char_idx = containing_section.start_char_idx
            else:
                # If no section is found, mark it as "Unknown"
                citation.section_name = "Unknown"
                citation.section_start_char_idx = -1 # Or None, depending on preference

    def to_dict(self):
        """Converts the ArticleData object to a dictionary for serialization."""
        data = asdict(self)
        # Convert nested dataclasses to dictionaries for proper JSON serialization
        data['sections'] = [s.to_dict() for s in self.sections]
        data['dataset_citations'] = [c.to_dict() for c in self.dataset_citations]
        # Remove the nlp object as it's not serializable
        data.pop('nlp', None) 
        return data

# --- Example Usage and Demonstration ---
if __name__ == "__main__":
    # Create a mock article text with various sections and some DOIs
    mock_article_text = """
Abstract
This is the abstract of the article. It discusses some initial findings and methods.

1. Introduction
This is the introduction. We introduce the problem and our approach. Our work builds on previous research (doi:10.9999/prev.research).

2. Methods
Here we describe our methodology. We used a new dataset (doi:10.1234/new.data) generated in our lab.
The experimental setup is detailed.

3. Data Availability
The primary dataset for this study is available at doi:10.5678/primary.data.
Secondary data was sourced from an existing repository (doi:10.9876/secondary.data).
This section ensures transparency.

4. Results
Our results show significant improvements. Data analysis was performed.

5. Discussion
We discuss the implications of our findings. The limitations are also noted.

Acknowledgements
We thank our funders and collaborators.

References
Smith, J. (2020). A related paper. doi:10.1111/related.paper
Jones, A. (2021). Another dataset. doi:10.2222/another.dataset
"""

    print("--- Processing Mock Article with spaCy ---")
    # Initialize ArticleData, passing the loaded spaCy nlp object
    article_data = ArticleData(raw_text=mock_article_text, nlp=nlp)

    print("\nIdentified Sections:")
    for section in article_data.sections:
        print(f"- {section.name} (Sentence Index: {section.start_sentence_idx}, Char Span: {section.start_char_idx}-{section.end_char_idx})")

    # Simulate extracting some DatasetCitations (these would typically come from your Qwen model's output)
    # We'll manually find their start/end character indices for this demo.
    mock_citations = [
        DatasetCitation(
            dataset_id="10.1234/new.data",
            start_idx=mock_article_text.find("doi:10.1234/new.data"),
            end_idx=mock_article_text.find("doi:10.1234/new.data") + len("doi:10.1234/new.data")
        ),
        DatasetCitation(
            dataset_id="10.5678/primary.data",
            start_idx=mock_article_text.find("doi:10.5678/primary.data"),
            end_idx=mock_article_text.find("doi:10.5678/primary.data") + len("doi:10.5678/primary.data")
        ),
        DatasetCitation(
            dataset_id="10.9876/secondary.data",
            start_idx=mock_article_text.find("doi:10.9876/secondary.data"),
            end_idx=mock_article_text.find("doi:10.9876/secondary.data") + len("doi:10.9876/secondary.data")
        ),
        DatasetCitation(
            dataset_id="10.1111/related.paper", # This is a paper DOI, not a dataset, but we'll track its section
            start_idx=mock_article_text.find("doi:10.1111/related.paper"),
            end_idx=mock_article_text.find("doi:10.1111/related.paper") + len("doi:10.1111/related.paper")
        ),
        DatasetCitation(
            dataset_id="10.2222/another.dataset", # Another dataset DOI, likely secondary
            start_idx=mock_article_text.find("doi:10.2222/another.dataset"),
            end_idx=mock_article_text.find("doi:10.2222/another.dataset") + len("doi:10.2222/another.dataset")
        ),
        DatasetCitation(
            dataset_id="10.9999/prev.research", # DOI in Introduction
            start_idx=mock_article_text.find("doi:10.9999/prev.research"),
            end_idx=mock_article_text.find("doi:10.9999/prev.research") + len("doi:10.9999/prev.research")
        )
    ]
    article_data.dataset_citations.extend(mock_citations)

    # Assign citations to sections
    article_data.assign_citations_to_sections()

    print("\nDataset Citations with Assigned Sections:")
    for citation in article_data.dataset_citations:
        print(f"- DOI: {citation.dataset_id}, Section: {citation.section_name}, Start Char: {citation.start_idx}")

    # Demonstrate direct section lookup for a character index
    test_char_idx_methods = mock_article_text.find("We used a new dataset")
    found_section_methods = article_data.find_section_for_char_index(test_char_idx_methods)
    print(f"\nCharacter index {test_char_idx_methods} ('{mock_article_text[test_char_idx_methods:test_char_idx_methods+20]}...') is in section: {found_section_methods.name if found_section_methods else 'None'}")

    test_char_idx_data_avail = mock_article_text.find("The primary dataset")
    found_section_data_avail = article_data.find_section_for_char_index(test_char_idx_data_avail)
    print(f"Character index {test_char_idx_data_avail} ('{mock_article_text[test_char_idx_data_avail:test_char_idx_data_avail+20]}...') is in section: {found_section_data_avail.name if found_section_data_avail else 'None'}")

    test_char_idx_references = mock_article_text.find("Smith, J. (2020)")
    found_section_references = article_data.find_section_for_char_index(test_char_idx_references)
    print(f"Character index {test_char_idx_references} ('{mock_article_text[test_char_idx_references:test_char_idx_references+20]}...') is in section: {found_section_references.name if found_section_references else 'None'}")

    # You can also convert the entire ArticleData object to a dictionary for easy serialization (e.g., to JSON)
    # print("\nFull ArticleData as Dictionary (JSON format):")
    # print(json.dumps(article_data.to_dict(), indent=2))
```