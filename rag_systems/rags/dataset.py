import datasets
import tqdm
from datasets import Dataset, load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class DocumentReader:
    def __init__(self, dataset_config):
        self.config = dataset_config
        self.datatype = dataset_config["type"]
        self.name = dataset_config["name"]
        self.split = dataset_config["split"]
        self.ds = None

    def load_dataset(self):
        """
        Load raw dataset
        """
        if self.datatype == "huggingface":
            self.ds = datasets.load_dataset(self.name, split=self.split)

        elif self.datatype == "pdf":
            reader = PdfReader(self.name)
            pdf_texts = [p.extract_text().strip() for p in reader.pages]

            # Filter the empty strings
            pdf_texts = [text for text in pdf_texts if text]
            self.ds = Dataset.from_dict({'text': pdf_texts})

        elif self.datatype in ["text", "json", "csv"]:
            self.ds = load_dataset(self.datatype, data_files=self.name)

        if self.ds is not None:
            self.langchain_docs = [LangchainDocument(page_content=doc["text"], metadata={
                                                     "source": doc["source"]}) for doc in tqdm(self.ds)]

        self.parse_dataset()

    def parse_dataset(self):
        """
        Parse dataset into langchain format
        """
        text_splitter_config = self.config["text_splitter"]
        if text_splitter_config["type"] == "recursive":

            self.text_splitter = RecursiveCharacterTextSplitter(
                add_start_index=True,
                separators=["\n\n", "\n", ".", " ", ""],
                **text_splitter_config["params"]
            )

        docs_processed = []
        for doc in self.langchain_docs:
            docs_processed += self.text_splitter.split_documents([doc])

        self.processed_docs = docs_processed
