import datasets
from datasets import Dataset, load_dataset
from pypdf import PdfReader


class DocumentReader:
    def __init__(self, dataset_config):
        self.datatype = dataset_config["type"]
        self.name = dataset_config["name"]
        self.split = dataset_config["split"]
        self.ds = None

    def load_dataset(self):
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
