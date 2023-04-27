# Lint as: python3
"""Reddit Topics Targz Demo Dataset."""

import json
import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
Demo
"""

_URL = "https://github.com/jamescalam/hf-datasets/raw/main/01_builder_script/dataset.tar.gz"
#replace this with the tar file for own dataset




class RedditTopicsTargz(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""



    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "sub": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "selftext": datasets.Value("string"),
                    "upvote_ratio": datasets.Value("float32"),
                    "id": datasets.Value("string"),
                    "created_utc": datasets.Value("float32")
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.download_and_extract(_URL)
        #takes a URL to a tar file and returns filepath in HF to that data

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"filepath": path+'/dataset.jsonl'})
        ] #filepath gets passed to generate examples

    def _generate_examples(self, filepath): #output rows of the dataset
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        idx = 0
        #open the file and read the lines
        with open(filepath, encoding="utf-8") as fp:
          for line in fp:
            #load json line
            print(json.loads(line))
            obj = json.loads(line)

            yield idx, obj
            idx += 1