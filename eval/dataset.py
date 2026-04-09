from datasets import Dataset

# Each entry = one test case
# question:       what you'd ask the pipeline
# ground_truth:   the correct answer (you write this based on your docs)
RAW_EVAL_DATA = [
    {
        "question":     "What is encoder-decoder attention?",
        "ground_truth": "Is a mechanism allowing the decoder to focus on specific, relevant parts of the input sequence while generating an output.",
    },
    {
        "question":     "What does an encoder-decoder do?",
        "ground_truth": "Handle sequential data, specifically mapping input sequences to output sequences of different lengths.",
    },
    {
        "question":     "What is an attention function?",
        "ground_truth": "Mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.",
    },
    # add 10-20 entries based on your actual documents
]


def build_eval_dataset() -> Dataset:
    """Returns a HuggingFace Dataset ready for RAGAs."""
    return Dataset.from_list(RAW_EVAL_DATA)
