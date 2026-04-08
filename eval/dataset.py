from datasets import Dataset

# Each entry = one test case
# question:       what you'd ask the pipeline
# ground_truth:   the correct answer (you write this based on your docs)
RAW_EVAL_DATA = [
    {
        "question":     "What is the refund policy?",
        "ground_truth": "Customers can request a refund within 30 days of purchase.",
    },
    {
        "question":     "How do I reset my password?",
        "ground_truth": "Click 'Forgot password' on the login page and follow the email instructions.",
    },
    {
        "question":     "What payment methods are accepted?",
        "ground_truth": "We accept Visa, Mastercard, and PayPal.",
    },
    # add 10-20 entries based on your actual documents
]


def build_eval_dataset() -> Dataset:
    """Returns a HuggingFace Dataset ready for RAGAs."""
    return Dataset.from_list(RAW_EVAL_DATA)
