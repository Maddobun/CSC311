from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_data(path: str) -> list:
    """
    parse text file to generate a list, preprocess it using sklearn feature extraction,
    and split it into 70% training, 15% validation, and 15% test sets
    """

    # preprocess data
    vectorizer = TfidfVectorizer(input='filename')
    data = vectorizer.fit_transform([path])

    # split data
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)

    return [train, val, test]




