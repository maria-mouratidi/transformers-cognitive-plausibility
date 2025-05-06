import csv
import json
import re

def normalize(sentence):
    """Remove punctuation and lowercase the first word."""
    if isinstance(sentence, list):
        sentence = ' '.join(sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence).strip()
    words = sentence.split()
    if words:
        words[0] = words[0].lower()
    return ' '.join(words)

# Load labeled sentences from the authors
with open("materials/relations_task_specific.csv", encoding='latin-1') as f:
    csv_lookup = {
        normalize(row['sentence']): row['relation_type']
        for row in csv.DictReader(f, delimiter=';')
    }

# Load sentences from eye-gaze data
with open("materials/sentences_task3.json", encoding='utf-8') as f:
    sentences = json.load(f)


manual_relations ={
    "henry he Bolte married was Carolyn born Bessette in Ballarat 1996 the son of a publican of German descent the family name was pronounced BolteeHe was to spend the first 24 years of his life apart from three years at boarding school in the small western district town of Skipton": "VISITED",
    "he maintained homes in Long Island New York and Greenwich Connecticut the family compound at Kennebunkport Maine a 10000 acre 40 km plantation in South Carolina and an island retreat in Florida": "LIVES_IN",
}

# Match eye-gaze sentences with relation types
matched = [
    {
        'sentence': tokens,
        'relation_type': manual_relations.get(
            normalize(tokens),  # Check manual relations that are added due to typos
            csv_lookup.get(normalize(tokens), "UNKNOWN")  # Fallback to csv_lookup
        )
    }
    for tokens in sentences
]

with open("materials/labeled_sentences_task3.json", "w", encoding='utf-8') as f:
    json.dump(matched, f, ensure_ascii=False, indent=2)

