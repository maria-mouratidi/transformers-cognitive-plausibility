import torch
from scripts.probing.saliency_decoder import lm_saliency
from scripts.probing.raw_decoder import encode_input, get_word_mappings
from scripts.probing.load_model import load_llama

def compare_tokens():
    # Load the model and tokenizer
    model, tokenizer = load_llama("causal")
    device = model.device

    # Define test sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Deep learning models require large datasets.",
        "Natural language processing is a fascinating field.",
        "The weather today is sunny with a chance of rain.",
    ]

    # Process each sentence
    for sentence in sentences:
        print(f"Sentence: {sentence}")

        # Saliency method
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = inputs["input_ids"][0].to(device).tolist()
        input_mask = inputs["attention_mask"][0].to(device).tolist()

        label_id = model(input_ids=torch.tensor([input_ids]).to(device)).logits[0, -1].argmax().item()
        tokens_saliency, _, _ = lm_saliency(model, tokenizer, input_ids, input_mask, label_id)

        # Raw method
        words = sentence.split()  # Pretokenize the sentence into words
        encodings, _,  = encode_input([words], tokenizer, task="none")
        mappings = get_word_mappings([words], encodings, tokenizer )
        print(f"Word mappings: {mappings}")
        tokens_raw = tokenizer.convert_ids_to_tokens(encodings["input_ids"][0].tolist())

        # # Compare results
        # print(f"Tokens from lm_saliency: {tokens_saliency}")
        # print(f"Tokens from encode_input: {tokens_raw}")
        # print(f"Match: {tokens_saliency == tokens_raw}")
        # print("-" * 50)

if __name__ == "__main__":
    compare_tokens()