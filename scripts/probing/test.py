import torch
from scripts.probing.raw import process_attention

def test_process_attention():
    # Mock attention tensor [num_layers, batch_size, num_heads, seq_len, seq_len]
    attention = torch.tensor([
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]
                ]
            ]
        ]
    ])

    # Mock word mappings: 2 tokens for the first word, 2 tokens for the second word
    word_mappings = [2, 2]

    word_mapping

    # Expected word attention: average of tokens for each word
    expected_word_attention = torch.tensor([
        [
            [
                [1.5, 3.5],
                [5.5, 7.5],
                [9.5, 11.5],
                [13.5, 15.5]
            ]
        ]
    ])

    # Process attention
    processed_attention = process_attention(attention, word_mappings)

    # Check if the processed attention matches the expected attention
    assert torch.allclose(processed_attention, expected_word_attention), f"Expected {expected_word_attention}, but got {processed_attention}"

if __name__ == "__main__":
    test_process_attention()
    print("All tests passed.")