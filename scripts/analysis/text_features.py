import numpy as np
import wordfreq

def get_frequency(word: str) -> int:
    """
    Get the frequency of a word.
    
    Args:
        word: The word to measure.
    
    Returns:
        The frequency of the word.
    """
    return 1 



# word length

def get_length(word: str) -> int:
    """
    Get the length of a word.
    
    Args:
        word: The word to measure.
    
    Returns:
        The length of the word.
    """
    return len(word)


def get_role(word: str) -> str:
    """
    Get the role of a word.
    
    Args:
        word: The word to measure.
    
    Returns:
        The role of the word (e.g., "content" or "function").
    """
    # Placeholder logic for determining word role
    if word in ["the", "is", "at", "which", "on"]:
        return "function"
    else:
        return "content"


def get_surprisal(word: str) -> float:
    """
    Get the surprisal of a word.
    
    Args:
        word: The word to measure.
    
    Returns:
        The surprisal of the word.
    """
    return -1.0 * np.log(1 / (1 + len(word)))