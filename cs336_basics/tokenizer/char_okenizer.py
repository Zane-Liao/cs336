
from tokenizer import Tokenizer

class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))
    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))
    
    
def character_tokenizer():
    raise NotImplementedError

def word_tokenizer():
    raise NotImplementedError