from dataclasses import asdict, dataclass, field
import json
from typing import Iterable

@dataclass
class Vocab:
    int_to_char: list[str] = field(default_factory=list)
    char_to_int: dict[str, int] = field(default_factory=dict)
    
    def __len__(self):
        return len(self.int_to_char)

    def encode(self, s: str) -> list[int]:
        return [self.char_to_int[x] if x in self.char_to_int else self.char_to_int[' '] for x in s]

    def decode(self, s: Iterable[int]) -> str:
        return ''.join([self.int_to_char[x] for x in s])
        
    @classmethod
    def load(cls, path: str):
        with open(path, mode='rt', encoding='utf-8') as f:
            return Vocab(**json.load(f))

    def save(self, path: str):
        with open(path, mode='wt', encoding='utf-8') as f:
            return json.dump(asdict(self), f)

def tokenize(text: str) -> Vocab:
    int_to_char = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(int_to_char))
    return Vocab(int_to_char, char_to_int)
