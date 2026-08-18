from .deps import tiktoken

_tokenizer = None
if tiktoken:
    _tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if _tokenizer:
        return len(_tokenizer.encode(text))
    return len(text.split())
