from config.settings import get_settings

from .deps import AutoModel, AutoTokenizer, CrossEncoder, SentenceTransformer, torch


class ModelHub:
    _embed_model = None
    _cross_encoder = None
    _splade_tokenizer = None
    _splade_model = None
    _colbert_searcher = None

    @classmethod
    def get_device(cls):
        if torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @classmethod
    def get_embed_model(cls):
        settings = get_settings()
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        if cls._embed_model is None:
            cls._embed_model = SentenceTransformer(settings.embed_model_name, device=cls.get_device())
            cls._embed_model.eval()
        return cls._embed_model

    @classmethod
    def get_cross_encoder(cls):
        settings = get_settings()
        if CrossEncoder is None:
            return None
        if cls._cross_encoder is None:
            cls._cross_encoder = CrossEncoder(settings.cross_encoder_name)
        return cls._cross_encoder

    @classmethod
    def get_splade(cls):
        settings = get_settings()
        if AutoTokenizer is None or AutoModel is None:
            raise RuntimeError("transformers not installed")
        if cls._splade_tokenizer is None:
            cls._splade_tokenizer = AutoTokenizer.from_pretrained(settings.splade_model_name)
            cls._splade_model = AutoModel.from_pretrained(settings.splade_model_name)
            if torch and torch.cuda.is_available():
                cls._splade_model = cls._splade_model.cuda()
            cls._splade_model.eval()
        return cls._splade_tokenizer, cls._splade_model

    @classmethod
    def get_colbert_searcher(cls):
        settings = get_settings()
        if cls._colbert_searcher is None:
            from colbert import Searcher
            from colbert.infra import ColBERTConfig

            config = ColBERTConfig(doc_maxlen=300, query_maxlen=32, nbits=2)
            cls._colbert_searcher = Searcher.checkpoint(settings.colbert_model_name, config=config)
        return cls._colbert_searcher
