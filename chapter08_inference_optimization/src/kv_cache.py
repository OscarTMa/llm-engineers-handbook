from loguru import logger
from pydantic import BaseModel


class KVCacheMetrics(BaseModel):
    num_tokens: int
    num_layers: int
    num_heads: int
    head_dim: int
    bytes_per_param: int
    cache_size_bytes: int
    cache_size_mb: float
    cache_size_gb: float


class KVCacheManager:

    @staticmethod
    def calculate_cache_size(
        num_tokens: int,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        bytes_per_param: int = 2,
    ) -> KVCacheMetrics:
        """
        Calcula el tamaño del KV cache:
        Size = 2 * n_tokens * n_layers * n_heads * head_dim * bytes_per_param
        """
        total_bytes = 2 * num_tokens * num_layers * num_heads * head_dim * bytes_per_param
        size_mb = total_bytes / (1024**2)
        size_gb = total_bytes / (1024**3)

        return KVCacheMetrics(
            num_tokens=num_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            bytes_per_param=bytes_per_param,
            cache_size_bytes=total_bytes,
            cache_size_mb=round(size_mb, 2),
            cache_size_gb=round(size_gb, 4),
        )

    @classmethod
    def evaluate_growth_profiles(cls, max_seq_len: int = 4096):
        profiles = [512, 1024, 2048, 4096, 8192, 16384]
        logger.info("Evaluating KV Cache memory footprints across sequence horizons:")
        for tokens in profiles:
            if tokens <= max_seq_len * 4:
                res = cls.calculate_cache_size(tokens)
                logger.info(f"Context: {tokens:>5} tokens -> KV Cache: {res.cache_size_mb:>8.2f} MB ({res.cache_size_gb:>6.3f} GB)")
