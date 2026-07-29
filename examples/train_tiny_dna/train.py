from fray.types import ResourceConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm

ARTIFACT_VERSION = "2026.07.29"

# 1. Tokenize a small sample of enhancer sequences.
enhancers_tokenized = tokenized(
    name="tokenized/zoonomia-ccre-non-promoter-tutorial",
    source="marin-dna/zoonomia-v1-v3_ccre_non_promoter-tutorial",
    tokenizer="marin-dna/tokenizer-char-bos",
    text_key="sequence",
    sample_count=64,
    version=ARTIFACT_VERSION,
)

# 2. Define a tiny Qwen3 decoder.
tiny_qwen3 = Qwen3Config(
    max_seq_len=256,  # 255 DNA bases plus the BOS token
    hidden_dim=32,
    intermediate_dim=128,
    num_heads=4,
    num_kv_heads=2,
    num_layers=2,
)

# 3. Train for a few steps on CPU.
tiny_enhancer_model = train_lm(
    name="checkpoints/tiny-dna-qwen3-cpu",
    version=ARTIFACT_VERSION,
    model=tiny_qwen3,
    optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
    datasets={enhancers_tokenized: 1.0},
    batch_size=4,
    seq_len=tiny_qwen3.max_seq_len,
    num_train_steps=10,
    z_loss_weight=None,
    evals=None,
    resources=ResourceConfig.with_cpu(),
)


if __name__ == "__main__":
    StepRunner().run([lower(tiny_enhancer_model)])
