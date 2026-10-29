"""Profile the compiled single-H100 PyTorch scorer."""

from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from exp586_moe.inference.contract import token_contract
from exp586_moe.inference.fixtures import load_fixture
from exp586_moe.inference.tokenizer import encode_dna, reduced_dna_tokenizer
from exp586_moe.inference.torch_backend import EagerGrugMoE


def profile(
    model_dir: str,
    fixture: str,
    output: str,
    *,
    batch_size: int = 64,
) -> None:
    """Capture one warmed compiled forward at a representative batch size."""
    device = torch.device("cuda:0")
    directory = Path(model_dir)
    sequences = load_fixture(Path(fixture))[:batch_size]
    if len(sequences) != batch_size:
        raise ValueError("Fixture is smaller than the requested batch")
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer = reduced_dna_tokenizer(tokenizer)
    contract = token_contract(Tokenizer.from_file(str(directory / "tokenizer.json")))
    base_ids = [contract.output_ids[base] for base in "ACGT"]
    ids = np.stack([encode_dna(tokenizer, sequence) for sequence in sequences])
    token_ids = torch.from_numpy(ids).to(device)
    model = EagerGrugMoE(
        directory,
        device,
        attention_backend="flex_expanded",
        global_attention_backend="flash",
    )

    def forward(tokens: torch.Tensor) -> torch.Tensor:
        return model(tokens, base_ids, variant="acgt_vllm_moe")

    compiled = torch.compile(
        forward,
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead",
    )
    with torch.inference_mode():
        compiled(token_ids)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
        ) as profiler:
            compiled(token_ids)
            torch.cuda.synchronize()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(destination.with_suffix(".json")))
    destination.write_text(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=100,
        )
        + "\n"
    )
    print(destination.read_text())


if __name__ == "__main__":
    import fire

    fire.Fire(profile)
