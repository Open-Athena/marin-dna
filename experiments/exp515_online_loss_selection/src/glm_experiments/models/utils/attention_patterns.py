"""Attention pattern generators for sliding window configurations.

These utility functions generate sliding_window lists for the Transformer class,
enabling common architectural patterns like alternating global/local attention.
"""


def alternating_global_local(
    n_layers: int, window_size: int, start_with_global: bool = True
) -> list[int | None]:
    """Generate alternating global and local attention pattern.

    Alternates between global attention (None) and local sliding window attention.
    Useful for balancing global context with efficient local attention.

    Args:
        n_layers: Number of transformer layers
        window_size: Window size for local attention layers
        start_with_global: If True, first layer is global. If False, first layer is local.
            (default: True)

    Returns:
        List of length n_layers with alternating None and window_size values.

    Examples:
        >>> alternating_global_local(4, 256, start_with_global=True)
        [None, 256, None, 256]

        >>> alternating_global_local(4, 256, start_with_global=False)
        [256, None, 256, None]
    """
    pattern = []
    for i in range(n_layers):
        if start_with_global:
            pattern.append(None if i % 2 == 0 else window_size)
        else:
            pattern.append(window_size if i % 2 == 0 else None)
    return pattern


def all_local(n_layers: int, window_size: int) -> list[int]:
    """Generate pattern with local attention for all layers.

    All layers use sliding window attention with the same window size.

    Args:
        n_layers: Number of transformer layers
        window_size: Window size for all layers

    Returns:
        List of length n_layers with all values equal to window_size.

    Examples:
        >>> all_local(4, 256)
        [256, 256, 256, 256]
    """
    return [window_size] * n_layers


def all_global(n_layers: int) -> list[None]:
    """Generate pattern with global attention for all layers.

    All layers use standard (global) attention. Equivalent to sliding_window=None
    but provides explicit pattern for consistency.

    Args:
        n_layers: Number of transformer layers

    Returns:
        List of length n_layers with all values None.

    Examples:
        >>> all_global(4)
        [None, None, None, None]
    """
    return [None] * n_layers


def sparse_transformer(n_layers: int, window_size: int, global_every: int = 3) -> list[int | None]:
    """Generate sparse transformer pattern with periodic global attention.

    Every Nth layer uses global attention, others use local sliding window.
    Based on "Generating Long Sequences with Sparse Transformers" (Child et al., 2019).

    Args:
        n_layers: Number of transformer layers
        window_size: Window size for local attention layers
        global_every: Use global attention every N layers (default: 3)

    Returns:
        List of length n_layers with None every global_every positions, window_size otherwise.

    Examples:
        >>> sparse_transformer(6, 256, global_every=3)
        [None, 256, 256, None, 256, 256]

        >>> sparse_transformer(4, 128, global_every=2)
        [None, 128, None, 128]
    """
    return [None if i % global_every == 0 else window_size for i in range(n_layers)]


def longformer_style(n_layers: int, base_window: int = 512) -> list[int]:
    """Generate Longformer-style decreasing window sizes.

    Earlier layers use larger windows, later layers use smaller windows.
    Window size halves every 4 layers, with a minimum of 64.

    Based on "Longformer: The Long-Document Transformer" (Beltagy et al., 2020).

    Args:
        n_layers: Number of transformer layers
        base_window: Starting window size for first layers (default: 512)

    Returns:
        List of length n_layers with decreasing window sizes.

    Examples:
        >>> longformer_style(12, base_window=512)
        [512, 512, 512, 512, 256, 256, 256, 256, 128, 128, 128, 128]

        >>> longformer_style(8, base_window=256)
        [256, 256, 256, 256, 128, 128, 128, 128]
    """
    pattern = []
    for i in range(n_layers):
        # Halve window every 4 layers, minimum 64
        window = max(64, base_window // (2 ** (i // 4)))
        pattern.append(window)
    return pattern


def first_k_global_rest_local(n_layers: int, k: int, window_size: int) -> list[int | None]:
    """Generate pattern with first k layers global, rest local.

    Useful for models that need strong global context in early layers
    but can use local attention in later layers.

    Args:
        n_layers: Number of transformer layers
        k: Number of initial global layers
        window_size: Window size for remaining local layers

    Returns:
        List of length n_layers with None for first k positions, window_size for rest.

    Examples:
        >>> first_k_global_rest_local(6, k=2, window_size=256)
        [None, None, 256, 256, 256, 256]

        >>> first_k_global_rest_local(4, k=1, window_size=128)
        [None, 128, 128, 128]
    """
    if k > n_layers:
        raise ValueError(f"k={k} cannot be greater than n_layers={n_layers}")
    return [None] * k + [window_size] * (n_layers - k)


def custom_pattern(n_layers: int, window_sizes: list[int | None]) -> list[int | None]:
    """Create custom pattern by repeating or truncating a window size list.

    Useful for creating complex patterns that repeat or for quickly prototyping.

    Args:
        n_layers: Number of transformer layers
        window_sizes: List of window sizes to use. If shorter than n_layers, will repeat.
            If longer, will truncate.

    Returns:
        List of length n_layers with pattern from window_sizes.

    Examples:
        >>> custom_pattern(6, [None, 256])  # Alternating pattern
        [None, 256, None, 256, None, 256]

        >>> custom_pattern(8, [None, 128, 128])  # 1 global, 2 local pattern
        [None, 128, 128, None, 128, 128, None, 128]

        >>> custom_pattern(3, [512, 256, 128, 64])  # Truncate
        [512, 256, 128]
    """
    if not window_sizes:
        raise ValueError("window_sizes cannot be empty")

    pattern = []
    for i in range(n_layers):
        pattern.append(window_sizes[i % len(window_sizes)])
    return pattern
