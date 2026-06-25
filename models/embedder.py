import torch
import torch.nn as nn


"""
Sinusoidal positional encoding used by NeuS/NeRF networks.

This file is essentially the standard NeRF embedding helper. It maps low
dimensional coordinates or view directions to a higher dimensional set of
sin/cos features so the MLPs can represent high-frequency geometry and color.
"""


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    """Build and apply a fixed bank of sinusoidal embedding functions."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """Create the list of identity, sine, and cosine feature functions."""
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = (2. ** torch.linspace(0., max_freq, N_freqs)).tolist()
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs).tolist()

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """Concatenate all positional encoding channels for `inputs`."""
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    """Return an embedding callable and its output dimension."""
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
