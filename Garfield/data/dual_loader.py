"""
Combined dataloader for edge-level and node-level graph tasks.
"""
import itertools
from typing import Iterator, Tuple


class DualGraphDataLoader:
    """
    Combines edge-level and node-level PyG dataloaders for joint training.

    Garfield trains on two tasks simultaneously:
    1. Edge-level: Graph reconstruction (edge prediction)
    2. Node-level: Gene expression reconstruction

    This loader zips the two dataloaders, cycling the node-level loader
    if it finishes before the edge-level loader.

    Parameters
    ----------
    edge_loader : DataLoader
        PyG LinkNeighborLoader for edge-level tasks.
    node_loader : DataLoader
        PyG NeighborLoader for node-level tasks.

    Examples
    --------
    >>> edge_loader = LinkNeighborLoader(...)
    >>> node_loader = NeighborLoader(...)
    >>> dual_loader = DualGraphDataLoader(edge_loader, node_loader)
    >>> for edge_batch, node_batch in dual_loader:
    ...     # Train on both tasks
    ...     edge_output = model(edge_batch, decoder_type='graph')
    ...     node_output = model(node_batch, decoder_type='omics')
    """

    def __init__(self, edge_loader, node_loader):
        self.edge_loader = edge_loader
        self.node_loader = node_loader
        self._node_iter = None

    def __iter__(self) -> Iterator[Tuple]:
        """
        Returns an iterator that yields (edge_batch, node_batch) tuples.

        The node loader is cycled, so it repeats until the edge loader
        finishes. This ensures balanced training between the two tasks.
        """
        # Create a new cycle iterator for each epoch
        self._node_iter = itertools.cycle(self.node_loader)
        return zip(self.edge_loader, self._node_iter)

    def __len__(self) -> int:
        """
        Returns the length of the edge loader (number of edge batches).
        """
        return len(self.edge_loader)
