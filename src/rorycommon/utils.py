import numpy as np
from typing import Tuple,Iterator
# import zarr 

class Utils:
    @staticmethod
    def read_chunks_numpy(filename: str, chunk_shape: Tuple[int, ...]) -> Iterator[np.ndarray]:
        """
        Lazily reads chunks from a .npy file using numpy.memmap.
        
        Parameters:
        - filename: path to the .npy file.
        - chunk_shape: shape of the chunk to return (e.g., (1000, 10)).
        
        Yields:
        - chunks of the array with shape <= chunk_shape.
        """
        print("AAA")
        mmap_array = np.load(filename, mmap_mode='r')
        full_shape = mmap_array.shape
        ndim = mmap_array.ndim
        print(full_shape,ndim)

        # Determine number of steps for each dimension
        steps = [range(0, full_shape[i], chunk_shape[i]) for i in range(ndim)]

        # Iterate over all chunks using nested loops
        from itertools import product
        for start_indices in product(*steps):
            slices = tuple(
                slice(start, min(start + chunk_shape[i], full_shape[i]))
                for i, start in enumerate(start_indices)
            )
            yield mmap_array[slices]
    # @staticmethod
    # def read_chunks_zarr(store_path: str, chunk_shape: Tuple[int, ...]) -> Iterator[np.ndarray]:
    #     """
    #     Lazily reads chunks from a .zarr array given a chunk shape.
        
    #     Parameters:
    #     - store_path: path to the Zarr store (directory or zip).
    #     - chunk_shape: shape of the chunks to yield.
        
    #     Yields:
    #     - chunks of the array with shape <= chunk_shape.
    #     """
    #     z = zarr.open(store_path, mode='r')
    #     full_shape = z.shape
    #     ndim = z.ndim

    #     steps = [range(0, full_shape[i], chunk_shape[i]) for i in range(ndim)]

    #     from itertools import product
    #     for start_indices in product(*steps):
    #         slices = tuple(
    #             slice(start, min(start + chunk_shape[i], full_shape[i]))
    #             for i, start in enumerate(start_indices)
    #         )
    #         yield z[slices]