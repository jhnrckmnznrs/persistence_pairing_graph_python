"""Image loading and CSV output compatible with the Rust command-line tool."""
import csv
from pathlib import Path

import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Read grayscale TIFF stacks or 8/16-bit PNG and JPEG images.

    TIFFs are loaded with tifffile; PIL is used for other formats. Arrays
    can also be passed directly to :func:`compute_ppg` without I/O.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in ('.tif', '.tiff'):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("TIFF support requires 'pip install tifffile'") from exc
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                raise ValueError("TIFF contains no image pages")
            if any(page.samplesperpixel != 1 for page in tif.pages):
                raise ValueError("Expected grayscale TIFF pages, not multichannel data")
            shapes = [page.shape for page in tif.pages]
            if any(shape != shapes[0] for shape in shapes):
                raise ValueError("All TIFF pages must have the same shape")
            if any(page.dtype != tif.pages[0].dtype for page in tif.pages):
                raise ValueError("All TIFF pages must have the same dtype")
            image = tif.asarray()
    elif ext in ('.png', '.jpg', '.jpeg'):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("PNG/JPEG support requires 'pip install Pillow'") from exc
        with Image.open(path) as im:
            if im.mode in ('RGB', 'RGBA', 'P', 'LA'):
                im = im.convert('L')
            elif im.mode not in ('L', 'I;16', 'I;16L', 'I;16B'):
                raise ValueError(f"Unsupported PNG/JPEG pixel format: {im.mode}")
            image = np.asarray(im)
    else:
        raise ValueError(f"Unsupported image type: {path.suffix}")
    if image.ndim not in (2, 3):
        raise ValueError("Expected a grayscale 2D image or 3D TIFF stack")
    if image.dtype.byteorder == '>' or (image.dtype.byteorder == '=' and not np.little_endian):
        image = image.astype(image.dtype.newbyteorder('='))
    return image


def export_csv(result, input_path: str | Path, *, output_dir: str | Path | None = None,
               export_cell_provenance: bool = False) -> tuple[Path, Path, Path | None]:
    """Write Rust-compatible node/edge CSVs and optional compressed provenance."""
    if export_cell_provenance and result.original_grid_indices is None:
        raise ValueError("Compressed cell provenance requires retained provenance (stream-block: set retain_cell_provenance=True)")
    path = Path(input_path)
    output = Path(output_dir) if output_dir is not None else path.parent
    output.mkdir(parents=True, exist_ok=True)
    base = path.stem
    node_path = output / f'{base}_persistence_pairing_graph_nodes.csv'
    edge_path = output / f'{base}_persistence_pairing_graph_edges.csv'
    with node_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(('node_id', 'dimension', 'birth_cell', 'death_cell', 'birth_value',
                         'death_value', 'persistence', 'essential'))
        for k, node in enumerate(result.nodes):
            writer.writerow((k, node.dimension, node.birth_cell,
                             node.death_cell if node.death_cell is not None else '',
                             node.birth_value, node.death_value, node.persistence,
                             int(node.essential)))
    if result.edge_output_path is not None and result.edge_output_path.resolve() != edge_path.resolve():
        raise ValueError('Streaming edge output path differs from the requested export destination')
    if result.edge_output_path is None:
        with edge_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(('dimension', 'from_node', 'to_node', 'from_birth_value',
                             'from_death_value', 'to_birth_value', 'to_death_value', 'coefficient'))
            for edge in result.edges:
                a = result.nodes[edge.from_node]
                b = result.nodes[edge.to_node]
                writer.writerow((edge.dimension, edge.from_node, edge.to_node,
                                 a.birth_value, a.death_value, b.birth_value,
                                 b.death_value, edge.coefficient))
    provenance_path = None
    if export_cell_provenance:
        provenance_path = output / f'{base}_compressed_cell_provenance.csv'
        with provenance_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(('compressed_cell', 'regular_grid_index', 'dimension', 'birth_value'))
            for k, old_grid in enumerate(result.original_grid_indices):
                writer.writerow((k, int(old_grid), int(result.cell_dimensions[k]),
                                 result.cell_values[k].item()))
    return node_path, edge_path, provenance_path
