"""python -m persistence_pairing_graph IMAGE_OR_DIRECTORY [options]."""
import argparse
from pathlib import Path

from . import compute_ppg, export_csv, load_image

_SUPPORTED = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compute persistence-pairing graphs from scalar images.')
    parser.add_argument('input', type=Path, help='Grayscale image, multipage TIFF, or image directory')
    parser.add_argument('--algorithm', choices=('stream-block', 'block', 'lazy'), default='block')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--max-cells', type=int, default=None)
    parser.add_argument('--temp-dir', type=Path, default=None, help='Directory for streaming order scratch file')
    parser.add_argument('--export-cell-provenance', action='store_true')
    parser.add_argument('--order-format', choices=('delta', 'u32'), default='u32')
    parser.add_argument('--order-builder', choices=('auto', 'sort', 'counting'), default='auto')
    parser.add_argument('--metadata-storage', choices=('memory', 'disk'), default='memory')
    parser.add_argument('--event-summary', action='store_true', help='Export death-event edge counts, including skipped events')
    parser.add_argument('--degrees', action='store_true', help='Export per-node in- and out-degree table')
    parser.add_argument('--reduction-storage', choices=('packed', 'disk', 'python'), default='packed')
    parser.add_argument('--node-storage', choices=('compact', 'python'), default='compact')
    parser.add_argument('--accelerator', choices=('python', 'numba'), default='python')
    parser.add_argument('--log-mode', choices=('full', 'selective'), default='full')
    parser.add_argument('--max-relation-cache-entries', type=int, default=100_000)
    parser.add_argument('--stream-edges', action='store_true', help='Write edges directly to CSV without keeping a Python edge list')
    args = parser.parse_args(argv)
    if args.export_cell_provenance and args.algorithm == 'lazy':
        parser.error('--export-cell-provenance requires --algorithm block or stream-block')
    if args.max_cells is not None and args.max_cells <= 0:
        parser.error('--max-cells must be positive')
    if args.max_relation_cache_entries < 0:
        parser.error('--max-relation-cache-entries must be nonnegative')
    path = args.input
    if path.is_dir():
        sources = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED)
        if not sources:
            parser.error('No supported images in directory')
    elif path.is_file():
        sources = [path]
    else:
        parser.error(f'No such file or directory: {path}')
    for source in sources:
        outdir = args.output_dir if args.output_dir is not None else source.parent
        edge_output = (outdir / f'{source.stem}_persistence_pairing_graph_edges.csv'
                       if args.stream_edges else None)
        graph = compute_ppg(load_image(source), algorithm=args.algorithm, max_cells=args.max_cells,
                            retain_cell_provenance=args.export_cell_provenance,
                            temp_dir=args.temp_dir, order_format=args.order_format,
                            reduction_storage=args.reduction_storage, log_mode=args.log_mode,
                            accelerator=args.accelerator, node_storage=args.node_storage,
                            max_relation_cache_entries=args.max_relation_cache_entries,
                            edge_output_path=edge_output, order_builder=args.order_builder,
                            metadata_storage=args.metadata_storage,
                            collect_event_summary=args.event_summary,
                            collect_degrees=args.degrees)
        node_path, edge_path, provenance_path = export_csv(
            graph, source, output_dir=args.output_dir,
            export_cell_provenance=args.export_cell_provenance,
        )
        print(f'{source}: {graph.original_cell_count:,} cells → '
              f'{graph.reduced_cell_count:,} generators; '
              f'{len(graph.nodes):,} nodes, {graph.edge_count:,} edges')
        print(f'  {node_path}\n  {edge_path}')
        if provenance_path:
            print(f'  {provenance_path}')
        if args.event_summary:
            summary_path = outdir / f'{source.stem}_ppg_event_summary.csv'
            graph.write_event_summaries(summary_path)
            print(f'  {summary_path}')
        if args.degrees:
            import csv
            outdir.mkdir(parents=True, exist_ok=True)
            degree_path = outdir / f'{source.stem}_ppg_degrees.csv'
            with degree_path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(('node_id', 'in_degree', 'out_degree'))
                for i, (indeg, outdeg) in enumerate(zip(graph.in_degrees, graph.out_degrees)):
                    writer.writerow((i, int(indeg), int(outdeg)))
            print(f'  {degree_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
