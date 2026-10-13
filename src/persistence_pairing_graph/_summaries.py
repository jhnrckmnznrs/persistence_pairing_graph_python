"""Exact event-level summaries with no retained graph-edge objects."""
from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True, slots=True)
class EventSummary:
    dimension: int
    death_value: int | float
    dying_classes: int
    outgoing_edges: int

    @property
    def mean_outgoing_edges(self) -> float:
        return self.outgoing_edges / self.dying_classes


def write_event_summaries(path: str | Path, events: list[EventSummary]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(('dimension', 'death_value', 'dying_classes',
                         'outgoing_edges', 'mean_outgoing_edges'))
        for e in events:
            writer.writerow((e.dimension, e.death_value, e.dying_classes,
                             e.outgoing_edges, e.mean_outgoing_edges))
    return path
