# -*- coding: utf-8 -*-
import os
import json
import time
import pprint
from typing import Any, Dict

from io import StringIO


class Logger(object):
    """
    Append-only metric logger.

    Metrics are flushed to JSONL as they arrive so a process crash does not
    discard the completed sample/batch records. ``save_json`` materializes the
    legacy JSON array at successful completion for existing analysis scripts.
    """

    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path
        self.json_file_path = os.path.join(folder_path, "log-1.json")
        self.jsonl_file_path = os.path.join(folder_path, "log.jsonl")
        self.txt_file_path = os.path.join(folder_path, "log.txt")
        self._jsonl_fp = open(self.jsonl_file_path, "a", buffering=1)
        self.pp = MyPrettyPrinter(indent=2, depth=3, compact=True)

    def log_metric(
        self,
        name: str,
        values: Dict[str, Any],
        tags: Dict[str, Any],
        display: bool = False,
    ) -> None:
        """
        Store a scalar metric

        :param name: measurement, like 'accuracy'
        :param values: dictionary, like { epoch: 3, value: 0.23 }
        :param tags: dictionary, like { split: train }
        """
        record = {"measurement": name, **values, **tags}
        self._jsonl_fp.write(json.dumps(record, separators=(",", ":")) + "\n")
        # Line buffering normally flushes at the newline; make this explicit so
        # records survive a Python/core-dump process failure.
        self._jsonl_fp.flush()

        if display:
            print(
                "{name}: {values} ({tags})".format(name=name, values=values, tags=tags)
            )

    def pretty_print(self, value: Any) -> None:
        self.pp.pprint(value)

    def log(self, value: str, display: bool = True, save: bool = None) -> None:
        """Log human-readable text.

        Calls made with ``display=False`` are normally per-sample debug messages in
        adaptation methods. Do not silently write those messages to disk unless the
        caller explicitly requests ``save=True``.
        """
        content = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + value
        if display:
            print(content)
        should_save = display if save is None else save
        if should_save:
            self.save_txt(content)

    def save_json(self) -> None:
        """Atomically materialize the legacy JSON array from the JSONL stream."""
        self.flush(sync=True)
        temporary_path = self.json_file_path + ".tmp"
        with open(self.jsonl_file_path, "r") as source, open(
            temporary_path, "w"
        ) as destination:
            destination.write("[\n")
            first_record = True
            for line in source:
                record = line.strip()
                if not record:
                    continue
                # Validate each line before placing it in the JSON array. A
                # truncated final line from an unclean failure is ignored.
                try:
                    json.loads(record)
                except json.JSONDecodeError:
                    continue
                if not first_record:
                    destination.write(",\n")
                destination.write(record)
                first_record = False
            destination.write("\n]\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, self.json_file_path)

    def flush(self, sync: bool = False) -> None:
        self._jsonl_fp.flush()
        if sync:
            os.fsync(self._jsonl_fp.fileno())

    def get_offsets(self) -> Dict[str, int]:
        self.flush(sync=True)
        return {
            "log.jsonl": os.path.getsize(self.jsonl_file_path),
            "log.txt": os.path.getsize(self.txt_file_path)
            if os.path.exists(self.txt_file_path)
            else 0,
        }

    def truncate_to_offsets(self, offsets: Dict[str, int]) -> Dict[str, str]:
        """Archive uncommitted tails, then roll logs back to a checkpoint offset."""
        self._jsonl_fp.close()
        archived_tails = {}
        for filename, offset in offsets.items():
            path = os.path.join(self.folder_path, filename)
            if not os.path.exists(path):
                if offset != 0:
                    raise RuntimeError(
                        f"Cannot restore log offset {offset}: missing {path}."
                    )
                continue
            with open(path, "r+b") as fp:
                current_size = os.path.getsize(path)
                if offset > current_size:
                    raise RuntimeError(
                        f"Checkpoint log offset {offset} exceeds {path} size "
                        f"{current_size}."
                    )
                if offset < current_size:
                    fp.seek(offset)
                    tail = fp.read()
                    archive_name = (
                        f"orphaned-{filename}-{time.time_ns()}"
                    )
                    archive_path = os.path.join(self.folder_path, archive_name)
                    with open(archive_path, "wb") as archive_fp:
                        archive_fp.write(tail)
                        archive_fp.flush()
                        os.fsync(archive_fp.fileno())
                    archived_tails[filename] = archive_path
                fp.truncate(offset)
                fp.flush()
                os.fsync(fp.fileno())
        self._jsonl_fp = open(self.jsonl_file_path, "a", buffering=1)
        return archived_tails

    def close(self) -> None:
        if not self._jsonl_fp.closed:
            self.flush()
            self._jsonl_fp.close()

    def save_txt(self, value: str) -> None:
        with open(self.txt_file_path, "a") as f:
            f.write(value + "\n")


class MyPrettyPrinter(pprint.PrettyPrinter):
    """Borrowed from
    https://stackoverflow.com/questions/30062384/pretty-print-namedtuple
    """

    def format_namedtuple(self, object, stream, indent, allowance, context, level):
        # Code almost equal to _format_dict, see pprint code
        write = stream.write
        write(object.__class__.__name__ + "(")
        object_dict = object._asdict()
        length = len(object_dict)
        if length:
            # We first try to print inline, and if it is too large then we print it on multiple lines
            inline_stream = StringIO()
            self.format_namedtuple_items(
                object_dict.items(),
                inline_stream,
                indent,
                allowance + 1,
                context,
                level,
                inline=True,
            )
            max_width = self._width - indent - allowance
            if len(inline_stream.getvalue()) > max_width:
                self.format_namedtuple_items(
                    object_dict.items(),
                    stream,
                    indent,
                    allowance + 1,
                    context,
                    level,
                    inline=False,
                )
            else:
                stream.write(inline_stream.getvalue())
        write(")")

    def format_namedtuple_items(
        self, items, stream, indent, allowance, context, level, inline=False
    ):
        # Code almost equal to _format_dict_items, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(items) - 1
        if inline:
            delimnl = ", "
        else:
            delimnl = ",\n" + " " * indent
            write("\n" + " " * indent)
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write(key + "=")
            self._format(
                ent,
                stream,
                indent + len(key) + 2,
                allowance if last else 1,
                context,
                level,
            )
            if not last:
                write(delimnl)

    def _format(self, object, stream, indent, allowance, context, level):
        # We dynamically add the types of our namedtuple and namedtuple like
        # classes to the _dispatch object of pprint that maps classes to
        # formatting methods
        # We use a simple criteria (_asdict method) that allows us to use the
        # same formatting on other classes but a more precise one is possible
        if hasattr(object, "_asdict") and type(object).__repr__ not in self._dispatch:
            self._dispatch[type(object).__repr__] = MyPrettyPrinter.format_namedtuple
        super()._format(object, stream, indent, allowance, context, level)
