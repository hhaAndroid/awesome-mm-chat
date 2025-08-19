from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os
import orjson
import argparse

# refer from https://github.com/fzyzcjy/torch_utils/blob/master/src/torch_profile_trace_merger/sglang_profiler_trace_merger.py
def main(
        dir_data: str = "",
        all_ranks: bool = True,
        ranks: Optional[List[int]] = None,
        start_time_ms: Optional[int] = 0,
        end_time_ms: Optional[int] = 999999999,
        thread_filters: str = None,
):
    all_file_path = os.listdir(dir_data)
    interesting_paths = [
        Path(dir_data) / p for p in all_file_path if p.endswith(".json")
    ]

    print(f"{interesting_paths}")

    output_path = Path(dir_data) / f"merged.trace.json"
    _merge_chrome_traces(
        interesting_paths, output_path,
        config=Config(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            thread_filters=thread_filters,
        ),
    )


@dataclass
class Config:
    start_time_ms: int
    end_time_ms: int
    thread_filters: str


def _merge_chrome_traces(interesting_paths: List[Path], output_path: Path, config: Config):
    merged_trace = {"traceEvents": []}

    # with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    #     for output_raw in executor.map(_handle_file, interesting_paths, [config] * len(interesting_paths)):
    for output_raw in map(_handle_file, interesting_paths, [config] * len(interesting_paths)):
        merged_trace['traceEvents'].extend(output_raw['traceEvents'])
        for key, value in output_raw.items():
            if key != 'traceEvents' and key not in merged_trace:
                merged_trace[key] = value

    print(f"Write output to {output_path}")
    # 写入到 json
    if output_path.exists():
        print(f"Output file {output_path} already exists, will be overwritten.")
    else:
        print(f"Output file {output_path} does not exist, will be created.")
    # 使用 orjson 序列化
    print(f"Writing merged trace to {output_path}")
    json_str = orjson.dumps(merged_trace, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE).decode('utf-8')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def _handle_file(path, config: Config):
    print(f"handle_file START {path=}")

    with open(path, 'rb') as f:
        trace = orjson.loads(f.read())
        output = {key: value for key, value in trace.items() if key != 'traceEvents'}
        output['traceEvents'] = _process_events(trace.get('traceEvents', []), config)

    print(f"handle_file END {path=}")
    return output


def _process_events(events, config):
    print(f"{len(events)=}")

    # format: us
    min_ts = min(e["ts"] for e in events)
    ts_interest_start = min_ts + 1000 * config.start_time_ms
    ts_interest_end = min_ts + 1000 * config.end_time_ms
    events = [e for e in events if ts_interest_start <= e["ts"] <= ts_interest_end]
    print(f"after filtering by timestamp {len(events)=} ({ts_interest_start=} {ts_interest_end})")

    if config.thread_filters is not None:
        thread_filters_list = config.thread_filters.split(',')

        thread_name_of_tid = {
            e["tid"]: e["args"]["name"]
            for e in events
            if e["name"] == "thread_name"
        }

        def _thread_name_filter_fn(thread_id):
            ans = False
            if 'gpu' in thread_filters_list:
                ans |= "stream" in str(thread_id)
            return ans

        remove_tids = [
            tid
            for tid, thread_name in thread_name_of_tid.items()
            if not _thread_name_filter_fn(thread_name)
        ]
        print(f"{remove_tids=}")

        events = [e for e in events if e["tid"] not in remove_tids]
        print(f"after filtering by thread_filters {len(events)=}")

    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_data")
    parser.add_argument("--all_ranks", type=bool, default=True)
    parser.add_argument("--ranks", type=int, nargs='+', help="List of integers")
    args = parser.parse_args()

    if args.all_ranks:
        args.ranks = None
    main(dir_data=args.dir_data, ranks=args.ranks, all_ranks=args.all_ranks)
