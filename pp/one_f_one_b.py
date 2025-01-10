pp_size = 16
pp_nmb = 64


def chunks_index(stage_index, num_stages, n_microbatches):
    warmup_chunks = min(
        n_microbatches,
        num_stages - stage_index,
    )

    fb_chunks = (n_microbatches - warmup_chunks)

    cool_down_chunks = n_microbatches - fb_chunks
    return warmup_chunks, fb_chunks, cool_down_chunks


def get_pp_stage(stage_index, warmup_chunks, fb_chunks, cool_down_chunks):
    out_str = ""
    fwd_count = 1
    for _ in range(warmup_chunks):
        out_str += f"{stage_index}F{fwd_count},"
        fwd_count += 1

    bwd_count = 1
    for _ in range(fb_chunks):
        out_str += f"{stage_index}B{bwd_count},"
        bwd_count += 1
        out_str += f"{stage_index}F{fwd_count},"
        fwd_count += 1

    for _ in range(cool_down_chunks):
        out_str += f"{stage_index}B{bwd_count},"
        bwd_count += 1
    return out_str


max_warmup_chunks = 0
max_fb_chunks = 0
max_cool_down_chunks = 0
for i in range(pp_size):
    warmup_chunks, fb_chunks, cool_down_chunks = chunks_index(i, pp_size, pp_nmb)
    out_str = get_pp_stage(i, warmup_chunks, fb_chunks, cool_down_chunks)
    print(out_str)
    max_warmup_chunks = max(max_warmup_chunks, warmup_chunks)
    max_fb_chunks = max(max_fb_chunks, fb_chunks)
    max_cool_down_chunks = max(max_cool_down_chunks, cool_down_chunks)

warmup_chunks, fb_chunks, cool_down_chunks = chunks_index(0, pp_size, pp_nmb)
# 假设 backward 时间是 forward 的 2 倍

forward_time = 1
backward_time = 2
read_times = (warmup_chunks * forward_time) + (
        fb_chunks * (forward_time + backward_time)) + cool_down_chunks * backward_time

# 由于在稳态阶段，forward 和 backward 是同时进行的，所以周期内中时间要以 backward 的时间为准
total_times = max_warmup_chunks * forward_time + max_fb_chunks * (
            backward_time + backward_time) + max_cool_down_chunks * backward_time
bubble_times = total_times - read_times

# 由于通信和计算无法重叠，所以如果考虑上通信的时间，那么空泡率会更高
print(f"单卡的空泡率: {bubble_times / total_times}")
