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


for i in range(pp_size):
    warmup_chunks, fb_chunks, cool_down_chunks = chunks_index(i, pp_size, pp_nmb)
    out_str = get_pp_stage(i, warmup_chunks, fb_chunks, cool_down_chunks)
    print(out_str)
