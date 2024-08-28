from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn


def demo_single_batch(model, tokenizer):
    # encode context the generation is conditioned on
    prompt = "I enjoy walking with my cute dog"
    model_inputs = tokenizer(prompt, return_tensors='pt').to(torch_device)

    input_ids = model_inputs['input_ids']
    print(input_ids.shape)  # (b, 7)

    # generate 40 new tokens
    # 注意 beam search 假设是 3，第一次运行时候会选择 top3，在第二次运行时候，此时 bs 就变成 3 了，然后在所有预测里面
    # 再选 top3，并不是说每个搜索路径都是 top3,这样路径会爆炸的，当然也不是每个搜索路径进行 top1
    # 所以如果某一条路径特别好，那么可能一直沿着这个路径进行 top3，其余路径等于没有用了额

    batch_size = 1
    num_beams = 5
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    # 在开始时候，由于实际上只有一条路径，其余 4 条路径的分数都是负无穷，后续累加时候有用
    beam_scores[:, 1:] = -1e9

    next_beam_tokens = torch.zeros((batch_size, num_beams), dtype=torch.int, device=input_ids.device)
    next_beam_indices = torch.zeros((batch_size, num_beams), dtype=torch.int, device=input_ids.device)

    # 为了统一计算，即使第一次运行时候，也会变成 (b*num_beams, seq_len)
    input_ids = input_ids.repeat_interleave(num_beams, dim=0)
    model_kwargs = {'input_ids': input_ids, 'use_cache': True}
    for _ in range(40):
        model_kwargs = model.prepare_inputs_for_generation(**model_kwargs)
        outputs = model(**model_kwargs)
        # logits (b, seq, vocab_size) -> (b, vocab_size)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        # 路径的 score 一直累加，最终选择的路径是 beam 里面 top1 的路径
        next_token_scores = next_token_scores + beam_scores.reshape(-1, 1).expand_as(
            next_token_scores
        )
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, num_beams, dim=1, largest=True, sorted=True
        )
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")  # 属于哪一条 beam 路径
        next_tokens = next_tokens % vocab_size  # 某一条路径中的某个预测 token

        # 如果考虑 bs=1，那么下面代码可以注释，进行简化
        # batch_idx = 0  # batch_size = 1
        # beam_idx = 0
        # for beam_token_rank, (next_token, next_score, next_index) in enumerate(
        #         zip(next_tokens[0], next_token_scores[0], next_indices[0])
        # ):
        #     batch_beam_idx = batch_idx * num_beams + next_index
        #     beam_scores[batch_idx, beam_idx] = next_score
        #     next_beam_tokens[batch_idx, beam_idx] = next_token
        #     next_beam_indices[batch_idx, beam_idx] = batch_beam_idx  # 最核心的是这个
        #     beam_idx += 1

        beam_scores = next_token_scores
        next_beam_tokens = next_tokens
        next_beam_indices = next_indices  # 最核心的是这个

        # 选择累加后 score 分数是 top5 的路径
        # 可能第一次跑是 [prompt,a],[prompt,b],[prompt,c],[prompt,d],[prompt,e]
        # 第二次变成 [prompt,a,a1],[prompt,a,a2],[prompt,d,d1],[prompt,d,d2],[prompt,e,e1]
        # 第三次变成 [prompt,d,d1,d11],[prompt,e,e1,e2],[prompt,e,e1,e3],[prompt,e,e1,e4]
        input_ids = torch.cat([input_ids[next_beam_indices.view(-1), :], next_beam_tokens.reshape(num_beams, 1)],
                              dim=-1)

        # 缓存需要
        # 由于 input_ids 第一维顺序变了，或者说 batch 维度变了，对应的 past_key_values 也要重新选择，否则是错的
        past_key_values = model._reorder_cache(outputs.past_key_values, next_beam_indices.view(-1))
        model_kwargs["past_key_values"] = past_key_values
        model_kwargs["input_ids"] = input_ids

    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))


if __name__ == '__main__':
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # add the EOS token as PAD token to avoid warnings
    model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)
    print(model)

    demo_single_batch(model, tokenizer)
