from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def demo_single_batch(model, tokenizer):
    # encode context the generation is conditioned on
    prompt = "I enjoy walking with my cute dog"
    model_inputs = tokenizer(prompt, return_tensors='pt').to(torch_device)

    input_ids = model_inputs['input_ids']
    print(input_ids.shape)  # (b, 7)

    model_kwargs = {'input_ids': input_ids, 'use_cache': True}
    # generate 40 new tokens
    for _ in range(40):
        model_kwargs = model.prepare_inputs_for_generation(**model_kwargs)
        outputs = model(**model_kwargs)
        # logits (b, seq, vocab_size) -> (b, vocab_size)
        next_tokens_scores = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)  # (b,)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # 缓存需要
        model_kwargs["past_key_values"] = outputs.past_key_values
        model_kwargs["input_ids"] = input_ids

    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))


if __name__ == '__main__':
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # add the EOS token as PAD token to avoid warnings
    model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)
    print(model)

    demo_single_batch(model, tokenizer)
