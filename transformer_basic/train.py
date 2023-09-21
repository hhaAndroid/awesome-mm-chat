import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Batch import create_masks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-trg_data', required=True)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)

    parser.add_argument('-no_cuda', action='store_true')

    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)

    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)

    parser.add_argument('-load_weights')

    args = parser.parse_args()
    return args


def train_model(model, opt):
    print("training model...")
    model.train()
    start = time.time()

    for epoch in range(opt.epochs):

        total_loss = 0
        print("%dm: epoch %d [%s]  %d%%  loss = %s" % \
              ((time.time() - start) // 60, epoch + 1, "".join(' ' * 20), 0, '...'), end='\r')

        for i, batch in enumerate(opt.train):

            src = batch.src.transpose(0, 1).to(opt.device)
            trg = batch.trg.transpose(0, 1).to(opt.device)
            # 假设 bs=200，那么其含义是 src 或者 tar 的元素个数最多是 200，并不是我们常规里面的 batch=200
            # 所以训练时候的 bs 其实是动态的，不是固定的。
            # 而且因为翻译任务，src 和 tar 不一样长，因此序列长度也是动态的
            # print(src.numel(), trg.numel())
            # print(src.shape, trg.shape)

            # TRG 的开始符和结束符为 '<sos>' '<eos>'， 分别位于词汇表的 2 和 3 索引位置
            # 在迭代每个样本训练时候，原始的 trg 样本都会自动追加开始符和结束符
            trg_input = trg[:, :-1]  # -1 就是结束符

            # 假设某个trg 样本序列长度是 11，并且是由 <sos> .... <eos> 构成
            # 那么在输入到model 时候，只需要输入前 10 个就可以，因此 eos 输入也没有意义
            # 模型 forward 后，也会输出长度为 10 的预测值，这个预测值的 trg 应该是 .... <eos>
            # 因此在计算 loss 时候，有 ys = trg[:, 1:] 操作。
            # 也就是说模型输入的 trg_input 和模型预测的 preds 相差了一位

            ######## 原始 target ：<sos> .... <eos>
            ######## 输入到模型里面的 query: <sos> ....
            ######## 模型预测输出的 GT:      .... <eos>
            # 在推理时候，初始化 query 设置为 <sos>，然后顺序解码，直到碰到 <eos> 为止
            src_mask, trg_mask = create_masks(src, trg_input, opt)  # '<eos>'创建 src 的 pad mask 和 trg 的下三角 mask
            src_mask.to(opt.device)
            trg_mask.to(opt.device)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)  # 计算 loss 时候开始符位置也不需要计算 loss
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)

            loss.backward()
            opt.optimizer.step()

            total_loss += loss.item()

            if (i + 1) % opt.printevery == 0:
                p = int(100 * (i + 1) / opt.train_len)
                avg_loss = total_loss / opt.printevery
                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, avg_loss), end='\r')
                total_loss = 0

        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
              ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
               avg_loss, epoch + 1, avg_loss))


def main():
    opt = parse_args()
    opt.device = 'cuda' if opt.no_cuda is False else 'cpu'
    if opt.device == 'cuda':
        assert torch.cuda.is_available()

    read_data(opt)
    # TRG 的开始符和结束符为 '<sos>' '<eos>'， 分别位于词汇表的 2 和 3 索引位置
    # 在迭代每个样本训练时候，原始的 trg 样本都会自动追加开始符和结束符
    SRC, TRG = create_fields(opt)
    opt.train = create_dataset(opt, SRC, TRG)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    train_model(model, opt)

    dst = 'weights'
    print("saving weights to " + dst + "/...")
    torch.save(model.state_dict(), f'{dst}/model_weights')


if __name__ == "__main__":
    main()
