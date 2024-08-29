import numpy as np
import math

def matrix_tie():
    # 创建两个 4x4 矩阵
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

    B = np.array([[16, 15, 14, 13],
                  [12, 11, 10, 9],
                  [8, 7, 6, 5],
                  [4, 3, 2, 1]])

    # 不分块计算
    C = np.dot(A, B)
    print("不分块计算结果:\n", C)

    """
    A= [[ A1, A2]
        [ A3, A4]
        ]
    B= [[ B1, B2]
        [ B3, B4]
        ]    

    C= [[ A1*B1 + A2*B3, A1*B2 + A2*B4]
        [ A3*B1 + A4*B3, A3*B2 + A4*B4]
     ]
    在分块前只需要计算一次，分块为 2x2 后，需要 4x2 次乘法     
    """

    # 定义矩阵分块大小
    block_size = 2

    # 初始化结果矩阵
    C_blocked = np.zeros((A.shape[0], B.shape[1]))

    # 分块计算
    for i in range(0, A.shape[0], block_size):
        for j in range(0, B.shape[1], block_size):
            for k in range(0, A.shape[1], block_size):
                # 可以并行算，最后相加
                A_block = A[i:i + block_size, k:k + block_size]
                B_block = B[k:k + block_size, j:j + block_size]
                C_blocked[i:i + block_size, j:j + block_size] += np.dot(A_block, B_block)

    print("分块计算结果:\n", C_blocked)


def numpy_softmax():
    def softmax(x):
        # 为了数值稳定性，减去最大值
        # 虽然这种实现特别快，但是如果 logits 长度是 1b，那么这个计算会 OOM
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # 示例输入
    logits = np.array([2.0, 1.0, 0.1, 0.8])

    # 计算 Softmax
    softmax_output = softmax(logits)
    print("Softmax 输出:", softmax_output)


def python_softmax():
    logits = [2.0, 1.0, 0.1, 0.8]

    softmax_output = []
    sum_exp = 0
    max_value = -math.inf
    # 相比于上面的实现，使用 for 计算方式有一个好处：
    # 如果 logits 超级长，或者说 logits 就无法一次性 load 到内存中
    # 那么我们可以 for 进行分段计算
    # 比如 0 cpu 先计算 0~1000，然后把 max_value 值传递给 1 cpu，1 cpu 计算 1000~2000
    # 通过这种串行方式就可以解决上述为问题。

    # 先求最大值
    for logit in logits:
        max_value = max(max_value, logit)
    # 分母减掉最大值
    for logit in logits:
        sum_exp += math.exp(logit - max_value)
    # 计算每个元素
    for logit in logits:
        softmax_output.append(np.exp(logit - max_value) / sum_exp)
    print("Softmax 输出:", softmax_output)


# Safe softmax with online normalizer calculation
def python_online_softmax():
    logits = [2.0, 1.0, 0.1, 0.8]
    softmax_output = []
    pre_max_value = -math.inf
    sum_exp = 0

    # 上述做法虽然可以解决问题，但是有 3次 for 训练，也就是 logits 要 load 3 次，如果可以减少 for 次数那么就更好了
    # online 做法就可以变成 2 次，采用了类似在线方式

    # 只有两次 for 循环
    for logit in logits:
        max_value_ = max(pre_max_value, logit)
        sum_exp = sum_exp*math.exp(pre_max_value - max_value_) + math.exp(logit - max_value_)
        pre_max_value = max_value_

    # 计算每个元素
    for logit in logits:
        softmax_output.append(np.exp(logit - pre_max_value) / sum_exp)
    print("Softmax 输出:", softmax_output)


def python_online_softmax_parallel():
    logits = [2.0, 1.0, 0.1, 0.8]
    softmax_output = []

    def online_softmax_update(m0, d0, m1, d1):
        m = max(m0, m1)
        d = d0 * np.exp(m0 - m) + d1 * np.exp(m1 - m)
        return m, d

    # 上述做法虽然少了一次 for，但是必须串行运行，无法真正并行
    # 假设下面代码是并行计算，一共 2 张卡，每张卡计算一半数据
    # 卡0-模拟
    pre_max_value1 = -math.inf
    sum_exp1 = 0
    for logit in logits[0:2]:
        pre_max_value1, sum_exp1 = online_softmax_update(pre_max_value1, sum_exp1, logit, 1)

    # 卡1-模拟
    pre_max_value2 = -math.inf
    sum_exp2 = 0
    for logit in logits[2:]:
        pre_max_value2, sum_exp2 = online_softmax_update(pre_max_value2, sum_exp2, logit, 1)

    # 合并结果
    # 假设一共切分为 8 份，那么每张卡需要计算 2*3=8 即一共计算 3+1 次得到最终的 pre_max_value, sum_exp
    # 但是相比于之前单卡要计算 8 次，还是少很多的，并且显存大幅减少
    pre_max_value, sum_exp = online_softmax_update(pre_max_value1, sum_exp1, pre_max_value2, sum_exp2)

    # 计算每个元素，这个自然的并行
    for logit in logits:
        softmax_output.append(np.exp(logit - pre_max_value) / sum_exp)

    print("Softmax 输出:", softmax_output)


if __name__ == '__main__':
    matrix_tie()
    numpy_softmax()
    python_softmax()
    python_online_softmax()
    python_online_softmax_parallel()
