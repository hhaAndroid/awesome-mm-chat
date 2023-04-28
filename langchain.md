# LangChain 说明

官方地址： https://github.com/hwchase17/langchain   
文档： https://python.langchain.com/en/latest/ 重要，特别是 agent 部分   
官方博客：https://blog.langchain.dev/   
非官方中文教程： https://liaokong.gitbook.io/llm-kai-fa-jiao-cheng/ 写的很好   
awesome: https://github.com/kyrolabs/awesome-langchain   
B 站上面有很多： https://search.bilibili.com/all?keyword=LangChain 但是没发现真正啥有用的   

LangChain 目标是将 LLM 与开发者现有的知识和系统相结合，以提供更智能化的服务。众所周知 OpenAI 的 API 无法联网的，所以如果只使用自己的功能实现联网搜索并给出回答、总结 PDF 文档、基于某个 Youtube 视频进行问答等等的功能肯定是无法实现的，强大的 LangChain 可以做到这个事情

它赋予LLM两大核心能力
- 数据感知，将语言模型与其他数据源相连接
- 代理能力，允许语言模型与其环境互动。

LangChain 是一个用于开发由语言模型驱动的应用程序的框架。他主要拥有 2 个能力：

- 可以将 LLM 模型与外部数据源进行连接
- 允许与 LLM 模型进行交互

LangChain 的主要应用场景包括个人助手、基于文档的问答、聊天机器人、查询表格数据、代码分析等。应用非常广泛，例如 VisualChatGPT 就是使用了这个库进行规划和执行特定函数。

它逻辑上主要有6大模块：llms、prompt、chains、agents、memory、indexes。

LangChain主要包括以下几个主要的模块：

- Prompt Templates：支持自定义Prompt工程的快速实现以及和LLMs的对接；
- LLMs：提供基于OpenAI API封装好的大模型，包含常见的OpenAI大模型，也支持自定义大模型的封装；
- Utils：大模型常见的植入能力的封装，比如搜索引擎、Python编译器、Bash编译器、数据库等等；
- Chains（重点）：大模型针对一系列任务的顺序执行逻辑链；
- Agents（重点）：通常 Utils 中的能力、Chains 中的各种逻辑链都会封装成一个个工具（Tools）供 Agents 进行智能化调用，从而具备了调用插件功能

其中，Chains 和 Agents 两个模块是 LangChain 的亮点。

注意： GPT3.5 模型本身是不支持多轮对话的，ChatGPT 其实是在 GPT3.5 之上植入了 Memory 实现了多轮对话的能力。也就是说我们经常和 LLM 进行对话，实际上他本身不具备连续对话能力，只不过程序员会维持一个可自定义的 memory bank
每次聊天时候都会把前后文发过去，所以才有了连续对话能力。

REACT 技术 (Reason+Act): https://arxiv.org/pdf/2210.03629.pdf   
LLM 的 ReAct 模式的 Python 实现: https://til.simonwillison.net/llms/python-react-pattern   
执行过程：https://blog.csdn.net/qq_35361412/article/details/129797199 写的非常详细，强烈推荐   

## langchain-ChatGLM 学习记录

基于 ChatGLM 的本地知识库 QA 应用

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/235078705-dc362fc9-21b4-45d3-ba2d-e40d49638f74.png"/>
</div>

项目地址：https://github.com/imClumsyPanda/langchain-ChatGLM   
在线可运行地址：https://www.heywhale.com/mw/project/643977aa446c45f4592a1e59   

实现原来看起来比较简单：

1. 下载 GanymedeNil/text2vec-large-chinese 支持中文的 Embedding 模型
2. 用户上传一份或者多份文档，注意由于内部是采用 langchain 来自动解析，应该需要符合一定格式。
3. 对上传文档进行一些处理，输入到 Embedding 模型中得到 vectors
4. 用户给定一个问题，先将该问题转换成 vector，然后计算与文档中每个 vector 的相似度，取 topk 个 vector。
5. 对查询出来的 topk 个 vector 反向查找文本字符串，将其和用户输入组成 prompt 输入给 ChatGLM 模型，得到答案。

如果失败，则原因在于 

- embedding 没找对 top-k
- chatglm 的 openqa 精度还有提升空间

将片段文转为向量，并将向量和原片段文本存到向量数据库，在用户提问时，先将提问的文本转为向量，在向量数据库查找相近向量的原片段，将原片段拼接为上下文，将用户问题和拼接的上下文发给chatgpt返回结果.

一个更大的类似的项目是： 

- https://github.com/yanqiangmiffy/Chinese-LangChain
- https://github.com/hwchase17/chat-langchain

## visual chatgpt

Visual ChatGPT 原理：https://medium.com/mlearning-ai/visual-chatgpt-paper-and-code-review-ffe69ff16671   
看完这个博客 https://blog.csdn.net/qq_35361412/article/details/129797199 就知道 visual chatgpt 运行原理了。核心还是构建 prompt ，然后一切交给 LLM 就行。langchain 的应用。

### hugginggpt

论文地址： https://arxiv.org/pdf/2303.17580.pdf   
官方地址：https://github.com/microsoft/JARVIS/blob/main/server/run_gradio_demo.py   

代码写的比较简单，没有用 langchain 库，看起来比较容易。做法实际上和 visualGPT 一样，但是 visualGPT 用了 langchain，代码虽然看起来比较简洁，但是比较难理解。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/235107725-848c048f-4724-4efe-b771-e7f61eb02d1c.png"/>
</div>

看图可以知道其干了啥。这个流程要 work，一个必备前提是超强 LLM，否则任何一步都可能失败了。

总的来说要实现任务自动调度，或者说 autoGPT 功能，需要按照顺序执行如下步骤：

1. 用户输入一个超长的包括各种要求的 prompt，Can you describe what this picture depicts  and count how many objects in the picture? 注意输入要给定图片，实际上只要给定真正存在的图片路径就行，不需要提前自己读取和解析啥的，因为后面的 task 会自动处理
2. 解析任务阶段：基于用户给定的 prompt 和预定的超长 prompt  #1 Task Planning Stage: The AI 来进行自动任务规划，这个 prompt 其实类似一个多选任务，里面包括了所有应该支持的任务。因为要明确的告诉 LLM 有哪些任务可以选择，不然没法处理
3. 上面的人为构建的 prompt 调用 llm 后就会输出所以应该需要的 task，并且会输出执行依赖和顺序。这个就非常考验 LLM 能力，如果能力稍微弱一点就做不到了。
4. 模型选择部分：注意不是所有模型都是要进行选择的，只有某些功能相同的模型需要再次提供 prompt 给 LLM 进行选择(选择已经是每个 model 的描述和 like 字段)，代码里面有部分 hard code。存在的问题是： 在进行模型选择时候，输入的只是 task 文本，按照道理来说选择的模型每次都一样才是(
可能还是有点动态，以 text2video 为例，不同的图片会产生不同的描述，这个描述而不同描述会输入到 prompt 中进行模型选择，由于描述参数不一样，可能选择的模型也不一样)。看样子目前无法处理：请选择一个最轻量化的检测模型，因为目标检测的输入只是图片，而没有文本，因此模型选择时候 prompt 应该是一样的。
5. 任务执行阶段： 基于前面规划好的任务，开启多进程执行即可，得到每个任务的输出
6. 结果收集阶段： 将前面收集到的所有数据，按照特定的 prompt形式组织再次输入给 llm，让他进行总结，最终就输出了一个包括详细过程的响应输出

可以发现整个过程的核心就是 prompt 构建，构建好后无脑的丢给 llm 就行了。用户其实 hard code 地方不多。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/235113570-f49d24bb-9f80-4a56-9d56-4394eaa6d00e.png"/>
</div>

上图就是所有的 prompt，理解了这个表就理解了这个做法。

**如果想让去其他模型也具备 langchain 功能，例如完成类似 hugginggpt 功能和 visual chatgpt 功能，只需要把模型部署到本地，直接通过 post 方式访问，就可以直接换掉 LLM 了。**

