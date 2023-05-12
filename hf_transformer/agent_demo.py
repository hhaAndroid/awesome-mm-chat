from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003")

# agent.run("Draw me a picture of rivers and lakes.", return_code=True)

# 一个更复杂的任务
agent.run("Draw me a picture of the sea then transform the picture to add an island", return_code=True)
