import random

# 可视化
import seaborn as sns

sns.set(style="ticks", color_codes=True)
# 数据科学
import pandas as pd
import yaml

# 加载对象
train = pd.read_pickle("../objects/train.pkl")

with open(r"../objects/entities.yml") as file:
    entities = yaml.load(file, Loader=yaml.FullLoader)

# with open(r"../objects/sorted_predictions.yml") as file:
#     entities = yaml.load(file, Loader=yaml.FullLoader)


# 制作一个类来定义当您
class Actions:
    memory = {"hardware": [], "app": []}

    def __init__(self, startup):
        # 初始提示
        self.startup = startup

    # 如果打招呼
    def utter_greet(self):
        # 存储响应库
        return random.choice(
            [
                "你好！我叫果bo。我今天可以怎么帮你？",
                "你好。我能帮上忙吗？",
            ]
        )

    # 如果再见
    def utter_goodbye(self):
        reaffirm = ["还有其他事我可以帮忙的吗？"]
        goodbye = [
            "感谢您的时间。祝您有个愉快的一天！",
            "很高兴能帮上忙，祝您有个愉快的一天！",
        ]
        return random.choice(goodbye)

    # 联系客服
    def link_to_human(self):
        return random.choice(["好的。让我把您转接到客服！"])

    def battery(self, entity):
        if entity == "none":
            return random.choice(["您使用的是什么设备？", "您使用的是什么设备？"])
        else:
            return random.choice(["很抱歉听到这个。您可以在设置中检查电池健康状况。如果低于75％，请考虑在您当地的苹果商店更换"])

    def forgot_pass(self):
        reset_appleid = "https://support.apple.com/en-us/HT201355"
        return f"很抱歉听到这个，前往{reset_appleid}"

    def payment(self):
        return random.choice(["使用您的Apple ID登录并更新您的付款方式"])

    def challenge_robot(self):
        return random.choice(
            [
                "我是果bo，您的个人助手，由Matthew设计来帮助您。",
            ]
        )

    def update(self, entity):
        # 确认硬件
        if entity == "none":
            return random.choice(["您使用的是什么设备？", "您使用的是什么设备？"])
        elif entity == "macbook pro":
            return random.choice(
                [
                    "在这里找到有关如何更新您的macbook pro的详细信息：https://support.apple.com/en-us/HT201541"
                ]
            )
        else:
            return random.choice(
                ["很抱歉，更新对您来说不起作用。请在这里找到更多信息：https://support.apple.com/en-us/HT201222"]
            )

    def info(self, entity):
        if entity == "macbook pro":
            return random.choice(
                [
                    "好的！现在我们有13英寸和16英寸的macbook pro。请在这里找到更多信息：https://www.apple.com/macbook-pro/"
                ]
            )
        if entity == "ipad":
            return random.choice(["我们有几种iPad可供选择，价格从"])
        if entity == "iphone":
            return random.choice(
                [
                    "我们最新的iPhone型号是iPhone 11。它有不同的型号尺寸。请在这里找到更多信息：https://www.apple.com/iphone/"
                ]
            )
        if entity == "none":
            return random.choice(["您想了解哪方面的信息？"])

    def fallback(self):
        return random.choice(["抱歉，我不太明白您想说什么。您能重新表达一下吗？"])
