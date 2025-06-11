import streamlit as st
import keyword_exploration
import generate_train
import bot


# 创建一个具有多个页面的演示应用
def main():
    # 将标志添加到侧边栏
    st.sidebar.image("images/apple-logo-transparent.png", width=100)
    # st.sidebar.image("images/eve-logo.png", width=200)

    # 在侧边栏添加导航选项
    selected = st.sidebar.radio("导航页面", options=["首页", "生成训练", "关键词探索"])

    # 根据选择的页面展示不同的内容
    if selected == "首页":
        home()
    if selected == "生成训练":
        # 运行生成训练页面
        def run_generate_train():
            generate_train.main()

        run_generate_train()
    elif selected == "关键词探索":
        # 运行关键词探索页面
        def run_keyword_explore():
            keyword_exploration.main()

        run_keyword_explore()


# 首页内容
def home():
    st.title("欢迎来到苹果服务中心")
    bot.main()


if __name__ == "__main__":
    main()
