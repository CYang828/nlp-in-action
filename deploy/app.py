import streamlit as st
import pandas as pd
import numpy as np
import keyword_exploration
import generate_train
import bot

# Creating a demo app with multiple pages
def main():
    # Adding logo to sidebar
    st.sidebar.image("images/apple-logo-transparent.png", width=200)
    # st.sidebar.image("images/eve-logo.png", width=200)

    selected = st.sidebar.radio(
        "Navigate pages", options=["首页", "生成训练", "关键词探索"]
    )

    if selected == "首页":
        home()
    if selected == "生成训练":

        def run_generate_train():
            generate_train.main()

        run_generate_train()
    elif selected == "关键词探索":

        def run_keyword_explore():
            keyword_exploration.main()

        run_keyword_explore()


def home():
    st.title("欢迎来到苹果服务中心")
    bot.main()


if __name__ == "__main__":
    main()
