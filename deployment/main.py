import streamlit as st
from matplotlib import pyplot as plt

st.title("Movie Review Sentiment Analysis")
t = st.text_input("Enter your movie review:")

def draw_horizontal_line(conf):
    conf = float(conf)
    neg_conf = 100 - conf
    st.write("<style> .element-container { display: flex; justify-content: center; } </style>", unsafe_allow_html=True)

    # Create a horizontal line with two different colors
    line_html = f'<hr style="width: {conf}%; border: none; height: 5px; background-color: green; display: inline-block; margin: 0;">'
    line_html += f'<hr style="width: {neg_conf}%; border: none; height: 5px; background-color: red; display: inline-block; margin: 0;">'

    # Display the line and labels
    st.write(f'''
             <div class="element-container">{line_html}</div>
                <div style="display: flex; justify-content: space-between;">
                    <div>{conf}%</div>
                    <div>{neg_conf}%</div>
                </div>
            </div>
             ''',unsafe_allow_html=True)

if st.button("Predict Sentiment"):
    draw_horizontal_line(t)