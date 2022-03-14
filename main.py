import streamlit as st
import pickle
from os.path import exists
from streamlit_drawable_canvas import st_canvas
import pandas as pd

st.set_page_config(page_title="Doodle Classifier", layout="centered", initial_sidebar_state="collapsed")

# Load model from .pkl file
model = None
if exists("./model.pkl"):
    model = pickle.load(open("model.pkl", 'rb'))
else:
    print("model.pkl not found, please run model.py file to generate this!")

def make_prediction(pred):
    """Makes a prediction using the model, takes in a pandas series, i.e. a single row from a pandas df, or an array of the pixel values"""
    return model.predict_proba(pred)

st.write("Hello world!")

# Define drawing mode for canvas
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

# Other canvas parameters
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.0)",  # Fixed fill color with some opacity
    stroke_width=15,
    stroke_color="#000",
    background_color="#eee",
    background_image=None,
    update_streamlit=realtime_update,
    height=500,
    width=500,
    drawing_mode="freedraw",
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)

print(canvas_result.image_data[0])