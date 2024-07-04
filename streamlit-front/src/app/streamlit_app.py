import streamlit as st
import requests
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io

api_url = "http://back:8000/api/v1/predict"

st.title("Handwritten Digit Recognition")

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#eee")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])


canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    height=500,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img_byte_arr = io.BytesIO()
        img = Image.fromarray(canvas_result.image_data.astype("uint8"))
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        

        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Error: Unable to get prediction")
    else:
        st.write("Please draw something on the canvas before predicting.")

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, use_column_width=True)