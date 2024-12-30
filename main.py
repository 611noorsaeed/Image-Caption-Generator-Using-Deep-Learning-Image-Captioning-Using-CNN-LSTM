import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle


# Function to generate and display caption
def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34,img_size=224):
    # Load the trained models and tokenizer
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Preprocess the image
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)  # Extract image features

    # Generate the caption
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()

    # Display the image with the generated caption
    img = load_img(image_path, target_size=(img_size, img_size))
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=16, color='blue')
    st.pyplot(plt)  # Display image in Streamlit


# Streamlit app interface
def main():
    st.title("Image Caption Generator")
    st.write("Upload an image and generate a caption using the trained model.")

    # Upload the image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save the uploaded image temporarily
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Paths for the saved models and tokenizer
        model_path = "models/model.keras"  # Replace with the actual path
        tokenizer_path = "models/tokenizer.pkl"  # Replace with the actual path
        feature_extractor_path = "models/feature_extractor.keras"  # Replace with the actual path

        # Generate caption and display image with caption
        generate_and_display_caption("uploaded_image.jpg", model_path, tokenizer_path, feature_extractor_path)


if __name__ == "__main__":
    main()
