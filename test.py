import torch
import streamlit as st
from pathlib import Path

from model.cycleGAN import generator
from model import utils


def show_example(modelA, modelB, image_path, resize_size, padding):
    image = utils.singe_image_transform(image_path, resize_size=resize_size,
                                        padding=padding)
    with torch.no_grad():
        generated = modelA(image)
        restored = modelB(generated)
    images = torch.cat((image.cpu().detach(), generated.cpu().detach(),
                        restored.cpu().detach()))
    st.pyplot(utils.show_images(images))


def show_samples(modelA, modelB, image_path, resize_size=256, padding=0):
    image_path = Path(image_path)
    files = list(image_path.rglob('*.jpg')) + list(image_path.rglob('*.png'))
    for file in files:
        show_example(modelA, modelB, file,
                     resize_size=resize_size, padding=padding)


@st.cache
def load_model(load_weights_path):
    weights_path = Path(load_weights_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_AB = generator().to(device)
    model_BA = generator().to(device)
    model_AB.load_state_dict(torch.load(weights_path/'AB_generator.pth',
                                        map_location=device))
    model_BA.load_state_dict(torch.load(weights_path/'BA_generator.pth',
                                        map_location=device))
    model_AB.eval()
    model_BA.eval()
    return model_AB, model_BA


def show_dataset_block(dataset_title, from_title, to_title,
                       examples_A_url, examples_B_url,
                       model_AB, model_BA,
                       resize_A_size=256,  resize_B_size=256,
                       padding_A=0,  padding_B=0):

    st.title(dataset_title)
    st.text("Find here some examples or upload \
your own images to test the model")

    # show examples
    with st.form(f"Show examples {dataset_title}"):
        st.text(f"Generate examples {from_title}2{to_title} \
and {to_title}2{from_title}")
        submitted = st.form_submit_button("Show Examples")
        if submitted:
            show_samples(model_AB, model_BA, examples_A_url,
                         resize_size=resize_A_size, padding=padding_A)
            show_samples(model_BA, model_AB, examples_B_url,
                         resize_size=resize_B_size, padding=padding_B)

    # Upload A images
    with st.form(f"From {from_title} to {to_title}"):
        st.text(f"Upload {from_title} images for \
{from_title}2{to_title} generator")
        files_A = st.file_uploader(f"upload your {from_title} images".upper(),
                                   accept_multiple_files=True)

        submitted = st.form_submit_button("Generate Images")
        if submitted:
            for file in files_A:
                show_example(model_AB, model_BA, file,
                             resize_size=resize_A_size, padding=padding_A)

    # Upload zebra images
    with st.form(f"From {to_title} to {from_title}"):
        st.text(f"Upload {to_title} images for \
{to_title}2{from_title} generator")
        files_B = st.file_uploader(f"upload your {to_title} images".upper(),
                                   accept_multiple_files=True)
        submitted = st.form_submit_button("Generate Images")
        if submitted:
            for file in files_B:
                show_example(model_BA, model_AB, file,
                             resize_size=resize_B_size, padding=padding_B)


model_AB_horses, model_BA_horses = load_model('weights/horse2zebra')
model_AB_photos, model_BA_photos = load_model('weights/photo2portrait')


# Horses -> Zebras

show_dataset_block(
    dataset_title="Horses -> Zebras",
    from_title="Horses", to_title="Zebras",
    examples_A_url='examples/horse2zebra/horses',
    examples_B_url='examples/horse2zebra/zebras',
    model_AB=model_AB_horses, model_BA=model_BA_horses
    )


# Portraits -> Photos

show_dataset_block(
    dataset_title="Photos -> Pencil Portraits",
    from_title="Photos", to_title="Portraits",
    examples_A_url='examples/photo2portrait/photos',
    examples_B_url='examples/photo2portrait/potraits',
    model_AB=model_AB_photos, model_BA=model_BA_photos,
    resize_B_size=400,
    padding_B=(0, 50, 0, 0)
    )
