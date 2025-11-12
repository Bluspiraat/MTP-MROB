import streamlit as st
import pandas as pd
import numpy as np
import json
import os


class Model:
    def __init__(self, model_folder):
        self.name = model_folder.split('/')[-1]
        self.confusion_matrix = os.path.join(model_folder, "confusion_matrix.png")
        self.loss_plot = os.path.join(model_folder, "loss_plot.png")
        self.loss_values = json.load(open(os.path.join(model_folder, "loss_values.json")))
        self.metrics = self.get_metrics(os.path.join(model_folder, "metrics.json"))
        self.parameters = json.load(open(os.path.join(model_folder, "parameters.json")))
        self.examples = self.get_examples(os.path.join(model_folder, "examples"))

    def get_metrics(self, metrics_file):
        data = json.load(open(metrics_file))
        data['Per_Class'] = pd.DataFrame.from_dict(data['Per_Class'], orient='index')
        return data

    def get_examples(self, example_folder):
        example_files = []
        files = os.listdir(example_folder)
        for file in os.listdir(example_folder):
            example_files.append(os.path.join(example_folder, file))
        return example_files
        # return [os.path.join(example_folder, file) for file in os.listdir(example_folder)]


def get_models(root_folder):
    model_folders = os.listdir(root_folder)
    models = []
    for model_folder in model_folders:
        models.append(Model(os.path.join(root_folder, model_folder)))
    return models


def update_index(forward=True):
    if forward:
        if st.session_state.image_index < 19:
            st.session_state.image_index += 1
            pass
        if st.session_state.image_index >= 19:
            st.session_state.image_index = 0
    if not forward:
        if st.session_state.image_index > 0:
            st.session_state.image_index -= 1
            pass
        if st.session_state.image_index == 0:
            st.session_state.image_index = 19


if __name__ == '__main__':
    data_folder = "C:/MTP-Data/trained_models/"
    models = get_models(data_folder)

    '''
    # --- Page setup --- #
    st.set_page_config(layout="wide")

    # --- Inspect individual model --- #
    st.title("Model comparison")
    st.subheader("Loss graph and confusion matrix")
    st.sidebar.header("Specific model")
    model_dict = {model.name: model for model in models}
    chosen_model = st.sidebar.selectbox("Choose which model to view", model_dict.keys())
    frame = pd.DataFrame([model_dict[chosen_model].parameters])
    st.table(frame)

    loss_plot, confusion_matrix = st.columns(2)
    with loss_plot:
        st.image(model_dict[chosen_model].loss_plot)
    with confusion_matrix:
        st.image(model_dict[chosen_model].confusion_matrix)

    st.subheader("Examples from the model")
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0
    st.image(model_dict[chosen_model].examples[st.session_state.image_index])
    previous_button, next_button = st.columns(2)
    with previous_button:
        st.button("Previous", on_click=lambda: update_index(forward=False), use_container_width=True)
    with next_button:
        st.button("Next", on_click=lambda: update_index(forward=True), use_container_width=True)



    # --- Compare models --- #
    st.sidebar.header("Compare model metrics")
    st.title("Compare model metrics")
    chosen_models = st.sidebar.multiselect("Select models", model_dict.keys())
    chosen_metrics = st.sidebar.multiselect("Select metric", list(models[0].metrics.keys())[0:-1])
    st.subheader(f"Selected models: {chosen_models}")
    st.subheader(f"Selected metrics: {chosen_metrics}")

    data = {
        m.name: {metric: m.metrics[metric] for metric in chosen_metrics}
        for m in models if m.name in chosen_models
    }

    st.dataframe(pd.DataFrame.from_dict(data).T
                 .style.highlight_max(axis=0, color='green'))

    # --- Compare performance per class
    st.title("Compare performance per model per class")
    st.sidebar.header("Compare class performance")
    class_metric = st.sidebar.radio("Select class metric", models[0].metrics['Per_Class'].columns)
    st.subheader(f"Selected evaluation metric: {class_metric}")

    data_per_class = pd.DataFrame()
    for model_name in chosen_models:
        model = model_dict[model_name]
        data_per_class[model_name] = model.metrics["Per_Class"][class_metric]
    data_per_class.replace(0, np.nan, inplace=True)
    st.dataframe(data_per_class.style.highlight_max(axis=1, color='green'), height=550)
    '''