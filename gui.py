import streamlit as st
import streamlit_toggle as tog
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from Sorters import Sorter_Framework
from preprocessor import preprocessor
from Sorters import make_sorter, get_predictions

models = os.listdir('models')
with st.expander("create new model"):
    with st.container():
        testop = tog.st_toggle_switch(label="early stop", default_value=True,inactive_color = '#D3D3D3', 
            active_color="#11567f", 
            track_color="#29B5E8")
        tkfold = tog.st_toggle_switch(label="kfold", default_value=True,inactive_color = '#D3D3D3', 
                active_color="#11567f", 
                track_color="#29B5E8")
    col1,col2 = st.columns(2)
    input_size = col1.slider("input size", 32,128, 48)
    classes = col1.slider("classes", 2,32,4)
    epochs = col1.slider("epochs", 10,100,24)
    batch_size = col1.slider("model batch size", 4, 128, 32)

    if(tkfold): kfold = col2.slider("model kfold", 2, 10, 4)
    lr_decay = col2.slider("lr decay", 0.75,0.99,0.95)
    validation_split = col2.slider("val split", 0.1, 0.5, 0.35)
    if(testop): estop = col2.slider("early stopping",0.25,0.9,0.5)

    if st.button('train new model'):
            new_sorter = Sorter_Framework(input_size=input_size, class_num=classes)
            if(tkfold): new_sorter.kfold = kfold
            if(testop): new_sorter.estop = estop
            new_sorter.batch_size = batch_size
            new_sorter.epochs = epochs
            new_sorter.validation_split = validation_split
            new_sorter.lr_decay = lr_decay
            
            make_sorter(new_sorter)
            history = new_sorter.history_ensemble
            loss = []; acc = []; val_loss = []; val_acc = []
            for h in history:
                loss.append(h.history['loss'])
                acc.append(h.history['accuracy'])
                val_loss.append(h.history['val_loss'])
                val_acc.append(h.history['val_accuracy'])

            df = pd.DataFrame({
                "loss":loss,
                "acc": acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            st.data_editor(
                df,
                column_config={
                    "loss": st.column_config.LineChartColumn("Loss for each member", width="medium",y_min=0, y_max=2),
                    "val_loss": st.column_config.LineChartColumn("Val-loss for each member", width="medium", y_min=0, y_max=2),
                    "acc": st.column_config.LineChartColumn("Accuracy for each member", width="medium", y_min=0, y_max=1),
                    "val_acc": st.column_config.LineChartColumn("Val-accuracy for each member", width="medium", y_min=0, y_max=1),
                },
            )

with st.expander("pretrained models"):
    select_model = st.selectbox(
        'trained models',
        tuple(models))  

    if select_model:
        size, classes = tuple(select_model.split('x'))
        classes=classes.split('_')
        sorter = Sorter_Framework(int(size),class_num=int(classes[0]))
        epochs = st.slider("epochs per member", 10,100,sorter.epochs)
        batch_size = st.slider("batch size", 4, 128, sorter.batch_size)
        kfold = st.slider("kfold", 2, 10, sorter.kfold)

        if st.button('retrain model'):
            sorter.kfold = kfold
            sorter.epochs = epochs
            sorter.batch_size = batch_size
                
            make_sorter(sorter)
            history = sorter.history_ensemble
            loss = []; acc = []; val_loss = []; val_acc = []
            for h in history:
                loss.append(h.history['loss'])
                acc.append(h.history['accuracy'])
                val_loss.append(h.history['val_loss'])
                val_acc.append(h.history['val_accuracy'])

            df = pd.DataFrame({
                "loss":loss,
                "acc": acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            st.data_editor(
                df,
                column_config={
                    "loss": st.column_config.LineChartColumn("Loss for each member", width="medium",y_min=0, y_max=2),
                    "val_loss": st.column_config.LineChartColumn("Val-loss for each member", width="medium", y_min=0, y_max=2),
                    "acc": st.column_config.LineChartColumn("Accuracy for each member", width="medium", y_min=0, y_max=1),
                    "val_acc": st.column_config.LineChartColumn("Val-accuracy for each member", width="medium", y_min=0, y_max=1),
                },
            )

        samples = int(st.slider("samples", 0,50,10))
        if st.button('run predictions'):
            fig = get_predictions(sorter, samples)
            st.pyplot(fig)