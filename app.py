import streamlit as st
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
import scipy
from pyEDA.main import *
from sklearn import preprocessing
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
import tsfresh
import seaborn as sns
import pickle
if not st.session_state:
    st.session_state.chart={"Mood":[],"Perfume":[]}

def upload_file() :
    body= st.empty()
    with body.container():
        st.title(f"""Hello! thank you for using this app""")
        st.subheader(f"This app will determine your mood based on you physiological data")
        st.write(f'## Please upload csv file contains the physiological data')
        with st.form(key="form1"):
            file=st.file_uploader("Please upload a csv file containing physiological data",type="csv",key="data")
            gsr_column=st.text_input("what is the column that contains GSR value?")
            ppg_column=st.text_input("what is the column that contains Heart Rate value?")
            submitted=st.form_submit_button()
            if submitted:
                # st.session_state.data=file
                st.session_state.gsr=gsr_column
                st.session_state.ppg=ppg_column
                # body.empty()
                # st.session_state.block=1

                table=pd.read_csv(file)
                
                try:
                    try:
                    # print(table["Perfume"].unique())
                        perfume=table["Perfume"].unique()
                        perfume=perfume[0]
                        table=table.drop("Perfume",axis=1)
                    # print(table)
                    except:
                        perfume="None"

                    cols=table.columns

                    for col in cols:
                        table[col]=pd.to_numeric(table[col])

                    try:
                        gsr=table[gsr_column].copy()
                    except:
                        gsr=pd.DataFrame()
                        st.error("GSR column not found")
                    try:
                        hr=table[ppg_column].copy()
                    except:
                        hr=pd.DataFrame()
                        st.error("HR column not found")
                    if(not gsr.empty and not hr.empty):
                        gsr=gsr.to_list()
                        hr=hr.to_list()
                        
                        
                            
                        

                        gsr=scipy.signal.resample(gsr,len(gsr)*128)
                        filt=scipy.signal.medfilt(gsr,kernel_size=128*8+1)
                        gsr=gsr-filt
                        gsr=(gsr-gsr.min())/(gsr.max()-gsr.min())
                        gsr=gsr.round(decimals=3)
                        
                        hr=scipy.signal.resample(hr,len(hr)*128)
                        hr=(hr-hr.min())/(hr.max()-hr.min())
                        hr=hr.round(decimals=3)
                        numpy_df=pd.DataFrame({'GSR':gsr,'HR':hr})
                        numpy_df["Session"]=1
                        st.markdown("## the GSR graph after normalized is:")
                        st.line_chart(gsr)
                        st.markdown("## the HR graph after normalized is:")
                        st.line_chart(hr)

                        with st.spinner(text="In progress..."):
                            progress= st.progress(0)
                            progress.progress(20)

                            settings = ComprehensiveFCParameters()
                            ex_data = extract_features(numpy_df, column_id="Session")
                            filter_cols=['GSR__fft_coefficient__attr_"real"__coeff_32',
                            'GSR__fft_coefficient__attr_"real"__coeff_64',
                            'HR__cwt_coefficients__coeff_7__w_2__widths_(2, 5, 10, 20)',
                            'HR__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)',
                            'HR__fft_coefficient__attr_"real"__coeff_7',
                            'HR__fft_coefficient__attr_"real"__coeff_11',
                            'HR__fft_coefficient__attr_"real"__coeff_13',
                            'HR__fft_coefficient__attr_"real"__coeff_69',
                            'HR__fft_coefficient__attr_"abs"__coeff_59']
                            sel_data= ex_data[filter_cols]
                            progress.progress(70)

                            data=sel_data.to_numpy()
                            loaded_model = pickle.load(open('FYP_RFmodel.pickle', "rb"))
                            progress.progress(80)

                            prediction=loaded_model.predict_proba(data)
                            classes = ["activation pleasant (Happy)", "activation unpleasant (Stressed)", "deactivation pleasant (Calm)", "deactivation unpleasant (Depressed)"]
                            class_simp=["Happy", "Stressed", "Calm", "Depressed"]
                            mood=classes[np.argmax(prediction)]
                            mood_simp=class_simp[np.argmax(prediction)]
                            confidence=np.max(prediction)
                            progress.progress(100)
                            progress.empty()
                            # st.success('Done!')
                            

                        st.markdown(f"# the predicted mood is {mood}")
                        st.markdown(f"# for perfume : {perfume}")
                        # st.markdown(f"# with confidence {confidence * 100:.2f}%")
                        print(mood)
                        print(perfume)
                        st.session_state.chart["Mood"].append(mood_simp)
                        st.session_state.chart["Perfume"].append(perfume)
                        
                        graph=pd.DataFrame(st.session_state.chart)
                        # print(graph)
                        # graph.hist()
                        graph=graph[graph["Perfume"]==perfume]["Mood"]

                        # print(graph)

                        st.markdown(f"# Mood distribution for Perfume: {perfume}")
                        fig, ax = plt.subplots()
                        ax.hist(graph)

                        st.pyplot(fig)


                except:
                    st.error("Something went wrong")
                
                



                

def process_file(file):
    body= st.empty()
    with body.container():
        st.title(f"""Please wait.. your file is being processed""")



with st.container():
   upload_file()

