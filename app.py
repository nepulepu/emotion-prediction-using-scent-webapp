import streamlit as st
import time
import pandas as pd
# import os
import matplotlib.pyplot as plt
import numpy as np
# import heartpy as hp
import scipy
# from pyEDA.main import *
# from sklearn import preprocessing
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
        st.write("Samples data can get it on : https://tinyurl.com/samples-data")

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

                        st.markdown("## the GSR graph is:")
                        st.line_chart(gsr)
                        st.markdown("## the HR graph is:")
                        st.line_chart(hr)                     
                                                 
                        gsr=scipy.signal.resample(gsr,len(gsr)*128)
                        filt=scipy.signal.medfilt(gsr,kernel_size=128*8+1)
                        gsr=gsr-filt
                        # gsr=(gsr-gsr.min())/(gsr.max()-gsr.min())
                        # gsr=gsr.round(decimals=3)
                        
                        hr=scipy.signal.resample(hr,len(hr)*128)
                        # hr=(hr-hr.min())/(hr.max()-hr.min())
                        # hr=hr.round(decimals=3)
                        numpy_df=pd.DataFrame({'GSR':gsr,'HR':hr})
                        numpy_df["Session"]=1
                        
                        with st.spinner(text="In progress..."):
                            progress= st.progress(0)
                            progress.progress(20)

                            settings = ComprehensiveFCParameters()
                            ex_data = extract_features(numpy_df, column_id="Session")
                            arousal_filter=['GSR__fft_coefficient__attr_"real"__coeff_78',
                            'GSR__approximate_entropy__m_2__r_0.3',
                            'GSR__approximate_entropy__m_2__r_0.9',
                            'GSR__energy_ratio_by_chunks__num_segments_10__segment_focus_6',
                            'HR__fft_coefficient__attr_"real"__coeff_19',
                            'HR__fft_coefficient__attr_"real"__coeff_52',
                            'HR__fft_coefficient__attr_"abs"__coeff_78',
                            'HR__fft_coefficient__attr_"angle"__coeff_72']
                            valence_filter =['GSR__large_standard_deviation__r_0.15000000000000002',
                            'GSR__binned_entropy__max_bins_10',
                            'GSR__fft_coefficient__attr_"real"__coeff_53',
                            'GSR__lempel_ziv_complexity__bins_2',
                            'GSR__lempel_ziv_complexity__bins_5',
                            'GSR__lempel_ziv_complexity__bins_10',
                            'GSR__lempel_ziv_complexity__bins_100',
                            'HR__fft_coefficient__attr_"real"__coeff_24',
                            'HR__fft_coefficient__attr_"imag"__coeff_37',
                            'HR__fft_coefficient__attr_"abs"__coeff_43',
                            'HR__fft_coefficient__attr_"angle"__coeff_37']
                            sel_A= ex_data[arousal_filter].copy()
                            sel_V= ex_data[valence_filter].copy()
                            progress.progress(70)

                            sel_A=sel_A.to_numpy()
                            sel_V=sel_V.to_numpy()
                            model_A = pickle.load(open('FYP_RF_A_slice_model.pkl', "rb"))
                            model_V = pickle.load(open('FYP_RF_V_slice_model.pkl', "rb"))
                            progress.progress(80)

                            A_predict=model_A.predict_proba(sel_A)
                            V_predict=model_V.predict_proba(sel_V)

                            def class_predict (A,V):
                                if A==0 and V==0:
                                    return "activation pleasant (Happy)","Happy"
                                if A==0 and V==1:
                                    return "activation unpleasant (Stressed)","Stressed"
                                if A==1 and V==0:
                                    return "deactivation pleasant (Calm)","Calm"
                                if A==1 and V==1:
                                    return "deactivation unpleasant (Sad)","Sad"
                                else:
                                    return None

                            # classes = ["activation pleasant (Happy)", "activation unpleasant (Stressed)", "deactivation pleasant (Calm)", "deactivation unpleasant (Depressed)"]
                            # class_simp=["Happy", "Stressed", "Calm", "Depressed"]
                            # mood=classes[np.argmax(prediction)]
                            mood,mood_simp=class_predict(np.argmax(A_predict),np.argmax(V_predict))
                            # mood_simp=class_simp[np.argmax(prediction)]
                            # confidence=np.max(prediction)
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

