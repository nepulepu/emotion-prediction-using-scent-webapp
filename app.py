import streamlit as st
import streamlit_ext as ste
import time
import pandas as pd
import os
import firebase_admin
from firebase_admin import firestore
import matplotlib.pyplot as plt
import numpy as np
# import heartpy as hp
import scipy
# from pyEDA.main import *
# from sklearn import preprocessing
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
import tsfresh
# import seaborn as sns
import pickle
from PIL import Image
import io
# from dotenv import load_dotenv

# load_dotenv()



def create_keyfile_dict():
    variables_keys = {
        "type": os.getenv("TYPE"),
        "project_id": os.getenv("PROJECT_ID"),
        "private_key_id": os.getenv("PRIVATE_KEY_ID"),
        "private_key": os.getenv("PRIVATE_KEY"),
        "client_email": os.getenv("CLIENT_EMAIL"),
        "client_id": os.getenv("CLIENT_ID"),
        "auth_uri": os.getenv("AUTH_URI"),
        "token_uri": os.getenv("TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL")
    }
    return variables_keys


if not firebase_admin._apps:
    cred_object = firebase_admin.credentials.Certificate(
        create_keyfile_dict())
    default_app = firebase_admin.initialize_app(cred_object)
    
db = firestore.client()

@st.cache(ttl=600)
def update_data():
    docs = db.collection("Perfume").get()
    for doc in docs:
        temp = doc.to_dict()
        st.session_state.chart["Mood"].append(temp["Mood"])
        st.session_state.chart["Perfume"].append(temp["Perfume"])

if "chart" not in st.session_state:
    st.session_state.chart={"Mood":[],"Perfume":[]}
    print("running")
    # update_data()
    # print (st.session_state.chart)


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
            if submitted and file:
                # st.session_state.data=file
                st.session_state.gsr=gsr_column
                st.session_state.ppg=ppg_column
                # body.empty()
                # st.session_state.block=1

                table=pd.read_csv(file)
                
                # try:
            elif submitted and not file:
                st.error("File not found")

    if submitted and file:

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

            fig, ax = plt.subplots()
            ax.plot(gsr)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("microSiemens (μS)")

            st.pyplot(fig)
            gsr_img = io.BytesIO()
            plt.savefig(gsr_img, format='png')
            
            btn = ste.download_button(
            label="Download image",
            data=gsr_img,
            file_name="gsr_graph.png",
            mime="image/png"
            )
            # st.line_chart(gsr)
            # st.line_chart(gsr_df, x ="time (s)",y="microSiemens (μS)")
            st.markdown("## the HR graph is:")
            fig, ax = plt.subplots()
            ax.plot(hr)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("heart rate (bpm)")

            st.pyplot(fig)

            hr_img = io.BytesIO()
            plt.savefig(hr_img, format='png')
            
            btn = ste.download_button(
            label="Download image",
            data=hr_img,
            file_name="hr_graph.png",
            mime="image/png"
            )

            # st.line_chart(hr)  
            # st.line_chart(hr_df, x="time (s)",y="heart rate (bpm)")  

            docs = db.collection("Perfume").get()
            for doc in docs:
                temp = doc.to_dict()
                st.session_state.chart["Mood"].append(temp["Mood"])
                st.session_state.chart["Perfume"].append(temp["Perfume"])

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

                
                mood,mood_simp=class_predict(np.argmax(A_predict),np.argmax(V_predict))
                
                progress.progress(100)
                progress.empty()
                # st.success('Done!')
            
            def mood_img(mood):
                if mood=="Happy":
                    st.image(happy)
                if mood=="Sad":
                    st.image(sad)
                if mood=="Stressed":
                    st.image(stressed)
                if mood=="Calm":
                    st.image(calm)
                

            st.markdown(f"# the predicted mood is {mood}")
            mood_img(mood_simp)
            st.markdown(f"# for perfume : {perfume}")
            
            # print(mood)
            # print(perfume)
            new_data = {"Mood":mood_simp,"Perfume":perfume}
            db.collection("Perfume").add(new_data)
            st.session_state.chart["Mood"].append(mood_simp)
            st.session_state.chart["Perfume"].append(perfume)
            
            graph=pd.DataFrame(st.session_state.chart)
            table=graph[graph["Perfume"]==perfume].copy()
            table= table.reset_index(drop=True)
            graph=graph[graph["Perfume"]==perfume]["Mood"]

            st.session_state.chart={"Mood":[],"Perfume":[]}
            
            st.markdown(f"# Mood distribution for Perfume: {perfume}")
            fig, ax = plt.subplots()
            ax.set_xlabel("Mood")
            ax.set_ylabel("Number of people")
            ax.hist(graph)

            st.pyplot(fig)

            mood_img = io.BytesIO()
            plt.savefig(mood_img, format='png')
            
            btn = ste.download_button(
            label="Download image",
            data=mood_img,
            file_name=f"mood_dist_{perfume}.png",
            mime="image/png"
            )

            st.dataframe(table.tail())

            mood_csv=io.BytesIO()
            
            table.to_csv(mood_csv,index=False)
            btn = ste.download_button(
            label="Download whole table",
            data=mood_csv,
            file_name=f"mood_dist_{perfume}_table.csv",
            mime="text/csv"
            
            )

            


                # except:
                #     st.error("Something went wrong")

            
                
                



                

def process_file(file):
    body= st.empty()
    with body.container():
        st.title(f"""Please wait.. your file is being processed""")



with st.container():
    happy=Image.open("./img/happy.png")
    sad=Image.open("./img/sad.png")
    calm=Image.open("./img/calm.png")
    stressed=Image.open("./img/stressed.png")
    upload_file()

