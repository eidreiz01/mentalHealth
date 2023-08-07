import numpy as np
import pandas as pd
import os
import pickle as pkl
from sklearn.cluster import KMeans
import streamlit as st

def Get_Pkl(pkl_path):
    '''Function to load a pickle file'''
    with open(pkl_path, 'rb') as f:
        scaler = pkl.load(f)
    return scaler

def Get_Label_Maps(map_path):
    '''Function to load a numpy array containing a dictionary'''
    return np.load(map_path, allow_pickle=True).item()

def Load_All_Needed_Pickles():
    '''Function to load all the pickles needed for inference'''
    cost_of_study = Get_Pkl('./Scalers/cost_of_study_Scaler.pkl')
    hours_per_week_university_work = Get_Pkl('./Scalers/hours_per_week_university_work_Scaler.pkl')
    total_device_hours = Get_Pkl('./Scalers/total_device_hours_Scaler.pkl')
    total_social_media_hours = Get_Pkl('./Scalers/total_social_media_hours_Scaler.pkl')
    exercise_per_week = Get_Pkl('./Scalers/exercise_per_week_Scaler.pkl')
    work_hours_per_week = Get_Pkl('./Scalers/work_hours_per_week_Scaler.pkl')
    Dict_Scalers = {
        'cost_of_study': cost_of_study,
        'hours_per_week_university_work': hours_per_week_university_work,
        'total_device_hours': total_device_hours,
        'total_social_media_hours': total_social_media_hours,
        'exercise_per_week': exercise_per_week,
        'work_hours_per_week': work_hours_per_week
    }

    known_disabilities = Get_Pkl('./Label Encoders/known_disabilities_LabelEncoder.pkl')
    stress_in_general = Get_Pkl('./Label Encoders/stress_in_general_LabelEncoder.pkl')
    well_hydrated = Get_Pkl('./Label Encoders/well_hydrated_LabelEncoder.pkl')
    ethnic_group = Get_Pkl('./Label Encoders/ethnic_group_LabelEncoder.pkl')
    home_country = Get_Pkl('./Label Encoders/home_country_LabelEncoder.pkl')
    course_of_study = Get_Pkl('./Label Encoders/course_of_study_LabelEncoder.pkl')
    personality_type = Get_Pkl('./Label Encoders/personality_type_LabelEncoder.pkl')
    institution_country = Get_Pkl('./Label Encoders/institution_country_LabelEncoder.pkl')
    year_of_birth = Get_Pkl('./Label Encoders/year_of_birth_LabelEncoder.pkl')
    student_type_location = Get_Pkl('./Label Encoders/student_type_location_LabelEncoder.pkl')
    Dict_LabelEncoders = {
        'known_disabilities': known_disabilities,
        'stress_in_general': stress_in_general,
        'well_hydrated' : well_hydrated,
        'ethnic_group' : ethnic_group,
        'home_country' : home_country,
        'course_of_study' : course_of_study,
        'personality_type' : personality_type,
        'institution_country' : institution_country,
        'year_of_birth' : year_of_birth,
        'student_type_location' : student_type_location
    }

    alcohol_consumption = Get_Label_Maps('./Label Maps/alcohol_consumption.npy')
    diet = Get_Label_Maps('./Label Maps/diet.npy')
    quality_of_life = Get_Label_Maps('./Label Maps/quality_of_life.npy')
    year_of_study = Get_Label_Maps('./Label Maps/year_of_study.npy')
    Dict_LabelMaps = {
        'alcohol_consumption': alcohol_consumption,
        'diet': diet,
        'quality_of_life': quality_of_life,
        'year_of_study': year_of_study
    }

    Model = Get_Pkl('./Models/SMOTE_BernoulliNB()_Unoptimized.pkl')

    mental_health_issues_map = Get_Label_Maps('./Label Maps/mental_health_issues_map.npy')

    return Dict_Scalers, Dict_LabelEncoders, Dict_LabelMaps, Model

# # Define the input form
# st.title(Mental Health GUI')
# st.subheader('Select an option')

def Preprocess_Inference_Dataset(Inference_DF,Dict_Scalers,Dict_LabelEncoders,Dict_LabelMaps):
    '''Function to preprocess the inference dataset'''
    for col in Inference_DF.select_dtypes(include=['int']).columns:
        if col != 'Response ID':
            Inference_DF[col] = Dict_Scalers[col].transform(Inference_DF[col].values.reshape(-1,1))

    for col in Inference_DF.select_dtypes(include=['object']).columns:
        if col not in ['alcohol_consumption','diet','quality_of_life','year_of_study']:
            for val in Inference_DF[col].unique():
                if val not in Dict_LabelEncoders[col].classes_:
                    Dict_LabelEncoders[col].classes_ = np.append(Dict_LabelEncoders[col].classes_, val)
            Inference_DF[col] = Dict_LabelEncoders[col].transform(Inference_DF[col])

    for col in Inference_DF.select_dtypes(include=['object']).columns:
        if col in ['alcohol_consumption','diet','quality_of_life','year_of_study']:
            Label_Map = Dict_LabelMaps[col]
            for val in Inference_DF[col].unique():
                if val not in Label_Map.keys():
                    Label_Map[val] = len(Label_Map)
            Dict_LabelMaps[col] = Label_Map
            Inference_DF[col] = Inference_DF[col].map(Label_Map)
    return Inference_DF

def Get_Kmeans_Prediction(XTest):
    '''Function to get the Kmeans prediction'''
    KMeans_Model = KMeans(n_clusters=2, random_state=0).fit(XTest)
    Kmeans_Pred = pd.Series(KMeans_Model.predict(XTest)).map({1:'Yes',0:'No'})
    return Kmeans_Pred

def Get_Model_Prediction(XTest,Model):
    '''Function to get the Model prediction'''
    Model_Pred = pd.Series(Model.predict(XTest)).map({1:'Yes',0:'No'})
    return Model_Pred

def Format_Results(Xtest,Kmeans_Pred,Model_Pred,df):
    '''Function to format the results'''
    if Kmeans_Pred != None: #if Kmeans_Pred != None i.e. if the user has uploaded a file
        Xtest['Response ID'] = df['Response ID']
        Xtest['Kmeans Predicted'] = Kmeans_Pred
        Xtest['Model Predicted'] = Model_Pred

        Results = Xtest.iloc[:,[-3,-2,-1]] 
        Xtest = Xtest.drop(Xtest.columns[[-1,-2,-3]], axis=1)
        Xtest = pd.concat([Results,Xtest], axis=1)
        return Xtest
    else: #if Kmeans_Pred == None i.e. if the user has inputted a record if
        Xtest['Response ID'] = df['Response ID']
        Xtest['Model Predicted'] = Model_Pred.values

        Results = Xtest.iloc[:,[-2,-1]] 
        Xtest = Xtest.drop(Xtest.columns[[-1,-2]], axis=1)
        Xtest = pd.concat([Results,Xtest], axis=1)
        return Xtest

def FileUploaded(file):
    '''Function to handle the file uploaded by the user'''
    df = None
    if 'csv' in file.name :
        df = pd.read_csv(file)
    elif 'xlsx' in file.name:
        df = pd.read_excel(file)
    
    Dict_Scalers, Dict_LabelEncoders, Dict_LabelMaps, Model = Load_All_Needed_Pickles()
    df = Preprocess_Inference_Dataset(df,Dict_Scalers,Dict_LabelEncoders,Dict_LabelMaps)

    Xtest = df.drop('Response ID',axis=1)
    Kmeans_Pred = Get_Kmeans_Prediction(Xtest)
    Model_Pred = Get_Model_Prediction(Xtest,Model)
    Results = Format_Results(Xtest,Kmeans_Pred,Model_Pred,df)
    st.write(Results)

def RecordInputted(record_id):
    '''Function to handle the record id inputted by the user'''
    df = pd.read_csv('Inference.csv')
    df = df[df['Response ID'] == record_id]

    Dict_Scalers, Dict_LabelEncoders, Dict_LabelMaps, Model = Load_All_Needed_Pickles()
    df = Preprocess_Inference_Dataset(df,Dict_Scalers,Dict_LabelEncoders,Dict_LabelMaps)
    if len(df) == 0: #if the record id is not found in the inference dataset
        st.write("Record ID not found. Hence Test Data is not available.")
        return

    Xtest = df.drop('Response ID',axis=1)

    Kmeans_Pred = None
    if len(df) >= 2:
        Kmeans_Pred = Get_Kmeans_Prediction(Xtest)
    Model_Pred = Get_Model_Prediction(Xtest,Model)
    Results = Format_Results(Xtest,Kmeans_Pred,Model_Pred,df)
    st.write(Results)

def main():
    '''Main function'''
    # Define the input form
    st.title(Mental Health GUI')
    st.subheader('Select an option')
    upload_file = st.radio("How would you like to input your data?", ["Upload File", "Enter Record ID"])
    if upload_file == "Upload File":
        file = st.file_uploader("Upload your file")
        if file is not None:
            FileUploaded(file)
    else:
        record_id = -1
        record_id = st.number_input("Enter Record ID",value=-1)
        if record_id != -1:
            RecordInputted(record_id)

if __name__ == "__main__":
    main()
