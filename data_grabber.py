import boto3 
import json  
from io import StringIO 
import pandas as pd    
import os 
import pickle 


def grab(MASTER_ACCESS_KEY, MASTER_SECRET):  
    
    print('data grabber was run')

    # --- s3 client --- 
    s3 = boto3.client('s3', region_name = 'us-west-1', 
          aws_access_key_id=MASTER_ACCESS_KEY, 
          aws_secret_access_key=MASTER_SECRET)  
    
    general = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/supply_general.csv')['Body'].read().decode('utf-8') 
    general = pd.read_csv(StringIO(general))  
    
    preds = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/predictions/all_predictions.csv')['Body'].read().decode('utf-8') 
    preds = pd.read_csv(StringIO(preds)) 
    
    geo = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/supply_geo.csv')['Body'].read().decode('utf-8')  
    geo = pd.read_csv(StringIO(geo))   
    
    clusters = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/clusters/clusters.csv')['Body'].read().decode('utf-8')   
    clusters = pd.read_csv(StringIO(clusters))   
    # clusters['cluster'] = clusters['cluster'].astype(str)  
    
    gdp = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/other/gdp_counties.csv')['Body'].read().decode('utf-8')   
    gdp = pd.read_csv(StringIO(gdp))   
    
    metro = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/ns/metro.csv')['Body'].read().decode('utf-8')   
    metro = pd.read_csv(StringIO(metro)) 
    
    # comp_dict = pickle.loads(s3.get_object(Bucket='acq-data-warehouse', Key='data_lake/other/comps_comps.pkl')['Body'].read()) 
    
    return {
        'general': general,
        'geo': geo, 
        'clusters': clusters, 
        'preds': preds, 
        'gdp': gdp, 
        'metro': metro, 
        'comp_full': None
    }
    
    
