import boto3 
import json  
from io import StringIO 
import pandas as pd    
import os 
import pickle  
from scipy import stats 



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
    gdp['per_cap_percentile'] = stats.rankdata(gdp['gdp_per_cap_20'], 'average') / gdp.shape[0]
    
    metro = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/ns/metro.csv')['Body'].read().decode('utf-8')   
    metro = pd.read_csv(StringIO(metro))  
    
    local_rent = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/other/rent_prices.csv')['Body'].read().decode('utf-8')   
    local_rent = pd.read_csv(StringIO(local_rent))  
    
    rev_shap = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/predictions/shapley/revenue/rev_shapley.csv')['Body'].read().decode('utf-8')   
    rev_shap = pd.read_csv(StringIO(rev_shap)) 
    
    bd_shap = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/predictions/shapley/bad_debt/bad_debt_shapley.csv')['Body'].read().decode('utf-8')   
    bd_shap = pd.read_csv(StringIO(bd_shap)) 
    
    rev_test = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/predictions/shapley/revenue/rev_training.csv')['Body'].read().decode('utf-8')   
    rev_test = pd.read_csv(StringIO(rev_test)) 
    
    bd_test = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/predictions/shapley/bad_debt/bad_debt_training.csv')['Body'].read().decode('utf-8')   
    bd_test = pd.read_csv(StringIO(bd_test)) 
    
    tax = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/other/tax_info.csv')['Body'].read().decode('utf-8')   
    tax = pd.read_csv(StringIO(tax)) 
    
    realtor = json.loads(s3.get_object(Bucket='acq-data-warehouse', 
                  Key=f'data_lake/other/realtor_demand.json')['Body'].read().decode('utf-8')) 
    
    realtor = pd.DataFrame(realtor).set_index('postal_code') 
    
    rev_coeffs = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/predictions/revenue_model_coefficients.csv')['Body'].read().decode('utf-8')   
    rev_coeffs = pd.read_csv(StringIO(rev_coeffs)).drop('Unnamed: 0', axis=1) 
    
    bd_coeffs = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/predictions/bad_debt_model_coefficients.csv')['Body'].read().decode('utf-8')   
    bd_coeffs = pd.read_csv(StringIO(bd_coeffs)).drop('Unnamed: 0', axis=1) 
    
    drivers = s3.get_object(Bucket='acq-data-warehouse', Key=f'data_lake/other/demographic_drivers_by_fips.csv')['Body'].read().decode('utf-8')   
    drivers = pd.read_csv(StringIO(drivers)) 
    
    
    # comp_dict = pickle.loads(s3.get_object(Bucket='acq-data-warehouse', Key='data_lake/other/comps_comps.pkl')['Body'].read()) 
    
    return {
        'general': general,
        'geo': geo, 
        'clusters': clusters, 
        'preds': preds, 
        'gdp': gdp, 
        'metro': metro,  
        'local_rent': local_rent,  
        'rev_shap': rev_shap,  
        'bd_shap': bd_shap,  
        'rev_test': rev_test, 
        'bd_test': bd_test,  
        'tax': tax,  
        'realtor': realtor, 
        'rev_coeffs': rev_coeffs, 
        'bd_coeffs': bd_coeffs,  
        'drivers': drivers, 
        'comp_full': None
    }
    
    
