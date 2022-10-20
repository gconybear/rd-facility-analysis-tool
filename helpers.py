import streamlit as st  
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 

def predict(m, new): 
    """
    m: statsmodels model obj 
    new: np array [1, x] --> x = ttm revenue 
    """ 
    
    return tuple(m.get_prediction(new).conf_int(alpha=.05)[0]) 

def predict_interval(m, new, alpha=.1, ci=False):  
    
    if ci: 
        return tuple(m.get_prediction(new).summary_frame(alpha=alpha)[['mean_ci_lower', 'mean_ci_upper']].values[0]) 
    
    return tuple(m.get_prediction(new).summary_frame(alpha=alpha)[['obs_ci_lower', 'obs_ci_upper']].values[0])

def create_shapley_table(coeff_df, shap_row, test_row, col_dict, model='rev'): 
    """
    creates pandas html formatted / styled table 
    
    params 
    ---- 
    
    shap_row : indexed row from shapley val df 
    test_row : indexed row from training dataset 
    col_dict : col name dict key:{'nice':val} 
    model : either 'rev' or 'bd'
    """ 
    
    cdf = coeff_df.copy() 
    
    shap_row_dict = dict(shap_row.reset_index().values) 
    test_row_dict = dict(test_row.T.reset_index().values)
    
    cdf['shapley_value'] = (cdf['variable'].apply(lambda x: shap_row_dict.get(f"{x}_{model}_shap"))) * 100 
    print(cdf['variable'])
    cdf['prediction_effect'] = cdf['variable'].apply(lambda x: 'down' if 
                                                               shap_row_dict.get(f"{x}_{model}_shap") < 0 else 'up')
    cdf['value'] = cdf['variable'].apply(lambda x: test_row_dict.get(x))
    cdf['z-score'] = cdf['variable'].apply(lambda x: test_row_dict.get(f"{x}_zscore")) # f"{x}_zscore" 
    cdf['percentile'] = (cdf['variable'].apply(lambda x: test_row_dict.get(f"{x}_percentile")) * 100 ).round()
    cdf['variable'] = cdf['variable'].apply(lambda x: col_dict[f"{x}_{model}_shap"]['nice'])

    cdf = cdf.sort_values('shapley_value', key=lambda x: abs(x), ascending=False).reset_index(drop=True) 
    cdf = cdf[['variable', 'value', 'z-score', 'percentile', 'prediction_effect', 'shapley_value', 'coefficient']]
    
    POS_COLOR = '#86f564' 
    NEG_COLOR = '#f52c54' 
    
    if model == 'bd': 
        POS_COLOR = '#f52c54'
        NEG_COLOR = '#86f564'

    def color_recommend(value):
        if value == 'down':
            color = NEG_COLOR
        elif value == 'up':
            color = POS_COLOR
        else:
            return
        return f'background-color: {color}'  

    def shap_color_highlight(value):
        if value < 0:
            color = NEG_COLOR
        elif value > 0:
            color = POS_COLOR
        else:
            color = 'black'
        return 'color: %s' % color



    table = (
        cdf 
        .style
        .format(
            formatter={  
                    'coefficient': '{:.5f}',
                    'shapley_value': '{:.2f}',
                    'value': '{:.2f}', 
                    'z-score': '{:.2f}',  
                    'percentile': '{:.0f}'
                      }
        )
        .applymap(shap_color_highlight, subset=['shapley_value'])
        .applymap(color_recommend, subset=['prediction_effect']) 

    ) 

    return table


def plot_demand(z, demand):  
    try:
        row = demand.loc[z, :]   

        city, state = tuple(row['zip_name'].split(', ')) 

        date_list = row['datetime'][::-1] 

        # want to show last obs in plot
        max_idx = len(date_list)

        fig, ax = plt.subplots(figsize=(9,4)) 
        sns.lineplot(date_list, row['demand_score'], ls='-', label='demand', ax=ax)  
        sns.lineplot(date_list, row['supply_score'], ls=':', label='supply', ax=ax) 
        ax.set_title(f"Housing Demand in {city.title()}, {state.upper()} – {z}") 


        ax.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.3)

        for i,tick in enumerate(ax.xaxis.get_ticklabels()): 
            if i == (max_idx-1): 
                tick.set_visible(True)
            elif i % 11 == 0:   
                if (max_idx - i) > 7:
                    tick.set_visible(True) 
                else: 
                    tick.set_visible(False) 
            else: 
                tick.set_visible(False) 

        return fig 
    except: 
        return None

tax_display_cols = ['TaxAmount', 'YoYChangeinTaxAmount',
       'ValueHistoryYear', 'AssessedLandValue', 'AssessedImprovementValue',
       'TotalAssessedValue', 'AppliedTaxRate', 'LandMarketValue',
       'ImprovementsMarketValue', 'TotalMarketValue']

def create_cols(row): 
    bad = -1
    cols = []
    for x in row: 
        if pd.isnull(x): 
            col = f"t{bad}" 
            bad -= 1 
        else: 
            col = str(int(x)) 
        
        cols.append(col)
    return cols

def get_tax_row(sid, tax):  
    
    row = tax[tax['StoreID'] == sid][tax_display_cols].T
    
    bad_cols = 1
    # if not pd.isnull(x) else x 
    #new_cols = [str(int(x)) for x in row.loc['ValueHistoryYear'].values]  
    new_cols = create_cols(row.loc['ValueHistoryYear'].values)
    row.columns = new_cols 
    
    return row.drop('ValueHistoryYear')

def compare_feature(f, row): 
    
    val = row[f].values[0] 
    z = row[f"{f}_zscore"].values[0] 
    p = row[f"{f}_percentile"].values[0]   
    
    p = round(p * 100)
    
#    return st.markdown(f"$\Rightarrow$ {col_dict[f]['nice]} has a value of **{round(val, 2)}**, a z-score of **{round(z,2)}** and in the **{p}{suffix(p)}** percentile") 
    
    return round(val, 2), round(z, 2), p

def suffix(d):
    return 'th' if 11<=d<=13 else {1:'st',2:'nd',3:'rd'}.get(d%10, 'th')

def parse_col_name(c): 
    dt = c.split('.')[-1][-6:] 
    
    return f"{dt[:4]} Q{dt[-1]}"

MHV_COLS = ['metro.real_estate.mhv_obs_2000_1',
 'metro.real_estate.mhv_obs_2000_2',
 'metro.real_estate.mhv_obs_2000_3',
 'metro.real_estate.mhv_obs_2000_4',
 'metro.real_estate.mhv_obs_2001_1',
 'metro.real_estate.mhv_obs_2001_2',
 'metro.real_estate.mhv_obs_2001_3',
 'metro.real_estate.mhv_obs_2001_4',
 'metro.real_estate.mhv_obs_2002_1',
 'metro.real_estate.mhv_obs_2002_2',
 'metro.real_estate.mhv_obs_2002_3',
 'metro.real_estate.mhv_obs_2002_4',
 'metro.real_estate.mhv_obs_2003_1',
 'metro.real_estate.mhv_obs_2003_2',
 'metro.real_estate.mhv_obs_2003_3',
 'metro.real_estate.mhv_obs_2003_4',
 'metro.real_estate.mhv_obs_2004_1',
 'metro.real_estate.mhv_obs_2004_2',
 'metro.real_estate.mhv_obs_2004_3',
 'metro.real_estate.mhv_obs_2004_4',
 'metro.real_estate.mhv_obs_2005_1',
 'metro.real_estate.mhv_obs_2005_2',
 'metro.real_estate.mhv_obs_2005_3',
 'metro.real_estate.mhv_obs_2005_4',
 'metro.real_estate.mhv_obs_2006_1',
 'metro.real_estate.mhv_obs_2006_2',
 'metro.real_estate.mhv_obs_2006_3',
 'metro.real_estate.mhv_obs_2006_4',
 'metro.real_estate.mhv_obs_2007_1',
 'metro.real_estate.mhv_obs_2007_2',
 'metro.real_estate.mhv_obs_2007_3',
 'metro.real_estate.mhv_obs_2007_4',
 'metro.real_estate.mhv_obs_2008_1',
 'metro.real_estate.mhv_obs_2008_2',
 'metro.real_estate.mhv_obs_2008_3',
 'metro.real_estate.mhv_obs_2008_4',
 'metro.real_estate.mhv_obs_2009_1',
 'metro.real_estate.mhv_obs_2009_2',
 'metro.real_estate.mhv_obs_2009_3',
 'metro.real_estate.mhv_obs_2009_4',
 'metro.real_estate.mhv_obs_2010_1',
 'metro.real_estate.mhv_obs_2010_2',
 'metro.real_estate.mhv_obs_2010_3',
 'metro.real_estate.mhv_obs_2010_4',
 'metro.real_estate.mhv_obs_2011_1',
 'metro.real_estate.mhv_obs_2011_2',
 'metro.real_estate.mhv_obs_2011_3',
 'metro.real_estate.mhv_obs_2011_4',
 'metro.real_estate.mhv_obs_2012_1',
 'metro.real_estate.mhv_obs_2012_2',
 'metro.real_estate.mhv_obs_2012_3',
 'metro.real_estate.mhv_obs_2012_4',
 'metro.real_estate.mhv_obs_2013_1',
 'metro.real_estate.mhv_obs_2013_2',
 'metro.real_estate.mhv_obs_2013_3',
 'metro.real_estate.mhv_obs_2013_4',
 'metro.real_estate.mhv_obs_2014_1',
 'metro.real_estate.mhv_obs_2014_2',
 'metro.real_estate.mhv_obs_2014_3',
 'metro.real_estate.mhv_obs_2014_4',
 'metro.real_estate.mhv_obs_2015_1',
 'metro.real_estate.mhv_obs_2015_2',
 'metro.real_estate.mhv_obs_2015_3',
 'metro.real_estate.mhv_obs_2015_4',
 'metro.real_estate.mhv_obs_2016_1',
 'metro.real_estate.mhv_obs_2016_2',
 'metro.real_estate.mhv_obs_2016_3',
 'metro.real_estate.mhv_obs_2016_4',
 'metro.real_estate.mhv_obs_2017_1',
 'metro.real_estate.mhv_obs_2017_2',
 'metro.real_estate.mhv_obs_2017_3',
 'metro.real_estate.mhv_obs_2017_4',
 'metro.real_estate.mhv_obs_2018_1',
 'metro.real_estate.mhv_obs_2018_2',
 'metro.real_estate.mhv_obs_2018_3',
 'metro.real_estate.mhv_obs_2018_4',
 'metro.real_estate.mhv_obs_2019_1',
 'metro.real_estate.mhv_obs_2019_2',
 'metro.real_estate.mhv_obs_2019_3',
 'metro.real_estate.mhv_obs_2019_4',
 'metro.real_estate.mhv_obs_2020_1',
 'metro.real_estate.mhv_obs_2020_2',
 'metro.real_estate.mhv_obs_2020_3',
 'metro.real_estate.mhv_obs_2020_4',
 'metro.real_estate.mhv_obs_2021_1',
 'metro.real_estate.mhv_obs_2021_2']

rev_model_col_dict = {'store': {'nice': 'store', 'db': 'store'},
 'full_fips': {'nice': 'full_fips', 'db': 'full_fips'},
 'neighborhood.real_estate.uscls_rent_rev_shap': {'nice': 'rent national class',
  'db': 'neighborhood.real_estate.uscls_rent'},
 'neighborhood.real_estate.grrent_ch_rev_shap': {'nice': 'change in rent price',
  'db': 'neighborhood.real_estate.grrent_ch'},
 'neighborhood.real_estate.his_pct_rev_shap': {'nice': 'historic homes pct',
  'db': 'neighborhood.real_estate.his_pct'},
 'neighborhood.crime.count_violent_rev_shap': {'nice': 'count of violent crimes',
  'db': 'neighborhood.crime.count_violent'},
 'neighborhood.crime.prop_frisk_ratio_rev_shap': {'nice': 'property crime ratio - present vs projected',
  'db': 'neighborhood.crime.prop_frisk_ratio'},
 'neighborhood.demographics.retire_fr_rev_shap': {'nice': 'retirement friendly score',
  'db': 'neighborhood.demographics.retire_fr'},
 'neighborhood.econ_and_empl.ind_manufacturing_pct_rev_shap': {'nice': 'manufacturing industry pct',
  'db': 'neighborhood.econ_and_empl.ind_manufacturing_pct'},
 'neighborhood.econ_and_empl.ind_transportation_pct_rev_shap': {'nice': 'transportation industry pct',
  'db': 'neighborhood.econ_and_empl.ind_transportation_pct'},
 'neighborhood.econ_and_empl.ind_realestate_pct_rev_shap': {'nice': 'real estate industry pct',
  'db': 'neighborhood.econ_and_empl.ind_realestate_pct'},
 'neighborhood.econ_and_empl.commtime60_pct_rev_shap': {'nice': '45-60 min commute time pct',
  'db': 'neighborhood.econ_and_empl.commtime60_pct'},
 'neighborhood.econ_and_empl.carpool_pct_rev_shap': {'nice': 'carpool pct',
  'db': 'neighborhood.econ_and_empl.carpool_pct'},
 'block_group.demographics.pop_est_1mile_rev_shap': {'nice': '1 mile population',
  'db': 'block_group.demographics.pop_est_1mile'},
 'block_group.demographics.pop_ch_25mile_rev_shap': {'nice': '25 mile population change',
  'db': 'block_group.demographics.pop_ch_25mile'}} 

bd_model_col_dict = {'store': {'nice': 'store', 'db': 'store'},
 'full_fips': {'nice': 'full_fips', 'db': 'full_fips'},
 'neighborhood.real_estate.grrent_ch_bd_shap': {'nice': 'change in rent price',
  'db': 'neighborhood.real_estate.grrent_ch'},
 'neighborhood.real_estate.own_ch_bd_shap': {'nice': 'change in homeownership rate',
  'db': 'neighborhood.real_estate.own_ch'},
 'neighborhood.real_estate.new_pct_bd_shap': {'nice': 'new homes pct',
  'db': 'neighborhood.real_estate.new_pct'},
 'neighborhood.real_estate.old_pct_bd_shap': {'nice': 'old homes pct',
  'db': 'neighborhood.real_estate.old_pct'},
 'neighborhood.real_estate.his_pct_bd_shap': {'nice': 'historic homes pct',
  'db': 'neighborhood.real_estate.his_pct'},
 'neighborhood.real_estate.att_pct_bd_shap': {'nice': 'row houses pct',
  'db': 'neighborhood.real_estate.att_pct'},
 'neighborhood.demographics.retire_fr_bd_shap': {'nice': 'retirement friendly score',
  'db': 'neighborhood.demographics.retire_fr'},
 'neighborhood.econ_and_empl.ind_wholesale_pct_bd_shap': {'nice': 'wholesale industry pct',
  'db': 'neighborhood.econ_and_empl.ind_wholesale_pct'},
 'neighborhood.econ_and_empl.ind_finance_pct_bd_shap': {'nice': 'finance industry pct',
  'db': 'neighborhood.econ_and_empl.ind_finance_pct'},
 'neighborhood.econ_and_empl.commtime90_pct_bd_shap': {'nice': '90+ min commute pct',
  'db': 'neighborhood.econ_and_empl.commtime90_pct'},
 'block_group.demographics.pop_past_50mile_bd_shap': {'nice': 'population past 50 mile',
  'db': 'block_group.demographics.pop_past_50mile'},
 'block_group.demographics.pop_ch_10mile_bd_shap': {'nice': '10 mile population change',
  'db': 'block_group.demographics.pop_ch_10mile'},
 'block_group.econ_and_empl.jobs_hipay_5min_bd_shap': {'nice': 'high paying jobs < 5 min',
  'db': 'block_group.econ_and_empl.jobs_hipay_5min'},
 'block_group.real_estate.app_2y_bd_shap': {'nice': 'hpa last 2 years',
  'db': 'block_group.real_estate.app_2y'},
 'block_group.real_estate.value_per_sqft_near_bd_shap': {'nice': 'home value per sq. ft nearby',
  'db': 'block_group.real_estate.value_per_sqft_near'},
 'num_multi_op_bd_shap': {'nice': 'number of multi-operators nearby',
  'db': 'num_multi_op'},
 'nrsf_median_bd_shap': {'nice': 'avg. nrsf nearby', 'db': 'nrsf_median'}}