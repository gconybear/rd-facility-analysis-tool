import streamlit as st 
import pandas as pd 
import numpy as np   
import folium
from folium import Marker
from folium.plugins import MarkerCluster 
import matplotlib.pyplot as plt 
import pickle 

import data_grabber  
import geomapping
from geocode import extract_lat_long_via_address 
import helpers 

st.set_page_config(layout='centered', page_icon='📈', page_title="Facility Analysis") 


def blank(): return st.text('')  

def comma_print(value, integer=False): 
    try:
        if integer: 
            return '{:,}'.format(int(value))     
        return '{:,}'.format(value)  
    except: 
        return None

MASTER_ACCESS_KEY = st.secrets['MASTER_ACCESS_KEY']
MASTER_SECRET = st.secrets['MASTER_SECRET']

@st.cache_data
def convert_df(df):
    # IMPORTANT: cache_data the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.cache_data()
def grab_data():  
    # print('function run') 
    reg = pickle.load(open('cp-models/cp_ttm_rev_simple.pkl', 'rb'))
    return data_grabber.grab(MASTER_ACCESS_KEY, MASTER_SECRET), reg

dat, reg = grab_data() 
gen = dat['general'] 
geo = dat['geo']
clusters = dat['clusters']  
preds = dat['preds'] 
gdp = dat['gdp']
metro = dat['metro'] 
local_rent = dat['local_rent'] 
rev_shap = dat['rev_shap'] 
bd_shap = dat['bd_shap'] 
rev_test = dat['rev_test'] 
bd_test = dat['bd_test'] 
tax = dat['tax'] 
realtor = dat['realtor'] 
rev_coeffs = dat['rev_coeffs'] 
bd_coeffs = dat['bd_coeffs'] 
drivers = dat['drivers']
#comp_full = dat['comp_full']

everthing_working = True 

st.header("Facility Analysis") 
blank()
with st.form(key='form'): 
    
    
    qtype = st.radio("Location type", ('address', 
                                       'coordinates'), help="coordinates should be in lat, long format") 
    
    query = st.text_input("Location") 
    
    with st.expander("Climate Control Modifer"): 
        
        st.caption("*Use the options below to add climate control revenue adjustment*") 
        cc_known = st.radio("Facility Climate Control Percentage", ['Known', 'Unknown'], index=1)
        cc_pct = st.number_input('Climate Control Percentage', min_value=0., max_value=100., step=5.)  
        
    with st.expander("Contract Price Estimation"): 
        
        st.caption("*Input TTM Revenue to get an estimate of contract price*")  
        help_str = """
        checking this box will produce a **prediction interval** for a new inputted TMM Revenue 
        
        for more info on prediction intervals vs confidence intervals, see this: https://www.statology.org/confidence-interval-vs-prediction-interval/
        """
        predict_cp = st.radio('Predict Contract Price', ['Yes', 'No'], index=1, help=help_str) 
        c1, c2 = st.columns(2)
        ttm_rev = c1.number_input('TTM Revenue ($)', min_value=0, max_value=None, step=10000)  
        alpha_level = c2.number_input('Confidence Level (%)', 
                                      min_value=0., 
                                      max_value=100., 
                                      step=5., 
                                      value=90., help="corresponds to the alpha level used for computing the prediction interval (e.g., 90% confidence level is a 0.1 alpha level)")
        
  
    
    blank()
    search_button = st.form_submit_button('search') 
    
    
if search_button:  
    
    # check to see if user asked for cp model 
    est_contract_price = predict_cp == 'Yes' 
    if est_contract_price:  
        a = (100 - alpha_level) / 100 
        low_cp, high_cp = helpers.predict_interval(reg, np.array([1., ttm_rev]), alpha=a, ci=True)  
        
        mssg = f"""
        
        ***Contract Price Estimation***
        
        ------ 
        
        $\Rightarrow$ **{round(alpha_level)}%** confidence interval: **\${int(low_cp):,}** to **\${int(high_cp):,}** 
         
        """  
        # $\Rightarrow$ there is a **{round(alpha_level)}%** probability that true contract price falls in the above range
        st.info(mssg)
        #st.info(f'contract price interval: **\${int(low_cp):,}** to **\${int(high_cp):,}**')
    
    
    if qtype == 'address': 
        coords = extract_lat_long_via_address(query)  
        if coords is None: 
            st.error("Unable to geocode address – **please input lat, long coordinates instead**") 
            everthing_working = False
        else:
            coords = tuple(coords.values())  

    else: 
        coords = tuple([float(x) for x in query.replace(' ','').split(',')])
    
    if everthing_working: 
        lat, long = coords  
        
        download_data = {}

        closest_store = geomapping.get_closest_store(lat, long, geo) 

        # --- index rows --- 

        # cluster
        crow = clusters[clusters['full_fips'] == closest_store['full_fips']].drop(['full_fips', 'Unnamed: 0'], axis=1).reset_index(drop=True).T.rename(columns={0:'values'}) 

        # general 
        gen_row = gen[gen['StoreID'] == closest_store.get('StoreID')]   
        
        if gen_row.empty: 
            everthing_working = False

        # predictions 
        preds_row = preds[preds['store'] == closest_store.get('StoreID')] 

        # gdp

        gdp_row = gdp[gdp['county_fips'] == closest_store.get('county_fips')]  
        
        
        # cc modifier 
        
        cc_known = cc_known == 'Known'
    
    if everthing_working: 
#        st.success(f"Store found! \n\nAddress mapped to: **{sname}** (**{closest_store['distance']}** miles away)")  

        # general vals
        sname = gen_row.loc[:, 'StoreName'].values[0]
        total_sf = gen_row.loc[:, 'TotalSqft'].values[0]
        nrsf = gen_row.loc[:, 'RentableSqft'].values[0]
        owner = gen_row.loc[:, 'OwnerCompanyName'].values[0]  
        company_type = gen_row.loc[:, 'CompanyType'].values[0]   
        
        st.success(f"Address mapped to: **{sname}** (**{closest_store['distance']}** miles away)") 
        
        download_data.update({
            'store name': sname, 
            'location': query, 
            'total sq ft': total_sf, 
            'rentable sq ft': nrsf, 
            'owner': owner, 
            'company type': company_type
        })
        
#        st.write(f"Address mapped to: **{sname}** (**{closest_store['distance']}** miles away)") 
        ##blank()
        
        # prediction vals 
        rev_pred = preds_row.loc[:, 'mean_rev_fit'].values[0] 
        bd_pred = preds_row.loc[:, 'bdebt_fit'].values[0]  
        nrsf_1_mile = preds_row.loc[:, 'nrsf_1mi_delta'].values[0]
        nrsf_3_mile = preds_row.loc[:, 'nrsf_3mi_delta'].values[0]
        nrsf_5_mile = preds_row.loc[:, 'nrsf_5mi_delta'].values[0] 
        nrsf_10_mile = preds_row.loc[:, 'nrsf_10mi_delta'].values[0] 
        
        # cc modifier 
        
        if cc_known: 
            cc_modifier = np.e**(0.003593*(cc_pct - 14.82))  
            base_rev = rev_pred 
            modified_rev = base_rev * cc_modifier
        else: 
            cc_modifier = 1
        
        download_data.update({
            'revenue_prediction': rev_pred * cc_modifier, 
            'cc_modifier': cc_modifier, 
            'bad_debt_prediction': bd_pred
        })
    

    #    comp_data = comp_full[closest_store['StoreID']].get('comps')
    #    kd_data = comp_full[closest_store['StoreID']].get('developments')

        # --- outputs ---- 

        #st.write("**Cluster Predictions**")   

        for row in crow.index: 
            if row == 'cluster': 
                crow.loc[row, :] = crow.loc[row, :].apply(lambda x: str(int(x))) 
            else: 
                crow.loc[row, :] = crow.loc[row, :].apply(lambda x: str(x)) 

        #st.dataframe(crow)   
        
        download_data.update({
            'cluster': crow.loc['cluster'].values[0], 
            'cluster 1 prob': crow.loc['cluster_1_probability'].values[0], 
            'cluster 2 prob': crow.loc['cluster_2_probability'].values[0], 
            'cluster 3 prob': crow.loc['cluster_3_probability'].values[0], 
            'cluster 4 prob': crow.loc['cluster_4_probability'].values[0], 
            'cluster 5 prob': crow.loc['cluster_5_probability'].values[0]
        })
        

        blank()  
        
        st.write("**Facility Info**")  
        
        with st.expander("General"):
        
            st.markdown(f"$\Rightarrow$ Name: **{sname}**", unsafe_allow_html=True)
            st.markdown(f"$\Rightarrow$ Owner: **{owner}**", unsafe_allow_html=True) 
            st.markdown(f"$\Rightarrow$ Company Type: **{company_type}**", unsafe_allow_html=True)
            st.markdown(f"$\Rightarrow$ Total Sq. Footage: **{comma_print(total_sf)}**", unsafe_allow_html=True)
            st.markdown(f"$\Rightarrow$ Rentable Sq. Footage: **{comma_print(nrsf)}**", unsafe_allow_html=True) 
            
        with st.expander("Taxes & Assessments"): 

            st.write('')  

            tax_table = helpers.get_tax_row(closest_store.get('StoreID'), tax)   
            print(tax_table.head())
            try:
                st.table(tax_table.style.set_precision(2)) 
            except: 
                st.write(tax_table.astype(str))  
                
            if not tax_table.empty: 
                download_data.update({
                    'last tax amount': tax_table.loc['TaxAmount'].values[0], 
                    'assessed land value': tax_table.loc['AssessedLandValue'].values[0], 
                    'assessed improvement value': tax_table.loc['AssessedImprovementValue'].values[0], 
                    'total assessed value': tax_table.loc['TotalAssessedValue'].values[0], 
                    'total market value': tax_table.loc['TotalMarketValue'].values[0]  
                })
        
        
        blank()

        st.write("**Model Predictions**")  

        with st.expander("Revenue"):  
            blank() 
            
            if cc_known: 
                st.info(f"Climate control modifier of **{round(cc_modifier, 5)}** applied to base revenue prediction")
            
            if not pd.isnull(rev_pred):  
                
                st.markdown("<u>Model Output</u>", unsafe_allow_html=True)  
                
                
                st.write(f"Predicted revenue per sq. foot: **${round(rev_pred, 2)}**")
                
                # POP_MEDIAN = preds['mean_rev_fit'].median()  
    
                fig, ax = plt.subplots(figsize=(6,3)) 
                # the histogram of the data 
                num_bins = 100
                n, bins, patches = ax.hist(preds['mean_rev_fit'], 
                                           num_bins, density=True, alpha=.2) 
                
                if cc_known:  
                    ax.axvline(base_rev, ls='dashdot', c='r', 
                               label=f'base rev / sq foot = ${round(base_rev, 2)}')  
                    ax.axvline(modified_rev, ls='dashdot', c='g', 
                               label=f'cc-adjusted rev / sq foot = ${round(modified_rev, 2)}') 
                else:
                    ax.axvline(rev_pred, ls='dashdot', c='r', 
                               label=f'predicted rev / sq foot = ${round(rev_pred, 2)}') 
#                ax.axvline(POP_MEDIAN, ls='dashdot', c='#9ea832', 
#                           label=f'population median = ${round(POP_MEDIAN, 2)}')
                ax.set_xlim(4.85, 21)  
                ax.legend(loc='upper right', fontsize='x-small') 
                ax.tick_params(axis='both', which='both', labelsize=6,
                   bottom=False, top=False, labelbottom=True,
                   left=False, right=False, labelleft=True)
                st.pyplot(fig)  
                
                if cc_known:
                    st.markdown("*Climate Control Adjustment*")  
                    
                    cc_delta = modified_rev - base_rev 
                    if cc_delta >= 0: 
                        sign = '+' 
                    else: 
                        sign = '-'

                    st.markdown(f""" 
                    - base revenue prediction: **${round(base_rev, 2)}**/sf \n 
                    - modified revenue prediction: **${round(modified_rev, 2)}**/sf\n 
                    - climate control revenue effect: **{sign} ${round(abs(cc_delta), 2)}**/sf\n
                    """)
                
                blank()  
                
                st.markdown("<u>Model Intepretation</u>", unsafe_allow_html=True)  
                
                st.write('*Variable Effects on Prediction*')
                
                rshap_row = rev_shap[rev_shap['store'] == closest_store.get('StoreID')]
                
                if rshap_row.shape[0] > 1: 
                    rshap_row = rshap_row.reset_index(drop=True).drop(1)
                 
                
                rev_shap_row = (
                    rshap_row
                    .drop(['store', 'full_fips', 'Unnamed: 0'], axis=1)
                    .squeeze().sort_values(key=lambda x: abs(x), ascending=True)
                )   
                
                fig, ax = plt.subplots(figsize=(6,3)) 
                xlabels = [helpers.rev_model_col_dict[x]['nice'] for x in rev_shap_row.index.tolist()] 

                ax.barh(y=xlabels, 
                       width=rev_shap_row.values, alpha=.3)  

                ax.set_xlabel('Effect on Revenue Prediction') 
                ax.grid(True, 'major', 'x', ls='--', lw=.5, c='k', alpha=.3) 
                
                st.pyplot(fig) 
                
#                N = 5
#                influential_features = rev_shap_row.sort_values(key=lambda x: abs(x), ascending=False)[:N]
#                fnames, fshaps = influential_features.index.tolist(), influential_features.values.tolist()  
#                
#                blank() 
#                
#                shap_string = '\n <br>'.join([f"$\Rightarrow$ **{helpers.rev_model_col_dict[f]['nice']}** drove the prediction **{'down' if v < 0 else 'up'}** <br>" for f,v in zip(fnames, fshaps)]) 
#                
#                st.write('*Directional Effects*')
#                st.markdown(shap_string, unsafe_allow_html=True) 
                
                blank()  
                
                st.write('*Variable Interpretation Table*')
                blank()
                
                rev_test_row = rev_test[rev_test['store'] == closest_store.get('StoreID')] 
                
#                for f in fnames:  
#                    f_feature = f.replace('_rev_shap', '')
#                    val, z, p = helpers.compare_feature(f_feature, rev_test_row) 
#                    #st.write(f"{f_feature} | {val} | {z} | {p}") 
#                    
#                    st.markdown(f"$\Rightarrow$ **{helpers.rev_model_col_dict[f]['nice']}** has a value of **{val}**, a z-score of **{z}**, and in the **{p}{helpers.suffix(p)}** percentile") 
                                
                shap_pretty_table = helpers.create_shapley_table(rev_coeffs, shap_row=rev_shap_row, 
                     test_row=rev_test_row, col_dict=helpers.rev_model_col_dict)
                
                st.dataframe(shap_pretty_table)

        with st.expander("Bad Debt"): 
            blank() 
            
            if not pd.isnull(bd_pred):  
                
                st.markdown("<u>Model Output</u>", unsafe_allow_html=True)  
                
                st.write(f"Predicted bad debt pct: **{round(bd_pred * 100, 2)}%**")
                
                # POP_MEDIAN = preds['bdebt_fit'].median() 
                
                fig, ax = plt.subplots(figsize=(5,3)) 
                # the histogram of the data 
                num_bins = 100
                n, bins, patches = ax.hist(preds['bdebt_fit'], 
                                           num_bins, density=True, alpha=.2) 
            
                ax.axvline(bd_pred, ls='dashdot', c='r', 
                           label=f'predicted bad debt % = {round(bd_pred * 100, 2)}%') 
#                ax.axvline(POP_MEDIAN, ls='dashdot', c='#9ea832', 
#                           label=f'population median = {round(POP_MEDIAN * 100, 2)}%')
                #ax.set_xlim(4.85, 21)  
                ax.legend(loc='upper right', fontsize='x-small') 
                ax.tick_params(axis='both', which='both', labelsize=6,
                   bottom=False, top=False, labelbottom=True,
                   left=False, right=False, labelleft=True)
                st.pyplot(fig) 
                
                blank()  
                
                st.markdown("<u>Model Intepretation</u>", unsafe_allow_html=True)  
                
                st.write('*Variable Interpretation*') 
                
                bd_row = bd_shap[bd_shap['store'] == closest_store.get('StoreID')]
                
                if bd_row.shape[0] > 1: 
                    bd_row = bd_row.reset_index(drop=True).drop(1)
                
                bd_shap_row = (
                    bd_row
                    .drop(['store', 'full_fips', 'Unnamed: 0'], axis=1)
                    .squeeze().sort_values(key=lambda x: abs(x), ascending=True)
                ) 
                 
                
                fig, ax = plt.subplots(figsize=(6,4)) 
                xlabels = [helpers.bd_model_col_dict[x]['nice'] for x in bd_shap_row.index.tolist()] 

                ax.barh(y=xlabels, 
                       width=bd_shap_row.values, alpha=.3)  

                ax.set_xlabel('Effect on Bad Debt Prediction') 
                ax.grid(True, 'major', 'x', ls='--', lw=.5, c='k', alpha=.3) 
                
                st.pyplot(fig) 
                
#                N = 5
#                influential_features = bd_shap_row.sort_values(key=lambda x: abs(x), ascending=False)[:N]
#                fnames, fshaps = influential_features.index.tolist(), influential_features.values.tolist()  
#                
#                blank() 
#                
#                shap_string = '\n <br>'.join([f"$\Rightarrow$ **{helpers.bd_model_col_dict[f]['nice']}** drove the prediction **{'down' if v < 0 else 'up'}** <br>" for f,v in zip(fnames, fshaps)]) 
#                
#                st.write('*Directional Effects*')
#                st.markdown(shap_string, unsafe_allow_html=True)
#                
#                blank() 
                blank()  
                
                st.write('*Variable Interpretation Table*') 
                blank()
                
                bd_test_row = bd_test[bd_test['store'] == closest_store.get('StoreID')] 
                 
                
#                for f in fnames:  
#                    f_feature = f.replace('_bd_shap', '')
#                    val, z, p = helpers.compare_feature(f_feature, bd_test_row) 
#                    #st.write(f"{f_feature} | {val} | {z} | {p}") 
#                    
#                    st.markdown(f"$\Rightarrow$ **{helpers.bd_model_col_dict[f]['nice']}** has a value of **{val}**, a z-score of **{z}**, and in the **{p}{helpers.suffix(p)}** percentile") 
                    
                shap_pretty_table = helpers.create_shapley_table(bd_coeffs, shap_row=bd_shap_row, 
                test_row=bd_test_row, col_dict=helpers.bd_model_col_dict, model='bd')
                
                st.dataframe(shap_pretty_table)

#        with st.expander("Supply"): 
#            st.write('')  
#            
#            one_mi_over = nrsf_1_mile > 0
#            three_mi_over = nrsf_3_mile > 0 
#            five_mi_over = nrsf_5_mile > 0 
#            ten_mi_over = nrsf_10_mile > 0  
#            
#            st.write(f"""This location is **{'over' if one_mi_over else 'under'}** supplied 
#            in a 1 mile radius, **{'over' if three_mi_over else 'under'}** supplied in a 3 mile radius, 
#            **{'over' if five_mi_over else 'under'}** supplied in a 5 mile radius, and
#            **{'over' if ten_mi_over else 'under'}** supplied in a 10 mile radius""")

        with st.expander("Clusters"):  
            st.write("**Cluster Predictions**")   

            st.dataframe(crow)   

        
        blank()  
        
        st.write("**Market Analytics**")   
        
        with st.expander("Housing & Economic Trends"): 
             
            st.markdown("<u>Housing Demand & Supply</u>", unsafe_allow_html=True) 
            st.caption("Source: realtor.com") 
            st.caption("Geography level: Zip Code") 
            
            zip_code = int(gen_row['ZipCode'].values[0]) 
            
            fig = helpers.plot_demand(zip_code, realtor)  
            if fig is not None:
                st.pyplot(fig) 
            else: 
                st.write("No housing demand data available for this zip code")
            
            blank() 
            
            st.markdown("<u>Median Home Values</u>", unsafe_allow_html=True) 
            st.caption("Source: NeighborhoodScout") 
            st.caption("Geography level: Metro area")
            
            state_fips = geo[geo['state'] == geo[geo['full_fips'] == closest_store.get('full_fips')]['state'].values[0]]['full_fips'].tolist()
            
            fig, ax = plt.subplots(figsize=(5,2)) 

            arr = metro[metro['full_fips'] == closest_store.get('full_fips')][helpers.MHV_COLS]  
            avg = metro[helpers.MHV_COLS].mean().values 
            state_avg = metro[metro['full_fips'].isin(state_fips)][helpers.MHV_COLS].mean().values
            x, y = [helpers.parse_col_name(i) for i in list(arr.columns)], arr.values[0]

            ax.plot(x, y, c='r', label='metro avg')
            ax.plot(x, avg, ls=':', label='national avg')  
            ax.plot(x, state_avg, ls=':', label='state avg')   
            
            ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3) 
            ax.tick_params(axis='both', which='both', labelsize=6,
               bottom=False, top=False, labelbottom=True,
               left=False, right=False, labelleft=True)

            
            ax.set_xticklabels(x, rotation=0) 
            ax.legend(loc='upper left', prop={'size': 6})
            
            for i,tick in enumerate(ax.xaxis.get_ticklabels()): 
                if i % 13 == 0:  
                    tick.set_visible(True) 
                else: 
                    tick.set_visible(False) 
            
            st.pyplot(fig)  
            
            blank()
            
            st.markdown("<u>GDP</u>", unsafe_allow_html=True) 
            st.caption("Source: U.S. Bureau of Economic Analysis") 
            st.caption("Geography level: County") 
            
            st.write("*Time Series*")
            
            gdp_chart_data = (
                gdp_row[['gdp_17', 'gdp_18', 'gdp_19', 'gdp_20']]
                .rename(columns={
                    'gdp_17': '2017', 
                    'gdp_18': '2018', 
                    'gdp_19': '2019', 
                    'gdp_20': '2020'
                })
                .T 
                .rename(columns={gdp_row.index[0] : 'GDP ($)'})

            ) 
            fig, ax = plt.subplots(figsize=(5,2))
            ax.plot(gdp_chart_data.index, gdp_chart_data['GDP ($)'], 
                    ls=':', marker='X', c='green', label='GDP ($)')   
            
            ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3) 
            ax.legend(loc='upper left', prop={'size': 6})
            ax.tick_params(axis='both', which='both', labelsize=6,
               bottom=False, top=False, labelbottom=True,
               left=False, right=False, labelleft=True)
            st.pyplot(fig)
            
            gdp_percap = gdp_row['gdp_per_cap_20'].values[0] 
            gdp_percentile = gdp_row['per_cap_percentile'].values[0]  
            if not pd.isnull(gdp_percentile):
                gdp_percentile = round(gdp_percentile*100) 
    #            GDP_AVG = 37.34
    #            pct_diff = round((1 - (GDP_AVG / gdp_percap)) * 100, 2) 
    #            delta = 'higher' if pct_diff > 0 else 'lower'  

                st.write("*Per Capita*")

                st.write(f"GDP per capita in this county is $**{round(gdp_percap, 2)}**, which is in the **{gdp_percentile}**{helpers.suffix(gdp_percentile)} percentile nationally")
            
            blank() 
            
            st.markdown("<u>Neighborhood Rent Prices </u>", unsafe_allow_html=True)   
            st.caption("Source: NeighborhoodScout") 
            st.caption("Geography level: Neighborhood (Census Tract)")
            
            rent_row = local_rent[local_rent['full_fips'] == closest_store.get('full_fips')] 
            rent_row = rent_row.drop(['Unnamed: 0', 'full_fips'], axis=1).rename(columns={'neighborhood.real_estate.grrent_yield':'rent to property value ratio',
                 'neighborhood.real_estate.rent': 'average rent',
                 'neighborhood.real_estate.rent_br1': '1 bedroom rent',
                 'neighborhood.real_estate.rent_br2': '2 bedroom rent',
                 'neighborhood.real_estate.rent_br3': '3 bedroom rent',
                 'neighborhood.real_estate.grrent_ch': 'change in rent price'}).T  
            rent_row.columns = ['value']
            
            st.dataframe(rent_row.style.set_precision(2))
            
#            _, col1, col2, col3, col4, _ = st.columns(6)
#            col1.metric("Median Rent", "$1.2K", "")
#            col2.metric("1 Bedroom Rent", "$1.5K", '')
#            col3.metric("2 Bedroom Rent", "$1.8K", '') 
#            col4.metric("3 Bedroom Rent", "$2.1K", '') 
#            
#            blank() 
#            
#            _, _, col1, col2, _, _ = st.columns(6) 
#            col1.metric("Gross Rent Yield", "4.25%") 
#            col2.metric("Rent Change", "5.2%", '')
#            
#            
#            blank() 
#            
#            st.markdown("<u>Neighborhood Rent Prices </u>", unsafe_allow_html=True) 
#            
#            
#            'average rent' 
#            
#            'income to rent'
#            
#            'per capita income'  
            
            
#        with st.expander("Search Demand"): 
#            st.write('Coming soon')
            
        with st.expander("Key Demographics"):   
            blank() 
            
            st.markdown("*Principal drivers categorized on a **nationwide** scale of 0 (lowest) to 100 (highest)*") 
            blank()
            
            drivers_row = drivers[drivers['full_fips'] == closest_store.get('full_fips')] 
            income_score = drivers_row['income_score'].values[0] 
            home_val_score = drivers_row['home_val_score'].values[0] 
            crime_score = drivers_row['crime_score'].values[0] 
            
            st.write(f"Income: **{income_score}** / 100") 
            st.write(f"Cost of Living: **{home_val_score}** / 100")
            st.write(f"Crime: **{crime_score}** / 100")  
            
            download_data.update({
                'income score': income_score, 
                'cost of living score': home_val_score, 
                'crime score': crime_score
            })
            
#            blank() 
#            _, c1, c2, c3, _ = st.columns(5) 
#            
#            c1.metric('Income', income_score) 
#            c2.metric('Cost of Living', home_val_score) 
#            c3.metric('Crime', crime_score)             
      
        
        blank()

        st.write("**Competitor Map**")




        blank() 
        
        DASH = 8
        OPACITY = 0.5
        COLOR = 'darkgreen'

        
        lat, long = closest_store['coords']
        m = folium.Map([lat, long], zoom_start=11)   

         
        folium.Circle(location=[lat, long], radius=1610, color=COLOR,
                      popup='1 mile radius', opacity=OPACITY, dash_array=DASH).add_to(m) 
        folium.Circle(location=[lat, long], radius=1610*3, color=COLOR,
                      popup='3 mile radius', opacity=OPACITY, dash_array=DASH).add_to(m) 
        folium.Circle(location=[lat, long], radius=1610*5, color=COLOR,
                      popup='5 mile radius', opacity=OPACITY, dash_array=DASH).add_to(m)
        folium.Circle(location=[lat, long], radius=1610*10, color=COLOR,
                      popup='10 mile radius', opacity=OPACITY, dash_array=DASH).add_to(m) 
        
        folium.LayerControl().add_to(m) 
        
        dists = geomapping.haversine_np(lat, long, geo['lat'].values, geo['long'].values) 

        comp_df = pd.merge(geo.loc[np.where(dists < 10)[0], :][['StoreID', 'lat', 'long']], gen)


        for store_name, addy, sf, ctype, lt, lng in comp_df[['StoreName', 'full_address', 
                                                        'RentableSqft', 'CompanyType',
                                                 'lat', 'long']].values:   
#            html = f""" 
#                    <h4>New Development</h4>\n
#                    <b>Name:</b> {kd_dict[s]['StoreName']}<br>
#                    <b>Acres:</b> {kd_dict[s]['propertyAcres']}<br>
#                    <b>Est. NRSF:</b> {kd_dict[s]['EstimatedRentableSquareFootage']}<br> 
#                    <b>Num. Buildings:</b> {kd_dict[s]['propertyNumberOfBuildings']}<br> 
#                    <b>Num. Floors:</b> {kd_dict[s]['propertyFloors']}<br>
#                    <b>Development stage:</b> {kd_dict[s]['propertyStage']}<br> 
#                    <b>Development type:</b> {kd_dict[s]['propertyProjectType']}<br>  
#                    <b>Est. Opening Date:</b> {kd_dict[s]['propertyExpectedToOpen']}<br> 
#                    """ 
#            iframe = folium.IFrame(html=html, width=250, height=100)
#            popup = folium.Popup(iframe, max_width=1000)
            
            html = f"""
            <b>{store_name}</b> <br> 
            ------------ <br> 
            {ctype} <br> 
            ------------ <br> 
            {addy} <br>  
            ------------ <br>
            {comma_print(sf)} rentable sq. ft
            """ 
            iframe = folium.IFrame(html=html, width=200, height=150)
            popup = folium.Popup(iframe, max_width=1000)
            folium.Marker(location=[lt, lng],  popup=popup, icon=folium.Icon()).add_to(m) 
            
        folium.Marker(location=[lat, long], draggable=False,
                    popup="""<b>{}</b><br> ---- <br> NRSF: {} """.format(sname, comma_print(nrsf)),
                                  icon=folium.Icon(color='red')).add_to(m)  

        st.markdown(m._repr_html_(), unsafe_allow_html=True)  
        
        blank() 
        blank()
        
        #st.download_button("Download full report", '')  
       
        download_data = pd.DataFrame({k:[v] for k,v in download_data.items()}).T  
        st.download_button(
                label="Download full report",
                data=convert_df(download_data),
                file_name=f"{sname}_{query}.csv",
                mime='text/csv'
            ) 
        
        
#        st.write(download_data.shape)
#        st.write(download_data)
    else: 
        st.warning("""
        We don't currently have data on this facility's neighborhood... 
        
        Instead, mapping to the closest location data is available for
        """)
        
        # TODO --> process to find cloest tract 