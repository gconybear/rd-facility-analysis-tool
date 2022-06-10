import numpy as np 
import pandas as pd 


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c 
    km_conv_fac = 0.621371

    return km * km_conv_fac

def get_closest_store(lat, long, geo): 
    
    dists = haversine_np(lat, long, geo['lat'].values, geo['long'].values) 
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx] 
    
    fips = geo.loc[min_idx, 'full_fips']
    sid = geo.loc[min_idx, 'StoreID'] 
    county_fips = geo.loc[min_idx, 'county_fips'] 
    coords = tuple(geo.loc[min_idx, ['lat', 'long']].values)
    
    return {
        'StoreID': float(sid), 
        'full_fips': float(fips),  
        'county_fips': float(county_fips), 
        'distance': round(min_dist, 2), 
        'coords': coords
    }