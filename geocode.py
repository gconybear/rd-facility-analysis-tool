import requests 
import json

def extract_lat_long_via_address(address_or_zipcode, rtype='coords'):
    lat, lng = None, None
    api_key = "AIzaSyDM38moX2k-1wDhRewE5thUJU9lbvTKnGg"
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    endpoint = f"{base_url}?address={address_or_zipcode}&key={api_key}"
    # see how our endpoint includes our API key? Yes this is yet another reason to restrict the key
    r = requests.get(endpoint) 
    
    if r.status_code == 200: 
        res = json.loads(r.content)  
        if rtype == 'coords': 
            return res['results'][0]['geometry']['location'] 
        else: 
            return res
    
    return None