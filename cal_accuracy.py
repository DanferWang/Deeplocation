import math
import pandas as pd



dists = pd.read_csv("./demo/data/result_325th.csv", usecols=["great_circle_distance"])
#dists = dists.to_csv(header=None)
print(dists)
dists = dists.astype(float)
result = {'continent': 0, 'country': 0, 'region': 0, 'city': 0, 'street': 0}

num_images = 0
for i in range(len(dists)):
    num_images +=1
    img = dists.iloc[i].values[0]
    if img <= 2500.0:
	    result['continent'] = result['continent'] + 1 
	    if img <= 750.0:
	        result['country'] = result['country'] + 1
	        if img <= 200.0:
	        	result['region'] = result['region'] + 1
	        	if img <= 25.0:
	        		result['city'] = result['city'] + 1
	        		if img <= 1.0:
	        			result['street'] = result['street'] + 1
	
print(num_images)
for thresh in result:
	result[thresh] = f"{(100 * result[thresh] / num_images):.1f}"

print(pd.DataFrame.from_dict(result, orient='index'))

