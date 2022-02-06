import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# function to pull in energy demand data
def get_data():

  df = pd.read_csv('2019.csv')
  return df.values

# function to pull in solar energy data
def get_solar():
    
    df = pd.read_csv("2019_solar_test.csv")
    return df.values


data = get_data()
solar = get_solar()

# function to integrat the area under the battery output curve

def curve_intergrate(curve, level):
    dif = curve - level
    dif[dif < 0] = 0
    intergrate = np.sum(dif/2)
 
    return(intergrate)

# function to calculate maximum demand reduction that can be achieved
# does this by iteratively dropping reduction target and intergrating 
# required battery to see if there is enough capacity.

def curve_max_dispatch(curve,capacity,max_out):
    
    level = np.max(curve)
    level_curve = np.repeat(level, len(curve))
    
        
    while (capacity > 0):
        kwh_used = curve_intergrate(curve,level_curve)
        if kwh_used < capacity:
            level = level - 10000
            level_curve = np.repeat(level, len(curve))
        else:
            break

    output_curve = curve - level_curve
    output_curve[output_curve < 0 ] = 0   
    
    if (np.max(curve)-np.max(curve - output_curve)) < max_out:
        return np.max(curve - output_curve) 
    return np.max(curve)-max_out   
        

# creating an array to hold each day's maximum reduction value

reduced_array = []

# working out the grid demand for each day, calculting the max
# reduction that can be achieved and storing it

for i in range(0,365):
    

    curve_i = [e-solar[i,][c] for c,e in enumerate(data[i,])] 
    reduced = curve_max_dispatch(curve_i,800000,200000)
    reduced_array.append(reduced)
    
print(reduced_array)    

pd.DataFrame(reduced_array).to_csv("baseline_results.csv")























  
   