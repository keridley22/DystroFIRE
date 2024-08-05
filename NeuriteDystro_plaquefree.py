##plaque free data

import numpy as np
import tifffile
from skimage.measure import label, regionprops, marching_cubes, mesh_surface_area
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, generate_binary_structure, distance_transform_edt
import pandas as pd
import os
import seaborn as sns

# Adjust the following paths as per your directory structure
image_dir = '/Users/katherineridley/Projects/NeuriteDystro/Dystrophies_WTFIRE/' #directory of ome tiff images labelled '{}_Dystrophies.tiff'

plotsfile = '/Users/katherineridley/Projects/NeuriteDystro/Dystrophies_WTFIRE/Plots/' #new folder to save plots

genotypekey = pd.read_csv('/Users/katherineridley/Projects/NeuriteDystro/genotypekey.csv') #direct path to genotype metadata
#make dir
os.makedirs(plotsfile, exist_ok=True)
sns.set_style()
#m04_filename = 'M04405_SMI312594_0424.lif__M04.tiff'
#dystrophies_filename = 'M04405_SMI312594_0424.lif__Dystrophies_reduced.tiff'

def load_image(filename):
    img = tifffile.imread(filename)
    #reduce image size to central 1/3
    
    return img


def process_m04_image(img):
    struct = generate_binary_structure(img.ndim, connectivity=3)
    closed_img = binary_closing(img, structure=struct)
    labeled_img = label(closed_img, connectivity=3)

    #Image with only largest object and in the center

    largest_obj = None
    largest_obj_area = 0
    for prop in regionprops(labeled_img):
        if prop.area > largest_obj_area:
            largest_obj = prop
            largest_obj_area = prop.area

    labeled_img[labeled_img != largest_obj.label] = 0
    labeled_img[labeled_img == largest_obj.label] = 1

    return labeled_img

def process_dystrophies_image(img, connectivity=3):
    struct = generate_binary_structure(img.ndim, connectivity) 
    closed_img = binary_closing(img, structure=struct)
    dilated_img = binary_dilation(closed_img, iterations=2)
    eroded_img = binary_erosion(dilated_img, iterations=1)
    
    
    # Use `connectivity` instead of `structure` for skimage.measure.label
    labeled_img = label(img, connectivity=connectivity)
    

    return labeled_img

def filter_and_count_dystrophies(labeled_dystrophies_img, voxel_size=1):
    dystrophy_props = regionprops(labeled_dystrophies_img)
    #m04_surface_distance = distance_transform_edt(~labeled_m04_img.astype(bool))
    
    filtered_dystrophy_info = []
    for prop in dystrophy_props:
        '''a = prop.major_axis_length / 2
        b = prop.minor_axis_length / 2
        # Here, `c` needs to be defined. Assuming it's the length along the Z-axis divided by 2. This must be obtained differently.
        c = (prop.bbox[3] - prop.bbox[0]) / 2  # Update this as per your Z-axis measurement method.

        # Calculate surface area using an approximate formula (Knud Thomsen's formula)
        p = 1.6075
        surface_area = 4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3) ** (1/p)
        complexity = prop.area / surface_area

        # Calculate sphericity
        sphericity = (np.pi ** (1.0 / 3.0)) * ((6 * prop.area) ** (2.0 / 3.0)) / surface_area'''
        if prop.area > 500:
            filtered_dystrophy_info.append({'Label': prop.label, 'Volume': prop.area})
            

        
    
    return pd.DataFrame(filtered_dystrophy_info)

def calculate_density(amount_of_dystrophies, volume_image, volume_plaque):
    available_volume = volume_image - volume_plaque
    if available_volume <= 0:
        raise ValueError("Available volume must be greater than zero")
    density = amount_of_dystrophies / available_volume
    return density

def calculate_m04_volume(labeled_m04_img):
    return np.sum(labeled_m04_img)

def calculate_z_amount(labeled_m04_img):
    return labeled_m04_img.shape[0]

def calculate_volume(width, height, depth):
    volume = width * height * depth
    return volume

def addregiondata(name):
    #if name contains 'L1, L23, L4'
    if ('L1' in name) or ('L23' in name) or ('L4' in name):
        return 'Cortex'
    #if name contains 'CA1, DGM, DGG'
    elif ('CA1' in name) or ('DGM' in name) or ('DGG' in name):
        return 'Hippocampus'

def main():
    
    #m04_suffix = '_M04.tiff'
    dystrophy_suffix = '_Dystrophies.tiff'

    os.chdir(image_dir)

            
    

    #m04_files = [f for f in os.listdir() if f.endswith(m04_suffix)]

    #print(f"Found {len(m04_files)} M04 files")

    
    #drop (2) from filenames
    
    dystrophy_files = [f for f in os.listdir() if f.endswith(dystrophy_suffix)]
    print(f"Found {len(dystrophy_files)} Dystrophy files")
    #drop (2) from filenames
    distancecsv = {'ID':[], 'Distance':[]}
    numobjs=[]
    dystrophies_all = []
    for dystrophy_filename in dystrophy_files:
        print(f"Processing {dystrophy_filename}")
        
        

        #m04_img = load_image(m04_filename)
        dystrophies_img = load_image(dystrophy_filename)
        
        

        #labeled_m04_img = process_m04_image(m04_img)
        labeled_dystrophies_img = process_dystrophies_image(dystrophies_img)

        #save m04 tiff
        #tifffile.imsave(os.path.join(f'{m04_filename[:-5]}_m04filtered.tiff'), labeled_m04_img)
       
        
        dystrophy_data = filter_and_count_dystrophies(labeled_dystrophies_img)

        #save filteredimg to tiff
        #tifffile.imsave(os.path.join(f'{dystrophy_filename[:-5]}_dystrofiltered.tiff'), filtered_img)

        print('Starting volume normalization')
    
    
        width = 183
        height = 183
        depth = dystrophies_img.shape[0]
        print('z:', depth)
        volume_image = calculate_volume(width, height, depth)
        print('volume:', volume_image)
        print()
        volume_plaque = 0
        z_amount = calculate_z_amount(labeled_dystrophies_img)
        density = calculate_density(len(dystrophy_data), volume_image, volume_plaque)

        print(f"Found {len(dystrophy_data)} dystrophies")

        #print(dystrophy_data.head(10))
        #for distance in dystrophy_data['Distance_to_Surface']:
         #   distancecsv['ID'].append(dystrophy_filename[23:-38])
          #  distancecsv['Distance'].append(distance)

        region = addregiondata(dystrophy_filename)

        uniqueid = dystrophy_filename[24:-48]
        print(uniqueid)
        print(uniqueid[:3])
        
        numobjs.append({'ID':uniqueid, 'dystrophy_count':len(dystrophy_data), 'volume':volume_image, 'dystrophy_density':density, 'region': region})

        

        dystrophies_all.append({'ID':uniqueid, 'dystrophy_count':len(dystrophy_data), 'dystrophy_data':dystrophy_data})
        
        # Saving results to CSV
        #results_csv_path = os.path.join(image_dir, f'{uniqueid}_dystrophy_distance_results.csv')
        #dystrophy_data.to_csv(results_csv_path, index=False)
        #print(f"Results saved to {results_csv_path}")

    #save numobjs to csv
    numobjs_df = pd.DataFrame(numobjs)

    numobjs_df['prefix'] = numobjs_df['ID'].str[:3]

    print(numobjs_df['prefix'])

    numobjs_df['genotype'] = numobjs_df['prefix'].map(genotypekey.set_index('prefix')['genotype'])

    numobjs_df.to_csv('numobjs.csv', index=False)

    #save distancecsv to csv
    #distancecsv_df = pd.DataFrame(distancecsv)
    #distancecsv_df.to_csv('distancecsv.csv', index=False)

if __name__ == '__main__':
    main()