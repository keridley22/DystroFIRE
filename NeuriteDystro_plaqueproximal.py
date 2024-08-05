##dystrophies proximal to plaques 

import numpy as np
import tifffile
from skimage.measure import label, regionprops, marching_cubes, mesh_surface_area
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, generate_binary_structure, distance_transform_edt, distance_transform_cdt
import pandas as pd
import os
import seaborn as sns

# Adjust the following paths as per your directory structure
## imagedir to your tiff images
## plotsfile to where to save your plots
image_dir = '/Users/katherineridley/Projects/NeuriteDystro/Dystrophies/Images/Manual/'

plotsfile = '/Users/katherineridley/Projects/NeuriteDystro/Dystrophies/Images/Manual/Plots/'

genotypekey = pd.read_csv('/Users/katherineridley/Projects/NeuriteDystro/genotypekey.csv')
#make dir
os.makedirs(plotsfile, exist_ok=True)
sns.set_style()
#m04_filename = 'M04405_SMI312594_0424.lif__M04.tiff'
#dystrophies_filename = 'M04405_SMI312594_0424.lif__Dystrophies_reduced.tiff'

def load_image(filename):
    img = tifffile.imread(filename)
    '''start_y, end_y = int(0.3 * img.shape[1]), int(0.7 * img.shape[1])
    start_x, end_x = int(0.3 * img.shape[2]), int(0.7 * img.shape[2])
    reduced_img = img[:, start_y:end_y, start_x:end_x]'''
    return img

    
    


def process_m04_image(img):
    print('Entered process_m04_image')
    '''struct = generate_binary_structure(img.ndim, connectivity=2)
    closed_img = binary_closing(img, structure=struct)'''
    labeled_img = label(img, connectivity=2)
    print('Unique labels in labeled_img:', np.unique(labeled_img))

    #Image with only largest object and in the center

    largest_obj = None
    largest_obj_area = 0
    for prop in regionprops(labeled_img):
        if prop.area > largest_obj_area:
            largest_obj = prop
            largest_obj_area = prop.area

    labeled_img[labeled_img != largest_obj.label] = 0
    labeled_img[labeled_img == largest_obj.label] = 1
    print('Finished processing M04 image')
    return labeled_img

def process_dystrophies_image(img, connectivity=1):
    struct = generate_binary_structure(img.ndim, connectivity) 
    closed_img = binary_closing(img, structure=struct)
    dilated_img = binary_dilation(closed_img, iterations=2)
    eroded_img = binary_erosion(dilated_img, iterations=1)
    
    
    # Use `connectivity` instead of `structure` for skimage.measure.label
    labeled_img = label(img, connectivity=connectivity)
    return labeled_img

def filter_and_count_dystrophies(labeled_dystrophies_img, labeled_m04_img):
    print('Entered filter_and_count_dystrophies')
    print('Unique labels in labeled_dystrophies_img:', np.unique(labeled_dystrophies_img))
    

    dystrophy_props = regionprops(labeled_dystrophies_img)
    print('Number of objects: ', len(dystrophy_props))

    print('Unique values in mask for distance_transform:', np.unique(~labeled_m04_img.astype(bool)))

    
    try:
        # Check the mask values
        mask_for_distance = ~labeled_m04_img.astype(bool)
        # Compute the distance transform
        m04_surface_distance = distance_transform_cdt(mask_for_distance)
        print('Distance transform computed successfully.')
    
        dystrophy_props = regionprops(labeled_dystrophies_img)
        print('Number of objects:', len(dystrophy_props))
        
        filtered_dystrophy_info = []
        for prop in dystrophy_props:
            print('Processing region with label:', prop.label)
            
            dst = np.min(m04_surface_distance[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]])
            print('Minimum distance to M04 plaque:', dst)
            
            if dst <= 50:
                filtered_dystrophy_info.append({'Label': prop.label, 'Distance_to_Surface': dst, 'Volume': prop.area})
    except Exception as e:
        print('Error in processing:', str(e))
    
    return pd.DataFrame(filtered_dystrophy_info)

def filter_and_count_dystrophies_from_csv(labeled_dystrophies_img, distance_csv):
    '''print('Entered filter_and_count_dystrophies')
    print('Unique labels in labeled_dystrophies_img:', np.unique(labeled_dystrophies_img))
    

    dystrophy_props = regionprops(labeled_dystrophies_img)
    print('Number of objects: ', len(dystrophy_props))

    print('Unique values in mask for distance_transform:', np.unique(~labeled_m04_img.astype(bool)))

    
    try:
        # Check the mask values
        mask_for_distance = ~labeled_m04_img.astype(bool)
        # Compute the distance transform
        m04_surface_distance = distance_transform_cdt(mask_for_distance)
        print('Distance transform computed successfully.')'''
    
    dystrophy_props = regionprops(labeled_dystrophies_img)
    print('Number of objects:', len(dystrophy_props))
    
    filtered_dystrophy_info = []
    for prop in dystrophy_props:
        print('Processing region with label:', prop.label)
        
        dst = distance_csv[distance_csv['Label'] == prop.label]['Distance_to_Surface'].values[0]
        
        if dst <= 50:
            filtered_dystrophy_info.append({'Label': prop.label, 'Distance_to_Surface': dst, 'Volume': prop.area})
    
    
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

def calculate_volume(width, height, depth, voxel_dimensions):
    # Calculate the volume by multiplying width, height, and depth by their respective voxel sizes
    voxel_size_z, voxel_size_y, voxel_size_x = voxel_dimensions
    volume = width * voxel_size_x * height * voxel_size_y * depth * voxel_size_z
    return volume

def calculate_volume_from_mask(mask, voxel_dimensions):
    # Calculate the total volume by counting all 'True' voxels and multiplying by the volume of one voxel
    return np.sum(mask) * np.prod(voxel_dimensions)

def dilate_mask(mask, voxel_dimensions, dilation_distance=50):
    # Calculate the number of voxels to dilate based on physical distance and voxel size
    structuring_element_size = [int(dilation_distance / dim) for dim in voxel_dimensions]
    structuring_element = np.ones(structuring_element_size, dtype=bool)
    dilated_mask = binary_dilation(mask, structure=structuring_element)
    return dilated_mask

def addregiondata(name):
    #if name contains 'L1, L23, L4'
    if ('L1' in name) or ('L23' in name) or ('L4' in name):
        return 'Cortex'
    #if name contains 'CA1, DGM, DGG'
    elif ('CA1' in name) or ('DGM' in name) or ('DGG' in name):
        return 'Hippocampus'


def main():


    m04_suffix = '_M04.tiff'
    dystrophy_suffix = '_Dystrophies_manual.tiff'


    os.chdir(image_dir)


    m04_files = [f for f in os.listdir() if f.endswith(m04_suffix)]

    #print(f"Found {len(m04_files)} M04 files")

    
    
    dystrophy_files = [f for f in os.listdir() if f.endswith(dystrophy_suffix)]

    print(f"Found {len(m04_files)} M04 files and {len(dystrophy_files)} Dystrophy files")

   
    #drop (2) from filenames
    distancecsv = {'ID':[], 'Distance':[]}
    numobjs=[]
    dystrophies_all = []
    for m04_filename in m04_files:
        print(f"Processing {m04_filename}")

        dystrophy_filename = m04_filename.replace(m04_suffix, dystrophy_suffix)
        m04_img = load_image(m04_filename)
        dystrophies_img = load_image(dystrophy_filename)
        print(f"M04 image shape: {m04_img.shape}")
        labeled_m04_img = process_m04_image(m04_img)
        print(f"Labeled M04 image shape: {labeled_m04_img.shape}")
        #labeled_dystrophies_img = process_dystrophies_image(dystrophies_img)
        
        '''#save m04 tiff
        #tifffile.imsave(os.path.join(f'{m04_filename[:-5]}_m04filtered.tiff'), labeled_m04_img)
        dystrophy_data= filter_and_count_dystrophies(labeled_dystrophies_img, labeled_m04_img)'''

        uniqueid = dystrophy_filename[23:-53]
        print(f"Processing {uniqueid}")
        # Saving results to CSV
        results_csv_path = os.path.join(image_dir, f'{uniqueid}_dystrophy_distance_results.csv')
        dystrophy_data = pd.read_csv(os.path.join(image_dir, f'{uniqueid}_dystrophy_distance_results.csv'))
        print(f"Results loaded from {results_csv_path}")

        print('Starting volume normalization')
        #save filteredimg to tiff
        #tifffile.imsave(os.path.join(f'{dystrophy_filename[:-5]}_dystrofiltered.tiff'), filtered_img)
        fov_width_um = 183  # width of the field of view in microns
        fov_height_um = 183 # height of the field of view in microns
        num_pixels_x = 1024  # number of pixels in X dimension
        num_pixels_y = 1024  # number of pixels in Y dimension
        z_step_size_um = 1  # Z step size in microns, change as per your setup
        width = dystrophies_img.shape[2]
        height = dystrophies_img.shape[1]
        depth = dystrophies_img.shape[0]
        # Calculate voxel dimensions
        voxel_size_x = fov_width_um / num_pixels_x
        voxel_size_y = fov_height_um / num_pixels_y
        voxel_size_z = z_step_size_um
        voxel_dimensions = (voxel_size_z, voxel_size_y, voxel_size_x)
        
        print('Calculating within 50um of plaque')
        volume_image = calculate_volume(width, height, depth, voxel_dimensions)
        print('Volume of image:', volume_image)
        plaque_mask = labeled_m04_img > 0  # Assuming nonzero labels for plaques
        dilated_mask = dilate_mask(plaque_mask, voxel_dimensions)

        volume_plaque = calculate_volume_from_mask(plaque_mask, voxel_dimensions)
        print('Volume of plaque:', volume_plaque)
        volume_dilated_plaque = calculate_volume_from_mask(dilated_mask, voxel_dimensions)
        volume_40um_surrounding = volume_dilated_plaque - volume_plaque
        
        


        # Update to calculate density using the volume within 40um of the plaque
        density = calculate_density(len(dystrophy_data), volume_40um_surrounding, volume_plaque)
        
        '''volume_image = calculate_volume(width, height, depth)
        volume_plaque = calculate_m04_volume(labeled_m04_img)
        z_amount = calculate_z_amount(labeled_dystrophies_img)
        density = calculate_density(len(dystrophy_data), volume_image, volume_plaque)'''

        print(f"Found {len(dystrophy_data)} dystrophies")

        
        print(dystrophy_data.head(10))
        #for distance in dystrophy_data['Distance_to_Surface']:
         #   distancecsv['ID'].append(dystrophy_filename[23:-38])
          #  distancecsv['Distance'].append(distance)
        if len(dystrophy_data) > 0:
            #dystrophy-data 0-10um from surface bin
            density0_10 = calculate_density(len(dystrophy_data[dystrophy_data['Distance_to_Surface'] <= 10]), volume_40um_surrounding, volume_plaque)
            #dystrophy-data 10-20um from surface bin
            density10_20 = calculate_density(len(dystrophy_data[(dystrophy_data['Distance_to_Surface'] > 10) & (dystrophy_data['Distance_to_Surface'] <= 20)]), volume_40um_surrounding, volume_plaque)
            #dystrophy-data 20-30um from surface bin
            density20_30 = calculate_density(len(dystrophy_data[(dystrophy_data['Distance_to_Surface'] > 20) & (dystrophy_data['Distance_to_Surface'] <= 30)]), volume_40um_surrounding, volume_plaque)
            #dystrophy-data 30-40um from surface bin
            density30_40 = calculate_density(len(dystrophy_data[(dystrophy_data['Distance_to_Surface'] > 30) & (dystrophy_data['Distance_to_Surface'] <= 40)]), volume_40um_surrounding, volume_plaque)
            #dystrophy-data 40-50um from surface bin
            density40_50 = calculate_density(len(dystrophy_data[(dystrophy_data['Distance_to_Surface'] > 40) & (dystrophy_data['Distance_to_Surface'] <= 50)]), volume_40um_surrounding, volume_plaque)
            
        else:
            density0_10 = 0
            density10_20 = 0
            density20_30 = 0
            density30_40 = 0
            density40_50 = 0
            

        print('density 0-10: ', density0_10, 'density 10-20: ', density10_20, 'density 20-30: ', density20_30, 'density 30-40: ', density30_40, 'density 40-50: ', density40_50)

        region = addregiondata(dystrophy_filename)

        uniqueid = dystrophy_filename[23:-38]
        numobjs.append({'ID':uniqueid, 'dystrophy_count':len(dystrophy_data), 'dystrophy_density':density, 'dystrophy_density0_10':density0_10, 'dystrophy_density10_20':density10_20, 'dystrophy_density20_30':density20_30, 'dystrophy_density30_40':density30_40, 'dystrophy_density40_50':density40_50, 
                        'region': region})


        

        dystrophies_all.append({'ID':uniqueid, 'dystrophy_count':len(dystrophy_data), 'dystrophy_data':dystrophy_data})
        
        # Saving results to CSV
        #results_csv_path = os.path.join(image_dir, f'{uniqueid}_dystrophy_distance_results.csv')
        #dystrophy_data.to_csv(results_csv_path, index=False)
        #print(f"Results saved to {results_csv_path}")

    #save numobjs to csv
        
    numobjs_df = pd.DataFrame(numobjs)
        
    numobjs_df['prefix'] = numobjs_df['ID'].str[:3]

    numobjs_df['genotype'] = numobjs_df['prefix'].map(genotypekey.set_index('prefix')['genotype'])

    numobjs_df.to_csv('numobjs2.csv', index=False)

    #save distancecsv to csv
    #distancecsv_df = pd.DataFrame(distancecsv)
    #distancecsv_df.to_csv('distancecsv.csv', index=False)

if __name__ == '__main__':
    main()