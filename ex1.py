"""
Author: Efraim Rahamim
ID: 315392621
"""

import matplotlib.pyplot as plt
import numpy
import numpy as np
import sys

image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.loadtxt(centroids_fname)  # load centroids

orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float) / 255
pixels = pixels.reshape(-1, 3)

# save the number of centroids in a var
cent_num = z.shape[0]

# save the number of pixels in a var
pix_num = pixels.shape[0]

# open out put file
output_file = open(out_fname, "w")

# running maximum 20 iteration
for i in range(20):
    # initialize empty matrix in size of (number of centroids)x3 which saves each update of every centroid
    cent_update = np.array([[0., 0., 0.]] * cent_num)
    # initialize counter array to count the pixels that belongs to each centroid
    cent_counter = np.array([0] * cent_num)
    # running on every pixel
    for pix_row in range(pix_num):
        # initialize the minimum distance to be the maximum distance
        min_distance = float('inf')
        # initialize the closest centroid
        closest_cent = None
        # running on every centroid
        for cent_row in range(cent_num):
            # get distance from current centroid to the pixel
            distance = np.linalg.norm(z[cent_row] - pixels[pix_row])
            # update the minimum distance
            if distance < min_distance:
                min_distance = distance
                # save the closest centroid
                closest_cent = cent_row
        # associating the pixel with it closest centroid
        # adding the RGB values
        cent_update[closest_cent] += pixels[pix_row]
        # updating the counter
        cent_counter[closest_cent] += 1

    # initialize flag variable to know if any centroid was updated
    flag = 0
    # running on the centroids
    for k in range(cent_num):
        # calculate the new position and make sure not to divide with zero
        if cent_counter[k] != 0:
            new_cent = np.array([cent_update[k][0] / cent_counter[k], cent_update[k][1] / cent_counter[k],
                                 cent_update[k][2] / cent_counter[k]]).round(4)
            cent_update[k] = new_cent

    # check if any position was updated
    if not (np.array_equal(z, cent_update)):
        # raise a flag that a change was made
        flag = 1
        # update the positions of the centroids
        z = cent_update
        # write the centroids update to the file
        output_file.write(f"[iter {i}]:{','.join([str(cent) for cent in cent_update])}\n")
    # if non of the centroid was updated - break the loop
    if flag == 0:
        # write the centroids update to the file
        output_file.write(f"[iter {i}]:{','.join([str(cent) for cent in cent_update])}\n")
        break
# close the out put file
output_file.close()
