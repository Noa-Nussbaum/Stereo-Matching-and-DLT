import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.ndimage import filters



def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    answer = np.zeros(img_l.shape)
    krows = k_size*2+1
    kcolumns = k_size*2+1
    for r in range(img_l.shape[0]):
        for c in range(img_l.shape[1]):
            b_offset = -1
            hold=np.inf
            for m in range(disp_range[0],disp_range[1]):
                ssd=0
                for i in range(krows):
                    for j in range(kcolumns):
                        if img_r.shape[0]>r+i-m and img_r.shape[1]>c+j-m:
                            ssd+=(img_l[r][c]-img_r[r+i-m,c+j-m])**2

                if ssd < hold:
                    hold = ssd
                    b_offset = m
                answer[r][c]= b_offset

    return answer

def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    # disparity range
    min_offset, max_offset = disp_range[0], disp_range[1]
    # disparity map
    disparity_map = np.zeros((img_l.shape[0], img_l.shape[1], max_offset))
    # calculate average value of our image kernel and normalize
    norm_l = img_l - filters.uniform_filter(img_l, k_size)
    norm_r = img_r - filters.uniform_filter(img_r, k_size)

    for offset in range(min_offset, max_offset):
        # move left img
        steps = offset + min_offset
        norm_l_to_r = np.roll(norm_l, -steps)
        # normalize
        sigma = filters.uniform_filter(norm_l_to_r * norm_r, k_size)
        sigma_l = filters.uniform_filter(np.square(norm_l_to_r), k_size)
        sigma_r = filters.uniform_filter(np.square(norm_r), k_size)

        # update disparity_map with NC score
        disparity_map[:, :, offset] = sigma / np.sqrt(sigma_l * sigma_r)

    # for each pixel choose maximum depth value
    return np.argmax(disparity_map, axis=2)

def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destination image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    answer = np.ones((3, 3))

    # initiate homography matrix
    A = []

    # create A
    for i in range(src_pnt.shape[0]):
        # init src vector
        x_s, y_s = src_pnt[i][0], src_pnt[i][1]
        # init dest vector
        x_d, y_d = dst_pnt[i][0], dst_pnt[i][1]
        # init A matrix
        A.append([x_s, y_s, 1, 0, 0, 0, -x_d * x_s, -x_d * y_s, -x_d])
        A.append([0, 0, 0, x_s, y_s, 1, -y_d * x_s, -y_d * y_s, -y_d])

    u, s, vh = np.linalg.svd(A)

    # find eigen vector with smallest eigen value - this is the answer, H
    answer = (vh[-1, :] / vh[-1, -1] ).reshape(3, 3)

    # find error
    src_pnt = np.hstack((src_pnt, np.ones((src_pnt.shape[0], 1)))).T
    h_src = answer.dot(src_pnt)
    h_src /= h_src[2, :]
    error = np.sqrt(np.sum(h_src[0:2, :] - dst_pnt.T) ** 2)

    return answer, error

def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######

    src_p = []
    fig2 = plt.figure()

    # same as onclick_1, but with the source image instead of the dest
    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display image 2, same operations as display image 1
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)
    RANSAC_REPROJ_THRESHOLD = 5.0

    homography = cv2.findHomography(src_p, dst_p,cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)[0]  # using cv to find the accurate homography

    source_height = src_img.shape[0]
    source_width = src_img.shape[1]
    warp = np.zeros_like(dst_img)
    # boundaries = np.zeros_like(dst_img)
    for i in range(source_height):
        for j in range(source_width):
            temp = np.array([j, i, 1])
            new_coor = np.dot(homography, temp)  # dot product for calculating the coordinates
            y = int(new_coor[0] / new_coor[new_coor.shape[0] - 1])  # getting the new y
            x = int(new_coor[1] / new_coor[new_coor.shape[0] - 1])  # getting the new x
            warp[x, y] = src_img[
                i, j]  # setting that the new coordinate will present the source in the original i and j
            # if warp[x, y] == :
            #     boundaries[x, y] = True
            # else:
            #     boundaries[x, y] = False
    boundaries = warp == 0  # getting where we want to "paste"
    outcome = dst_img * boundaries + (1 - boundaries) * warp  # using a formula we have been taught
    plt.imshow(outcome)
    plt.show()



