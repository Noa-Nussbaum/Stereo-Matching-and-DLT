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

    disp_low = disp_range[0]
    disp_high = disp_range[1]

    if disp_high - disp_low + 1 > 80:
        raise ValueError("display range must be lower than 80")

    # initialize disparity map with zeros
    answer = np.zeros(img_l.shape)

    rows = img_l.shape[0]
    columns = img_l.shape[1]

    for i in range(k_size, rows - k_size):
        for j in range(k_size, columns - k_size):
            sxl, exl = i - k_size, i + k_size + 1
            syl, eyl = j - k_size, j + k_size + 1
            patch_left = img_l[sxl:exl, syl:eyl]
            best = np.inf
            for offset in range(disp_low, disp_high):
                sxrr, exrr = i - k_size, i + k_size + 1
                syrr, eyrr = j - k_size + offset, j + k_size + 1 + offset
                sxrl, exrl = i - k_size, i + k_size + 1
                syrl, eyrl = j - k_size - offset, j + k_size + 1 - offset
                right_r = img_r[sxrr:exrr, syrr:eyrr]
                right_l = img_r[sxrl:exrl, syrl:eyrl]
                if right_r.shape == patch_left.shape:
                    ssd = np.sum((patch_left - right_r) ** 2)
                    if ssd < best:
                        best = ssd
                        answer[i, j] = offset
                if right_l.shape[0] == patch_left.shape[0] and right_l.shape[1] == patch_left.shape[1]:
                    ssd = np.sum((patch_left - right_l) ** 2)
                    if ssd < best:
                        answer[i, j] = offset
                        best = ssd
    return answer

def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    disp_low, disp_high = disp_range[0], disp_range[1]
    rows = img_l.shape[0]
    columns = img_l.shape[1]

    # initialize disparity map with zeros
    answer = np.zeros((rows, columns, disp_high))

    l_norm = img_l - filters.uniform_filter(img_l, k_size)
    r_norm = img_r - filters.uniform_filter(img_r, k_size)

    for offset in range(disp_low, disp_high):
        # move and normalize
        steps = offset + disp_low
        norm_l_to_r = np.roll(l_norm, -steps)
        sigma = filters.uniform_filter(norm_l_to_r * r_norm, k_size)
        sigma_l = filters.uniform_filter(np.square(norm_l_to_r), k_size)
        sigma_r = filters.uniform_filter(np.square(r_norm), k_size)

        # update disparity_map with NC score
        answer[:, :, offset] = sigma / np.sqrt(sigma_l * sigma_r)

    # for each pixel choose maximum depth value
    return np.argmax(answer, axis=2)

def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destination image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """

    # initiate homography matrix
    homography = []

    # create A
    for i in range(src_pnt.shape[0]):
        # init src vector
        x_s, y_s = src_pnt[i][0], src_pnt[i][1]
        # init dest vector
        x_d, y_d = dst_pnt[i][0], dst_pnt[i][1]
        # init homography matrix
        homography.append([x_s, y_s, 1, 0, 0, 0, -x_d * x_s, -x_d * y_s, -x_d])
        homography.append([0, 0, 0, x_s, y_s, 1, -y_d * x_s, -y_d * y_s, -y_d])

    u, s, vh = np.linalg.svd(homography)

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

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)
    RANSAC_REPROJ_THRESHOLD = 5.0

    homography, _ = cv2.findHomography(src_p, dst_p,cv2.RANSAC, RANSAC_REPROJ_THRESHOLD) # using cv to find the accurate homography

    height = src_img.shape[0]
    width = src_img.shape[1]
    warped = np.zeros_like(dst_img)

    for i in range(height):
        for j in range(width):
            hold = np.array([j, i, 1])
            # multiply matrices
            mat = np.dot(homography, hold)
            # new y value
            y = int(mat[0] / mat[mat.shape[0] - 1])
            # new x value
            x = int(mat[1] / mat[mat.shape[0] - 1])
            # assign new values
            warped[x, y] = src_img[i, j]

    bound = warped == 0

    # find the answer by combining both images
    answer = dst_img * bound + (1 - bound) * warped
    plt.imshow(answer)
    plt.show()



