

import numpy as np
import nibabel as nib
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth

def main():
        img = nib.load("ScanName.nii").get_fdata()

        mShiftImg = np.empty(img.shape)

        for sliceNum in (range(img.shape[2])):
        #for sliceNum in range(31,32):
        #for sliceNum in range(18,19):
            print(sliceNum)
            slice = img[:,:,sliceNum]
            print("slice shape: ", slice.shape)

            from PIL import Image
            im = Image.fromarray(slice)
            new_im = im.convert('RGB')
            new_im.save("slice.png")


            imgS = cv.imread("slice.png")
            imgS = cv.medianBlur(imgS, 3)
            print("imgS shape: ", imgS.shape)
            flat_image = imgS.reshape((-1,3))
            flat_image = np.float32(flat_image)
            #self = flat_image

            quantileVal = 0.057
            #quantileVal = 0.5
            bandwidth = estimate_bandwidth(flat_image, quantile=quantileVal, n_samples=3000)
            print(bandwidth)
            if(bandwidth >= 5):
                #ms = MeanShift(bandwidth, max_iter=800, bin_seeding=True)
                ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
                print("fitting")
                ms.fit(flat_image)
                labeled=ms.labels_
    
                segments = np.unique(labeled)
                print('Number of segments: ', segments.shape[0])

                total = np.zeros((segments.shape[0], 3), dtype=float)
                count = np.zeros(total.shape, dtype=float)
             
                for i, label in enumerate(labeled):
                    #print(label)
                    total[label] = total[label] + flat_image[i]
                    count[label] += 1
                avg = total/count
                avg = np.uint8(avg)
                # cast the labeled image into the corresponding average color
                res = avg[labeled]
                res = res.reshape(imgS.shape)
                print("res shape: ", res.shape)
                #print((res))
                resImg = Image.fromarray(res)
                resImg = resImg.convert('L')
                
                result = np.array(resImg)
                mShiftImg[:,:,sliceNum] = result
            else:
                print("no image change")
                mShiftImg[:,:,sliceNum] = im




        new_image = nib.Nifti1Image(mShiftImg, affine=np.eye(4))
        nib.save(new_image, 'ScanName_mShift.nii') 

if __name__== "__main__":
    main()
