import numpy as np
import cv2 as cv2
import torch
import torch.nn as nn
import gudhi as gd  
import gudhi.representations
from sklearn.neighbors import KernelDensity

class PIAtt:
    def __init__(self, bandwidth, resolution, weight, life_val, density_bandwidth, imgshape):
        self.bandwidth = bandwidth
        self.resolution = resolution
        self.weight = weight
        self.life_val = life_val
        self.density_bandwidth = density_bandwidth
        self.imgshape = imgshape

    def create_complex(self, points):
        #Selected Grid
        xval = np.arange(0, self.imgshape[0], 1)
        yval = np.arange(0, self.imgshape[1], 1)
        nx = len(xval)
        ny = len(yval)

        positions = np.array([[u, v] for u in xval for v in yval])
        
        if len(points) != 0:
            kde = KernelDensity(kernel = 'gaussian', bandwidth = self.density_bandwidth).fit(points)
            filt_values = -kde.score_samples(X = positions)
        else:
            filt_values = np.zeros((int(self.imgshape[0]*self.imgshape[1])))
        
        #Create Cubical Complex
        complex_cubical = gd.CubicalComplex(dimensions = [nx ,ny], 
                                            top_dimensional_cells = filt_values)
        return complex_cubical

    
    def create_image(self, complex, determined_range = None):
        complex.persistence()
        if determined_range is None:
            PI = gd.representations.PersistenceImage(bandwidth = self.bandwidth, resolution = [self.resolution, self.resolution], \
                                                 weight = lambda x: x[1]**self.weight, im_range = [0, np.nan, 0, np.nan])
        else:
            PI = gd.representations.PersistenceImage(bandwidth = self.bandwidth, resolution = [self.resolution, self.resolution], \
                                                 weight = lambda x: x[1]**self.weight, im_range = np.array(determined_range))
        
        #To control the case when there is no persistence interval in dimension 1 (especially for predictions)
        if len(complex.persistence_intervals_in_dimension(1))>=1:  
            current_points = complex.persistence_intervals_in_dimension(1)
            life_values = current_points[:, 1] - current_points[:, 0]
            pers_im = PI.fit_transform([current_points[life_values > self.life_val]])
        else:
            pers_im = np.zeros(shape = (1, self.resolution**2))
            
        image = np.flip(np.reshape(pers_im[0], [self.resolution, self.resolution]), 0)

        return image, PI.im_range

    def create_persistence_img_training(self, complex_target, complex_pred):
        complex_target.persistence()
        complex_pred.persistence()

        PI_target = gd.representations.PersistenceImage(bandwidth = self.bandwidth, resolution = [self.resolution, self.resolution], \
                                                weight = lambda x: x[1]**self.weight, im_range = [0, np.nan, 0, np.nan])

        PI_pred = gd.representations.PersistenceImage(bandwidth = self.bandwidth, resolution = [self.resolution, self.resolution], \
                                                weight = lambda x: x[1]**self.weight, im_range = [0, np.nan, 0, np.nan])

        # Persistence Image for target
        current_pred_target = complex_target.persistence_intervals_in_dimension(1)
        life_values_target = current_pred_target[:, 1] - current_pred_target[:, 0]
        pers_img_target = PI_target.fit_transform([current_pred_target[life_values_target > self.life_val]])

        # Persistence Image for prediction
        current_pred_points = complex_pred.persistence_intervals_in_dimension(1)
        life_values_pred = current_pred_points[:, 1] - current_pred_points[:, 0]
        pers_img_pred = PI_pred.fit_transform([current_pred_points[life_values_pred > self.life_val]])

        #Create new range
        new_range = np.where(PI_target.im_range > PI_pred.im_range, PI_target.im_range, PI_pred.im_range)

        #Create new persistence image with new range
        PI_last_target = gd.representations.PersistenceImage(bandwidth = self.bandwidth, resolution = [self.resolution, self.resolution], \
                                            weight = lambda x: x[1]**self.weight, im_range = new_range)
        
        PI_last_pred = gd.representations.PersistenceImage(bandwidth = self.bandwidth, resolution = [self.resolution, self.resolution], \
                                            weight = lambda x: x[1]**self.weight, im_range = new_range)
                    
            
        #To control the case when there is no persistence interval in dimension 1
        if len(complex_target.persistence_intervals_in_dimension(1))>=1:  
            current_label_points = complex_target.persistence_intervals_in_dimension(1)
            life_values_label = current_label_points[:, 1] - current_label_points[:, 0]
            pers_img_target = PI_last_target.fit_transform([current_label_points[life_values_label > self.life_val]])
        else:
            pers_img_target = np.zeros(shape = (1, self.resolution**2))     

        #To control the case when there is no persistence interval in dimension 1
        if len(complex_pred.persistence_intervals_in_dimension(1))>=1:  
            current_pred_points = complex_pred.persistence_intervals_in_dimension(1)
            life_values_pred = current_pred_points[:, 1] - current_pred_points[:, 0]
            pers_img_pred = PI_last_pred.fit_transform([current_pred_points[life_values_pred > self.life_val]])
        else:
            pers_img_pred = np.zeros(shape = (1, self.resolution**2)) 
            

        #Create proper shape for persistence images
        pers_img_target = np.flip(np.reshape(pers_img_target[0], [self.resolution, self.resolution]), 0)
        pers_img_pred = np.flip(np.reshape(pers_img_pred[0], [self.resolution, self.resolution]), 0)      
        
        return pers_img_target, pers_img_pred


    def create_pi_training_phase(self, points_target, points_pred):
        # Create CubicalComplex
        complex_target = self.create_complex(points_target)
        complex_pred = self.create_complex(points_pred)

        #Get persistence images for target and predictions
        pers_img_target, pers_img_pred = self.create_persistence_img_training(complex_target, complex_pred)

        return pers_img_target, pers_img_pred
    
     
    def calculate_pi_difference(self, target_persistence_image, pred_persistence_image):
        #Calculation of difference between images
        difference = np.abs(target_persistence_image - pred_persistence_image)
        difference = np.mean(difference)

        return difference


class piatt_loss(nn.Module):    
    def __init__(self, args):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(args.loss_beta_term))
        self.beta.requires_grad = True
        self.device = device
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, target, persistence_images, device):
        td_term = torch.abs(persistence_images[0] - persistence_images[1]).mean(axis = [1, 2]) #Topological dissimilarity term
        piatt_coeff = 1 + self.beta * td_term

        ce_loss = self.ce(outputs, targets)
        ce_loss = torch.mean(ce_loss, dim = [1,2])

        piatt_loss = (piatt_coeff * ce_loss).mean()
        return piatt_loss


def contour_points_object(image):
    background_image = np.zeros_like(image)

    image = np.where(image >= 0.5, 1.0, 0.0)

    contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]   
    background_image = cv2.drawContours(background_image, contours, -1, (255, 0, 0), 1)

    X, Y = np.where(background_image > 0)
    points = np.concatenate((X[:, None], Y[:, None]), axis=1) 

    return points