import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import cos, sin

def draw_bbox(ax, center, size, angle=0, color='r'):
    # Calculate the corner points of the box before rotation
    x, y, z = center
    dx, dy, dz = size
    corners = np.array([[x - dx / 2, y - dy / 2, z - dz / 2],
                        [x + dx / 2, y - dy / 2, z - dz / 2],
                        [x + dx / 2, y + dy / 2, z - dz / 2],
                        [x - dx / 2, y + dy / 2, z - dz / 2],
                        [x - dx / 2, y - dy / 2, z + dz / 2],
                        [x + dx / 2, y - dy / 2, z + dz / 2],
                        [x + dx / 2, y + dy / 2, z + dz / 2],
                        [x - dx / 2, y + dy / 2, z + dz / 2]])

    # Apply rotation around the Z-axis (yaw rotation)
    rotation_matrix = np.array([[cos(angle), -sin(angle), 0],
                                [sin(angle), cos(angle), 0],
                                [0, 0, 1]])

    # Rotate each corner
    rotated_corners = np.dot(corners - np.array(center), rotation_matrix.T) + np.array(center)

    # Create a list of 6 faces using the corner points
    faces = [[rotated_corners[j] for j in [0, 1, 2, 3]],
             [rotated_corners[j] for j in [4, 5, 6, 7]], 
             [rotated_corners[j] for j in [0, 1, 5, 4]], 
             [rotated_corners[j] for j in [2, 3, 7, 6]], 
             [rotated_corners[j] for j in [0, 3, 7, 4]], 
             [rotated_corners[j] for j in [1, 2, 6, 5]]]

    # Plot each face as a polygon
    ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=0.3))

def visualize_pointcloud_with_bboxes(point_cloud, pred_bboxes, gt_bboxes, pred_labels, save_path="output_image.png"):
    """
    Visualizes point cloud with predicted and ground truth bounding boxes and saves it as an image.
    
    Parameters:
    - point_cloud: np.array of shape (N, 3) where N is the number of points.
    - pred_bboxes: np.array of shape (n1, 7) where n1 is the number of predicted bounding boxes. 
                   The 7 elements represent (x_center, y_center, z_center, width, height, depth, rotation_angle).
    - gt_bboxes: np.array of shape (n2, 7) where n2 is the number of ground truth bounding boxes. 
                 The 7 elements represent (x_center, y_center, z_center, width, height, depth, rotation_angle).
    - pred_labels: np.array of shape (n1, 1) representing the labels of the predicted bounding boxes.
    - save_path: Path to save the generated image. Default is "output_image.png".
    """
    
    # Create a Matplotlib figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', s=1)

    # Plot predicted bounding boxes (in red) with labels
    for i, bbox in enumerate(pred_bboxes):
        center = bbox[:3]
        size = bbox[3:6]
        angle = bbox[6]  # rotation angle
        label = pred_labels[i]  # get the predicted label for the box (optional)
        draw_bbox(ax, center, size, angle, color='r')

    # Plot ground truth bounding boxes (in green)
    for bbox in gt_bboxes:
        center = bbox[:3]
        size = bbox[3:6]
        angle = bbox[6]  # rotation angle
        draw_bbox(ax, center, size, angle, color='g')

    # Set labels and plot limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the figure as a PNG file
    plt.savefig(save_path)

    # Close the figure after saving to avoid displaying in environments with no GUI
    plt.close()
