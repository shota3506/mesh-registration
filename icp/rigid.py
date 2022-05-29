import copy
import open3d
import numpy as np


# Rigid ICP Registration
def rigid(source, target, threshold=0.02):
    sourceply = open3d.geometry.PointCloud()
    targetply = open3d.geometry.PointCloud()
    sourceply.points = copy.deepcopy(source.vertices)
    targetply.points = copy.deepcopy(target.vertices)
    sourceply.normals = copy.deepcopy(source.vertex_normals)
    targetply.normals = copy.deepcopy(target.vertex_normals)

    p2p = open3d.pipelines.registration.registration_icp(
        sourceply,
        targetply,
        threshold,
        np.eye(4),
        open3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    affine = p2p.transformation

    refined = copy.deepcopy(source)
    refined.transform(affine)
    return refined
