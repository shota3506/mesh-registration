import copy
import open3d
import numpy as np
from sklearn import neighbors
from sksparse import cholmod
from scipy import sparse


def cholesky_solve(M, b):
    factor = cholmod.cholesky_AAt(M.T)
    return factor(M.T.dot(b)).toarray()


# Non-Ridid ICP Registratoin
def nricp(
    source,
    target,
    alphas=np.linspace(50, 1, 20),
    gamma=1,
    eps=1e-2,
):
    # creat a new mesh for non rigid transform estimation
    source = copy.deepcopy(source)

    # vertices
    source_vertices = np.array(source.vertices)
    target_vertices = np.array(target.vertices)
    # num of source mesh vertices
    n_source_vertices = source_vertices.shape[0]

    knn = neighbors.NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        target_vertices
    )

    # calculating edge info
    source_edges = set()
    for face in np.array(source.triangles):
        face = np.sort(face)
        source_edges.add((face[0], face[1]))
        source_edges.add((face[0], face[2]))
        source_edges.add((face[1], face[2]))
    # num of source mesh edges
    n_source_edges = len(source_edges)

    # node-arc incidence matrix
    M = sparse.lil_matrix((n_source_edges, n_source_vertices), dtype=np.float32)
    for i, t in enumerate(source_edges):
        M[i, t[0]] = -1
        M[i, t[1]] = 1

    G = np.diag([1, 1, 1, gamma]).astype(np.float32)

    kron_M_G = sparse.kron(M, G)

    # affine transformations stored in the 4n*3 format
    X = np.tile(
        np.concatenate((np.eye(3), np.array([[0, 0, 0]])), axis=0),
        (n_source_vertices, 1),
    )

    D = sparse.lil_matrix((n_source_vertices, n_source_vertices * 4), dtype=np.float32)
    j_ = 0
    for i in range(n_source_vertices):
        D[i, j_ : j_ + 3] = source_vertices[i, :]
        D[i, j_ + 3] = 1
        j_ += 4

    n_max_inner = 10
    for num, alpha_stiffness in enumerate(alphas):
        for i in range(n_max_inner):
            X_prev = X

            transformed = D.dot(X)

            distances, indices = knn.kneighbors(transformed)
            indices = indices.squeeze()

            # 1. Distance term
            W = sparse.diags(np.ones((n_source_vertices)))  # weight
            U = target_vertices[indices]
            A_d = W.dot(D)
            B_d = W.dot(U)

            # 2. Stiffness term
            A_s = alpha_stiffness * kron_M_G
            B_s = sparse.lil_matrix((4 * n_source_edges, 3), dtype=np.float32)

            # Equation (12): E(X) = || AX-B ||^2
            A = sparse.csr_matrix(sparse.vstack([A_s, A_d]))
            B = sparse.csr_matrix(sparse.vstack([B_s, B_d]))
            X = cholesky_solve(A, B)

            if np.mean(np.linalg.norm(X - X_prev, axis=1)) < eps:
                break

    transformed = D.dot(X)
    source.vertices = open3d.utility.Vector3dVector(transformed)
    return source
