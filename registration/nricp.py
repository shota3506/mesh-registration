import copy
import open3d
import numpy as np
from sksparse import cholmod
from sklearn import neighbors
from scipy import sparse


def cholesky_solve(M, b):
    factor = cholmod.cholesky_AAt(M.T)
    return factor(M.T.dot(b)).toarray()


def distance_loss(D, U, W=None):
    if W is None:
        return D, U
    return W.dot(D), W.dot(U)


def stiffness_loss(kron_M_G, alpha_stiffness):
    return alpha_stiffness * kron_M_G, sparse.lil_matrix(
        (kron_M_G.shape[0], 3), dtype=np.float32
    )


# Non-Ridid ICP Registratoin
def nricp(
    source,
    target,
    threshold=1e8,
    alphas=np.linspace(50, 1, 20),
    gamma=1,
    eps=1e-2,
    coverage=False,
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

    n_max_inner = 50
    for num, alpha_stiffness in enumerate(alphas):
        for i in range(n_max_inner):
            X_prev = X

            transformed = D.dot(X)

            A, B = [], []

            # 1. Distance loss
            distances, indices = knn.kneighbors(transformed)
            distances, indices = distances.squeeze(), indices.squeeze()
            W = sparse.diags((distances < threshold).astype(np.float32))  # weight
            U = target_vertices[indices]
            A_d, B_d = distance_loss(D, U, W)
            A.append(A_d)
            B.append(B_d)

            # 2. Stiffness loss
            A_s, B_s = stiffness_loss(kron_M_G, alpha_stiffness)
            A.append(A_s)
            B.append(B_s)

            # 3. Coverage loss
            if coverage:
                knn_c = neighbors.NearestNeighbors(
                    n_neighbors=1, algorithm="kd_tree"
                ).fit(transformed)
                distances_c, indices_c = knn_c.kneighbors(target_vertices)
                distances_c, indices_c = distances_c.squeeze(), indices_c.squeeze()
                D_c = D[indices_c]
                U_c = target_vertices
                A_c, B_c = distance_loss(D_c, U_c)
                A.append(A_c)
                B.append(B_c)

            # Equation: E(X) = || AX - B ||^2
            X = cholesky_solve(
                sparse.csr_matrix(sparse.vstack(A)),
                sparse.csr_matrix(sparse.vstack(B)),
            )

            if np.mean(np.linalg.norm(X - X_prev, axis=1)) < eps:
                break

    transformed = D.dot(X)
    source.vertices = open3d.utility.Vector3dVector(transformed)
    return source
