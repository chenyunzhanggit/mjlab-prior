import numpy as np

def quat_conjugate(q):
    """Computes the conjugate of a quaternion.
    Args:
        q: (..., 4) array in (w, x, y, z) format
    Returns:
        (..., 4) array
    """
    q = np.asarray(q)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)

def quat_inv(q, eps=1e-9):
    """Computes the inverse of a quaternion.
    Args:
        q: (..., 4) array
        eps: small value to avoid division by zero
    Returns:
        (..., 4) array
    """
    q = np.asarray(q)
    norm_sq = np.sum(q**2, axis=-1, keepdims=True)
    return quat_conjugate(q) / np.clip(norm_sq, eps, None)

def quat_mul(q1, q2):
    """Multiply two quaternions together.
    Args:
        q1: (..., 4) array
        q2: (..., 4) array
    Returns:
        (..., 4) array
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def matrix_from_quat(q):
    """Convert quaternion to rotation matrix.
    Args:
        q: (..., 4) array
    Returns:
        (..., 3, 3) array
    """
    q = np.asarray(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    mat = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    mat[..., 0, 0] = 1 - 2*(yy + zz)
    mat[..., 0, 1] = 2*(xy - wz)
    mat[..., 0, 2] = 2*(xz + wy)
    mat[..., 1, 0] = 2*(xy + wz)
    mat[..., 1, 1] = 1 - 2*(xx + zz)
    mat[..., 1, 2] = 2*(yz - wx)
    mat[..., 2, 0] = 2*(xz - wy)
    mat[..., 2, 1] = 2*(yz + wx)
    mat[..., 2, 2] = 1 - 2*(xx + yy)
    return mat

def quat_apply(q, v):
    """Apply quaternion rotation to vector(s).
    Args:
        q: (..., 4) array
        v: (..., 3) array
    Returns:
        (..., 3) array
    """
    q = np.asarray(q)
    v = np.asarray(v)
    rot = matrix_from_quat(q)
    return np.einsum('...ij,...j->...i', rot, v)

def subtract_frame_transforms(t01, q01, t02=None, q02=None):
    """Subtract transformations between two reference frames into a stationary frame.
    Args:
        t01: (..., 3) array
        q01: (..., 4) array
        t02: (..., 3) array or None
        q02: (..., 4) array or None
    Returns:
        t12: (..., 3) array
        q12: (..., 4) array
    """
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12