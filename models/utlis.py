import numpy as np
import torch
import os
import torch.distributed as dist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def kernel_point_optimization_debug(radius, num_points,
                                    num_kernels=1, dimension=3,
                                    fixed='center', ratio=1.0, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while (kernel_points.shape[0] < num_kernels * num_points):
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initiate figure
    # if verbose>1:
    #     fig = plt.figure()

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10 * kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        if verbose:
            print('iter {:5d} / max grad = {:f}'.format(iter, np.max(gradients_norms[:, 3:])))
        # if verbose > 1:
        #     plt.clf()
        #     plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')
        #     circle = plt.Circle((0, 0), radius, color='r', fill=False)
        #     fig.axes[0].add_artist(circle)
        #     fig.axes[0].set_xlim((-radius*1.1, radius*1.1))
        #     fig.axes[0].set_ylim((-radius*1.1, radius*1.1))
        #     fig.axes[0].set_aspect('equal')
        #     plt.draw()
        #     plt.pause(0.001)
        #     plt.show(block=False)
        #     print(moving_factor)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def create_kernel_points(radius, num_kpoints, num_kernels, dimension, fixed):
    # Number of tries in the optimization process, to ensure we get the most stable disposition
    num_tries = 100

    # Kernel directory
    log_dir = os.environ.get("JOB_LOG_DIR", None)
    load_dir = os.environ.get("JOB_LOAD_DIR", None)
    if load_dir is not None:
        kernel_dir = os.path.join(load_dir, 'kernels', 'dispositions')
    elif log_dir is not None:
        kernel_dir = os.path.join(log_dir, 'kernels', 'dispositions')
    else:
        kernel_dir = os.path.join(ROOT_DIR, 'kernels', 'dispositions')
    if not os.path.exists(kernel_dir):
        os.makedirs(kernel_dir, exist_ok=True)

    prefix_name = 'pt'

    if dimension == 3:
        specific_kernel_file = os.path.join(kernel_dir,
                                            'sk_{}_{:04f}_{:03d}_{:s}.npy'.format(prefix_name, radius, num_kpoints,
                                                                                  fixed))
    elif dimension == 2:
        specific_kernel_file = os.path.join(kernel_dir,
                                            'sk_{}_{:04f}_{:03d}_{:s}_2D.npy'.format(prefix_name, radius, num_kpoints,
                                                                                     fixed))
    else:
        raise ValueError('Unsupported dimpension of kernel : ' + str(dimension))

    if os.path.exists(specific_kernel_file):
        kernels = np.load(specific_kernel_file)
        print(f"============================>{specific_kernel_file} loaded <=======================")
        # print(kernels)
    elif dist.get_rank() != 0:
        while True:
            if os.path.exists(specific_kernel_file + '_finish'):
                kernels = np.load(specific_kernel_file)
                # print(f"============================>r{radius} rank{dist.get_rank()} loaded <=====================")
                # print(kernels)
                break
    else:
        # Kernel_file
        if dimension == 3:
            kernel_file = os.path.join(kernel_dir, 'k_{:03d}_{:s}.ply'.format(num_kpoints, fixed))
        elif dimension == 2:
            kernel_file = os.path.join(kernel_dir, 'k_{:03d}_{:s}_2D.ply'.format(num_kpoints, fixed))
        else:
            raise ValueError('Unsupported dimpension of kernel : ' + str(dimension))

        # Check if already done
        if not os.path.exists(kernel_file):

            # Create kernels
            kernel_points, grad_norms = kernel_point_optimization_debug(1.0,
                                                                        num_kpoints,
                                                                        num_kernels=num_tries,
                                                                        dimension=dimension,
                                                                        fixed=fixed,
                                                                        verbose=0)

            # Find best candidate
            best_k = np.argmin(grad_norms[-1, :])

            # Save points
            original_kernel = kernel_points[best_k, :, :]
            np.save(kernel_file, original_kernel)
        else:
            original_kernel = np.load(kernel_file)

        # N.B. 2D kernels are not supported yet
        if dimension == 2:
            return original_kernel

        # Random rotations depending of the fixed points
        if fixed == 'verticals':

            # Create random rotations
            thetas = np.random.rand(num_kernels) * 2 * np.pi
            c, s = np.cos(thetas), np.sin(thetas)
            R = np.zeros((num_kernels, 3, 3), dtype=np.float32)
            R[:, 0, 0] = c
            R[:, 1, 1] = c
            R[:, 2, 2] = 1
            R[:, 0, 1] = s
            R[:, 1, 0] = -s

            # Scale kernels
            original_kernel = radius * np.expand_dims(original_kernel, 0)

            # Rotate kernels
            kernels = np.matmul(original_kernel, R)

        else:

            # Create random rotations
            u = np.ones((num_kernels, 3))
            v = np.ones((num_kernels, 3))
            wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99
            while np.any(wrongs):
                new_u = np.random.rand(num_kernels, 3) * 2 - 1
                new_u = new_u / np.expand_dims(np.linalg.norm(new_u, axis=1) + 1e-9, -1)
                u[wrongs, :] = new_u[wrongs, :]
                new_v = np.random.rand(num_kernels, 3) * 2 - 1
                new_v = new_v / np.expand_dims(np.linalg.norm(new_v, axis=1) + 1e-9, -1)
                v[wrongs, :] = new_v[wrongs, :]
                wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99

            # Make v perpendicular to u
            v -= np.expand_dims(np.sum(u * v, axis=1), -1) * u
            v = v / np.expand_dims(np.linalg.norm(v, axis=1) + 1e-9, -1)

            # Last rotation vector
            w = np.cross(u, v)
            R = np.stack((u, v, w), axis=-1)

            # Scale kernels
            original_kernel = radius * np.expand_dims(original_kernel, 0)

            # Rotate kernels
            kernels = np.matmul(original_kernel, R)

            # Add a small noise
            kernels = kernels
            kernels = kernels + np.random.normal(scale=radius * 0.01, size=kernels.shape)

        np.save(specific_kernel_file, kernels)
        with open(specific_kernel_file + '_finish', 'w') as fo:
            fo.write('finish!')
            fo.close()
        # print(f"=================================>r{radius} rank{dist.get_rank()} saved <=========================")
        # print(kernels)
    return kernels


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * torch.pow(sig, 2) + eps))


def weight_variable(size):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/21
    initial = np.random.normal(scale=np.sqrt(2 / size[-1]), size=size)
    initial[initial > 2 * np.sqrt(2 / size[-1])] = 0  # truncated
    initial[initial < -2 * np.sqrt(2 / size[-1])] = 0  # truncated
    weight = torch.nn.Parameter(torch.from_numpy(initial).float(), requires_grad=True)
    return weight
