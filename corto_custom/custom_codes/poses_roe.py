import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import json

"""
Generate a CORTO spacecraft trajectory from relative orbital elements.

The script constructs a safety-ellipse-like relative trajectory in the RTN
frame, converts sampled camera positions to the LVLH convention used for the
CORTO scene, orients each camera toward the target, and writes the resulting
camera/body/Sun poses to a CORTO `geometry.json` file.

Quaternions written to CORTO are stored in scalar-first order [w, x, y, z].
"""


def RTN_from_ROE(ROE, t0, n, v, t, a):
    """
    Convert relative orbital elements to an RTN relative state at time `t`.

    ROE = [delta_a/a, delta_lambda_0, delta_ex, delta_ey, delta_ix, delta_iy]
    and the returned state is [R, T, N, v_R, v_T, v_N].
    """

    tau = n * (t - t0)
    u0 = 0
    u = u0 + tau

    c = np.cos(u)
    s = np.sin(u)

    delta_a_norm, delta_lambda_0, delta_ex, delta_ey, delta_ix, delta_iy = ROE

    delta_r_R_norm = delta_a_norm - delta_ex*c - delta_ey*s
    delta_r_T_norm = delta_lambda_0 - 3/2*delta_a_norm*(u - u0) - 2*delta_ey*c + 2*delta_ex*s
    delta_r_N_norm = -delta_iy*c + delta_ix*s
    delta_r = np.array([delta_r_R_norm, delta_r_T_norm, delta_r_N_norm])*a

    delta_v_R_norm = delta_ex*s - delta_ey*c
    delta_v_T_norm = -3/2*delta_a_norm + 2*delta_ex*c + 2*delta_ey*s
    delta_v_N_norm = delta_ix*c + delta_iy*s
    delta_v = np.array([delta_v_R_norm, delta_v_T_norm, delta_v_N_norm])*v

    x = np.concatenate((delta_r, delta_v))
    return x
    

def safety_ellipse(a, ROE_0, plot_trajectory=True):
    """Propagate one orbit of relative motion and optionally plot diagnostics."""
    mu = 3.986e14  # Earth's gravitational parameter in m^3/s^2

    # Target circular-orbit parameters.
    n = np.sqrt(mu / a**3) # Mean motion in rad/s.

    r_ECI = np.array([a, 0, 0])
    v_ECI = np.array([0, np.sqrt(mu/a), 0])

    # RTN frame: R is radial outward, T is along-track, N is orbit-normal.

    e_R = r_ECI/np.linalg.norm(r_ECI)
    e_N = np.cross(r_ECI, v_ECI)/np.linalg.norm(np.cross(r_ECI, v_ECI))
    e_T = np.cross(e_N, e_R)

    t0 = 0
    T = 2*np.pi/n  # orbital period in s

    ROE_0 = np.array(ROE_0, dtype=float)
    x = []
    time = np.linspace(0, T, 100)

    for t in time:
        x.append(RTN_from_ROE(ROE_0, t0, n, np.linalg.norm(v_ECI), t, a))

    x = np.array(x)

    if plot_trajectory is True:
        plt.figure(figsize=(12, 8))
        plt.plot(x[:, 0], x[:, 1], label='Relative Trajectory')
        plt.xlabel('Radial (R) [m]')
        plt.ylabel('Along-track (T) [m]')
        plt.title('Relative Trajectory in RTN Frame')
        plt.grid()
        plt.axis('equal')
        plt.legend()
        plt.show()

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x[:,0], x[:,1], x[:,2], label='Chaser trajectory')

        ax.scatter(0, 0, 0, color='red', s=100, label='Target')

        # Reference axes through the target.
        ax.plot([0,0],[0,0],[min(x[:,2]), max(x[:,2])], linestyle='--')
        ax.plot([min(x[:,0]), max(x[:,0])],[0,0],[0,0], linestyle='--')
        ax.plot([0,0],[min(x[:,1]), max(x[:,1])],[0,0], linestyle='--')

        ax.set_xlabel('Radial R [m]')
        ax.set_ylabel('Along-track T [m]')
        ax.set_zlabel('Normal N [m]')

        ax.set_title('Relative Orbit in RTN Frame')
        ax.legend()

        # Equal scale on all 3D axes.
        max_range = np.array([
            x[:,0].max()-x[:,0].min(),
            x[:,1].max()-x[:,1].min(),
            x[:,2].max()-x[:,2].min()
        ]).max()/2

        mid_x = (x[:,0].max()+x[:,0].min())*0.5
        mid_y = (x[:,1].max()+x[:,1].min())*0.5
        mid_z = (x[:,2].max()+x[:,2].min())*0.5

        ax.set_xlim(mid_x-max_range, mid_x+max_range)
        ax.set_ylim(mid_y-max_range, mid_y+max_range)
        ax.set_zlim(mid_z-max_range, mid_z+max_range)

        plt.show()
    return x, time


def camera_poses(positions, plot_poses=True):
    """
    Build camera poses from LVLH positions, with cameras looking at the target.

    The returned pose array stores position followed by scalar-first quaternion
    [w, x, y, z], matching the CORTO geometry convention.
    """

    positions = np.asarray(positions, dtype=float)

    # Up direction is the negative H-bar axis in this LVLH convention.
    up = np.array([0, -1, 0])

    poses = []
    rot_mats = []

    for p in positions:
        p_norm = np.linalg.norm(p)

        # The camera looks toward the target, so +Z_cam points away from it.
        z_cam = p / p_norm 
        x_cam = np.cross(up, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam /= np.linalg.norm(y_cam)

        R_cam_to_world = np.column_stack((x_cam, y_cam, z_cam))

        q_cam_to_world = R.from_matrix(R_cam_to_world).as_quat()  # [qx, qy, qz, qw]
        q = (q_cam_to_world[3], q_cam_to_world[0], q_cam_to_world[1], q_cam_to_world[2])
        q_cam_to_world = np.array(q)

        poses.append(np.hstack((p, q_cam_to_world)))
        rot_mats.append(R_cam_to_world)

    poses = np.array(poses)
    rot_mats = np.array(rot_mats)

    if plot_poses:

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                   color='blue', label='Cameras')

        ax.scatter(0,0,0,color='red',s=120,label='Target')

        for p, Rcw in zip(positions, rot_mats):

            view_dir = -Rcw[:,2]  # camera viewing direction

            ax.quiver(
                p[0], p[1], p[2],
                view_dir[0], view_dir[1], view_dir[2],
                length=10,
                normalize=True,
                color='green'
            )

        # Camera trajectory line.
        ax.plot(positions[:,0], positions[:,1], positions[:,2],
                linestyle='--', alpha=0.5)

        ax.set_xlabel("V-BAR (x)")
        ax.set_ylabel("-H-BAR (y)")
        ax.set_zlabel("R-BAR (z)")

        ax.set_title("Camera Poses and Viewing Directions")
        ax.legend()

        # Equal scale on all 3D axes.
        max_range = np.ptp(positions, axis=0).max()/2
        mid = positions.mean(axis=0)

        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

        plt.show()


    return poses, rot_mats

def target_rotation(t_samples, w_body, axis_vec):
    """Generate body quaternions for a constant angular velocity around an axis."""
    t_samples = np.asarray(t_samples, dtype=float)

    axis_vec = np.asarray(axis_vec, dtype=float)   
    axis_vec /= np.linalg.norm(axis_vec)            

    R0 = R.identity()

    body_quaternions = []
    for t in t_samples:
        angle = w_body * t
        R_t = R.from_rotvec(angle * axis_vec)
        q_t = (R_t * R0).as_quat()
        q = (q_t[3], q_t[0], q_t[1], q_t[2])
        q_t = np.array(q)
        body_quaternions.append(q_t.tolist())

    return body_quaternions

def RTN_to_LVLH(trajectory):
    """
    Convert RTN positions to the LVLH axes used by the CORTO scenario.

    LVLH axes:
    - x: V-bar, aligned with +T;
    - y: H-bar, aligned with -N;
    - z: R-bar, aligned with -R toward Earth.
    """

    trajectory_RTN = np.asarray(trajectory, dtype=float)
    R_RTN_to_LVLH = np.array([[0, 1, 0],
                              [0, 0, -1],
                              [-1, 0, 0]])
    trajectory_LVLH = trajectory_RTN @ R_RTN_to_LVLH.T

    return trajectory_LVLH

def sun_behind_camera(camera_positions, camera_rot_mats, sun_distance=3):
    """Place the Sun behind each camera along the opposite viewing direction."""
    sun_positions = []
    sun_orientations = []
    
    for pos, Rcw in zip(camera_positions, camera_rot_mats):
        view_dir = -Rcw[:, 2]  # camera looks along -Z_cam in world
        sun_pos = pos - sun_distance * view_dir  # behind the camera
        sun_positions.append(sun_pos.tolist())
        
        # Match camera orientation so the Sun local -Z points toward the target.
        q = R.from_matrix(Rcw).as_quat()  # [qx, qy, qz, qw]
        q_wxyz = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        sun_orientations.append(q_wxyz)
    
    return sun_positions, sun_orientations


# ============================================================
# Scenario configuration
# ============================================================

R_E = 6371e3  # Earth radius in m
h = 500e3     # altitude in m
a = R_E + h  # semi-major axis in m

# Safety ellipse ROE parameters.
delta_a = 0
delta_lambda_0 = 0 / a
delta_ex = 0 / a
delta_ey = 15 / a
delta_ix = 0 / a
delta_iy = 15 / a


# Secondary trajectory ROE parameters.
#delta_a = 0
#delta_lambda_0 = 0 / a
#delta_ex = 5 / a
#delta_ey = 15 / a
#delta_ix = -10 / a
#delta_iy = 15 / a

ROE_0 = [delta_a/a, delta_lambda_0, delta_ex, delta_ey, delta_ix, delta_iy]
trajectory, time = safety_ellipse(a, ROE_0, plot_trajectory=True)

# Camera poses in LVLH frame.
N_cameras = 100
sampled_idx = np.linspace(0, len(trajectory)-1, N_cameras, dtype=int)

sampled_times = time[sampled_idx]

camera_positions_RTN = trajectory[sampled_idx, :3]
camera_positions = RTN_to_LVLH(camera_positions_RTN)
poses, rot_mats = camera_poses(camera_positions)

position_camera = poses[:, :3]
orientation_camera = poses[:, 3:]

# Body poses in LVLH frame.
body_position = [0, 0, 0]  # Target at the origin in LVLH frame
body_position = [body_position for _ in range(N_cameras)]

w_body = np.deg2rad(0.0)
body_orientation = target_rotation(sampled_times, w_body, axis_vec=[1, 0, 0])

# Sun direction examples in LVLH frame.

# Frame 30
#sun_position = [11.454296579136235, -5.727148289568118, 16.547508253780414]
#sun_orientation = [0.041196401836761584, 0.29527501802266465, -0.13189642990487144, 0.9453670461708344]

# Frame 60
#sun_position = [26.112891737705336, -13.056445868852668, -10.267702520241535]
#sun_orientation = [0.17854235299508195, 0.8069007163465081, -0.12164506476453522, 0.5497602571711632]

# Frame 70
#sun_position = [9.405184040057277, -4.702592020028638, -17.009387829317753]
#sun_orientation = [0.11465767141019656, 0.961466119617321, -0.02958844337093845, 0.24811497986556103]

# Frame 80
#sun_position = [-12.449471239119474, 6.224735619559737, -16.293213657257493]
#sun_orientation = [-0.13908756996937627, 0.9369907825469415, -0.04705562986741773, -0.31699950946312944]

#sun_position = [sun_position for _ in range(N_cameras)]
#sun_orientation = [sun_orientation for _ in range(N_cameras)]



# Final Sun placement: behind each camera along the viewing direction.
sun_position, sun_orientation = sun_behind_camera(position_camera, rot_mats, sun_distance=3)

# CORTO geometry output.
geometry = {
        "sun": {
            "position": sun_position,
            "orientation": sun_orientation
        },
        "camera": {
            "position": position_camera.tolist(),
            "orientation": orientation_camera.tolist()
        },
        "body": {
            "position": body_position,
            "orientation": body_orientation  
        }
    }

with open("/Users/martino/Desktop/Tesi/codes/corto/input/S10_Spacecraft_Complex_Sat/geometry/geometry.json", "w") as f:
       json.dump(geometry, f, indent=4) 
