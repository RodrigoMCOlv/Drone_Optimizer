import sys
import numpy as np
import cma
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import signal
import json
import multiprocessing
import time
import os
import re

# --- Target Maneuver Settings ---
TARGET_POS_X = 1.0
TARGET_POS_Y = 0.5
TARGET_POS_Z_START = 1.0
TARGET_POS_Z_END = 1.5
TARGET_PITCH_DEG = 0.0
TARGET_ROLL_DEG = 0.0
TARGET_YAW_DEG = 45.0

model = None
data = None
drone_config = {}
B_pinv = None
drone_body_id = None
num_motors = 0
motors = []
max_torque_available = None

def init_worker(xml_path):
    global model, data, drone_config, B_pinv, drone_body_id, num_motors, motors, max_torque_available
    """Ignore SIGINT in worker processes so the main process can handle Ctrl+C cleanly."""
    if multiprocessing.current_process().name != 'MainProcess':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    try:
        xml_content = open(xml_path, "r", encoding="utf-8").read()
        if xml_path.endswith('.sdf'):
            import re
            match = re.search(r'<mujoco_xml>(.*?)</mujoco_xml>', xml_content, re.DOTALL)
            if not match:
                print("Error: No embedded <mujoco_xml> found in SDF file.")
                sys.exit(1)
            mjcf_string = match.group(1).strip()
            model = mujoco.MjModel.from_xml_string(mjcf_string)
        else:
            model = mujoco.MjModel.from_xml_string(xml_content)
    except ValueError as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    num_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "transport_delay_ms")
    transport_delay_ms = model.numeric_data[model.numeric_adr[num_id1]] if num_id1 != -1 else 10.0
    
    num_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "motor_tau")
    motor_tau = model.numeric_data[model.numeric_adr[num_id2]] if num_id2 != -1 else 0.085
    
    drone_config["transport_delay_ms"] = transport_delay_ms
    drone_config["motor_tau"] = motor_tau
    
    drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    drone_pos = data.xipos[drone_body_id]
    
    num_motors = model.nu
    motors = []
    
    num_id_dof = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "dof_mask")
    if num_id_dof != -1:
        adr = model.numeric_adr[num_id_dof]
        size = model.numeric_size[num_id_dof]
        dof_mask = model.numeric_data[adr:adr+size].copy()
        if len(dof_mask) < 6:
            dof_mask = np.pad(dof_mask, (0, 6 - len(dof_mask)), constant_values=0)
    else:
        dof_mask = np.array([0, 0, 1, 1, 1, 1])
    drone_config["dof_mask"] = dof_mask
    
    B_full = np.zeros((6, num_motors))
    
    max_thrust_all = np.max(model.actuator_ctrlrange[:, 1])
    
    for i in range(num_motors):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        act_id = i
        site_id = model.actuator_trnid[act_id, 0]
        
        pos_rel = data.site_xpos[site_id] - drone_pos
        site_mat = data.site_xmat[site_id].reshape(3, 3)
        z_axis = site_mat @ np.array([0.0, 0.0, 1.0])
        
        c_m = model.actuator_gear[act_id, 5]
        force = z_axis.copy()
        torque = np.cross(pos_rel, z_axis) + c_m * z_axis
        
        # Heuristic: If an actuator is < 20% as strong as the main lifter, it's a maneuvering thruster.
        if model.actuator_ctrlrange[act_id, 1] < 0.2 * max_thrust_all:
            force[2] = 0.0
            
        B_full[0:3, i] = force
        B_full[3:6, i] = torque

        # Parse prop inertia from the motor mount body
        mount_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"motor_mount_{i}")
        prop_inertia = 0.0
        if mount_id != -1:
            prop_inertia = np.max(model.body_inertia[mount_id])
            
        # Parse Performance Data
        rpms = []
        thrusts = []
        try:
            match = re.search(r'<!--\s*Motor\s*' + str(i) + r':.*?\[Data:\s*(.*?)\s*Unit:\s*([^\]]+)\].*?-->', xml_content, re.IGNORECASE)
            if match:
                data_str = match.group(1)
                unit_str = match.group(2).strip()
                if data_str.strip().lower() != "none":
                    for point in data_str.split(';'):
                        parts = point.strip().replace(',', ' ').split()
                        if len(parts) >= 2:
                            try:
                                rpm_val = float(parts[0])
                                thrust_val = float(parts[1])
                                if unit_str == 'gf': thrust_val *= 0.00980665
                                elif unit_str == 'kgf': thrust_val *= 9.80665
                                rpms.append(rpm_val)
                                thrusts.append(thrust_val)
                            except ValueError:
                                pass
        except Exception:
            pass

        motors.append({
            "name": act_name,
            "pos": pos_rel.tolist(),
            "z_axis": z_axis.tolist(),
            "c_m": c_m,
            "spin": -1 if c_m > 0 else 1,
            "inertia": prop_inertia,
            "rpms": rpms,
            "thrusts": thrusts
        })

    active_dofs = np.where(dof_mask == 1)[0]
    B_active = B_full[active_dofs, :]
    B_pinv = np.linalg.pinv(B_active, rcond=1e-4)
    
    # Calculate available dynamic torque limits (approximate maximum torque per axis)
    max_thrusts = model.actuator_ctrlrange[:, 1]
    max_torque_available = np.sum(np.abs(B_full[3:6, :]) * max_thrusts, axis=1) / 2.0

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def apply_quat(q, v):
    vq = np.array([0.0, v[0], v[1], v[2]])
    res = quat_mult(quat_mult(q, vq), quat_conj(q))
    return res[1:]

def evaluate_flight(params):
    Kp = params[0:4]
    Ki = params[4:8]
    Kd = params[8:12]
    Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y = params[12:18]
    max_int = params[18:24]
    
    mixer = B_pinv
    total_loss = 0.0
    dt = model.opt.timestep
    
    num_id_dur = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "sim_duration_s")
    sim_duration_s = model.numeric_data[model.numeric_adr[num_id_dur]] if num_id_dur != -1 else 5.0
    steps = int(sim_duration_s / dt) 
    delay_steps = int((drone_config["transport_delay_ms"] / 1000.0) / dt)
    delay_steps = max(1, delay_steps)
    alpha_motor = dt / (drone_config["motor_tau"] + dt)

    test_cases = [
        (1.5, 1.5, 90.0),
        (-1.0, -1.0, -90.0),
        (0.0, 1.5, 180.0),
        (1.0, -0.5, 45.0)
    ]
    
    max_thrusts = model.actuator_ctrlrange[:, 1]
    min_thrusts = model.actuator_ctrlrange[:, 0]
    
    for target_x, target_y, target_yaw_deg in test_cases:
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = [0.0, 0.0, TARGET_POS_Z_START]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        
        nv = model.nv
        M_dense = np.zeros((nv, nv))
        mujoco.mj_fullM(model, data, M_dense)
        mass = M_dense[0, 0]
        I_body_fixed = M_dense[3:6, 3:6]
        
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = [0.0, 0.0, TARGET_POS_Z_START]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        
        loss = 0.0
        integral_error_z = 0.0
        integral_error_ang = np.zeros(3)
        integral_error_body_x = 0.0
        integral_error_body_y = 0.0
        prev_thrusts = np.zeros(num_motors)
        
        prev_rpms = np.zeros(num_motors)
        for i in range(num_motors):
            if len(motors[i]["rpms"]) >= 2:
                prev_rpms[i] = motors[i]["rpms"][0]
        
        base_thrusts = mixer @ np.array([mass * 9.81, 0.0, 0.0, 0.0])
        safe_max_thrusts = max_thrusts + 1e-6
        actual_motor_pwm_arr = np.sign(base_thrusts) * np.sqrt(np.abs(base_thrusts) / safe_max_thrusts)
        action_queue = [actual_motor_pwm_arr.copy()] * delay_steps
        
        for step in range(steps):
            t = step * dt
            t_rise = 2.0
            
            if t < t_rise:
                tau = t / t_rise
                s = 10*tau**3 - 15*tau**4 + 6*tau**5
                ds = (30*tau**2 - 60*tau**3 + 30*tau**4) / t_rise
                target_pos = np.array([target_x * s, target_y * s, TARGET_POS_Z_START + (TARGET_POS_Z_END - TARGET_POS_Z_START) * s])
                target_vel = np.array([target_x * ds, target_y * ds, (TARGET_POS_Z_END - TARGET_POS_Z_START) * ds])
                current_target_yaw = np.radians(target_yaw_deg) * s
                target_ang_vel = np.array([0.0, 0.0, np.radians(target_yaw_deg) * ds])
            else:
                target_pos = np.array([target_x, target_y, TARGET_POS_Z_END])
                target_vel = np.zeros(3)
                current_target_yaw = np.radians(target_yaw_deg)
                target_ang_vel = np.zeros(3)
                
            # Clean trajectory for training (no chirp)
                
            pos = data.qpos[:3].copy()
            quat = data.qpos[3:7].copy()
            vel = data.qvel[:3].copy()
            ang_vel = data.qvel[3:6].copy()
            
            up_world = apply_quat(quat, np.array([0.0, 0.0, 1.0]))
            if up_world[2] < 0.5 or pos[2] < 0.02 or np.any(np.isnan(pos)):
                loss += 1e5 + (steps - step) * 10.0
                break
                
            pos_err = target_pos - pos
            vel_err = target_vel - vel
            
            # Rotate world errors into body yaw frame
            w, x_quat, y_quat, z_quat = quat
            current_yaw_actual = np.arctan2(2*(w*z_quat + x_quat*y_quat), 1 - 2*(y_quat**2 + z_quat**2))
            cos_y = np.cos(current_yaw_actual)
            sin_y = np.sin(current_yaw_actual)
            
            pos_err_body_x = pos_err[0] * cos_y + pos_err[1] * sin_y
            pos_err_body_y = -pos_err[0] * sin_y + pos_err[1] * cos_y
            vel_err_body_x = vel_err[0] * cos_y + vel_err[1] * sin_y
            vel_err_body_y = -vel_err[0] * sin_y + vel_err[1] * cos_y
            
            dof_mask = drone_config["dof_mask"]
            max_tilt = np.radians(20.0)
            wrench_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
            
            if dof_mask[0] == 1:
                wrench_dict[0] = (Kp_x * pos_err_body_x + Kd_x * vel_err_body_x) * mass
                target_pitch = np.radians(TARGET_PITCH_DEG)
            else:
                integral_error_body_x += pos_err_body_x * dt
                integral_error_body_x = np.clip(integral_error_body_x, -max_int[4], max_int[4])
                accel_x_cmd = Kp_x * pos_err_body_x + Ki_x * integral_error_body_x + Kd_x * vel_err_body_x
                # Positive pitch = +X body acceleration
                target_pitch = np.clip(accel_x_cmd / 9.81, -max_tilt, max_tilt)
                
            if dof_mask[1] == 1:
                wrench_dict[1] = (Kp_y * pos_err_body_y + Kd_y * vel_err_body_y) * mass
                target_roll = np.radians(TARGET_ROLL_DEG)
            else:
                integral_error_body_y += pos_err_body_y * dt
                integral_error_body_y = np.clip(integral_error_body_y, -max_int[5], max_int[5])
                accel_y_cmd = Kp_y * pos_err_body_y + Ki_y * integral_error_body_y + Kd_y * vel_err_body_y
                # Positive roll = roll right = +Y body acceleration (Wait, left +Y, roll right is negative Y... let's re-verify:
                # Right hand rule: Thumb along X (forward). Fingers curl from Y (left) to Z (up).
                # So +Roll rotates Y (left) towards Z (up). The left side goes up. The right side goes down. 
                # Thrust vector points to the left (+Y body). So +Roll gives +Y acceleration.
                target_roll = np.clip(-accel_y_cmd / 9.81, -max_tilt, max_tilt)

            cr, sr = np.cos(target_roll * 0.5), np.sin(target_roll * 0.5)
            cp, sp = np.cos(target_pitch * 0.5), np.sin(target_pitch * 0.5)
            cy, sy = np.cos(current_target_yaw * 0.5), np.sin(current_target_yaw * 0.5)

            target_quat = np.array([
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy
            ])
            
            q_err = quat_mult(quat_conj(quat), target_quat)
            if q_err[0] < 0:
                q_err = -q_err
            ang_err = 2.0 * q_err[1:]
            body_mat = data.xmat[drone_body_id].reshape(3, 3)
            target_ang_vel_body = body_mat.T @ target_ang_vel
            ang_vel_err = target_ang_vel_body - ang_vel
            
            integral_error_z += pos_err[2] * dt
            integral_error_z = np.clip(integral_error_z, -max_int[0], max_int[0])
            
            desired_accel_z = Kp[0] * pos_err[2] + Ki[0] * integral_error_z + Kd[0] * vel_err[2]
            up_world = body_mat @ np.array([0.0, 0.0, 1.0])
            desired_force_z = (desired_accel_z + 9.81) * mass / max(0.5, up_world[2])
            
            integral_error_ang += ang_err * dt
            integral_error_ang = np.clip(integral_error_ang, -max_int[1:4], max_int[1:4])
            
            desired_ang_accel = Kp[1:] * ang_err + Ki[1:] * integral_error_ang + Kd[1:] * ang_vel_err
            
            I_body = I_body_fixed
            gyro_term = np.cross(ang_vel, I_body @ ang_vel)
            desired_torque_body = I_body @ desired_ang_accel + gyro_term
            
            wrench_dict[2] = desired_force_z
            wrench_dict[3] = desired_torque_body[0]
            wrench_dict[4] = desired_torque_body[1]
            wrench_dict[5] = desired_torque_body[2]
            
            active_dofs = np.where(dof_mask == 1)[0]
            
            # Attitude wrench (Roll, Pitch, Yaw)
            att_wrench = np.array([wrench_dict[dof] if dof >= 3 else 0.0 for dof in active_dofs])
            thrusts_att = B_pinv @ att_wrench
            
            max_att = np.max(np.abs(thrusts_att) / (safe_max_thrusts + 1e-6))
            if max_att > 1.0:
                thrusts_att /= max_att
                
            # Position wrench (X, Y, Z)
            pos_wrench = np.array([wrench_dict[dof] if dof < 3 else 0.0 for dof in active_dofs])
            thrusts_pos = B_pinv @ pos_wrench
            
            scale_pos = 1.0
            for i in range(num_motors):
                if thrusts_pos[i] > 0:
                    margin = max_thrusts[i] - thrusts_att[i]
                    if margin < 0: margin = 0
                    if thrusts_pos[i] > margin:
                        scale_pos = min(scale_pos, margin / thrusts_pos[i])
                elif thrusts_pos[i] < 0:
                    margin = min_thrusts[i] - thrusts_att[i]
                    if margin > 0: margin = 0
                    if thrusts_pos[i] < margin:
                        scale_pos = min(scale_pos, margin / thrusts_pos[i])
                        
            thrusts = thrusts_att + scale_pos * thrusts_pos
            thrusts = np.clip(thrusts, min_thrusts, max_thrusts) 
            
            pwm_cmd = np.sign(thrusts) * np.sqrt(np.abs(thrusts) / safe_max_thrusts)
            
            action_queue.append(pwm_cmd)
            delayed_pwm = action_queue.pop(0)
            
            actual_motor_pwm_arr = (1.0 - alpha_motor) * actual_motor_pwm_arr + alpha_motor * delayed_pwm
            
            physical_thrusts = np.sign(actual_motor_pwm_arr) * (actual_motor_pwm_arr ** 2) * max_thrusts
            
            data.ctrl[:] = physical_thrusts
            
            # Simulate RPM transient torque (RPM jerk)
            transient_torque_z = 0.0
            for i in range(num_motors):
                m = motors[i]
                if m["inertia"] > 0 and len(m["rpms"]) >= 2:
                    # np.interp requires strictly increasing x, thrusts array from XML should be increasing
                    target_rpm = np.interp(abs(physical_thrusts[i]), m["thrusts"], m["rpms"])
                    alpha_rpm = dt / (0.02 + dt)
                    current_rpm = (1.0 - alpha_rpm) * prev_rpms[i] + alpha_rpm * target_rpm
                    rpm_diff = current_rpm - prev_rpms[i]
                    prev_rpms[i] = current_rpm
                    
                    alpha = (rpm_diff * 2.0 * np.pi / 60.0) / dt
                    # T = - I * alpha * spin (CW is -1, CCW is +1)
                    t_jerk = - m["inertia"] * alpha * m["spin"]
                    transient_torque_z += t_jerk
                    
            if transient_torque_z != 0.0:
                body_mat = data.xmat[drone_body_id].reshape(3, 3)
                z_axis_world = body_mat @ np.array([0.0, 0.0, 1.0])
                data.xfrc_applied[drone_body_id, 3:6] = z_axis_world * transient_torque_z
                
            mujoco.mj_step(model, data)
            
            # Heavily penalize position error to force the drone to fly to the target
            # Reduce attitude penalty so it doesn't just hover safely to avoid attitude loss
            loss += (np.sum(pos_err[:2]**2)*100.0 + pos_err[2]**2 * 10.0 + np.sum(ang_err**2) * 5.0 + np.sum(ang_vel_err**2) * 0.5) * dt
            # Penalize high-frequency chatter but don't penalize smooth maneuvering
            loss += np.sum((thrusts - prev_thrusts)**2) * 0.05
            prev_thrusts = thrusts
            
        # Add a massive penalty if the final position is far from the target
        final_dist = np.linalg.norm(pos_err[:2])
        loss += final_dist * 1000.0
            
        total_loss += loss 

    return total_loss

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Universal UAV Control Optimizer")
    parser.add_argument("--model", type=str, required=True, help="Path to MuJoCo XML")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (default: dynamic based on cores)")
    args = parser.parse_args()

    # Initialize the globals for the main process so it can print/save matrix and visualize correctly
    init_worker(args.model)
    
    x0 = [
        150.0, 40.0, 40.0, 40.0,       # Kp (Z, Roll, Pitch, Yaw)
        1.0, 0.5, 0.5, 0.5,     # Ki
        30.0, 10.0, 10.0, 10.0,        # Kd
        1.5, 0.05, 1.5, 1.5, 0.05, 1.5, # Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y
        10.0, 10.0, 10.0, 10.0, 5.0, 5.0  # max_int z, r, p, y, x, y
    ]
    
    stds = [
        50.0, 20.0, 20.0, 20.0,
        0.5, 0.5, 0.5, 0.5,
        10.0, 5.0, 5.0, 5.0,
        1.0, 0.1, 1.0, 1.0, 0.1, 1.0,
        5.0, 5.0, 5.0, 5.0, 2.0, 2.0
    ]
    
    bounds_lower = [0.0] * 24
    
    bounds_upper = [
        300.0, 300.0, 300.0, 300.0,  # Kp
        50.0, 50.0, 50.0, 50.0,      # Ki
        150.0, 150.0, 150.0, 150.0,  # Kd
        30.0, 20.0, 30.0, 30.0, 20.0, 30.0, # Outer Kp, Ki, Kd
        300.0, 300.0, 300.0, 300.0, 100.0, 100.0 # max_int
    ]
    
    options = {
        'bounds': [bounds_lower, bounds_upper],
        'CMA_stds': stds,
        'maxiter': 1000,  
        'popsize': 24
    }
    
    es = cma.CMAEvolutionStrategy(x0, 1.0, options)
    
    cpu_usage_percent = 60.0
    num_cores = multiprocessing.cpu_count()
    if args.workers > 0:
        max_workers = args.workers
    else:
        max_workers = max(1, int(num_cores * (cpu_usage_percent / 100.0)))
    
    loss_history_best = []
    loss_history_avg = []
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line_best, = ax.plot([], [], 'b-', label='Best Loss')
    line_avg, = ax.plot([], [], 'r--', label='Avg Loss', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Loss (Log Scale)')
    ax.set_title('CMA-ES Optimization Progress (16D Decoupled PID)')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    print("Starting optimization. Press Ctrl+C to stop early and visualize the best flight so far.")
    print(f"Mixer matrix:\n{B_pinv}")
    
    gen = 0
    start_time = time.time()
    with multiprocessing.Pool(processes=max_workers, initializer=init_worker, initargs=(args.model,)) as pool:
        try:
            while not es.stop():
                gen_start_time = time.time()
                solutions = es.ask()
                fitnesses = pool.map(evaluate_flight, solutions)
                es.tell(solutions, fitnesses)
                
                best_loss = np.min(fitnesses)
                avg_loss = np.mean(fitnesses)
                loss_history_best.append(best_loss)
                loss_history_avg.append(avg_loss)
                
                gen_time = time.time() - gen_start_time
                total_time = time.time() - start_time
                
                print(f"Generation {gen:03d} | Best Loss: {best_loss:10.4f} | Avg Loss: {avg_loss:10.4f} | Gen Time: {gen_time:.2f}s | Total Time: {total_time:.2f}s")
                
                if gen % 10 == 0:
                    line_best.set_xdata(range(len(loss_history_best)))
                    line_best.set_ydata(loss_history_best)
                    line_avg.set_xdata(range(len(loss_history_avg)))
                    line_avg.set_ydata(loss_history_avg)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
                if gen >= 300 and len(loss_history_best) >= 20:
                    recent_losses = loss_history_best[-20:]
                    variation = (max(recent_losses) - min(recent_losses)) / (min(recent_losses) + 1e-6)
                    if variation < 0.001: # 0.1% variation
                        print(f"\nEarly stopping triggered: Loss variation over last 20 gens is under 0.1% ({variation*100:.4f}%).")
                        break
                    
                gen += 1
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Proceeding with the best parameters found so far...")
            pool.terminate()
            pool.join()
        
    if es.result.xbest is None:
        print("No generations completed. Exiting without saving.")
        return
        
    best_params = es.result.xbest
    best_Kp = best_params[0:4]
    best_Ki = best_params[4:8]
    best_Kd = best_params[8:12]
    best_outer = best_params[12:18]
    best_max_int = best_params[18:24]
    best_mixer = B_pinv
    
    print("\n" + "="*50)
    print("=== OPTIMIZATION COMPLETE ===")
    print("="*50)
    print("\nFrozen Geometric Mixer Matrix:")
    np.set_printoptions(precision=3, suppress=True)
    print(best_mixer)

    plt.ioff()
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    xml_name = os.path.splitext(os.path.basename(args.model))[0]
    
    evo_plot_path = os.path.join(out_dir, f"{xml_name}_evolution.png")
    plt.savefig(evo_plot_path)
    print(f"\n-> Saved evolution plot to '{evo_plot_path}'")
    plt.close(fig)

    print("\nOptimized PID Gains:")
    axes = ["Z", "Roll", "Pitch", "Yaw"]
    print(f"{'Axis':<6} | {'Kp':<8} | {'Ki':<8} | {'Kd':<8}")
    print("-" * 40)
    for i, axis in enumerate(axes):
        print(f"{axis:<6} | {best_Kp[i]:8.2f} | {best_Ki[i]:8.2f} | {best_Kd[i]:8.2f}")
        
    print(f"\nOuter Loop Gains:\nKp_x: {best_outer[0]:.2f} | Ki_x: {best_outer[1]:.2f} | Kd_x: {best_outer[2]:.2f}\nKp_y: {best_outer[3]:.2f} | Ki_y: {best_outer[4]:.2f} | Kd_y: {best_outer[5]:.2f}")
        
    print(f"\nMax Integrators:\nZ: {best_max_int[0]:.2f} | Roll: {best_max_int[1]:.2f} | Pitch: {best_max_int[2]:.2f} | Yaw: {best_max_int[3]:.2f}\nX: {best_max_int[4]:.2f} | Y: {best_max_int[5]:.2f}")
        
    print("\nSaving configuration to file...")
    output_data = {
        "motors": motors,
        "mixer_matrix": best_mixer.tolist(),
        "pid_gains": {
            "z": [best_Kp[0], best_Ki[0], best_Kd[0]],
            "roll": [best_Kp[1], best_Ki[1], best_Kd[1]],
            "pitch": [best_Kp[2], best_Ki[2], best_Kd[2]],
            "yaw": [best_Kp[3], best_Ki[3], best_Kd[3]]
        },
        "outer_gains": {
            "x": [best_outer[0], best_outer[1], best_outer[2]],
            "y": [best_outer[3], best_outer[4], best_outer[5]]
        },
        "max_int": {
            "z": best_max_int[0],
            "roll": best_max_int[1],
            "pitch": best_max_int[2],
            "yaw": best_max_int[3],
            "x": best_max_int[4],
            "y": best_max_int[5]
        }
    }
    out_path = os.path.join(out_dir, f"{xml_name}_opt.json")
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"-> Saved to '{out_path}'")
    
    print("\nLaunching Visualizer and generating plots...")
    visualize_best_flight(best_params, xml_name)

def visualize_best_flight(best_params, xml_name):
    Kp = best_params[0:4]
    Ki = best_params[4:8]
    Kd = best_params[8:12]
    Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y = best_params[12:18]
    max_int = best_params[18:24]
    mixer = B_pinv
    
    max_thrusts = model.actuator_ctrlrange[:, 1]
    min_thrusts = model.actuator_ctrlrange[:, 0]
    safe_max_thrusts = max_thrusts + 1e-6
    max_torque_available = np.sum(np.abs(mixer[1:4, :]) * max_thrusts, axis=1) / 2.0
    max_thrusts = model.actuator_ctrlrange[:, 1]
    min_thrusts = model.actuator_ctrlrange[:, 0]
    safe_max_thrusts = max_thrusts + 1e-6
    
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, TARGET_POS_Z_START]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    
    nv = model.nv
    M_dense = np.zeros((nv, nv))
    mujoco.mj_fullM(model, data, M_dense)
    mass = M_dense[0, 0]
    I_body = M_dense[3:6, 3:6]
    
    dt = model.opt.timestep
    steps = int(5.0 / dt) 
    
    delay_steps = int((drone_config["transport_delay_ms"] / 1000.0) / dt)
    delay_steps = max(1, delay_steps)
    alpha_motor = dt / (drone_config["motor_tau"] + dt)
    
    integral_error_z = 0.0
    integral_error_ang = np.zeros(3)
    integral_error_body_x = 0.0
    integral_error_body_y = 0.0
    
    base_thrusts = mixer @ np.array([mass * 9.81, 0.0, 0.0, 0.0])
    actual_motor_pwm = np.sign(base_thrusts) * np.sqrt(np.abs(base_thrusts) / safe_max_thrusts)
    actual_motor_pwm_arr = actual_motor_pwm.copy()
    action_queue = [actual_motor_pwm_arr.copy()] * delay_steps
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    history_t = []
    history_z = []
    history_target_z = []
    history_roll = []
    history_pitch = []
    history_yaw = []
    history_x = []
    history_target_x = []
    
    target_x = 1.0
    target_y = 0.0
    target_yaw_deg = 45.0
    
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, TARGET_POS_Z_START]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    
    for step in range(steps):
        t = step * dt
        t_rise = 2.0
        
        if t < t_rise:
            tau = t / t_rise
            s = 10*tau**3 - 15*tau**4 + 6*tau**5
            ds = (30*tau**2 - 60*tau**3 + 30*tau**4) / t_rise
            target_pos = np.array([target_x * s, target_y * s, TARGET_POS_Z_START + (TARGET_POS_Z_END - TARGET_POS_Z_START) * s])
            target_vel = np.array([target_x * ds, target_y * ds, (TARGET_POS_Z_END - TARGET_POS_Z_START) * ds])
            current_target_yaw = np.radians(target_yaw_deg) * s
            target_ang_vel = np.array([0.0, 0.0, np.radians(target_yaw_deg) * ds])
        else:
            target_pos = np.array([target_x, target_y, TARGET_POS_Z_END])
            target_vel = np.zeros(3)
            current_target_yaw = np.radians(target_yaw_deg)
            target_ang_vel = np.zeros(3)
            
        # Clean trajectory for testing (no chirp in optimizer test flight)
            
        pos = data.qpos[:3]
        quat = data.qpos[3:7]
        vel = data.qvel[:3]
        ang_vel = data.qvel[3:6]
        
        pos_err = target_pos - pos
        vel_err = target_vel - vel
        
        # Rotate world errors into body yaw frame
        w, x_quat, y_quat, z_quat = quat
        current_yaw_actual = np.arctan2(2*(w*z_quat + x_quat*y_quat), 1 - 2*(y_quat**2 + z_quat**2))
        cos_y = np.cos(current_yaw_actual)
        sin_y = np.sin(current_yaw_actual)
        
        pos_err_body_x = pos_err[0] * cos_y + pos_err[1] * sin_y
        pos_err_body_y = -pos_err[0] * sin_y + pos_err[1] * cos_y
        vel_err_body_x = vel_err[0] * cos_y + vel_err[1] * sin_y
        vel_err_body_y = -vel_err[0] * sin_y + vel_err[1] * cos_y
        
        dof_mask = drone_config["dof_mask"]
        max_tilt = np.radians(20.0)
        wrench_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        
        if dof_mask[0] == 1:
            wrench_dict[0] = (Kp_x * pos_err_body_x + Kd_x * vel_err_body_x) * mass
            target_pitch = np.radians(TARGET_PITCH_DEG)
        else:
            integral_error_body_x += pos_err_body_x * dt
            integral_error_body_x = np.clip(integral_error_body_x, -max_int[4], max_int[4])
            accel_x_cmd = Kp_x * pos_err_body_x + Ki_x * integral_error_body_x + Kd_x * vel_err_body_x
            target_pitch = np.clip(accel_x_cmd / 9.81, -max_tilt, max_tilt)
            
        if dof_mask[1] == 1:
            wrench_dict[1] = (Kp_y * pos_err_body_y + Kd_y * vel_err_body_y) * mass
            target_roll = np.radians(TARGET_ROLL_DEG)
        else:
            integral_error_body_y += pos_err_body_y * dt
            integral_error_body_y = np.clip(integral_error_body_y, -max_int[5], max_int[5])
            accel_y_cmd = Kp_y * pos_err_body_y + Ki_y * integral_error_body_y + Kd_y * vel_err_body_y
            target_roll = np.clip(-accel_y_cmd / 9.81, -max_tilt, max_tilt)
        
        cr, sr = np.cos(target_roll * 0.5), np.sin(target_roll * 0.5)
        cp, sp = np.cos(target_pitch * 0.5), np.sin(target_pitch * 0.5)
        cy, sy = np.cos(current_target_yaw * 0.5), np.sin(current_target_yaw * 0.5)

        target_quat = np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])
        
        q_err = quat_mult(quat_conj(quat), target_quat)
        if q_err[0] < 0: q_err = -q_err
        ang_err = 2.0 * q_err[1:]
        
        body_mat = data.xmat[drone_body_id].reshape(3, 3)
        target_ang_vel_body = body_mat.T @ target_ang_vel
        ang_vel_err = target_ang_vel_body - ang_vel
        
        integral_error_z = np.clip(integral_error_z + pos_err[2] * dt, -max_int[0], max_int[0])
        desired_accel_z = Kp[0]*pos_err[2] + Ki[0]*integral_error_z + Kd[0]*vel_err[2]
        
        up_world = body_mat @ np.array([0.0, 0.0, 1.0])
        desired_force_z = (desired_accel_z + 9.81) * mass / max(0.5, up_world[2])
        
        integral_error_ang = np.clip(integral_error_ang + ang_err * dt, -max_int[1:4], max_int[1:4])
        desired_ang_accel = Kp[1:]*ang_err + Ki[1:]*integral_error_ang + Kd[1:]*ang_vel_err
        
        max_ang_accel_limit = (max_torque_available / np.diag(I_body)) * 0.8
        desired_ang_accel[0] = np.clip(desired_ang_accel[0], -max_ang_accel_limit[0], max_ang_accel_limit[0])
        desired_ang_accel[1] = np.clip(desired_ang_accel[1], -max_ang_accel_limit[1], max_ang_accel_limit[1])
        desired_ang_accel[2] = np.clip(desired_ang_accel[2], -max_ang_accel_limit[2], max_ang_accel_limit[2])
        
        gyro_term = np.cross(ang_vel, I_body @ ang_vel)
        desired_torque_body = I_body @ desired_ang_accel + gyro_term
        
        wrench_dict[2] = desired_force_z
        wrench_dict[3] = desired_torque_body[0]
        wrench_dict[4] = desired_torque_body[1]
        wrench_dict[5] = desired_torque_body[2]
        
        active_dofs = np.where(dof_mask == 1)[0]
        
        # Attitude wrench (Roll, Pitch, Yaw)
        att_wrench = np.array([wrench_dict[dof] if dof >= 3 else 0.0 for dof in active_dofs])
        thrusts_att = B_pinv @ att_wrench
        
        max_att = np.max(np.abs(thrusts_att) / (safe_max_thrusts + 1e-6))
        if max_att > 1.0:
            thrusts_att /= max_att
            
        # Position wrench (X, Y, Z)
        pos_wrench = np.array([wrench_dict[dof] if dof < 3 else 0.0 for dof in active_dofs])
        thrusts_pos = B_pinv @ pos_wrench
        
        scale_pos = 1.0
        for i in range(num_motors):
            if thrusts_pos[i] > 0:
                margin = max_thrusts[i] - thrusts_att[i]
                if margin < 0: margin = 0
                if thrusts_pos[i] > margin:
                    scale_pos = min(scale_pos, margin / thrusts_pos[i])
            elif thrusts_pos[i] < 0:
                margin = min_thrusts[i] - thrusts_att[i]
                if margin > 0: margin = 0
                if thrusts_pos[i] < margin:
                    scale_pos = min(scale_pos, margin / thrusts_pos[i])
                    
        thrusts = thrusts_att + scale_pos * thrusts_pos
        thrusts = np.clip(thrusts, min_thrusts, max_thrusts) 
        
        pwm_cmd = np.sign(thrusts) * np.sqrt(np.abs(thrusts) / safe_max_thrusts)
        
        action_queue.append(pwm_cmd)
        delayed_pwm = action_queue.pop(0)
        
        actual_motor_pwm_arr = (1.0 - alpha_motor) * actual_motor_pwm_arr + alpha_motor * delayed_pwm
        physical_thrusts = np.sign(actual_motor_pwm_arr) * (actual_motor_pwm_arr ** 2) * max_thrusts
        
        data.ctrl[:] = physical_thrusts
        mujoco.mj_step(model, data)
        
        if step % (int(0.02 / dt)) == 0:
            history_t.append(t)
            history_x.append(pos[0])
            history_target_x.append(target_pos[0])
            history_z.append(pos[2])
            history_target_z.append(target_pos[2])
            
            w, x, y, z = quat
            actual_roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
            actual_pitch = np.arcsin(2*(w*y - z*x))
            actual_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
            history_roll.append(np.degrees(actual_roll))
            history_pitch.append(np.degrees(actual_pitch))
            history_yaw.append(np.degrees(actual_yaw))
            
            viewer.sync()
            time.sleep(0.02)
            
    viewer.close()

    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(history_t, history_z, label='Actual Z')
    plt.plot(history_t, history_target_z, 'r--', label='Target Z')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(history_t, history_roll, label='Roll')
    plt.plot(history_t, history_pitch, label='Pitch')
    plt.plot(history_t, history_yaw, label='Yaw')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(history_t, history_x, label='Actual X')
    plt.plot(history_t, history_target_x, 'r--', label='Target X')
    plt.ylabel('Position X (m)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    flight_plot_path = os.path.join("output", f"{xml_name}_flight.png")
    plt.savefig(flight_plot_path)
    print(f"-> Saved flight plot to '{flight_plot_path}'")

if __name__ == '__main__':
    main()
